#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Multiple land-cover/land-use Maps Translation (MMT)

Main agent. Performs training and testing of the auto-encoders on pair of land cover patches.
"""
import json
import os
import shutil
import time

import numpy as np
import torch
import torch.optim as optim
import torch.distributed as dist # LUMI-multi-GPU
from torch.nn.parallel import DistributedDataParallel as DDP # LUMI-multi-GPU
from sklearn.metrics import confusion_matrix

from mmt.agents import base
from mmt.datasets import landcover_to_landcover
from mmt.graphs.models import attention_autoencoder, universal_embedding, position_encoding, autoencoder_wrapper

from mmt.utils import misc, plt_utils

timeit = misc.timeit
plot_loss = plt_utils.plot_loss

default_bestmodel_filename = "model_best.ckpt"
default_checkpoint_filename = "checkpoint.ckpt"


class MultiLULCAgent(base.BaseAgent):
    """Multi Land Use/Land Cover agent: the main agent to train the auto-encoders doing map translation.
    
    Following the formatting of [this template](https://github.com/moemen95/Pytorch-Project-Template),
    the agent performs traing, validation and testing of the models.
    
    The models are auto-encoders. The input and output layers of these auto-encoders
    depend on the resolution of the map. Therefore, there is one model per map.
    However, the latent space is the same for all maps.
    
    The dataloaders are providing pairs of patches from two different maps
    (the target and the source), as further explained in mmt.datasets.landcover_to_landcover
    
    Also see mmt.agents.base to a lighter code with the same interface
    """
    def __init__(self, config, startfrom=None):
        """Build the agent according to the config.
        
        Instanciates models (one auto-encoder per map), optimizer and dataloaders.
        If startfrom is provided, the training starts from this checkpoint.


        Parameters
        ----------
        config: dict
            The configuration parameters for the agent.
        
        startfrom: str, optional    # TODO: Not accurate, used only for phase 2 when starting from other experiment. If None will resume from current experiment
            The name of the experiment directory to start from. If None,
            the agent will start from scratch. Defaults to None.
        """
        super().__init__(config)

        # LUMI-multi-GPU: fetch world_size and rank
        world_size = int(os.environ['WORLD_SIZE'])
        self.rank = int(os.environ['LOCAL_RANK'])
        print(f"Hello from process rank {self.rank}!")

        # Set device and RNG seed
        self.cuda = torch.cuda.is_available() & self.config.cuda
        self.manual_seed = self.config.seed
        if self.cuda:
            torch.cuda.manual_seed(self.manual_seed)
            self.device = torch.device(f"cuda:{self.rank}")
            self.logger.info("Program will run on *****GPU-CUDA***** ")
            print(f"| Number of GPUs: {torch.cuda.device_count()} | Number of processes: {world_size} |")
        else:
            self.device = torch.device("cpu")
            torch.manual_seed(self.manual_seed)
            self.logger.info("Program will run on *****CPU*****\n")

        # Define data_loader
        DataLoader = getattr(landcover_to_landcover, self.config.dataloader.type)
        self.data_loader = DataLoader(
            config=self.config,
            world_size=world_size, # LUMI-multi-GPU: Pass the world_size (number of processes) for data distribution
            rank=self.rank, # LUMI-multi-GPU: Pass the rank of this processes for data distribution
            **self.config.dataloader.params)
        self.datasets = self.data_loader.datasets  # shortcut

        # Initialize counters
        self.current_epoch = 0
        self.current_iteration = 0
        self.best_metric = 0

        # Initialize a dictionary to hold loss values
        self.loss_log = {
            "training": {
                "reconstruction": {d: [] for d in self.datasets},
                "total_average": [],
        }, "validation": {
                "reconstruction": {d: [] for d in self.datasets},
                "total_average": []
        }}

        # Get required param for network initialisation
        input_channels = self.data_loader.input_channels
        output_channels = self.data_loader.output_channels
        resizes = self.config.dimensions.n_px_embedding // np.array(self.data_loader.real_patch_sizes)
        resizes = np.where(resizes == 1, None, resizes)

        # Initialize the model
        self.models_wrapper = autoencoder_wrapper.AutoencoderWrapper(        
            in_channels=input_channels,
            out_channels=output_channels,
            n_px_inputs=self.data_loader.real_patch_sizes,
            resizes=resizes,
            config=config
        ).to(self.device)
        self.models_wrapper = DDP(self.models_wrapper, device_ids=[self.rank], find_unused_parameters=True)

        # Define optimizers:
        optim_class = getattr(optim, self.config.optimizer.type)
        self.optimizers = [
            optim_class(net.parameters(), **self.config.optimizer.params)
            for net in self.models_wrapper.module.models
        ]
        if self.config.model.use_pos:
            self.coord_optimizer = optim_class(
                self.models_wrapper.module.coord_model.parameters(), **self.config.optimizer.params
            )
        
        # Load checkpoints consecutively for each process:
        for i in range(0,world_size):
            if self.rank == i:
                self.load_checkpoint(startfrom)#checkpoint_filename)
                print(f"Checkpoint loaded for process rank {self.rank}")
            dist.barrier()

    def load_checkpoint(self, startfrom) -> None:
        """Latest checkpoint loader

        Parameters
        ----------
        file_name: str
            Name of the checkpoint file  #TODO: Update
        """

        # Start from scratch or resume training if checkpoint exists:
        filename = os.path.join(self.config.paths.checkpoint_dir, default_checkpoint_filename)
        begin_phase_2 = False
        if startfrom is not None and not os.path.isfile(filename):
            # If startfrom is specified and default checkpoint file is not found, begin phase 2. Otherwise resume from checkpoint or begin phase 1.
            filename = os.path.join(
                self.config.paths.experiments_dir,
                startfrom,
                "checkpoints",
                default_bestmodel_filename,
            )
            begin_phase_2 = True
        # Try to load and restore checkpoint:
        try:
            checkpoint = torch.load(filename)

            self.current_iteration = checkpoint["iteration"]    # Currently unused variable
            self.manual_seed = checkpoint["manual_seed"]

            if begin_phase_2:
                # In phase two, there are fewer autoencoders in the models_wrapper and the parameters of these excluded AEs has to be filtered out of the state_dict
                # Exclude parameters where the key starts with "model.i" where i is not an index of an existing AE in the ModuleList within models_wrapper
                # Phase 2 also resets the current_epoch counter, loss_log data and optimizer states, therefore these are not recovered.
                n_autoencoders = len(self.models_wrapper.module.models)
                filtered_state_dict = {k: v for k, v in checkpoint["model"].items() if not k.startswith(f"models.") or any(k.startswith(f"models.{i}.") for i in range(n_autoencoders))}
                self.models_wrapper.module.load_state_dict(filtered_state_dict)
                self.logger.info("**Beginning of phase 2 of training**")
            else:
                self.models_wrapper.module.load_state_dict(checkpoint["model"])
                self.current_epoch = checkpoint["epoch"] + 1
                self.loss_log = checkpoint["loss_log"]
                if self.config.model.use_pos:
                    self.coord_optimizer.load_state_dict(checkpoint["coord_optimizer"])
                for i, d in enumerate(self.datasets):
                    self.optimizers[i].load_state_dict(checkpoint["encoder_optimizer_" + d])

            self.logger.info(
                "Checkpoint loaded successfully from '{}' at (epoch {}) at (iteration {})\n".format(
                    filename,
                    checkpoint["epoch"]+1,
                    checkpoint["iteration"],
                )
            )
        # Loading the checkpoint failed, continue to train from scratch:
        except OSError as e:
            self.logger.info("No checkpoint exists at '{}'. Skipping...".format(filename))
            self.logger.info("**First time to train**")

    def save_checkpoint(self, file_name=default_checkpoint_filename, is_best=0,) -> None:
        """Checkpoint saver

        Parameters
        ----------
        file_name: str
            Name of the checkpoint file
        is_best: bool
            Flag to indicate whether current checkpoint's metric is the best so far
        """

        state = {
            "epoch": self.current_epoch,
            "iteration": self.current_iteration,
            "manual_seed": self.manual_seed,
            "loss_log": self.loss_log
        }

        state["model"] = self.models_wrapper.module.state_dict()
        if self.config.model.use_pos:
            state["coord_optimizer"] = self.coord_optimizer.state_dict()
        for i, d in enumerate(self.datasets):
            state["encoder_optimizer_" + d] = self.optimizers[i].state_dict()

        # Save the state
        torch.save(state, os.path.join(self.config.paths.checkpoint_dir, file_name))
        # If it is the best copy it to another file 'model_best.ckpt'
        if is_best:
            shutil.copyfile(
                os.path.join(self.config.paths.checkpoint_dir, file_name),
                os.path.join(
                    self.config.paths.checkpoint_dir, default_bestmodel_filename
                ),
            )

    def run(self) -> None:
        """The main operator"""
        try:
            torch.cuda.empty_cache()
            self.train()
            torch.cuda.empty_cache()
            if self.rank == 0:
                self.test()
                torch.cuda.empty_cache()
        except KeyboardInterrupt:
            self.logger.info("You have entered CTRL+C.. Wait to finalize")

    def train(self) -> None:
        """Main training loop"""
        loss_ref = 1000

        #for epoch in range(1, self.config.training.n_epochs + 1):
        for epoch in range(self.current_epoch+1, self.config.training.n_epochs+1): # LUMI-multi-GPU: Start epoch from current_epoch, Start from 1 for more intuitive logs
            if self.rank == 0:
                self.logger.info("")
                self.logger.info(
                    " ------- Training epoch {}/{} ({:.0f}%) ------- ".format(
                        epoch,
                        self.config.training.n_epochs,
                        100 * epoch / self.config.training.n_epochs,
                    )
                )

            # LUMI-multi-GPU: Train for one epoch
            self.train_one_epoch()
            torch.cuda.empty_cache()
            # LUMI-multi-GPU: Validate the model only in the rank 0 process.
            if self.rank == 0 and epoch % self.config.training.validate_every == 0:
                rank_0_t = time.time()
                self.logger.info(
                    " - - - - Validation epoch {}/{} ({:.0f}%) - - - - ".format(
                        epoch,
                        self.config.training.n_epochs,
                        100 * epoch / self.config.training.n_epochs,
                    )
                )
                # LUMI-multi-GPU: Validate the model
                self.validate()
                print(f"Rank 0 process spent {time.time()-rank_0_t} seconds validating")
                # LUMI-multi-GPU: Check if this is the best model so far (based on total average loss)
                last_validation_loss = self.loss_log["validation"]["total_average"][-1][1]
                if last_validation_loss == min([item[1] for item in self.loss_log["validation"]["total_average"]]):
                    self.logger.info("Best model for now: saved")
                    self.save_checkpoint(is_best=1)
                torch.cuda.empty_cache()
            
            # LUMI-multi-GPU: Check if the training and validation loss logging lists has enough data (>1 data points) to plot and proceed accordingly:
            enough_data = all(len(self.loss_log["training"]["reconstruction"][d])>1 for d in self.datasets) and all(len(self.loss_log["validation"]["reconstruction"][d])>1 for d in self.datasets)
            if self.rank == 0 and enough_data:
                plot_loss(
                    self.loss_log["training"]["reconstruction"],
                    self.loss_log["validation"]["reconstruction"],
                    savefig=os.path.join(self.config.paths.out_dir, "reconstruction_loss.png"),
                )
            
            # LUMI-multi-GPU: Save checkpoint only on process rank 0, after training and validation losses has been stored
            if self.rank == 0:
                self.save_checkpoint()
                print("Chekpoint saved!")

            self.current_epoch += 1

            # LUMI-multi-GPU: The processes with rank != 0 stop here, and continue when the rank 0 process reaches this point after validation/plotting losses
            dist.barrier()
        if self.rank == 0:
            self.logger.info("Training ended!")

    @timeit
    def train_one_epoch(self) -> None:
        """One epoch of training"""
        loss_arrays = {d: [] for d in self.datasets}

        # LUMI-multi-GPU: Set the models to training mode:
        for model in self.models_wrapper.module.models:
            model.train()
        if self.config.model.use_pos:
            self.models_wrapper.module.coord_model.train()

        # LUMI-multi-GPU: call set_epoch on the DistributedSampler
        for _, targetval in self.data_loader.train_loader.items():
            for _, val in targetval.items():
                val.sampler.set_epoch(self.current_epoch)

        data_loader = {
            source: {target: iter(val) for target, val in targetval.items()}
            for source, targetval in self.data_loader.train_loader.items()
        }

        dlcount = {}
        for source, targetval in data_loader.items():
            for target, dl in targetval.items():
                dlcount[dl] = 0

        epoch_running_loss, n_loss_items = 0, 0

        end = False
        while not end:
            for source, targetval in data_loader.items():   # Iterate over source maps
                i_source = self.datasets.index(source)
                for target, dl in targetval.items():        # Iterate over target maps of current source
                    i_target = self.datasets.index(target)

                    ### Load data
                    try:
                        data = next(dl)                     # Fetch one batch corresponding to the current source and target map
                        dlcount[dl] += 1
                    except:
                        end = True
                        break

                    pos_enc = data.get("coordenc").float().to(self.device)
                    # LUMI: also move all data below to device
                    source_patch = data.get("source_one_hot").to(self.device)
                    target_patch = data.get("target_one_hot").to(self.device)
                    sv = data.get("source_data")[:, 0].to(self.device)
                    tv = data.get("target_data")[:, 0].to(self.device)

                    self.optimizers[i_source].zero_grad(set_to_none=True)
                    self.optimizers[i_target].zero_grad(set_to_none=True)
                    if self.config.model.use_pos:
                        self.coord_optimizer.zero_grad(set_to_none=True)

                    ### LUMI-multi-GPU: Forward pass, call forward only on the AutoencoderWrapper
                    rec_source, rec_target, embedding_source, embedding_target, src_to_target, target_to_src = self.models_wrapper(i_source, i_target, source_patch, target_patch, pos_enc)

                    # Calculate and add source reconstruction error to reconstruction loss:
                    loss_rec_source = torch.nn.CrossEntropyLoss(ignore_index=0)(rec_source, sv)        # self reconstruction loss

                    # Calculate and add target reconstruction error to reconstruction loss:
                    loss_rec_target = torch.nn.CrossEntropyLoss(ignore_index=0)(rec_target, tv)       # self reconstruction loss

                    loss_rec = loss_rec_source + loss_rec_target

                    # Calculate embedding loss:
                    loss_emb = torch.nn.MSELoss()(embedding_source, embedding_target)           # similar embedding loss

                    # Calculate translation loss from source->target:
                    loss_tra = torch.nn.CrossEntropyLoss(ignore_index=0)(src_to_target, tv)     # translation loss

                    # Calculate translation loss from target->source:
                    loss_tra += torch.nn.CrossEntropyLoss(ignore_index=0)(target_to_src, sv)    # translation loss

                    # Combine all losses:
                    loss = loss_rec + loss_emb + loss_tra

                    # Divide rec and trans losses by two as they are sum of two losses?
                    if self.rank == 0 and dlcount[dl] % self.config.training.print_inc == 0:
                        self.logger.info(
                            f"[ep {self.current_epoch+1}, i={self.current_iteration}][batch {dlcount[dl]}/{len(dl)}] train\t {source} -> {target} \t Losses: rec={loss_rec.item()}, emb={loss_emb.item()}, tra={loss_tra.item()}"
                        )

                    ### Backward propagation
                    loss.backward()
                    self.optimizers[i_source].step()
                    self.optimizers[i_target].step()
                    if self.config.model.use_pos:
                        self.coord_optimizer.step()

                    # Accumulate the running loss and count loss items:
                    epoch_running_loss += loss.item()
                    n_loss_items += 1

                    # LUMI-multi-GPU: collect model specific loss data, currently collects reconstruction losses.
                    loss_arrays[source].append(loss_rec_source.item())
                    loss_arrays[target].append(loss_rec_target.item())
                if end:
                    break
            # One iteration here corresponds to fetching one batch from all soure-target dataloaders.
            self.current_iteration += 1
        # LUMI-multi-GPU: Store loss values to the loss_log:
        for d, l in loss_arrays.items():
            self.loss_log["training"]["reconstruction"][d].append([self.current_epoch+1, np.mean(l)])
        self.loss_log["training"]["total_average"].append([self.current_epoch+1, epoch_running_loss/n_loss_items])
    
    def validate(self) -> None:
        """One cycle of model validation"""
        loss_arrays = {d: [] for d in self.datasets}

        # LUMI-multi-GPU: Set the models to evaluation mode:
        for model in self.models_wrapper.module.models:
            model.eval()
        if self.config.model.use_pos:
            self.models_wrapper.module.coord_model.eval()

        test_loss = 0
        with torch.no_grad():
            im_save = {d: {j: 0 for j in self.datasets} for d in self.datasets}
            data_loader = {
                source: {target: iter(val) for target, val in targetval.items()}
                for source, targetval in self.data_loader.valid_loader.items()
            }
            epoch_running_loss, n_loss_items = 0, 0
            end = False
            while not end:
                for source, targetval in data_loader.items():
                    i_source = self.datasets.index(source)
                    for target, dl in targetval.items():
                        i_target = self.datasets.index(target)
                        try:
                            data = next(dl)
                        except:
                            end = True
                            break
                        pos_enc = data.get("coordenc").float().to(self.device)
                        source_patch = data.get("source_one_hot").to(self.device)
                        target_patch = data.get("target_one_hot").to(self.device)
                        sv = data.get("source_data")[:, 0].to(self.device)
                        tv = data.get("target_data")[:, 0].to(self.device)

                        """
                        if self.config.model.use_pos:
                            #pos_enc = (self.coord_model(pos_enc.float()).unsqueeze(2).unsqueeze(3))
                            #embedding, rec = self.models[i_source](source_patch.float(), full=True, res=pos_enc)
                            pos_enc =  self.coord_model(pos_enc.float())
                            embedding, rec = self.models[i_source](source_patch.float(), full=True, res=pos_enc)
                        else:
                            embedding, rec = self.models[i_source](source_patch.float(), full=True)

                        if self.config.model.type == "attention_autoencoder":
                            trad = self.models[i_target].decoder(embedding)
                        else:
                            _, trad = self.models[i_target](embedding)
                        """
                        # LUMI-multi-GPU:
                        rec_source, rec_target, embedding_source, embedding_target, src_to_target, target_to_src = self.models_wrapper(i_source, i_target, source_patch, target_patch, pos_enc)

                        """
                        loss = torch.nn.CrossEntropyLoss(ignore_index=0)(trad, torch.argmax(target_patch, 1)) # TODO: Make sure why argmax is here
                        """
                        # LUMI-multi-GPU: For plotting, the training and validation losses has to be the same.
                        # Calculate and add source reconstruction error to reconstruction loss:
                        loss_rec_source = torch.nn.CrossEntropyLoss(ignore_index=0)(rec_source, sv)        # self reconstruction loss
                        # Calculate and add target reconstruction error to reconstruction loss:
                        loss_rec_target = torch.nn.CrossEntropyLoss(ignore_index=0)(rec_target, tv)       # self reconstruction loss
                        loss_rec = loss_rec_source + loss_rec_target
                        # Calculate embedding loss:
                        loss_emb = torch.nn.MSELoss()(embedding_source, embedding_target)           # similar embedding loss
                        # Calculate translation loss from source->target:
                        loss_tra = torch.nn.CrossEntropyLoss(ignore_index=0)(src_to_target, tv)     # translation loss
                        # Calculate translation loss from target->source:
                        loss_tra += torch.nn.CrossEntropyLoss(ignore_index=0)(target_to_src, sv)    # translation loss
                        # Combine all losses:
                        loss = loss_rec + loss_emb + loss_tra

                        if im_save[source][target] == 0:
                            out_img = self.data_loader.plot_samples_per_epoch(
                                source_patch,
                                target_patch,
                                #trad,
                                src_to_target,
                                #embedding,
                                embedding_source,
                                source,
                                target,
                                self.current_epoch+1,
                                data.get("coordinate"),
                            )
                            im_save[source][target] = 1
                            self.logger.info(
                                f"Figure saved (patch plot {source} -> {target})"
                            )
                        if im_save[source][source] == 0:
                            out_img = self.data_loader.plot_samples_per_epoch(
                                source_patch,
                                source_patch,
                                #rec,
                                rec_source,
                                #embedding,
                                embedding_source,
                                source,
                                source,
                                self.current_epoch+1,
                                data.get("coordinate"),
                            )
                            im_save[source][source] = 1
                        
                        # Accumulate the running loss and count loss items:
                        epoch_running_loss += loss.item()
                        n_loss_items += 1

                        # LUMI-multi-GPU: Collect reconstruction losses for each model 
                        loss_arrays[source].append(loss_rec_source.item())
                        loss_arrays[target].append(loss_rec_target.item())
                    if end:
                        break
        for d, l in loss_arrays.items():
            self.loss_log["validation"]["reconstruction"][d].append([self.current_epoch+1, np.mean(l)])
        self.loss_log["validation"]["total_average"].append([self.current_epoch+1, epoch_running_loss/n_loss_items])

    def test(self) -> None:
        """Final testing on left-out dataset"""
        self.logger.info(
            f"Start testing on {len(self.data_loader.test_loader)} items..."
        )
        with torch.no_grad():
            ##### Read ground_truth_file
            self.load_checkpoint(default_bestmodel_filename)
            # LUMI-multi-GPU: Set the models to evaluation mode:
            for model in self.models_wrapper.module.models:
                model.eval()
            if self.config.model.use_pos:
                self.models_wrapper.module.coord_model.eval()

            res_oa = {d: {j: [0, 0] for j in self.datasets} for d in self.datasets}
            conf_matrix = {
                d: {
                    j: np.zeros(
                        (
                            self.data_loader.n_classes[j] + 1,
                            self.data_loader.n_classes[j] + 1,
                        )
                    )
                    for j in self.datasets
                }
                for d in self.datasets
            }

            for source, targetval in self.data_loader.test_loader.items():
                i_source = self.datasets.index(source)
                for target, val in targetval.items():
                    i_target = self.datasets.index(target)
                    for nb_it, data in enumerate(val):
                        pos_enc = data.get("coordenc").float().to(self.device)
                        source_patch = data.get("source_one_hot").to(self.device)
                        target_patch = data.get("target_one_hot").to(self.device)
                        sv = data.get("source_data")[:, 0].to(self.device)
                        tv = data.get("target_data")[:, 0].to(self.device)
                        """
                        pos_enc = data.get("coordenc").to(self.device)
                        source_patch = data.get("source_one_hot")
                        tv = data.get("target_data")[:, 0]

                        if self.config.model.use_pos:
                            pos_enc = (
                                self.coord_model(pos_enc.float())
                                .unsqueeze(2)
                                .unsqueeze(3)
                            )
                            embedding, _ = self.models[i_source](
                                source_patch.float(), full=True, res=pos_enc
                            )
                        else:
                            embedding, _ = self.models[i_source](
                                source_patch.float(), full=True
                            )

                        if self.config.model.type == "attention_autoencoder":
                            trad = self.models[i_target].decoder(embedding)
                        else:
                            _, trad = self.models[i_target](embedding)
                        """
                        # LUMI-multi-GPU:
                        rec_source, rec_target, embedding_source, embedding_target, src_to_target, target_to_src = self.models_wrapper(i_source, i_target, source_patch, target_patch, pos_enc)
                        embedding = embedding_source
                        trad = src_to_target

                        y_pred = torch.argmax(trad, dim=1)

                        y_pred = y_pred.int().view(-1).cpu().numpy()
                        y_targ = tv.int().view(-1).cpu().numpy()

                        y_pred = y_pred[y_targ != 0]
                        y_targ = y_targ[y_targ != 0]

                        where_id = y_pred == y_targ
                        T = np.sum(where_id)
                        nb = len(y_pred)

                        res_oa[source][target][0] += T
                        res_oa[source][target][1] += nb

                        labels = range(self.data_loader.n_classes[target] + 1)
                        conf_matrix[source][target] += confusion_matrix(
                            y_targ, y_pred, labels=labels
                        )

            res = {
                d: {
                    j: res_oa[d][j][0] / (res_oa[d][j][1] + 0.00001)
                    for j in self.datasets
                }
                for d in self.datasets
            }
            with open(
                os.path.join(self.config.paths.out_dir, "accuracy_assessement.json"),
                "w",
            ) as fp:
                json.dump(res, fp)

            res = {
                d: {j: conf_matrix[d][j].tolist() for j in self.datasets}
                for d in self.datasets
            }
            with open(
                os.path.join(
                    self.config.paths.out_dir, "per_class_accuracy_assessement.json"
                ),
                "w",
            ) as fp:
                json.dump(res, fp)

            plt_utils.PltPerClassMetrics()(
                conf_matrix,
                savefig=os.path.join(self.config.paths.out_dir, "per_class"),
            )

    def finalize(self) -> None:
        """Finalizes all the operations of the 2 Main classes of the process, the operator and the data loader"""
        self.logger.info("Please wait while finalizing the operation.. Thank you")
        torch.cuda.empty_cache()
        dist.destroy_process_group()
        if self.config.training.tensorboard:
            self.tensorboard_process.kill()
            self.summary_writer.close()
