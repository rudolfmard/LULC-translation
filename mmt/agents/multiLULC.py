#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Multiple land-cover/land-use Maps Translation (MMT)

Main agent. Performs training and testing of the auto-encoders on pair of land cover patches.
"""
import json
import os
import shutil
import time
import itertools

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
        self.rank = int(os.environ['RANK'])
        self.local_rank = int(os.environ['LOCAL_RANK'])
        print(f"Hello from process rank {self.rank}!")

        # Set device and RNG seed
        self.cuda = torch.cuda.is_available() & self.config.cuda
        self.manual_seed = self.config.seed
        if self.cuda:
            torch.cuda.manual_seed(self.manual_seed)
            self.device = torch.device(f"cuda:{self.local_rank}")
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
            rank=self.rank, # LUMI-multi-GPU: Pass the global rank of this processes to distribute data across all GPUs
            **self.config.dataloader.params)
        self.datasets = self.data_loader.datasets  # shortcut

        # Initialize counters
        self.current_epoch = 0
        self.current_iteration = 0
        self.best_metric = 0

        # Initialize a dictionary to hold loss values
        self.loss_log, self.loss_log_lookup_table = self.initialize_loss_log(key_lookup=True)

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
        self.models_wrapper = DDP(self.models_wrapper, device_ids=[self.local_rank], find_unused_parameters=True)

        # Define optimizers:
        optim_class = getattr(optim, self.config.optimizer.type)
        self.optimizers = [
            optim_class(net.parameters(), **self.config.optimizer.params)
            for net in self.models_wrapper.module.models
        ]
        if self.config.model.use_pos == "sinusoidal":
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
                # TODO: Possible bug! Currently only keep range(n_autoencoders) but are the AEs of interest actually the first n_autoencoders?
                n_autoencoders = len(self.models_wrapper.module.models)
                filtered_state_dict = {k: v for k, v in checkpoint["model"].items() if not k.startswith(f"models.") or any(k.startswith(f"models.{i}.") for i in range(n_autoencoders))}
                self.models_wrapper.module.load_state_dict(filtered_state_dict)
                self.logger.info("**Beginning of phase 2 of training**")
            else:
                self.models_wrapper.module.load_state_dict(checkpoint["model"])
                self.current_epoch = checkpoint["epoch"] + 1
                self.loss_log = checkpoint["loss_log"]
                if self.config.model.use_pos == "sinusoidal":
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
        if self.config.model.use_pos == "sinusoidal":
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
            """
            if self.rank == 0:
                self.test()
                torch.cuda.empty_cache()
            """
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
                # LUMI-multi-GPU: Check if this is the best model so far (based on total average loss)
                last_validation_loss = self.loss_log["validation"]["total_average"][-1][1]
                if last_validation_loss == min([item[1] for item in self.loss_log["validation"]["total_average"]]):
                    self.logger.info("Best model for now: saved")
                    self.save_checkpoint(is_best=1)
                torch.cuda.empty_cache()
                print(f"Rank 0 process spent {time.time()-rank_0_t} seconds validating")
        
            #TODO: Only plot every n epochs
            """
            if self.rank == 0 and self.current_epoch > 1:
                plot_loss(
                    self.loss_log["training"]["reconstruction"],
                    self.loss_log["validation"]["reconstruction"],
                    savefig=os.path.join(self.config.paths.out_dir, "reconstruction_loss.png"),
                )
            """
            
            # LUMI-multi-GPU: Save checkpoint only on process rank 0, after training and validation losses has been stored
            if self.rank == 0:
                self.save_checkpoint()
                print("Chekpoint saved!")

            # LUMI-multi-GPU: The processes with rank != 0 stop here, and continue when the rank 0 process reaches this point after validation/plotting losses
            dist.barrier()

            # Stop training if early stopping criterion is met:
            if self.early_stopping():
                print(f"TRAINING STOPPED AFTER {self.current_epoch+1} EPOCHS DUE TO EARLY STOPPING POLICY!")
                #TODO: plot the losses once more
                break
            self.current_epoch += 1
        if self.rank == 0:
            self.logger.info("Training ended!")

    @timeit
    def train_one_epoch(self) -> None:
        """One epoch of training"""

        # Initialize log for collecting running losses and loss item counts throughout the epoch:
        running_loss = self.initialize_loss_log(running=True)

        # LUMI-multi-GPU: Set the models to training mode:
        self.models_wrapper.train()
        """
        for model in self.models_wrapper.module.models:
            model.train()
        if self.config.model.use_pos == "sinusoidal":
            self.models_wrapper.module.coord_model.train()
        """

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

                    # If use_pos model config is "sinusoidal", pass sinusoidal encoding of coordinates to model instead of raw coordinates:
                    if self.config.model.use_pos == "sinusoidal":
                        coordinates = data.get("coordenc").float().to(self.device)
                    else:
                        coordinates = data.get("coordinate_tensor")
                    #TODO: Moving data to device should not matter, as dataloader already does it
                    source_patch = data.get("source_one_hot")#.to(self.device)
                    target_patch = data.get("target_one_hot")#.to(self.device)
                    sv = data.get("source_data")[:, 0]#.to(self.device)
                    tv = data.get("target_data")[:, 0]#.to(self.device)

                    self.optimizers[i_source].zero_grad(set_to_none=True)
                    self.optimizers[i_target].zero_grad(set_to_none=True)
                    if self.config.model.use_pos == "sinusoidal":
                        self.coord_optimizer.zero_grad(set_to_none=True)

                    ### LUMI-multi-GPU: Forward pass, call forward only on the AutoencoderWrapper
                    rec_source, rec_target, embedding_source, embedding_target, src_to_target, target_to_src = self.models_wrapper(i_source, i_target, source_patch, target_patch, coordinates)

                    # Calculate and add source reconstruction error to reconstruction loss:
                    loss_rec_source = torch.nn.CrossEntropyLoss(ignore_index=0)(rec_source, sv)        # self reconstruction loss

                    # Calculate and add target reconstruction error to reconstruction loss:
                    loss_rec_target = torch.nn.CrossEntropyLoss(ignore_index=0)(rec_target, tv)       # self reconstruction loss

                    # Calculate embedding loss:
                    loss_emb = torch.nn.MSELoss()(embedding_source, embedding_target)           # similar embedding loss

                    # Calculate translation loss from source->target:
                    loss_tra_src_to_target = torch.nn.CrossEntropyLoss(ignore_index=0)(src_to_target, tv)     # translation loss

                    # Calculate translation loss from target->source:
                    loss_tra_target_to_src = torch.nn.CrossEntropyLoss(ignore_index=0)(target_to_src, sv)    # translation loss

                    # Combine all losses:
                    loss_rec = loss_rec_source + loss_rec_target
                    loss_tra = loss_tra_src_to_target + loss_tra_target_to_src
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
                    if self.config.model.use_pos == "sinusoidal":
                        self.coord_optimizer.step()

                    # Accumulate the running losses and count loss items
                    # Loss averages:
                    running_loss["total_average"] = [running_loss["total_average"][0]+loss.item(), running_loss["total_average"][1]+1]
                    running_loss["reconstruction_average"] = [running_loss["reconstruction_average"][0]+loss_rec.item(), running_loss["reconstruction_average"][1]+1]
                    running_loss["embedding_average"] = [running_loss["embedding_average"][0]+loss_emb.item(), running_loss["embedding_average"][1]+1]
                    running_loss["translation_average"] = [running_loss["translation_average"][0]+loss_tra.item(), running_loss["translation_average"][1]+1]
                    # Model specific reconstruction losses:
                    rec_key_source = self.loss_log_lookup_table["reconstruction_keys"][i_source]
                    rec_key_target = self.loss_log_lookup_table["reconstruction_keys"][i_target]
                    running_loss["reconstruction"][rec_key_source] = [running_loss["reconstruction"][rec_key_source][0]+loss_rec_source.item(), running_loss["reconstruction"][rec_key_source][1]+1]
                    running_loss["reconstruction"][rec_key_target] = [running_loss["reconstruction"][rec_key_target][0]+loss_rec_target.item(), running_loss["reconstruction"][rec_key_target][1]+1]
                    # Model specific embedding losses:
                    embedding_key = self.loss_log_lookup_table["embedding_keys"][(i_source, i_target)]
                    running_loss["embedding"][embedding_key] = [running_loss["embedding"][embedding_key][0]+loss_emb.item(), running_loss["embedding"][embedding_key][1]+1]
                    # Model specific translation losses:
                    source_to_target_key = self.loss_log_lookup_table["translation_keys"][(i_source, i_target)]
                    target_to_source_key = self.loss_log_lookup_table["translation_keys"][(i_target, i_source)]
                    running_loss["translation"][source_to_target_key] = [running_loss["translation"][source_to_target_key][0]+loss_tra_src_to_target.item(), running_loss["translation"][source_to_target_key][1]+1]
                    running_loss["translation"][target_to_source_key] = [running_loss["translation"][target_to_source_key][0]+loss_tra_target_to_src.item(), running_loss["translation"][target_to_source_key][1]+1]
                if end:
                    break
            # One iteration here corresponds to fetching one batch from all soure-target dataloaders.
            self.current_iteration += 1
        # LUMI-multi-GPU: Store epoch average loss values to the loss_log:
        self.loss_log["training"]["total_average"].append([self.current_epoch+1, running_loss["total_average"][0]/running_loss["total_average"][1]])
        self.loss_log["training"]["reconstruction_average"].append([self.current_epoch+1, running_loss["reconstruction_average"][0]/running_loss["reconstruction_average"][1]])
        self.loss_log["training"]["embedding_average"].append([self.current_epoch+1, running_loss["embedding_average"][0]/running_loss["embedding_average"][1]])
        self.loss_log["training"]["translation_average"].append([self.current_epoch+1, running_loss["translation_average"][0]/running_loss["translation_average"][1]])
        for key in self.loss_log["training"]["reconstruction"].keys():
            self.loss_log["training"]["reconstruction"][key].append([self.current_epoch+1, running_loss["reconstruction"][key][0]/running_loss["reconstruction"][key][1]])
        for key in self.loss_log["training"]["embedding"].keys():
            self.loss_log["training"]["embedding"][key].append([self.current_epoch+1, running_loss["embedding"][key][0]/running_loss["embedding"][key][1]])
        for key in self.loss_log["training"]["translation"].keys():
            self.loss_log["training"]["translation"][key].append([self.current_epoch+1, running_loss["translation"][key][0]/running_loss["translation"][key][1]])
    
    def validate(self) -> None:
        """One cycle of model validation"""
        # Initialize log for collecting running losses and loss item counts throughout the epoch:
        running_loss = self.initialize_loss_log(running=True)

        loss_arrays = {d: [] for d in self.datasets}

        # LUMI-multi-GPU: Set the models to evaluation mode:
        self.models_wrapper.eval()
        """
        for model in self.models_wrapper.module.models:
            model.eval()
        if self.config.model.use_pos == "sinusoidal":
            self.models_wrapper.module.coord_model.eval()
        """

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
                        if self.config.model.use_pos == "sinusoidal":
                            coordinates = data.get("coordenc").float().to(self.device)
                        else:
                            coordinates = data.get("coordinate_tensor")
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
                        rec_source, rec_target, embedding_source, embedding_target, src_to_target, target_to_src = self.models_wrapper(i_source, i_target, source_patch, target_patch, coordinates)

                        """
                        loss = torch.nn.CrossEntropyLoss(ignore_index=0)(trad, torch.argmax(target_patch, 1)) # TODO: Make sure why argmax is here
                        """
                        # LUMI-multi-GPU: For plotting, the training and validation losses has to be the same.
                        # Calculate and add source reconstruction error to reconstruction loss:
                        loss_rec_source = torch.nn.CrossEntropyLoss(ignore_index=0)(rec_source, sv)        # self reconstruction loss
                        # Calculate and add target reconstruction error to reconstruction loss:
                        loss_rec_target = torch.nn.CrossEntropyLoss(ignore_index=0)(rec_target, tv)       # self reconstruction loss
                        # Calculate embedding loss:
                        loss_emb = torch.nn.MSELoss()(embedding_source, embedding_target)           # similar embedding loss
                        # Calculate translation loss from source->target:
                        loss_tra_src_to_target = torch.nn.CrossEntropyLoss(ignore_index=0)(src_to_target, tv)     # translation loss
                        # Calculate translation loss from target->source:
                        loss_tra_target_to_src = torch.nn.CrossEntropyLoss(ignore_index=0)(target_to_src, sv)    # translation loss
                        # Combine all losses:
                        loss_rec = loss_rec_source + loss_rec_target
                        loss_tra = loss_tra_src_to_target + loss_tra_target_to_src
                        loss = 0.75*loss_rec + 0.75*loss_emb + loss_tra

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
                        
                        # Accumulate the running losses and count loss items
                        # Loss averages:
                        running_loss["total_average"] = [running_loss["total_average"][0]+loss.item(), running_loss["total_average"][1]+1]
                        running_loss["reconstruction_average"] = [running_loss["reconstruction_average"][0]+loss_rec.item(), running_loss["reconstruction_average"][1]+1]
                        running_loss["embedding_average"] = [running_loss["embedding_average"][0]+loss_emb.item(), running_loss["embedding_average"][1]+1]
                        running_loss["translation_average"] = [running_loss["translation_average"][0]+loss_tra.item(), running_loss["translation_average"][1]+1]
                        # Model specific reconstruction losses:
                        rec_key_source = self.loss_log_lookup_table["reconstruction_keys"][i_source]
                        rec_key_target = self.loss_log_lookup_table["reconstruction_keys"][i_target]
                        running_loss["reconstruction"][rec_key_source] = [running_loss["reconstruction"][rec_key_source][0]+loss_rec_source.item(), running_loss["reconstruction"][rec_key_source][1]+1]
                        running_loss["reconstruction"][rec_key_target] = [running_loss["reconstruction"][rec_key_target][0]+loss_rec_target.item(), running_loss["reconstruction"][rec_key_target][1]+1]
                        # Model specific embedding losses:
                        embedding_key = self.loss_log_lookup_table["embedding_keys"][(i_source, i_target)]
                        running_loss["embedding"][embedding_key] = [running_loss["embedding"][embedding_key][0]+loss_emb.item(), running_loss["embedding"][embedding_key][1]+1]
                        # Model specific translation losses:
                        source_to_target_key = self.loss_log_lookup_table["translation_keys"][(i_source, i_target)]
                        target_to_source_key = self.loss_log_lookup_table["translation_keys"][(i_target, i_source)]
                        running_loss["translation"][source_to_target_key] = [running_loss["translation"][source_to_target_key][0]+loss_tra_src_to_target.item(), running_loss["translation"][source_to_target_key][1]+1]
                        running_loss["translation"][target_to_source_key] = [running_loss["translation"][target_to_source_key][0]+loss_tra_target_to_src.item(), running_loss["translation"][target_to_source_key][1]+1]
                    if end:
                        break
        # LUMI-multi-GPU: Store epoch average loss values to the loss_log:
        self.loss_log["validation"]["total_average"].append([self.current_epoch+1, running_loss["total_average"][0]/running_loss["total_average"][1]])
        self.loss_log["validation"]["reconstruction_average"].append([self.current_epoch+1, running_loss["reconstruction_average"][0]/running_loss["reconstruction_average"][1]])
        self.loss_log["validation"]["embedding_average"].append([self.current_epoch+1, running_loss["embedding_average"][0]/running_loss["embedding_average"][1]])
        self.loss_log["validation"]["translation_average"].append([self.current_epoch+1, running_loss["translation_average"][0]/running_loss["translation_average"][1]])
        for key in self.loss_log["validation"]["reconstruction"].keys():
            self.loss_log["validation"]["reconstruction"][key].append([self.current_epoch+1, running_loss["reconstruction"][key][0]/running_loss["reconstruction"][key][1]])
        for key in self.loss_log["validation"]["embedding"].keys():
            self.loss_log["validation"]["embedding"][key].append([self.current_epoch+1, running_loss["embedding"][key][0]/running_loss["embedding"][key][1]])
        for key in self.loss_log["validation"]["translation"].keys():
            self.loss_log["validation"]["translation"][key].append([self.current_epoch+1, running_loss["translation"][key][0]/running_loss["translation"][key][1]])

    def test(self) -> None:
        """Final testing on left-out dataset"""
        self.logger.info(
            f"Start testing on {len(self.data_loader.test_loader)} items..."
        )
        with torch.no_grad():
            ##### Read ground_truth_file
            self.load_checkpoint(default_bestmodel_filename)
            # LUMI-multi-GPU: Set the models to evaluation mode:
            self.models_wrapper.eval()
            """
            for model in self.models_wrapper.module.models:
                model.eval()
            if self.config.model.use_pos == "sinusoidal":
                self.models_wrapper.module.coord_model.eval()
            """

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
                        #pos_enc = data.get("coordenc").float().to(self.device)
                        if self.config.model.use_pos == "sinusoidal":
                            coordinates = data.get("coordenc").float().to(self.device)
                        else:
                            coordinates = data.get("coordinate_tensor")
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
                        rec_source, rec_target, embedding_source, embedding_target, src_to_target, target_to_src = self.models_wrapper(i_source, i_target, source_patch, target_patch, coordinates)
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

    def early_stopping(self) -> bool:
        """
        Inspects the epoch-wise validation losses
        and determines whether early stopping criterion is met.

        Constants:
            patience (int): Number of consecutive epochs without improvement (as defined by delta) in the validation loss before training stops.
            delta (float):  Minimum absolute change required in the validation loss to qualify as an improvement.
        Returns:
            stop_training (bool):  True if early stopping criterion is met, False otherwise.
        """
        patience = 10
        delta = 0.001

        validation_epoch_averages = self.loss_log["validation"]["total_average"]

        if len(validation_epoch_averages) <= patience:
            return False  # Not enough data to make decision

        # Best loss, excluding last n (=patience) elements:
        best_loss = min([i[1] for i in validation_epoch_averages[:-patience]])
        # Last n (=patience) elements:
        last_n = [i[1] for i in validation_epoch_averages[-patience:]]
        # Check if any of the last n (=patience) loss values is better (by >delta) than the best_loss:
        all_worse = all(loss > (best_loss - delta) for loss in last_n)

        return all_worse

    def initialize_loss_log(self, running=False, key_lookup=False) -> dict:
        """
        Creates and empty dict to hold loss values.

        Parameters:
            running (bool):    If True, initialize the lists as [0,0] for collecting running loss information. False by default.
        """
        dataset_names = [os.path.splitext(name)[0] for name in self.datasets]

        name_combinations_ = list(itertools.combinations(dataset_names, 2))
        name_combinations= [f"{d[0]}-{d[1]}" for d in name_combinations_]

        name_permutations_ = list(itertools.permutations(dataset_names, 2))
        name_permutations = [f"{d[0]}-{d[1]}" for d in name_permutations_]

        if running:
            loss_log = {
                "total_average": [0,0],
                "reconstruction_average": [0,0],
                "embedding_average": [0,0],
                "translation_average": [0,0],
                "reconstruction": {d: [0,0] for d in dataset_names},
                "embedding": {d: [0,0] for d in name_combinations},
                "translation": {d: [0,0] for d in name_permutations},
            }
        else:
            loss_log = {
                "training": {
                    "total_average": [],
                    "reconstruction_average": [],
                    "embedding_average": [],
                    "translation_average": [],
                    "reconstruction": {d: [] for d in dataset_names},
                    "embedding": {d: [] for d in name_combinations},
                    "translation": {d: [] for d in name_permutations},
            }, "validation": {
                    "total_average": [],
                    "reconstruction_average": [],
                    "embedding_average": [],
                    "translation_average": [],
                    "reconstruction": {d: [] for d in dataset_names},
                    "embedding": {d: [] for d in name_combinations},
                    "translation": {d: [] for d in name_permutations},
            }}

        if key_lookup:
            # A lookup table is created to get the names corresponding to each loss component to store.
            # Embedding loss:
            #   Calculated pair-wise but is permutation invariant, source-target ordering does not matter. The permutations of (source_index, target_index) point to the same pair or maps and share the name.
            # Translation loss: 
            #   Calculated pair-wise but depends on which input is source and which is target. Each pair of (source_index, target_index) and their permutations correspond to a unique loss component and name.
            dataset_indices = [dataset_names.index(d) for d in dataset_names]
            index_combinations = [(i,j) for i,j in list(itertools.combinations(dataset_indices, 2))]
            index_permutations = [(i,j) for i,j in list(itertools.permutations(dataset_indices, 2))]
            lookup_table = {
                "reconstruction_keys": {i: d for i, d in zip(dataset_indices, dataset_names)},
                "embedding_keys": {i: d for i, d in zip(index_combinations, name_combinations)},
                "translation_keys": {i: d for i, d in zip(index_permutations, name_permutations)},
            }
            for i, d in zip(index_combinations, name_combinations):
                lookup_table["embedding_keys"][(i[1],i[0])] = d
            return loss_log, lookup_table
        return loss_log

    def finalize(self) -> None:
        """Finalizes all the operations of the 2 Main classes of the process, the operator and the data loader"""
        self.logger.info("Please wait while finalizing the operation.. Thank you")
        torch.cuda.empty_cache()
        dist.destroy_process_group()
        if self.config.training.tensorboard:
            self.tensorboard_process.kill()
            self.summary_writer.close()
