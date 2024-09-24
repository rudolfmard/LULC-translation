#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Multiple land-cover/land-use Maps Translation (MMT)

Main agent. Performs training and testing of the auto-encoders on pair of land cover patches.
"""
import json
import os
import shutil

import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import confusion_matrix

from mmt.agents import base
from mmt.datasets import landcover_to_landcover
from mmt.graphs.models import attention_autoencoder, universal_embedding, position_encoding

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
        
        startfrom: str, optional
            The name of the experiment directory to start from. If None,
            the agent will start from scratch. Defaults to None.
        """
        super().__init__(config)

        # Set device and RNG seed
        self.cuda = torch.cuda.is_available() & self.config.cuda
        self.manual_seed = self.config.seed
        if self.cuda:
            torch.cuda.manual_seed(self.manual_seed)
            self.device = torch.device("cuda")
            self.logger.info("Program will run on *****GPU-CUDA***** ")
            # LUMI: Omit calling print_cuda_statistics() as LUMI is equipped with AMD hardware and doesn't provide all NVIDIA tools.
            #misc.print_cuda_statistics()
        else:
            self.device = torch.device("cpu")
            torch.manual_seed(self.manual_seed)
            self.logger.info("Program will run on *****CPU*****\n")

        # Define data_loader
        DataLoader = getattr(landcover_to_landcover, self.config.dataloader.type)
        self.data_loader = DataLoader(
            config=self.config, **self.config.dataloader.params
        )
        self.datasets = self.data_loader.datasets  # shortcut

        # Get required param for network initialisation
        input_channels = self.data_loader.input_channels
        output_channels = self.data_loader.output_channels
        resizes = self.config.dimensions.n_px_embedding // np.array(
            self.data_loader.real_patch_sizes
        )
        resizes = np.where(resizes == 1, None, resizes)

        # Define models
        if config.model.type == "universal_embedding":
            EncDec = getattr(universal_embedding, config.model.name)
        elif config.model.type == "attention_autoencoder":
            EncDec = getattr(attention_autoencoder, config.model.name)
        else:
            raise ValueError(
                f"Unknown model.type = {config.model.type}. Please change config to one among ['universal_embedding', 'attention_autoencoder']"
            )

        #LUMI: add the number of elements position encoding to input_channel
        if self.config.model.use_pos:
            pos_enc_dim = self.config.model.pos_enc_dim
        else:
            pos_enc_dim = 0

        self.models = [
            EncDec(
                in_channels=input_channel + pos_enc_dim,
                out_channels=output_channel,
                n_px_input=self.data_loader.real_patch_sizes[i_model],
                resize=resizes[i_model],
                n_px_embedding=self.config.dimensions.n_px_embedding,
                n_channels_hiddenlay=self.config.dimensions.n_channels_hiddenlay,
                n_channels_embedding=self.config.dimensions.n_channels_embedding,
                **self.config.model.params,
            )
            for i_model, (input_channel, output_channel) in enumerate(
                zip(input_channels, output_channels)
            )
        ]

        # LUMI: Added a new parameter dictating the dimensionality of the position encoding:
        #self.coord_model = position_encoding.PositionEncoder(n_channels_embedding=self.config.dimensions.n_channels_embedding)
        self.coord_model = position_encoding.PositionEncoder(n_channels_embedding=self.config.model.pos_enc_dim)

        # Define optimizer
        optim_class = getattr(optim, self.config.optimizer.type)
        self.coord_optimizer = optim_class(
            self.coord_model.parameters(), **self.config.optimizer.params
        )
        self.optimizers = [
            optim_class(net.parameters(), **self.config.optimizer.params)
            for i, net in enumerate(self.models)
        ]

        # Initialize counters
        self.current_epoch = 0
        self.current_iteration = 0
        self.best_metric = 0

        if self.cuda:
            self.models = [net.to(self.device) for net in self.models]
            self.coord_model = self.coord_model.to(self.device)
            # LUMI: Omit calling print_cuda_statistics() as LUMI is equipped with AMD hardware and doesn't provide all NVIDIA tools.
            #misc.print_cuda_statistics()

        # Model Loading from the latest checkpoint if not found start from scratch.
        if startfrom is None:
            checkpoint_filename = default_checkpoint_filename
        else:
            checkpoint_filename = os.path.join(
                self.config.paths.experiments_dir,
                startfrom,
                "checkpoints",
                default_checkpoint_filename,
            )

        self.load_checkpoint(checkpoint_filename)

        print("Let's use", torch.cuda.device_count(), "GPUs!")
        if self.cuda and torch.cuda.device_count() > 1:
            # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            self.models = [torch.nn.DataParallel(net) for net in self.models]
            # LUMI: also wrap DataParallel to coord_model
            self.coord_model = torch.nn.DataParallel(self.coord_model)

    def load_checkpoint(self, file_name) -> None:
        """Latest checkpoint loader


        Parameters
        ----------
        file_name: str
            Name of the checkpoint file
        """
        if os.path.isfile(file_name):
            filename = file_name
        else:
            filename = os.path.join(self.config.paths.checkpoint_dir, file_name)

        try:
            self.logger.info(f"Loading checkpoint '{filename}'")
            checkpoint = torch.load(filename)

            self.current_epoch = checkpoint["epoch"]
            self.current_iteration = checkpoint["iteration"]
            for i, d in enumerate(self.datasets):
                self.models[i].load_state_dict(checkpoint["encoder_state_dict_" + d])
                self.optimizers[i].load_state_dict(checkpoint["encoder_optimizer_" + d])
                if self.cuda and torch.cuda.device_count() > 1:
                    print("Let's use", torch.cuda.device_count(), "GPUs!")
                    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
                    self.models[i] = torch.nn.DataParallel(self.models[i])
                    self.coord_model = torch.nn.DataParallel(self.coord_model)
            self.manual_seed = checkpoint["manual_seed"]

            self.logger.info(
                "Checkpoint loaded successfully from '{}' at (epoch {}) at (iteration {})\n".format(
                    filename,
                    checkpoint["epoch"],
                    checkpoint["iteration"],
                )
            )
        except OSError as e:
            self.logger.info(
                "No checkpoint exists from '{}'. Skipping...".format(
                    self.config.paths.checkpoint_dir
                )
            )
            self.logger.info("**First time to train**")

    def save_checkpoint(
        self,
        file_name=default_checkpoint_filename,
        is_best=0,
    ) -> None:
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
        }
        for i, d in enumerate(self.datasets):
            state["encoder_optimizer_" + d] = self.optimizers[i].state_dict()
            state["coord_optimizer_" + d] = self.coord_optimizer.state_dict()
            if torch.cuda.device_count() > 1 and self.cuda:
                state["encoder_state_dict_" + d] = self.models[i].module.state_dict()
                state["image_state_dict_" + d] = self.coord_model.module.state_dict()
            else:
                state["encoder_state_dict_" + d] = self.models[i].state_dict()
                state["image_state_dict_" + d] = self.coord_model.state_dict()

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
            self.test()
            torch.cuda.empty_cache()
        except KeyboardInterrupt:
            self.logger.info("You have entered CTRL+C.. Wait to finalize")

    def train(self) -> None:
        """Main training loop"""
        loss_ref = 1000
        loss_log_training = {d: [] for d in self.datasets}
        loss_log_validation = {d: [] for d in self.datasets}

        self.logger.info("Start training !")
        for epoch in range(1, self.config.training.n_epochs + 1):
            self.logger.info("")
            self.logger.info(
                " ------- Training epoch {}/{} ({:.0f}%) ------- ".format(
                    epoch,
                    self.config.training.n_epochs,
                    100 * epoch / self.config.training.n_epochs,
                )
            )

            train_loss = self.train_one_epoch()

            for d, l in train_loss.items():
                loss_log_training[d].extend(l)

            torch.cuda.empty_cache()
            if epoch % self.config.training.validate_every == 0:
                self.logger.info(
                    " - - - - Validation epoch {}/{} ({:.0f}%) - - - - ".format(
                        epoch,
                        self.config.training.n_epochs,
                        100 * epoch / self.config.training.n_epochs,
                    )
                )

                validation_loss = self.validate()

                for d, l in validation_loss.items():
                    loss_log_validation[d].append([self.current_iteration, np.mean(l)])

                tmp = [v for v in validation_loss.values()]
                vl = np.mean([item for elem in tmp for item in elem])
                if vl < loss_ref:
                    self.logger.info("Best model for now  : saved ")
                    loss_ref = vl
                    self.save_checkpoint(is_best=1)
                
                torch.cuda.empty_cache()

            self.current_epoch += 1
            if epoch > 1 and epoch >= 2 * self.config.training.validate_every:
                plot_loss(
                    loss_log_training,
                    loss_log_validation,
                    savefig=os.path.join(self.config.paths.out_dir, "loss.png"),
                )
        self.logger.info("Training ended!")

    @timeit
    def train_one_epoch(self) -> dict:
        """One epoch of training"""
        loss_log = {d: [] for d in self.datasets}

        [model.train() for model in self.models]
        self.coord_model.train()

        batch_idx = 0
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
            for source, targetval in data_loader.items():
                i_source = self.datasets.index(source)
                for target, dl in targetval.items():
                    i_target = self.datasets.index(target)

                    ### Load data
                    try:
                        data = next(dl)
                        dlcount[dl] += 1
                    except:
                        end = True
                        break

                    pos_enc = data.get("coordenc").to(self.device)
                    # LUMI: also move all data below to device
                    source_patch = data.get("source_one_hot").to(self.device)
                    target_patch = data.get("target_one_hot").to(self.device)
                    sv = data.get("source_data")[:, 0].to(self.device)
                    tv = data.get("target_data")[:, 0].to(self.device)

                    self.optimizers[i_source].zero_grad(set_to_none=True)
                    self.coord_optimizer.zero_grad(set_to_none=True)
                    self.optimizers[i_target].zero_grad(set_to_none=True)

                    ### Forward pass

                    # Encode+Decode the source patches:
                    if self.config.model.use_pos:
                        # LUMI: Pass the raw output from coord_model to autoencoder (no unsqueezing here)
                        #pos_enc = (self.coord_model(pos_enc.float()).unsqueeze(2).unsqueeze(3))
                        pos_enc =  self.coord_model(pos_enc.float())
                        embedding, rec = self.models[i_source](source_patch, full=True, res=pos_enc)
                    else:
                        embedding, rec = self.models[i_source](source_patch, full=True)

                    # Calculate and add source reconstruction error to reconstruction loss:
                    loss_rec = torch.nn.CrossEntropyLoss(ignore_index=0)(rec, sv)  # self reconstruction loss

                    # Encode+Decode the target patches:
                    if self.config.model.use_pos:
                        embedding2, rec = self.models[i_target](target_patch, full=True, res=pos_enc)
                    else:
                        embedding2, rec = self.models[i_target](target_patch, full=True)

                    # Calculate and add target reconstruction error to reconstruction loss:
                    loss_rec += torch.nn.CrossEntropyLoss(ignore_index=0)(rec, tv)  # self reconstruction loss
                    
                    # Calculate embedding loss:
                    loss_emb = torch.nn.MSELoss()(embedding, embedding2)  # similar embedding loss

                    # Translation decode the source embedding:
                    if self.config.model.type == "attention_autoencoder":
                        rec = self.models[i_target].decoder(embedding)
                    else:
                        _, rec = self.models[i_target](embedding)

                    # Calculate translation loss from source->target:
                    loss_tra = torch.nn.CrossEntropyLoss(ignore_index=0)(rec, tv)  # translation loss

                    # Translation decode the target embedding:
                    if self.config.model.type == "attention_autoencoder":
                        rec = self.models[i_source].decoder(embedding2)
                    else:
                        _, rec = self.models[i_source](embedding2)
                    
                    # Calculate translation loss from target->source:
                    loss_tra += torch.nn.CrossEntropyLoss(ignore_index=0)(rec, sv)  # translation loss

                    # Combine all losses:
                    loss = loss_rec + loss_emb + loss_tra

                    if dlcount[dl] % self.config.training.print_inc == 0:
                        self.logger.info(
                            f"[ep {self.current_epoch}, i={self.current_iteration}][batch {dlcount[dl]}/{len(dl)}] train\t {source} -> {target} \t Losses: rec={loss_rec.item()}, emb={loss_emb.item()}, tra={loss_tra.item()}"
                        )

                    ### Backward propagation
                    loss.backward()
                    self.optimizers[i_source].step()
                    self.optimizers[i_target].step()
                    self.coord_optimizer.step()
                if end:
                    break
                batch_idx += 1

                loss_log[source].append([self.current_iteration, loss.item()])

            self.current_iteration += 1
        self.save_checkpoint()
        return loss_log

    def validate(self) -> dict:
        """One cycle of model validation"""
        loss_log = {d: [] for d in self.datasets}
        [model.eval() for model in self.models]
        self.coord_model.eval()

        test_loss = 0
        with torch.no_grad():
            im_save = {d: {j: 0 for j in self.datasets} for d in self.datasets}
            data_loader = {
                source: {target: iter(val) for target, val in targetval.items()}
                for source, targetval in self.data_loader.valid_loader.items()
            }
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
                        pos_enc = data.get("coordenc").to(self.device)
                        source_patch = data.get("source_one_hot")
                        target_patch = data.get("target_one_hot")

                        if self.config.model.use_pos:
                            pos_enc = (
                                self.coord_model(pos_enc.float())
                                .unsqueeze(2)
                                .unsqueeze(3)
                            )
                            embedding, rec = self.models[i_source](
                                source_patch.float(), full=True, res=pos_enc
                            )
                        else:
                            embedding, rec = self.models[i_source](
                                source_patch.float(), full=True
                            )

                        if self.config.model.type == "attention_autoencoder":
                            trad = self.models[i_target].decoder(embedding)
                        else:
                            _, trad = self.models[i_target](embedding)

                        loss = torch.nn.CrossEntropyLoss(ignore_index=0)(
                            trad, torch.argmax(target_patch, 1)
                        )

                        if im_save[source][target] == 0:
                            out_img = self.data_loader.plot_samples_per_epoch(
                                source_patch,
                                target_patch,
                                trad,
                                embedding,
                                source,
                                target,
                                self.current_epoch,
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
                                rec,
                                embedding,
                                source,
                                source,
                                self.current_epoch,
                                data.get("coordinate"),
                            )
                            im_save[source][source] = 1
                        loss_log[target].append(loss.item())

                if end:
                    break
        return loss_log

    def test(self) -> None:
        """Final testing on left-out dataset"""
        self.logger.info(
            f"Start testing on {len(self.data_loader.test_loader)} items..."
        )
        with torch.no_grad():
            ##### Read ground_truth_file
            self.load_checkpoint(default_bestmodel_filename)
            [model.eval() for model in self.models]
            self.coord_model.eval()

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
        if self.config.training.tensorboard:
            self.tensorboard_process.kill()
            self.summary_writer.close()
