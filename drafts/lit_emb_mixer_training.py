#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Multiple land-cover/land-use Maps Translation (MMT)

Training embedding mixers with Lightning
"""

import os
import torch
import lightning.pytorch as pl

from mmt import _repopath_ as mmt_repopath
from mmt.datasets import landcover_to_landcover
from mmt.datasets import transforms as mmt_transforms
from mmt.utils import config as utilconf
from mmt.inference import io
from mmt.graphs.models import embedding_mixer


# Configs
#------------
device = torch.device("cuda")
batch_size = 16
n_epochs = 150

config = utilconf.get_config(
    os.path.join(
        mmt_repopath,
        "configs",
        "new_config_template.yaml",
    )
)


# Loading models
#----------------
xp_name = "vanilla"

# Pytorch
esawc_encoder = io.load_pytorch_model(xp_name, lc_in = "esawc", lc_out = "encoder")
ecosg_encoder = io.load_pytorch_model(xp_name, lc_in = "ecosg", lc_out = "encoder")
esgp_encoder = io.load_pytorch_model(xp_name, lc_in = "esgp", lc_out = "encoder")
esgp_decoder = io.load_pytorch_model(xp_name, lc_in = "esgp", lc_out = "decoder")

esawc_encoder.to(device)
ecosg_encoder.to(device)
esgp_encoder.to(device)
esgp_decoder.to(device)

# Non-trained model
emb_mixer = embedding_mixer.MLP(n_channels_embedding = config.dimensions.n_channels_embedding, h_channels = 64)

esawc_transform = mmt_transforms.OneHot(13, device = device)
ecosg_transform = mmt_transforms.OneHot(35, device = device)
esgp_transform = mmt_transforms.OneHot(35, device = device)

# Define the Lightning module
#----------------
class LitEmbMixer(pl.LightningModule):
    def __init__(self, emb_mixer):
        super().__init__()
        self.emb_mixer = emb_mixer
    
    def training_step(self, batch, batch_idx):
        x_esawc = batch["esawc"]
        x_ecosg = batch["ecosg"]
        y = batch["esgp"]
        
        x_esawc = esawc_transform(x_esawc)
        x_ecosg = ecosg_transform(x_ecosg)
        x_esgp = esgp_transform(y)
        
        with torch.no_grad():
            emba = esawc_encoder(x_esawc.float())
            embo = ecosg_encoder(x_ecosg.float())
            embp = esgp_encoder(x_esgp.float())
            
        embc = torch.cat([emba, embo], dim = 1)
        emb = self.emb_mixer(embc)
        loss_emb = torch.nn.MSELoss()(emb, embp)
        
        with torch.no_grad():
            y_pred = esgp_decoder(emb)
        
        loss_tra = torch.nn.CrossEntropyLoss(ignore_index=0)(y_pred, y)
        self.log("train_loss_emb", loss_emb)
        self.log("train_loss_tra", loss_tra)
        return loss_emb + loss_tra
    
    def validation_step(self, batch, batch_idx):
        x_esawc = batch["esawc"]
        x_ecosg = batch["ecosg"]
        y = batch["esgp"]
        
        x_esawc = esawc_transform(x_esawc)
        x_ecosg = ecosg_transform(x_ecosg)
        x_esgp = esgp_transform(y)
        
        with torch.no_grad():
            emba = esawc_encoder(x_esawc.float())
            embo = ecosg_encoder(x_ecosg.float())
            embp = esgp_encoder(x_esgp.float())
            
        embc = torch.cat([emba, embo], dim = 1)
        emb = self.emb_mixer(embc)
        loss_emb = torch.nn.MSELoss()(emb, embp)
        
        with torch.no_grad():
            y_pred = esgp_decoder(emb)
        
        loss_tra = torch.nn.CrossEntropyLoss(ignore_index=0)(y_pred, y)
        self.log("val_loss_emb", loss_emb)
        self.log("val_loss_tra", loss_tra)
        return loss_emb + loss_tra
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    

# Instanciate
#-------------
model = LitEmbMixer(emb_mixer)
train_ds = landcover_to_landcover.EEEmapsDataset(path = os.path.join(config.paths.data_dir, "hdf5_data"), mode = "train")
train_dl = torch.utils.data.DataLoader(train_ds, batch_size = batch_size, shuffle = True, num_workers = 8, pin_memory = True)
val_ds = landcover_to_landcover.EEEmapsDataset(path = os.path.join(config.paths.data_dir, "hdf5_data"), mode = "val")
val_dl = torch.utils.data.DataLoader(val_ds, batch_size = batch_size, shuffle = False, num_workers = 8, pin_memory = True)

trainer = pl.Trainer(default_root_dir=os.path.join(mmt_repopath, "experiments"), max_epochs=n_epochs, accelerator='gpu', devices=1)


# Launch training
#-------------
trainer.fit(model, train_dataloaders = train_dl, val_dataloaders = val_dl)
print("Training complete.")

# Save model
#-------------
checkpoint_path = os.path.join(
    mmt_repopath,
    "experiments",
    xp_name,
    "checkpoints",
    "emb_mixer_state_dict.pt",
)
torch.save(model.emb_mixer.state_dict(), checkpoint_path)
print(f"State dict saved in {checkpoint_path}.")

emb_mixer = embedding_mixer.MLP(n_channels_embedding = config.dimensions.n_channels_embedding, h_channels = 64)
state_dict = torch.load(checkpoint_path)
emb_mixer.load_state_dict(state_dict)
