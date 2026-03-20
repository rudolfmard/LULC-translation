import torch
import torch.nn as nn

from mmt.graphs.models import attention_autoencoder, universal_embedding, position_encoding

class AutoencoderWrapper(nn.Module):
    # Wrapper for holding all autoencoders to perform parallel training
    # Accept the same arguments as EncDec modules
    def __init__(
        self,
        in_channels,
        out_channels,
        n_px_inputs,
        resizes,
        config,
    ):
        super(AutoencoderWrapper, self).__init__()

        self.config = config

        # Define models:
        if config.model.type == "universal_embedding":
            EncDec = getattr(universal_embedding, config.model.name)
        elif config.model.type == "attention_autoencoder":
            EncDec = getattr(attention_autoencoder, config.model.name)
        else:
            raise ValueError(
                f"Unknown model.type = {config.model.type}. Please change config to one among ['universal_embedding', 'attention_autoencoder']"
            )
        
        # Initialize autoencoders:
        self.models = nn.ModuleList(
            [EncDec(
                in_channels=input_channel,
                out_channels=output_channel,
                n_px_input=n_px_inputs[i_model],
                resize=resizes[i_model],
                n_px_embedding=config.dimensions.n_px_embedding,
                n_channels_hiddenlay=config.dimensions.n_channels_hiddenlay,
                n_channels_embedding=config.dimensions.n_channels_embedding,
                use_pos=config.model.use_pos,
                **config.model.params,
            )
            for i_model, (input_channel, output_channel) in enumerate(zip(in_channels, out_channels))]
        )

    def forward(self, i_source, i_target, source_patch, target_patch, coordinates):
        """
        Forward is always performed between two autoencoders
        """

        # Encode+Decode the source patches:
        if self.config.model.use_pos in ["embed_layer", "sinusoidal", "elevation_and_coordinates"]:
            embedding_source, rec_source = self.models[i_source](source_patch, full=True, coordinates=coordinates)
        else:
            embedding_source, rec_source = self.models[i_source](source_patch, full=True)

        # Encode+Decode the target patches:
        if self.config.model.use_pos in ["embed_layer", "sinusoidal", "elevation_and_coordinates"]:
            embedding_target, rec_target = self.models[i_target](target_patch, full=True, coordinates=coordinates)
        else:
            embedding_target, rec_target = self.models[i_target](target_patch, full=True)

        # Translation decode the source embedding:
        if self.config.model.type == "attention_autoencoder":
            src_to_target = self.models[i_target].decoder(embedding_source)
        else:
            _, src_to_target = self.models[i_target](embedding_source)

        # Translation decode the target embedding:
        if self.config.model.type == "attention_autoencoder":
            target_to_src = self.models[i_source].decoder(embedding_target)
        else:
            _, target_to_src = self.models[i_source](embedding_target)
        
        return rec_source, rec_target, embedding_source, embedding_target, src_to_target, target_to_src