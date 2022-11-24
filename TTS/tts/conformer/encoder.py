import torch
import torch.nn as nn

from TTS.tts.conformer.conformer import Encoder

"""
    adim: 384         # attention dimension
    aheads: 2         # number of attention heads
    elayers: 4        # number of encoder layers
    eunits: 1536      # number of encoder ff units
    dlayers: 4        # number of decoder layers
    dunits: 1536      # number of decoder ff units
    positionwise_layer_type: conv1d   # type of position-wise layer
    positionwise_conv_kernel_size: 3  # kernel size of position wise conv layer
    duration_predictor_layers: 2      # number of layers of duration predictor
    duration_predictor_chans: 256     # number of channels of duration predictor
    duration_predictor_kernel_size: 3 # filter size of duration predictor
    postnet_layers: 5                 # number of layers of postnset
    postnet_filts: 5                  # filter size of conv layers in postnet
    postnet_chans: 256                # number of channels of conv layers in postnet
    use_masking: True                 # whether to apply masking for padded part in loss calculation
    encoder_normalize_before: True    # whether to perform layer normalization before the input
    decoder_normalize_before: True    # whether to perform layer normalization before the input
    reduction_factor: 1               # reduction factor
    encoder_type: conformer           # encoder type
    decoder_type: conformer           # decoder type
    conformer_rel_pos_type: latest               # relative positional encoding type
    conformer_pos_enc_layer_type: rel_pos        # conformer positional encoding type
    conformer_self_attn_layer_type: rel_selfattn # conformer self-attention type
    conformer_activation_type: swish             # conformer activation type
    use_macaron_style_in_conformer: true         # whether to use macaron style in conformer
    use_cnn_in_conformer: true                   # whether to use CNN in conformer
    conformer_enc_kernel_size: 7                 # kernel size in CNN module of conformer-based encoder
    conformer_dec_kernel_size: 31                # kernel size in CNN module of conformer-based decoder
    init_type: xavier_uniform                    # initialization type
    transformer_enc_dropout_rate: 0.2            # dropout rate for transformer encoder layer
    transformer_enc_positional_dropout_rate: 0.2 # dropout rate for transformer encoder positional encoding
    transformer_enc_attn_dropout_rate: 0.2       # dropout rate for transformer encoder attention layer
    transformer_dec_dropout_rate: 0.2            # dropout rate for transformer decoder layer
    transformer_dec_positional_dropout_rate: 0.2 # dropout rate for transformer decoder positional encoding
    transformer_dec_attn_dropout_rate: 0.2       # dropout rate for transformer decoder attention layer
    pitch_predictor_layers: 5                    # number of conv layers in pitch predictor
    pitch_predictor_chans: 256                   # number of channels of conv layers in pitch predictor
    pitch_predictor_kernel_size: 5               # kernel size of conv leyers in pitch predictor
    pitch_predictor_dropout: 0.5                 # dropout rate in pitch predictor
    pitch_embed_kernel_size: 1                   # kernel size of conv embedding layer for pitch
    pitch_embed_dropout: 0.0                     # dropout rate after conv embedding layer for pitch
    stop_gradient_from_pitch_predictor: true     # whether to stop the gradient from pitch predictor to encoder
    energy_predictor_layers: 2                   # number of conv layers in energy predictor
    energy_predictor_chans: 256                  # number of channels of conv layers in energy predictor
    energy_predictor_kernel_size: 3              # kernel size of conv leyers in energy predictor
    energy_predictor_dropout: 0.5                # dropout rate in energy predictor
    energy_embed_kernel_size: 1                  # kernel size of conv embedding layer for energy
    energy_embed_dropout: 0.0                    # dropout rate after conv embedding layer for energy
    stop_gradient_from_energy_predictor: false   # whether to stop the gradient from energy predictor to encoder
"""


class ConformerEncoder(nn.Module):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            params: dict,
    ) -> None:
        super(ConformerEncoder, self).__init__()

        self._conformer_encoder = Encoder(
            idim=0,  # Unused, because input_layer=None
            input_layer="linear",
            cnn_module_kernel=7,
            **params
        )

        self._output_projection = None
        if out_channels != in_channels:
            self._output_projection = nn.Linear(in_channels, out_channels)

    def forward(self, x, x_mask, g=None):
        x = x.transpose(1, 2)

        if x_mask.ndim == 2:
            x_mask = x_mask.unsqueeze(1)

        x = self._conformer_encoder(x, x_mask)[0]

        if self._output_projection is not None:
            x = self._output_projection(x)

        x = x.transpose(1, 2)

        return x


class ConformerDecoder(nn.Module):
    """Yeah... just an alias. Live with this information."""
    def __init__(self, in_channels: int, out_channels: int, params: dict) -> None:
        super(ConformerDecoder, self).__init__()

        self._conformer_encoder = Encoder(
            idim=0,  # Unused, because input_layer=None
            input_layer=None,
            cnn_module_kernel=31,
            **params
        )

        self._output_projection = None
        if out_channels != in_channels:
            self._output_projection = nn.Linear(in_channels, out_channels)

    def forward(self, x, x_mask, g=None):
        x = x.transpose(1, 2)

        if x_mask.ndim == 2:
            x_mask = x_mask.unsqueeze(1)

        x = self._conformer_encoder(x, x_mask)[0]

        if self._output_projection is not None:
            x = self._output_projection(x)

        x = x.transpose(1, 2)

        return x
