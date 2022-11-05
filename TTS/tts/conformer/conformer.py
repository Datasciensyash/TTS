# Copyright 2020 Johns Hopkins University (Shinji Watanabe)
#                Northwestern Polytechnical University (Pengcheng Guo)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Encoder definition."""

import logging

import torch

from TTS.tts.conformer.modules.convolution import ConvolutionModule
from TTS.tts.conformer.modules.encoder_layer import EncoderLayer
from TTS.tts.conformer.nets_utils import get_activation
from TTS.tts.conformer.modules.vggl2 import VGG2L
from TTS.tts.conformer.modules.attention import (
    LegacyRelPositionMultiHeadedAttention,
    MultiHeadedAttention,
    RelPositionMultiHeadedAttention,
)
from TTS.tts.conformer.modules.embedding import (
    LegacyRelPositionalEncoding,
    PositionalEncoding,
    RelPositionalEncoding,
    ScaledPositionalEncoding,
)
from TTS.tts.conformer.modules.layer_norm import LayerNorm
from TTS.tts.conformer.modules.multi_layer_conv import (
    Conv1dLinear,
    MultiLayeredConv1d,
)
from TTS.tts.conformer.modules.position_wise_feed_forward import (
    PositionwiseFeedForward,
)
from TTS.tts.conformer.modules.repeat import repeat
from TTS.tts.conformer.modules.subsampling import Conv2dSubsampling


class Encoder(torch.nn.Module):
    """Conformer encoder module.
    Args:
        idim (int): Input dimension.
        attention_dim (int): Dimension of attention.
        attention_heads (int): The number of heads of multi head attention.
        linear_units (int): The number of units of position-wise feed forward.
        num_blocks (int): The number of decoder blocks.
        dropout_rate (float): Dropout rate.
        positional_dropout_rate (float): Dropout rate after adding positional encoding.
        attention_dropout_rate (float): Dropout rate in attention.
        input_layer (Union[str, torch.nn.Module]): Input layer type.
        normalize_before (bool): Whether to use layer_norm before the first block.
        concat_after (bool): Whether to concat attention layer's input and output.
            if True, additional linear will be applied.
            i.e. x -> x + linear(concat(x, att(x)))
            if False, no additional linear will be applied. i.e. x -> x + att(x)
        positionwise_layer_type (str): "linear", "conv1d", or "conv1d-linear".
        positionwise_conv_kernel_size (int): Kernel size of positionwise conv1d layer.
        macaron_style (bool): Whether to use macaron style for positionwise layer.
        pos_enc_layer_type (str): Encoder positional encoding layer type.
        selfattention_layer_type (str): Encoder attention layer type.
        activation_type (str): Encoder activation function type.
        use_cnn_module (bool): Whether to use convolution module.
        zero_triu (bool): Whether to zero the upper triangular part of attention matrix.
        cnn_module_kernel (int): Kernerl size of convolution module.
        padding_idx (int): Padding idx for input_layer=embed.
        stochastic_depth_rate (float): Maximum probability to skip the encoder layer.
        intermediate_layers (Union[List[int], None]): indices of intermediate CTC layer.
            indices start from 1.
            if not None, intermediate outputs are returned (which changes return type
            signature.)
    """

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

    def __init__(
        self,
        idim,
        attention_dim=384,
        attention_heads=2,
        linear_units=1536,
        num_blocks=4,
        dropout_rate=0.2,
        positional_dropout_rate=0.2,
        attention_dropout_rate=0.2,
        input_layer=None,
        normalize_before=True,
        concat_after=False,
        positionwise_layer_type="conv1d",
        positionwise_conv_kernel_size=3,
        macaron_style=True,
        pos_enc_layer_type="rel_pos",
        selfattention_layer_type="rel_selfattn",
        activation_type="swish",
        use_cnn_module=True,
        zero_triu=False,
        cnn_module_kernel=31,
        padding_idx=-1,
        stochastic_depth_rate=0.0,
        intermediate_layers=None,
        ctc_softmax=None,
        conditioning_layer_dim=None,
    ):
        """Construct an Encoder object."""
        super(Encoder, self).__init__()

        activation = get_activation(activation_type)
        if pos_enc_layer_type == "abs_pos":
            pos_enc_class = PositionalEncoding
        elif pos_enc_layer_type == "scaled_abs_pos":
            pos_enc_class = ScaledPositionalEncoding
        elif pos_enc_layer_type == "rel_pos":
            assert selfattention_layer_type == "rel_selfattn"
            pos_enc_class = RelPositionalEncoding
        elif pos_enc_layer_type == "legacy_rel_pos":
            pos_enc_class = LegacyRelPositionalEncoding
            assert selfattention_layer_type == "legacy_rel_selfattn"
        else:
            raise ValueError("unknown pos_enc_layer: " + pos_enc_layer_type)

        self.conv_subsampling_factor = 1
        if input_layer == "linear":
            self.embed = torch.nn.Sequential(
                torch.nn.Linear(idim, attention_dim),
                torch.nn.LayerNorm(attention_dim),
                torch.nn.Dropout(dropout_rate),
                pos_enc_class(attention_dim, positional_dropout_rate),
            )
        elif input_layer == "conv2d":
            self.embed = Conv2dSubsampling(
                idim,
                attention_dim,
                dropout_rate,
                pos_enc_class(attention_dim, positional_dropout_rate),
            )
            self.conv_subsampling_factor = 4
        elif input_layer == "vgg2l":
            self.embed = VGG2L(idim, attention_dim)
            self.conv_subsampling_factor = 4
        elif input_layer == "embed":
            self.embed = torch.nn.Sequential(
                torch.nn.Embedding(idim, attention_dim, padding_idx=padding_idx),
                pos_enc_class(attention_dim, positional_dropout_rate),
            )
        elif isinstance(input_layer, torch.nn.Module):
            self.embed = torch.nn.Sequential(
                input_layer,
                pos_enc_class(attention_dim, positional_dropout_rate),
            )
        elif input_layer is None:
            self.embed = torch.nn.Sequential(
                pos_enc_class(attention_dim, positional_dropout_rate)
            )
        else:
            raise ValueError("unknown input_layer: " + input_layer)
        self.normalize_before = normalize_before

        # self-attention module definition
        if selfattention_layer_type == "selfattn":
            logging.info("encoder self-attention layer type = self-attention")
            encoder_selfattn_layer = MultiHeadedAttention
            encoder_selfattn_layer_args = (
                attention_heads,
                attention_dim,
                attention_dropout_rate,
            )
        elif selfattention_layer_type == "legacy_rel_selfattn":
            assert pos_enc_layer_type == "legacy_rel_pos"
            encoder_selfattn_layer = LegacyRelPositionMultiHeadedAttention
            encoder_selfattn_layer_args = (
                attention_heads,
                attention_dim,
                attention_dropout_rate,
            )
        elif selfattention_layer_type == "rel_selfattn":
            logging.info("encoder self-attention layer type = relative self-attention")
            assert pos_enc_layer_type == "rel_pos"
            encoder_selfattn_layer = RelPositionMultiHeadedAttention
            encoder_selfattn_layer_args = (
                attention_heads,
                attention_dim,
                attention_dropout_rate,
                zero_triu,
            )
        else:
            raise ValueError("unknown encoder_attn_layer: " + selfattention_layer_type)

        # feed-forward module definition
        if positionwise_layer_type == "linear":
            positionwise_layer = PositionwiseFeedForward
            positionwise_layer_args = (
                attention_dim,
                linear_units,
                dropout_rate,
                activation,
            )
        elif positionwise_layer_type == "conv1d":
            positionwise_layer = MultiLayeredConv1d
            positionwise_layer_args = (
                attention_dim,
                linear_units,
                positionwise_conv_kernel_size,
                dropout_rate,
            )
        elif positionwise_layer_type == "conv1d-linear":
            positionwise_layer = Conv1dLinear
            positionwise_layer_args = (
                attention_dim,
                linear_units,
                positionwise_conv_kernel_size,
                dropout_rate,
            )
        else:
            raise NotImplementedError("Support only linear or conv1d.")

        # convolution module definition
        convolution_layer = ConvolutionModule
        convolution_layer_args = (attention_dim, cnn_module_kernel, activation)

        self.encoders = repeat(
            num_blocks,
            lambda lnum: EncoderLayer(
                attention_dim,
                encoder_selfattn_layer(*encoder_selfattn_layer_args),
                positionwise_layer(*positionwise_layer_args),
                positionwise_layer(*positionwise_layer_args) if macaron_style else None,
                convolution_layer(*convolution_layer_args) if use_cnn_module else None,
                dropout_rate,
                normalize_before,
                concat_after,
                stochastic_depth_rate * float(1 + lnum) / num_blocks,
            ),
        )
        if self.normalize_before:
            self.after_norm = LayerNorm(attention_dim)

        self.intermediate_layers = intermediate_layers
        self.use_conditioning = True if ctc_softmax is not None else False
        if self.use_conditioning:
            self.ctc_softmax = ctc_softmax
            self.conditioning_layer = torch.nn.Linear(
                conditioning_layer_dim, attention_dim
            )

    def forward(self, xs, masks):
        """Encode input sequence.
        Args:
            xs (torch.Tensor): Input tensor (#batch, time, idim).
            masks (torch.Tensor): Mask tensor (#batch, 1, time).
        Returns:
            torch.Tensor: Output tensor (#batch, time, attention_dim).
            torch.Tensor: Mask tensor (#batch, time).
        """
        if isinstance(self.embed, (Conv2dSubsampling, VGG2L)):
            print(self.embed.__class__.__name__)
            xs, masks = self.embed(xs, masks)
        else:
            xs = self.embed(xs)

        if self.intermediate_layers is None:
            xs, masks = self.encoders(xs, masks)
        else:
            intermediate_outputs = []
            for layer_idx, encoder_layer in enumerate(self.encoders):
                xs, masks = encoder_layer(xs, masks)

                if (
                    self.intermediate_layers is not None
                    and layer_idx + 1 in self.intermediate_layers
                ):
                    # intermediate branches also require normalization.
                    encoder_output = xs
                    if isinstance(encoder_output, tuple):
                        encoder_output = encoder_output[0]

                    if self.normalize_before:
                        encoder_output = self.after_norm(encoder_output)

                    intermediate_outputs.append(encoder_output)

                    if self.use_conditioning:
                        intermediate_result = self.ctc_softmax(encoder_output)

                        if isinstance(xs, tuple):
                            x, pos_emb = xs[0], xs[1]
                            x = x + self.conditioning_layer(intermediate_result)
                            xs = (x, pos_emb)
                        else:
                            xs = xs + self.conditioning_layer(intermediate_result)

        if isinstance(xs, tuple):
            xs = xs[0]

        if self.normalize_before:
            xs = self.after_norm(xs)

        if self.intermediate_layers is not None:
            return xs, masks, intermediate_outputs
        return xs, masks


if __name__ == '__main__':
    m = Encoder(128)
    m(torch.randn(2, 32, 128), torch.ones([2, 1, 32]) > 0)
