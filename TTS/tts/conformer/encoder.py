import torch
import torch.nn as nn

from TTS.tts.conformer.conformer import Encoder


class ConformerEncoder(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, params: dict) -> None:
        super(ConformerEncoder, self).__init__()

        self._conformer_encoder = Encoder(in_channels, **params)

        self._output_projection = None
        if out_channels != in_channels:
            self._output_projection = nn.Linear(in_channels, out_channels)

    def forward(self, x, x_mask):
        print('X: ', x.shape)
        print('X_MASK: ', x_mask.shape)

        x = x.transpose(1, 2)

        if x_mask.ndim == 2:
            x_mask = x_mask.unsqueeze(1)

        print('X_MASK: ', x.shape)
        x = self._conformer_encoder(x, x_mask)
        x = x.transpose(1, 2)

        if self._output_projection is not None:
            x = self._output_projection(x)
        return x


class ConformerDecoder(ConformerEncoder):
    """Yeah... just an alias. Live with this information."""
    pass

