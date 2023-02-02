import sys

import torch
import torch.nn.functional as functional

from torch import nn
from torch.nn.modules.conv import Conv1d

from TTS.vocoder.models.hifigan_discriminator import DiscriminatorP, MultiPeriodDiscriminator


class DiscriminatorS(torch.nn.Module):
    """HiFiGAN Scale Discriminator. Channel sizes are different from the original HiFiGAN.

    Args:
        use_spectral_norm (bool): if `True` swith to spectral norm instead of weight norm.
    """

    def __init__(self, use_spectral_norm=False):
        super().__init__()
        norm_f = nn.utils.spectral_norm if use_spectral_norm else nn.utils.weight_norm
        self.convs = nn.ModuleList(
            [
                norm_f(Conv1d(1, 16, 15, 1, padding=7)),
                norm_f(Conv1d(16, 64, 41, 4, groups=4, padding=20)),
                norm_f(Conv1d(64, 256, 41, 4, groups=16, padding=20)),
                norm_f(Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
                norm_f(Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
                norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
            ]
        )
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        """
        Args:
            x (Tensor): input waveform.

        Returns:
            Tensor: discriminator scores.
            List[Tensor]: list of features from the convolutiona layers.
        """
        feat = []
        for l in self.convs:
            x = l(x)
            x = torch.nn.functional.leaky_relu(x, 0.1)
            feat.append(x)
        x = self.conv_post(x)
        feat.append(x)
        x = torch.flatten(x, 1, -1)
        return x, feat


class DiscriminatorR(torch.nn.Module):
    """UnivNet Discriminator. Computes the discriminator score for a given input."""

    def __init__(
            self,
            use_spectral_norm: bool = False,
            fft_size: int = 1024,
            win_size: int = 800,
            hop_size: int = 200,
            use_harmonic_conv: bool = True,
    ):
        super(DiscriminatorR, self).__init__()

        self.fft_size = fft_size
        self.win_size = win_size
        self.hop_size = hop_size

        self.activation = nn.LeakyReLU(0.2)

        norm_f = nn.utils.spectral_norm if use_spectral_norm else nn.utils.weight_norm

        self.convs = nn.ModuleList([
            norm_f(nn.Conv2d(1, 32, (3, 9), padding=(1, 4))),
            norm_f(nn.Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4))),
            norm_f(nn.Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4))),
            norm_f(nn.Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4))),
            norm_f(nn.Conv2d(32, 32, (3, 3), padding=(1, 1))),
        ])
        self.conv_post = norm_f(nn.Conv2d(32, 1, (3, 3), padding=(1, 1)))

        if use_harmonic_conv:
            from pathlib import Path
            sys.path.append(str(Path(__file__).parent))
            import harmonic_conv
            self.convs[0] = harmonic_conv.SingleHarmonicConv2d(
                1, 32, 3, anchor=1, stride=1, padding=(0, 1), padding_mode="zeros"
            )


    def forward(self, x: torch.Tensor):
        """
        Args:
            x (Tensor): input waveform.

        Returns:
            Tensor: discriminator scores.
            List[Tensor]: list of features from the convolutiona layers.
        """
        fmap = []

        x = self._compute_spectrogram(x)
        x = x.unsqueeze(1)
        for l in self.convs:
            x = l(x)
            x = self.activation(x)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap

    def _compute_spectrogram(self, x: torch.Tensor) -> torch.Tensor:
        x = functional.pad(x, (int((self.fft_size - self.hop_size) / 2), int((self.fft_size - self.hop_size) / 2)), mode='reflect')
        x = x.squeeze(1)
        x = torch.stft(x, n_fft=self.fft_size, hop_length=self.hop_size, win_length=self.win_size, center=False) #[B, F, TT, 2]
        return torch.norm(x, p=2, dim=-1)


class VitsDiscriminator(nn.Module):
    """VITS discriminator wrapping one Scale Discriminator and a stack of Period Discriminator.

    ::
        waveform -> ScaleDiscriminator() -> scores_sd, feats_sd --> append() -> scores, feats
               |--> MultiPeriodDiscriminator() -> scores_mpd, feats_mpd ^

    Args:
        use_spectral_norm (bool): if `True` swith to spectral norm instead of weight norm.
    """

    def __init__(self, periods=(2, 3, 5, 7, 11), use_spectral_norm=False, use_r_discriminator=True):
        super().__init__()
        self.nets = nn.ModuleList()
        self.nets.append(DiscriminatorS(use_spectral_norm=use_spectral_norm))
        self.nets.extend([DiscriminatorP(i, use_spectral_norm=use_spectral_norm) for i in periods])

        if use_r_discriminator:
            fft_params = [(1024, 120, 600), (2048, 240, 1200), (512, 50, 240)]
            self.nets.extend([
                DiscriminatorR(
                    use_spectral_norm=use_spectral_norm,
                    fft_size=fft_size,
                    win_size=win_size,
                    hop_size=hop_size
                ) for fft_size, win_size, hop_size in fft_params
            ])

    def forward(self, x, x_hat=None):
        """
        Args:
            x (Tensor): ground truth waveform.
            x_hat (Tensor): predicted waveform.

        Returns:
            List[Tensor]: discriminator scores.
            List[List[Tensor]]: list of list of features from each layers of each discriminator.
        """
        x_scores = []
        x_hat_scores = [] if x_hat is not None else None
        x_feats = []
        x_hat_feats = [] if x_hat is not None else None

        # TODO: Remove self.nets[1] (DiscriminatorS)
        for net in self.nets[1:]:
            x_score, x_feat = net(x)
            x_scores.append(x_score)
            x_feats.append(x_feat)
            if x_hat is not None:
                x_hat_score, x_hat_feat = net(x_hat)
                x_hat_scores.append(x_hat_score)
                x_hat_feats.append(x_hat_feat)
        return x_scores, x_feats, x_hat_scores, x_hat_feats
