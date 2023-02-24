from pathlib import Path
from typing import List, Union

import torch
import numpy as np

from TTS.tts.utils.synthesis import synthesis
from TTS.config import load_config
from TTS.tts.models import setup_model

CONFIG_FILE_NAME = "config.json"
LANGUAGE_IDS_FILE_NAME = "language_ids.json"


class VITSEvalInterfaceV2:
    def __init__(
            self,
            device: str,
            model_checkpoint_path: Path,
    ):
        self.device = device

        self.config = load_config(str(model_checkpoint_path.parent / CONFIG_FILE_NAME))
        self.sampling_rate = self.config.audio["sample_rate"]

        self.config.d_vector_file = None
        self.config.model_args.speakers_file = None
        self.config.model_args.d_vector_file = None

        if self.config.model_args.language_ids_file is not None:
            self.config.model_args.language_ids_file = str(model_checkpoint_path.parent / LANGUAGE_IDS_FILE_NAME)

        if self.config.language_ids_file is not None:
            self.config.language_ids_file = str(model_checkpoint_path.parent / LANGUAGE_IDS_FILE_NAME)

        self.config.phoneme_cache_path = None

        setup_model(config=self.config)

        self.model = setup_model(config=self.config)
        checkpoint = torch.load(model_checkpoint_path, map_location=torch.device('cpu'))
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()

        self.model.to(device)

    def __call__(
            self,
            text: str,
            speaker_embedding: Union[torch.Tensor, np.ndarray],
            length_scale: float = 1.0,
            noise_scale: float = 0.0,
    ) -> np.ndarray:

        speaker_embedding = torch.Tensor(speaker_embedding)
        self.model.length_scale = length_scale  # set speed of the speech.
        self.model.noise_scale = noise_scale  # set speech variation

        res = synthesis(
            self.model,
            text,
            self.config,
            use_cuda="cuda" in self.device,
            d_vector=speaker_embedding,
        )

        return (res["wav"] * 32768).astype("int16")
