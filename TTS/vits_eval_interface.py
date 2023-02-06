from pathlib import Path
from typing import List

import torch
import numpy as np

from TTS.tts.utils.synthesis import synthesis
from TTS.config import load_config
from TTS.tts.models import setup_model
from TTS.tts.utils.speakers import SpeakerManager


CONFIG_FILE_NAME = "config.json"
LANGUAGE_IDS_FILE_NAME = "language_ids.json"
SPEAKER_ENCODER_CONFIG_FILE_NAME = "config_se.json"


class VITSEvalInterface:
    def __init__(
            self,
            device: str,
            model_root_dir: Path,
            speaker_encoder_checkpoint_path: Path,
            checkpoint_name: str = "best_model.pth",
    ):
        self.device = device

        self.config = load_config(str(model_root_dir / CONFIG_FILE_NAME))
        self.sampling_rate = self.config.audio["sample_rate"]

        self.config.d_vector_file = None
        self.config.model_args.speakers_file = None
        self.config.model_args.d_vector_file = None

        if self.config.model_args.language_ids_file is not None:
            self.config.model_args.language_ids_file = str(model_root_dir / LANGUAGE_IDS_FILE_NAME)

        if self.config.language_ids_file is not None:
            self.config.language_ids_file = str(model_root_dir / LANGUAGE_IDS_FILE_NAME)

        self.config.phoneme_cache_path = None

        setup_model(config=self.config)

        self.model = setup_model(config=self.config)
        checkpoint = torch.load(model_root_dir / checkpoint_name, map_location=torch.device('cpu'))
        self.model.load_state_dict(checkpoint["model"], strict=False)
        self.model.eval()

        self.model.to(device)

        self.speaker_manager = SpeakerManager(
            encoder_model_path=str(speaker_encoder_checkpoint_path),
            encoder_config_path=str(
                speaker_encoder_checkpoint_path.with_name(SPEAKER_ENCODER_CONFIG_FILE_NAME)
            ),
        )

    def __call__(
            self,
            text: str,
            speaker_embedding: torch.Tensor,
            length_scale: float = 1.0,
            noise_scale: float = 0.0,
        ) -> np.ndarray:

        self.model.length_scale = length_scale  # set speed of the speech.
        self.model.noise_scale = noise_scale  # set speech variation

        res = synthesis(
            self.model,
            text,
            self.config,
            use_cuda="cuda" in self.device,
            d_vector=speaker_embedding,
        )

        return (res['wav'] * 32768).astype('int16')

    def get_speaker_embedding_from_file(self, file_path: Path) -> torch.Tensor:
        return self.get_speaker_embedding([file_path])

    def get_speaker_embedding(self, file_list: List[Path]) -> torch.Tensor:

        # extract embedding from wav files
        speaker_embeddings = []
        for file_path in file_list:
            embedding = self.speaker_manager.compute_embedding_from_clip(str(file_path))
            speaker_embeddings.append(embedding)

        # takes the average of the embeddings samples of the speaker
        speaker_embedding = np.mean(np.array(speaker_embeddings), axis=0).tolist()
        return torch.Tensor(speaker_embedding)
