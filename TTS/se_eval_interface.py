import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from TTS.tts.utils.speakers import SpeakerManager

SPEAKER_ENCODER_CONFIG_FILE_NAME = "config_se.json"


class SpeakerEncoderEvalInterface:
    def __init__(self, checkpoint_path: Path):
        self.speaker_manager = SpeakerManager(
            encoder_model_path=str(checkpoint_path),
            encoder_config_path=str(
                checkpoint_path.with_name(SPEAKER_ENCODER_CONFIG_FILE_NAME)
            ),
        )

    def __call__(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        # Save audio to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav") as temp_file:
            sf.write(temp_file.name, audio, sample_rate)
            return np.array(self.speaker_manager.compute_embedding_from_clip(temp_file.name))
