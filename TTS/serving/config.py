import json
from dataclasses import dataclass
from pathlib import Path

from TTS.serving.constants import TEXT_INPUT_NAME, SPEAKER_EMBEDDING_INPUT_NAME, \
    AUDIO_SIGNAL_OUTPUT_NAME


@dataclass
class TTSServingConfig:
    """The configuration for a TTS model serving instance."""
    model_root_dir: Path
    speaker_encoder_checkpoint_path: Path
    
    device: str = "cuda:0"
    checkpoint_name: str = "best_model.pth"
    
    source_text_field: str = TEXT_INPUT_NAME
    speaker_embedding_field: str = SPEAKER_EMBEDDING_INPUT_NAME
    target_audio_field: str = AUDIO_SIGNAL_OUTPUT_NAME

    def to_json(self, path: Path) -> None:
        """Save the TTSServingConfig to a JSON file at the specified path.
        Args:
            path: The path to the JSON file to save to.
        """
        self.model_root_dir = str(self.model_root_dir)  # type: ignore
        path.write_text(json.dumps(self.__dict__))

    @classmethod
    def from_json(cls, path: Path) -> "TTSServingConfig":
        """Load a TTSServingConfig from a JSON file at the specified path.
        Args:
            path: The path to the JSON file to load from.
        Returns:
            The loaded VCServingConfig.
        """
        config = cls(**json.loads(path.read_text()))
        config.model_root_dir = Path(config.model_root_dir)
        return config
