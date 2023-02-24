import logging
import shutil
import time
from typing import Optional

from TTS.serving.config import TTSServingConfig, SEServingConfig
from TTS.serving.constants import *
from TTS.serving.template import copy_template_config, copy_template_config_se
from TTS.vits_eval_interface import SPEAKER_ENCODER_CONFIG_FILE_NAME


class TTSTritonExporter:
    def __init__(
        self,
        checkpoint_path: Path,
        speaker_encoder_checkpoint_path: Path,
    ) -> None:
        self._logger = logging.getLogger("tts_triton_exporter")
        self._checkpoint_path = checkpoint_path
        self._speaker_encoder_checkpoint_path = speaker_encoder_checkpoint_path

    @staticmethod
    def get_version() -> str:
        """
        Get dummy version number for Triton model.
        """
        return str(int(time.time()))

    def export(
        self,
        export_dir: Path,
        version: Optional[int] = None,
    ) -> None:

        # VERSION IS SAME FOR BOTH MODELS
        version = str(version) if version is not None else self.get_version()

        # Export TTS model
        model_dir = export_dir / MODEL_NAME
        model_dir.mkdir(exist_ok=True, parents=True)

        version_dir = model_dir / version
        version_dir.mkdir(exist_ok=True, parents=True)

        # Create config.pbtxt file
        config_filename = model_dir / TTS_MODEL_CONFIG_NAME
        copy_template_config(config_filename)

        # Export model (move checkpoint)
        tts_model_export_path = version_dir / TTS_CHECKPOINT_DIR_NAME
        tts_model_export_path.mkdir(exist_ok=True, parents=True)
        tts_model_checkpoint_path = tts_model_export_path / DEFAULT_TTS_CHECKPOINT_NAME
        tts_model_config_path = tts_model_export_path / DEFAULT_TTS_CONFIG_NAME
        shutil.copy(self._checkpoint_path, tts_model_checkpoint_path)
        shutil.copy(self._checkpoint_path.with_name(CONFIG_FILE_NAME), tts_model_config_path)

        # Export model.py
        model_export_path = version_dir / EXPORTED_MODEL_PY_NAME
        shutil.copy(TTS_PYTHON_BACKEND_MODEL_TEMPLATE_PATH, model_export_path)

        # Export serving_config.json
        serving_config = TTSServingConfig(
            model_checkpoint_path=tts_model_checkpoint_path,
        )
        serving_config.to_json(version_dir / SERVING_CONFIG_NAME)

        # Export Speaker Encoder model
        model_dir = export_dir / MODEL_SE_NAME
        model_dir.mkdir(exist_ok=True, parents=True)

        version_dir = model_dir / version
        version_dir.mkdir(exist_ok=True, parents=True)

        # Create config.pbtxt file
        config_filename = model_dir / SE_MODEL_CONFIG_NAME
        copy_template_config_se(config_filename)

        speaker_encoder_model_export_path = version_dir / SPEAKER_ENCODER_CHECKPOINT_DIR_NAME
        speaker_encoder_model_export_path.mkdir(exist_ok=True, parents=True)
        speaker_encoder_model_checkpoint_path = speaker_encoder_model_export_path / DEFAULT_SPEAKER_ENCODER_CHECKPOINT_NAME
        speaker_encoder_model_config_path = speaker_encoder_model_export_path / DEFAULT_SPEAKER_ENCODER_CONFIG_NAME
        shutil.copy(self._speaker_encoder_checkpoint_path, speaker_encoder_model_checkpoint_path)
        shutil.copy(self._speaker_encoder_checkpoint_path.with_name(SPEAKER_ENCODER_CONFIG_FILE_NAME), speaker_encoder_model_config_path)

        # Export serving_config.json
        serving_config = SEServingConfig(
            model_checkpoint_path=speaker_encoder_model_checkpoint_path,
        )
        serving_config.to_json(version_dir / SERVING_CONFIG_NAME)

        # Export model.py
        model_export_path = version_dir / EXPORTED_MODEL_PY_NAME
        shutil.copy(SE_PYTHON_BACKEND_MODEL_TEMPLATE_PATH, model_export_path)
