import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

import triton_python_backend_utils as pb_utils

from TTS.serving.constants import SERVING_CONFIG_NAME
from TTS.serving.config import TTSServingConfig
from TTS.vits_eval_interface_v2 import VITSEvalInterfaceV2

MODEL_CONFIG = "model_config"
DATA_TYPE = "data_type"

THIS_DIR = Path(__file__).parent


class TritonPythonModel:
    _eval_interface: VITSEvalInterfaceV2
    _serving_config: TTSServingConfig
    _model_config: Dict
    _target_audio_dtype: Any

    def initialize(self, args: Dict[str, Any]):
        self._serving_config = TTSServingConfig.from_json(
            THIS_DIR / SERVING_CONFIG_NAME
        )

        self._eval_interface = VITSEvalInterfaceV2(
            model_checkpoint_path=self._serving_config.model_checkpoint_path,
            device=self._serving_config.device,
        )

        self._model_config = json.loads(args[MODEL_CONFIG])
        target_audio_config = pb_utils.get_output_config_by_name(
            self._model_config,
            self._serving_config.target_audio_field,
        )
        self._target_audio_dtype = pb_utils.triton_string_to_numpy(
            target_audio_config[DATA_TYPE],
        )

    def _forward(
            self, text_input: str, speaker_embedding: np.ndarray
    ) -> np.ndarray:
        return self._eval_interface(
            text=text_input,
            speaker_embedding=speaker_embedding,
        )

    def execute(self, requests: List[Any]):
        output0_dtype = self._target_audio_dtype
        responses = []

        for request in requests:
            source_text = pb_utils.get_input_tensor_by_name(
                request,
                self._serving_config.source_text_field,
            ).as_numpy()

            speaker_embedding = pb_utils.get_input_tensor_by_name(
                request,
                self._serving_config.speaker_embedding_field,
            ).as_numpy()

            # TODO: Somehow we need to support batched inference
            target_audio = self._forward(source_text.astype('U')[0], speaker_embedding)
            target_audio_out = pb_utils.Tensor(
                self._serving_config.target_audio_field,
                target_audio.astype(output0_dtype),
            )
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[target_audio_out]
            )
            responses.append(inference_response)

        return responses
