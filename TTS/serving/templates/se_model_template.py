import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

import triton_python_backend_utils as pb_utils
from TTS.se_eval_interface import SpeakerEncoderEvalInterface

from TTS.serving.constants import SERVING_CONFIG_NAME
from TTS.serving.config import SEServingConfig

MODEL_CONFIG = "model_config"
DATA_TYPE = "data_type"

THIS_DIR = Path(__file__).parent


class TritonPythonModel:
    _eval_interface: SpeakerEncoderEvalInterface
    _serving_config: SEServingConfig
    _model_config: Dict
    _target_embedding_dtype: Any

    def initialize(self, args: Dict[str, Any]):
        self._serving_config = SEServingConfig.from_json(
            THIS_DIR / SERVING_CONFIG_NAME
        )

        self._eval_interface = SpeakerEncoderEvalInterface(
            checkpoint_path=THIS_DIR / self._serving_config.model_checkpoint_path,
        )

        self._model_config = json.loads(args[MODEL_CONFIG])
        target_embedding_config = pb_utils.get_output_config_by_name(
            self._model_config,
            self._serving_config.target_embedding_field,
        )
        self._target_embedding_dtype = pb_utils.triton_string_to_numpy(
            target_embedding_config[DATA_TYPE],
        )

    def _forward(self, audio: np.ndarray, sampling_rate: int) -> np.ndarray:
        return self._eval_interface(audio, sampling_rate)

    def execute(self, requests: List[Any]):
        output0_dtype = self._target_embedding_dtype
        responses = []

        for request in requests:
            source_audio = pb_utils.get_input_tensor_by_name(
                request,
                self._serving_config.source_audio_field,
            ).as_numpy()

            source_sample_rate = pb_utils.get_input_tensor_by_name(
                request,
                self._serving_config.source_sampling_rate_field,
            ).as_numpy()

            # TODO: Somehow we need to support batched inference
            target_embedding = self._forward(source_audio, source_sample_rate[0])
            target_embedding_out = pb_utils.Tensor(
                self._serving_config.target_embedding_field,
                target_embedding.astype(output0_dtype),
            )
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[target_embedding_out]
            )
            responses.append(inference_response)

        return responses
