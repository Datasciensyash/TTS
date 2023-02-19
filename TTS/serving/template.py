from pathlib import Path

from TTS.serving.constants import TEXT_INPUT_NAME, SPEAKER_EMBEDDING_INPUT_NAME, \
    AUDIO_SIGNAL_OUTPUT_NAME, MODEL_CONFIG_TEMPLATE_PATH, MODEL_NAME


def copy_template_config(copy_path: Path) -> None:
    config = MODEL_CONFIG_TEMPLATE_PATH.read_text()
    config = config.replace("MODEL_NAME", MODEL_NAME)
    config = config.replace("TEXT_INPUT_NAME", TEXT_INPUT_NAME)
    config = config.replace("SPEAKER_EMBEDDING_INPUT_NAME", SPEAKER_EMBEDDING_INPUT_NAME)
    config = config.replace(
        "AUDIO_SIGNAL_OUTPUT_NAME", AUDIO_SIGNAL_OUTPUT_NAME
    )
    copy_path.write_text(config)
