import argparse
import warnings
from pathlib import Path
from typing import Tuple

import fastwer
import numpy as np
import soundfile as sf
import whisperx
from tqdm import tqdm

from TTS.vits_eval_interface import VITSEvalInterface

warnings.filterwarnings("ignore", category=UserWarning)


# TODO: I think that is bad idea
TMP_FILENAME = "tmp.wav"


def parse_args() -> argparse.Namespace:
    arguments_parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    arguments_parser.add_argument(
        "-d",
        "--device",
        help="Computing device",
        type=str,
        required=False,
        default="cpu",
    )
    arguments_parser.add_argument(
        "-m",
        "--model_root_dir",
        help="Path to the model root directory",
        type=Path,
        required=True,
    )
    arguments_parser.add_argument(
        "-se",
        "--speaker_encoder_checkpoint_path",
        help="Path to the speaker encoder checkpoint",
        type=Path,
        required=True,
    )
    arguments_parser.add_argument(
        "-c",
        "--checkpoint_name",
        help="Name of the checkpoint",
        type=str,
        required=False,
        default="best_model.pth",
    )
    arguments_parser.add_argument(
        "-dir",
        "--input_dir",
        help="Path to the directory with input files",
        type=Path,
        required=True,
    )
    arguments_parser.add_argument(
        "-ms",
        "--max_speakers_num",
        help="Maximum number of speakers to compare",
        type=int,
        required=False,
        default=10,
    )
    arguments_parser.add_argument(
        "-txt",
        "--texts_file",
        help="Path to the file with texts",
        type=Path,
        required=False,
        default=Path(__file__).parent / "script_data" / "wer_default.txt",
    )
    arguments_parser.add_argument(
        "-w",
        "--whisper_model",
        help="Name of the whisper model",
        type=str,
        default="tiny.en",
        required=False,
    )
    return arguments_parser.parse_args()


def compute_wer(
    input_dir: Path,
    device: str,
    model_root_dir: Path,
    speaker_encoder_checkpoint_path: Path,
    checkpoint_name: str = "best_model.pth",
    whisper_model: str = "tiny.en",
    max_speakers_num: int = 10,
    texts_file: Path = Path(__file__).parent
    / "script_data"
    / "wer_default.txt",
) -> Tuple[float, float]:
    vits_eval_interface = VITSEvalInterface(
        device=device,
        model_root_dir=model_root_dir,
        speaker_encoder_checkpoint_path=speaker_encoder_checkpoint_path,
        checkpoint_name=checkpoint_name,
    )

    asr_model = whisperx.load_model(whisper_model, device)

    test_texts = texts_file.read_text().splitlines()

    original_texts, predicted_texts = [], []
    speaker_directories = [
        spk_dir for spk_dir in input_dir.iterdir() if spk_dir.is_dir()
    ]
    for speaker_dir in tqdm(
        speaker_directories[:max_speakers_num], desc="Computing WER..."
    ):

        audio_files = list(speaker_dir.rglob("*.wav"))
        speaker_embedding_path = speaker_dir / "speaker_embedding.npy"
        if not speaker_embedding_path.exists():
            speaker_embedding = vits_eval_interface.get_speaker_embedding(
                audio_files
            )
            np.save(speaker_embedding_path, speaker_embedding)
        else:
            speaker_embedding = np.load(speaker_embedding_path)

        for text in test_texts:
            audio = vits_eval_interface(text, speaker_embedding)
            sf.write(TMP_FILENAME, audio, vits_eval_interface.sampling_rate)

            try:
                predicted_text = asr_model.transcribe(TMP_FILENAME)["segments"][
                    0
                ]["text"]

            # When ASR model failed to transcribe audio
            except IndexError:
                predicted_text = ""

            original_texts.append(text)
            predicted_texts.append(predicted_text)

    cer = fastwer.score(original_texts, predicted_texts, char_level=True)
    wer = fastwer.score(original_texts, predicted_texts, char_level=False)

    print(f"WER: {wer} | CER: {cer}")

    return wer, cer


def main():
    args = parse_args()
    compute_wer(
        input_dir=args.input_dir,
        device=args.device,
        model_root_dir=args.model_root_dir,
        speaker_encoder_checkpoint_path=args.speaker_encoder_checkpoint_path,
        checkpoint_name=args.checkpoint_name,
        whisper_model=args.whisper_model,
        texts_file=args.texts_file,
        max_speakers_num=args.max_speakers_num,
    )


if __name__ == "__main__":
    main()
