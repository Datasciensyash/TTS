import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm

from thefuzz import fuzz

from TTS.vits_eval_interface import VITSEvalInterface

import whisperx

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
        "-w",
        "--whisper_model",
        help="Name of the whisper model",
        type=str,
        default="base.en",
        required=False,
    )
    return arguments_parser.parse_args()


def compute_wer(
    input_dir: Path,
    device: str,
    model_root_dir: Path,
    speaker_encoder_checkpoint_path: Path,
    checkpoint_name: str = "best_model.pth",
    whisper_model: str = "base.en",
) -> None:
    vits_eval_interface = VITSEvalInterface(
        device=device,
        model_root_dir=model_root_dir,
        speaker_encoder_checkpoint_path=speaker_encoder_checkpoint_path,
        checkpoint_name=checkpoint_name,
    )

    # Compute levenshetin distance between original and generated texts
    model = whisperx.load_model(whisper_model, device)

    fuzzy_ratios = []
    for speaker_dir in tqdm(
        input_dir.iterdir(),
        desc="Computing WER for each speaker...",
    ):
        if not speaker_dir.is_dir():
            continue

        fuzzy_ratios_speaker = []

        # TODO: flac -> wav & flac & mp3
        audio_files = list(speaker_dir.rglob("*.wav"))

        speaker_embedding_path = speaker_dir / "speaker_embedding.npy"

        if not speaker_embedding_path.exists():
            speaker_embedding = vits_eval_interface.get_speaker_embedding(
                audio_files
            )
            np.save(speaker_embedding_path, speaker_embedding)
        else:
            speaker_embedding = np.load(speaker_embedding_path)

        for audio_file in audio_files:

            if audio_file.with_suffix(".txt").exists():
                text = audio_file.with_suffix(".txt").read_text()
            else:
                text = model.transcribe(audio_file)['segments'][0]['text']
                audio_file.with_suffix(".txt").write_text(text)

            # Infer TTS model
            audio = vits_eval_interface(text, speaker_embedding)
            sf.write(TMP_FILENAME, audio, vits_eval_interface.sampling_rate)

            # Infer whisper model
            text_predicted = model.transcribe(TMP_FILENAME)['segments']

            # Compute levenshtein distance
            fuzzy_ratio = fuzz.ratio(text, text_predicted)
            fuzzy_ratios_speaker.append(fuzzy_ratio)

        fuzzy_ratios.extend(fuzzy_ratios_speaker)

    print(f"Fuzzy ratio (higher is better): {np.mean(fuzzy_ratios)}")


def main():
    args = parse_args()
    compute_wer(
        input_dir=args.input_dir,
        device=args.device,
        model_root_dir=args.model_root_dir,
        speaker_encoder_checkpoint_path=args.speaker_encoder_checkpoint_path,
        checkpoint_name=args.checkpoint_name,
        whisper_model=args.whisper_model,
    )


if __name__ == "__main__":
    main()
