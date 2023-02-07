import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.spatial.distance import cosine
from tqdm import tqdm

from TTS.vits_eval_interface import VITSEvalInterface


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
        "-spk",
        "--speaker_embedding_path",
        help="Path to the speaker embedding",
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
        "-o",
        "--output_dir",
        help="Path to output directory",
        type=Path,
        required=True,
    )
    arguments_parser.add_argument(
        "-txt",
        "--texts_file",
        help="Path to the file with texts",
        type=Path,
        required=False,
        default=Path(__file__).parent / "script_data" / "texts_spk_sim.txt",
    )
    return arguments_parser.parse_args()


def infer_vits_model(
    device: str,
    model_root_dir: Path,
    speaker_encoder_checkpoint_path: Path,
    speaker_embedding_path: Path,
    checkpoint_name: str,
    output_dir: Path,
    texts_file: Path,
) -> None:
    vits_eval_interface = VITSEvalInterface(
        device=device,
        model_root_dir=model_root_dir,
        speaker_encoder_checkpoint_path=speaker_encoder_checkpoint_path,
        checkpoint_name=checkpoint_name,
    )
    texts = texts_file.read_text().splitlines()

    output_dir.mkdir(parents=True, exist_ok=True)
    speaker_embedding = np.load(speaker_embedding_path)
    for i, text in tqdm(enumerate(texts), desc="TTS Inference..."):
        audio = vits_eval_interface(text, speaker_embedding)
        sf.write(output_dir / f"{i}.wav", audio, vits_eval_interface.sampling_rate)

    return None


def main():
    args = parse_args()
    infer_vits_model(
        device=args.device,
        model_root_dir=args.model_root_dir,
        speaker_encoder_checkpoint_path=args.speaker_encoder_checkpoint_path,
        speaker_embedding_path=args.speaker_embedding_path,
        checkpoint_name=args.checkpoint_name,
        output_dir=args.output_dir,
        texts_file=args.texts_file,
    )


if __name__ == "__main__":
    main()
