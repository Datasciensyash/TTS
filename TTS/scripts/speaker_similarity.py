import argparse
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.spatial.distance import cosine
from tqdm import tqdm

from TTS.vits_eval_interface import VITSEvalInterface

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
        "-txt",
        "--texts_file",
        help="Path to the file with texts",
        type=Path,
        required=False,
        default=Path(__file__).parent / "texts_spk_sim.txt",
    )
    return arguments_parser.parse_args()


def compute_speaker_similarity(
    input_dir: Path,
    device: str,
    model_root_dir: Path,
    speaker_encoder_checkpoint_path: Path,
    checkpoint_name: str = "best_model.pth",
    texts_file=Path(__file__).parent / "texts_spk_sim.txt",
) -> None:
    vits_eval_interface = VITSEvalInterface(
        device=device,
        model_root_dir=model_root_dir,
        speaker_encoder_checkpoint_path=speaker_encoder_checkpoint_path,
        checkpoint_name=checkpoint_name,
    )
    texts = texts_file.read_text().splitlines()

    speaker_distances = []
    for speaker_dir in tqdm(
        input_dir.iterdir(),
        desc="Computing speaker similarity for each speaker...",
    ):
        if not speaker_dir.is_dir():
            continue

        distances = []

        # TODO: flac -> wav & flac & mp3
        audio_files = list(speaker_dir.glob("*.wav"))

        speaker_embedding_path = speaker_dir / "speaker_embedding.npy"

        if not speaker_embedding_path.exists():
            speaker_embedding = vits_eval_interface.get_speaker_embedding(
                audio_files
            )
            np.save(speaker_embedding_path, speaker_embedding)
        else:
            speaker_embedding = np.load(speaker_embedding_path)

        for text in texts:
            audio = vits_eval_interface(text, speaker_embedding)
            sf.write(TMP_FILENAME, audio, vits_eval_interface.sampling_rate)
            _speaker_embedding = (
                vits_eval_interface.get_speaker_embedding_from_file(
                    Path(TMP_FILENAME)
                )
            )
            distance = cosine(speaker_embedding, _speaker_embedding)
            distances.append(distance)

        speaker_distances.append(np.mean(distances))

    print(f"Speaker similarity: {np.mean(speaker_distances)}")


def main():
    args = parse_args()
    compute_speaker_similarity(
        input_dir=args.input_dir,
        device=args.device,
        model_root_dir=args.model_root_dir,
        speaker_encoder_checkpoint_path=args.speaker_encoder_checkpoint_path,
        checkpoint_name=args.checkpoint_name,
        texts_file=args.texts_file,
    )


if __name__ == "__main__":
    main()
