import argparse
import warnings
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import soundfile as sf
from nisqa import NISQA_WEIGHTS_DIR
from nisqa.model import nisqaModel
from tqdm import tqdm

from TTS.vits_eval_interface import VITSEvalInterface

warnings.filterwarnings("ignore", category=UserWarning)


# TODO: I think that is bad idea
TMP_FILENAME = "tmp.wav"
TMP_CSV_FILE_NAME = "nisqa.csv"
OUTPUT_DIR = "nisqa_results"
NISQA_MODEL_NAME = "tmp_model_name"
COLUMN_NAME = "audio"
OUT_COLUMNS = ["mos_pred", "noi_pred", "dis_pred", "col_pred", "loud_pred"]

NISQA_ARGS = {
    "pretrained_model": str(NISQA_WEIGHTS_DIR / "nisqa.tar"),
    "mode": "predict_csv",
    "csv_deg": COLUMN_NAME,
    "csv_file": TMP_CSV_FILE_NAME,
    "data_dir": NISQA_MODEL_NAME,
    "ms_channel": 0,
    "output_dir": OUTPUT_DIR,
}


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
        default=Path(__file__).parent / "script_data" / "mos_corpus.txt",
    )
    return arguments_parser.parse_args()


def compute_mos_nisqa(
    input_dir: Path,
    device: str,
    model_root_dir: Path,
    speaker_encoder_checkpoint_path: Path,
    checkpoint_name: str = "best_model.pth",
    max_speakers_num: int = 10,
    texts_file: Path = Path(__file__).parent / "script_data" / "mos_corpus.txt",
) -> Tuple[float, float, float, float, float]:
    vits_eval_interface = VITSEvalInterface(
        device=device,
        model_root_dir=model_root_dir,
        speaker_encoder_checkpoint_path=speaker_encoder_checkpoint_path,
        checkpoint_name=checkpoint_name,
    )

    tmp_output_dir = Path(NISQA_MODEL_NAME)
    tmp_output_dir.mkdir(parents=True, exist_ok=True)

    test_texts = texts_file.read_text().splitlines()

    output_file_names = []
    speaker_directories = [
        spk_dir for spk_dir in input_dir.iterdir() if spk_dir.is_dir()
    ]
    for speaker_dir in tqdm(
        speaker_directories[:max_speakers_num], desc="TTS Inference..."
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

        for i, text in enumerate(test_texts):
            audio = vits_eval_interface(text, speaker_embedding)

            audio_name = f"{speaker_dir.name}_{i}.wav"
            output_file_names.append(audio_name)

            sf.write(
                tmp_output_dir / audio_name,
                audio,
                vits_eval_interface.sampling_rate,
            )

    df = pd.DataFrame(output_file_names, columns=[COLUMN_NAME])
    df.to_csv(Path(NISQA_MODEL_NAME) / TMP_CSV_FILE_NAME)

    nisqa = nisqaModel(NISQA_ARGS)
    nisqa_predictions = nisqa.predict()[OUT_COLUMNS].mean()

    print("\nPredictions:")
    print(nisqa_predictions)

    return nisqa_predictions.tolist()


def main():
    args = parse_args()
    compute_mos_nisqa(
        input_dir=args.input_dir,
        device=args.device,
        model_root_dir=args.model_root_dir,
        speaker_encoder_checkpoint_path=args.speaker_encoder_checkpoint_path,
        checkpoint_name=args.checkpoint_name,
        texts_file=args.texts_file,
        max_speakers_num=args.max_speakers_num,
    )


if __name__ == "__main__":
    main()
