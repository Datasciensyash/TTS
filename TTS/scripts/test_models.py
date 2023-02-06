import argparse
import json
from csv import writer
from pathlib import Path
from typing import List

from TTS.scripts.mos import compute_mos_nisqa
from TTS.scripts.speaker_similarity import compute_speaker_similarity
from TTS.scripts.wer import compute_wer


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
        "--model_dirs",
        help="Paths to the models directories",
        type=Path,
        required=True,
        nargs="+",
    )
    arguments_parser.add_argument(
        "-se",
        "--speaker_encoder_checkpoint_path",
        help="Path to the speaker encoder checkpoint",
        type=Path,
        required=True,
    )
    arguments_parser.add_argument(
        "-dir",
        "--input_dir",
        help="Path to the directory with input files",
        type=Path,
        required=True,
    )
    arguments_parser.add_argument(
        "-out",
        "--out_csv",
        help="Path to the output .csv file",
        type=Path,
        required=False,
        default=Path("test_output.csv"),
    )
    return arguments_parser.parse_args()


def default_model_test(
    device: str,
    model_dirs: List[Path],
    speaker_encoder_checkpoint_path: Path,
    input_dir: Path,
    out_csv: Path,
) -> None:
    with open(out_csv, "w", newline="") as csvfile:
        csvwriter = writer(csvfile)

        wer_files = list((Path(__file__).parent / "script_data").glob("wer*.txt"))
        wer_names = [i.stem for i in wer_files]

        columns = [
            "model",
            "mos_pred",
            "noi_pred",
            "dis_pred",
            "col_pred",
            "loud_pred",
            "speaker_similarity",
        ]
        columns += wer_names

        csvwriter.writerow(columns)

        for model_dir in model_dirs:
            # Find all checkpoints in the model root directory
            checkpoint_paths = [i for i in model_dir.rglob("*.pth") if i.stem != "speakers"]

            for checkpoint_path in checkpoint_paths:
                checkpoint = checkpoint_path.name
                model_root_dir = checkpoint_path.parent

                sentinel = (model_root_dir / checkpoint).with_suffix(".test.json")
                if sentinel.exists():
                    continue
                # Run MOS testing
                # NOTE: compute_mos_nisqa returns Tuple[float, float, float, float, float]
                # NOTE: compute_mos_nisqa returns Tuple of "mos_pred", "noi_pred", "dis_pred", "col_pred", "loud_pred"
                mos_output = compute_mos_nisqa(
                    input_dir=input_dir,
                    model_root_dir=model_root_dir,
                    speaker_encoder_checkpoint_path=speaker_encoder_checkpoint_path,
                    checkpoint_name=checkpoint,
                    device=device,
                )
                mos_pred, noi_pred, dis_pred, col_pred, loud_pred = mos_output

                # Run speaker similarity testing
                speaker_similarity = compute_speaker_similarity(
                    input_dir=input_dir,
                    device=device,
                    model_root_dir=model_root_dir,
                    speaker_encoder_checkpoint_path=speaker_encoder_checkpoint_path,
                    checkpoint_name=checkpoint,
                )
                data_row = [
                    str(model_root_dir / checkpoint),
                    mos_pred,
                    noi_pred,
                    dis_pred,
                    col_pred,
                    loud_pred,
                    speaker_similarity,
                ]

                # Run WER testing
                for file in wer_files:
                    wer, cer = compute_wer(
                        input_dir=input_dir,
                        model_root_dir=model_root_dir,
                        speaker_encoder_checkpoint_path=speaker_encoder_checkpoint_path,
                        checkpoint_name=checkpoint,
                        device=device,
                        texts_file=file,
                        max_speakers_num=2,
                    )
                    data_row.append(wer)
                    data_row.append(cer)

                csvwriter.writerow(data_row)
                with sentinel.open("w") as f:
                    json.dump(dict(zip(columns, data_row)), f)

        print(f"Test results are saved in {out_csv}")


if __name__ == "__main__":
    args = parse_args()
    default_model_test(
        args.device,
        args.model_dirs,
        args.speaker_encoder_checkpoint_path,
        args.input_dir,
        args.out_csv,
    )