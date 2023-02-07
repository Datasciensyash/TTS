import argparse
import warnings
from pathlib import Path

import pandas as pd
from nisqa import NISQA_WEIGHTS_DIR
from nisqa.model import nisqaModel

warnings.filterwarnings("ignore", category=UserWarning)


# TODO: I think that is bad idea
TMP_FILENAME = "tmp.wav"
TMP_CSV_FILE_NAME = "tmp_nisqa.csv"
OUTPUT_DIR = "nisqa_results"
COLUMN_NAME = "audio"
OUT_COLUMNS = ["mos_pred", "noi_pred", "dis_pred", "col_pred", "loud_pred"]

NISQA_ARGS = {
    "pretrained_model": str(NISQA_WEIGHTS_DIR / "nisqa.tar"),
    "mode": "predict_csv",
    "csv_deg": COLUMN_NAME,
    "csv_file": TMP_CSV_FILE_NAME,
    "data_dir": None,
    "ms_channel": 0,
    "output_dir": None,
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
        "-i",
        "--input_dir",
        help="Path to the directory with input files",
        type=Path,
        required=True,
    )
    arguments_parser.add_argument(
        "-o",
        "--output_dir",
        help="Path to the output directory",
        type=Path,
        required=False,
        default=Path("nisqa_results")
    )
    return arguments_parser.parse_args()


def compute_mos_nisqa(
        input_dir: Path,
        output_dir: Path,
        device: str,
) -> None:

    df = pd.DataFrame([i.relative_to(input_dir) for i in input_dir.rglob("*.wav")], columns=[COLUMN_NAME])
    df.to_csv(input_dir / TMP_CSV_FILE_NAME, index=False)

    NISQA_ARGS["data_dir"] = str(input_dir)
    NISQA_ARGS["output_dir"] = str(output_dir)

    nisqa = nisqaModel(NISQA_ARGS)
    nisqa_predictions = nisqa.predict()[OUT_COLUMNS].mean()

    Path(input_dir / TMP_CSV_FILE_NAME).unlink()

    nisqa_predictions.to_csv(output_dir / f"nisqa_predictions_{input_dir.name}.csv", index=False)


def main():
    args = parse_args()
    compute_mos_nisqa(
        input_dir=args.input_dir,
        device=args.device,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
