import argparse
from pathlib import Path

from TTS.serving.exporter import TTSTritonExporter


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--export_dir",
        type=Path,
        default=Path.cwd() / "model_repository",
    )
    parser.add_argument(
        "-ckpt",
        "--checkpoint_path",
        type=Path,
        help="Path to VITS checkpoint.",
        required=True,
    )
    parser.add_argument(
        "-sckpt",
        "--speaker_encoder_checkpoint_path",
        type=Path,
        help="Path to speaker encoder checkpoint.",
        required=True,
    )
    parser.add_argument(
        "-v",
        "--version",
        help="Optional version of exporting model.",
        type=str,
        default=None,
        required=False,
    )
    parser.add_argument(
        "--device",
        help="Specify device e.g. cpu or cuda:<gpu_num>",
        type=str,
        default="cuda:0",
    )
    return parser.parse_args()


def export_vc_model(
    export_dir: Path,
    checkpoint_path: Path,
    speaker_encoder_checkpoint_path: Path = None,
) -> None:
    exporter = TTSTritonExporter(
        checkpoint_path=checkpoint_path,
        speaker_encoder_checkpoint_path=speaker_encoder_checkpoint_path,
    )
    exporter.export(export_dir)


def main() -> None:
    args = _parse_args()
    export_vc_model(
        export_dir=args.export_dir,
        checkpoint_path=args.checkpoint_path,
        speaker_encoder_checkpoint_path=args.speaker_encoder_checkpoint_path,
    )


if __name__ == "__main__":
    main()
