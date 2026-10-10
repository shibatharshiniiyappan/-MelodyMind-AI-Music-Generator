import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.preprocessing import prepare_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract pitch/duration LSTM sequences from a MIDI dataset."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "midi",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "data"
        / "processed"
        / "sequences.npz",
    )
    parser.add_argument("--sequence-length", type=int, default=32)
    args = parser.parse_args()

    summary = prepare_dataset(args.input, args.output, args.sequence_length)
    print(
        f"Prepared {summary.sequences} sequences from {summary.notes} notes "
        f"across {summary.files} MIDI files."
    )


if __name__ == "__main__":
    main()
