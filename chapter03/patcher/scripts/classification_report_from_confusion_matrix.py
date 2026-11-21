#!/usr/bin/env python3
"""Print an sklearn classification report from a confusion matrix.

Rows are true labels and columns are predicted labels.
"""

from __future__ import annotations

import argparse

from sklearn.metrics import classification_report


def expand_confusion_matrix(
    matrix: list[list[int]], labels: list[str]
) -> tuple[list[str], list[str]]:
    y_true: list[str] = []
    y_pred: list[str] = []

    for true_idx, row in enumerate(matrix):
        for pred_idx, count in enumerate(row):
            if count < 0:
                raise ValueError("confusion matrix counts must be non-negative")

            y_true.extend([labels[true_idx]] * count)
            y_pred.extend([labels[pred_idx]] * count)

    return y_true, y_pred


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Produce an sklearn classification report from a confusion matrix."
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        required=True,
        help="Class labels, in the same order as the confusion matrix rows/columns.",
    )
    parser.add_argument(
        "--matrix",
        nargs="+",
        type=int,
        required=True,
        help="Flattened square confusion matrix in row-major order.",
    )
    parser.add_argument(
        "--digits",
        type=int,
        default=2,
        help="Number of digits for sklearn's text report.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    label_count = len(args.labels)
    expected_values = label_count * label_count

    if len(args.matrix) != expected_values:
        raise SystemExit(
            f"expected {expected_values} matrix values for {label_count} labels, "
            f"got {len(args.matrix)}"
        )

    values = iter(args.matrix)
    matrix = [list(row) for row in zip(*[values] * label_count, strict=True)]
    y_true, y_pred = expand_confusion_matrix(matrix, args.labels)

    print(
        classification_report(
            y_true,
            y_pred,
            labels=args.labels,
            target_names=args.labels,
            digits=args.digits,
            zero_division=0,
        )
    )


if __name__ == "__main__":
    main()
