from collections import Counter

import pytest

from synpath.utility.resnet import stratified_limit_rows


def rows(counts: list[int]) -> list[dict[str, str]]:
    return [
        {"image_path": f"{label}-{index}.png", "label_id": str(label)}
        for label, count in enumerate(counts)
        for index in range(count)
    ]


def test_stratified_limit_is_exact_deterministic_and_proportional() -> None:
    population = rows([80, 20])

    first = stratified_limit_rows(population, limit=50, seed=7)
    second = stratified_limit_rows(population, limit=50, seed=7)

    assert first == second
    assert len(first) == 50
    assert Counter(row["label_id"] for row in first) == {"0": 40, "1": 10}


def test_stratified_limit_rejects_oversized_request() -> None:
    with pytest.raises(ValueError, match="Cannot select"):
        stratified_limit_rows(rows([5, 5]), limit=11, seed=0)


def test_stratified_limit_rejects_non_positive_request() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        stratified_limit_rows(rows([5, 5]), limit=0, seed=0)
