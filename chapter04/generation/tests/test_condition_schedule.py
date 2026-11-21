from collections import Counter

from synpath.manifests import proportional_condition_schedule


def test_proportional_condition_schedule_uses_largest_remainder() -> None:
    rows = [
        {"condition": condition, "image_path": f"{condition}-{index}"}
        for condition, count in (("a", 50), ("b", 30), ("c", 20))
        for index in range(count)
    ]

    selected = proportional_condition_schedule(rows, 10, seed=0)

    assert Counter(row["condition"] for row in selected) == {
        "a": 5,
        "b": 3,
        "c": 2,
    }
    assert selected == proportional_condition_schedule(rows, 10, seed=0)
    assert len({row["image_path"] for row in selected}) == 10


def test_proportional_condition_schedule_resolves_rounding() -> None:
    rows = [
        {"condition": condition, "image_path": condition}
        for condition in ("a", "b", "c")
    ]

    selected = proportional_condition_schedule(rows, 2, seed=0)

    assert Counter(row["condition"] for row in selected) == {"a": 1, "b": 1}
