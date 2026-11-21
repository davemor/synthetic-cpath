import pytest
import torch
from PIL import Image

from icaird_cervical_benchmark.images import (
    _ControlledAugmentationSampler,
    _augmentation_operations_for_suite,
    _build_transform,
)


@pytest.mark.parametrize("suite", ["none", "geometric", "colour", "hed", "combined"])
def test_image_transform_suites_return_normalized_tensor(suite):
    image = Image.new("RGB", (32, 40), color=(180, 80, 120))
    transform = _build_transform(
        24,
        suite,
        train=True,
        probability=1.0,
        hed_sigma=0.03,
    )

    transformed = transform(image)

    assert isinstance(transformed, torch.Tensor)
    assert transformed.shape == (3, 24, 24)
    assert torch.isfinite(transformed).all()


def test_image_transform_rejects_unknown_augmentation_suite():
    with pytest.raises(ValueError, match="Unknown image augmentation suite"):
        _build_transform(24, "unknown", train=True)


def test_eval_transform_does_not_include_augmentation_sampler():
    transform = _build_transform(24, "combined", train=False, probability=1.0)

    assert all(
        not isinstance(step, _ControlledAugmentationSampler)
        for step in transform.transforms
    )


def test_controlled_sampler_respects_probability_zero():
    calls = []

    class Recorder:
        def __call__(self, image):
            calls.append("called")
            return image

    sampler = _ControlledAugmentationSampler([Recorder()], probability=0.0)
    image = Image.new("RGB", (32, 32))

    assert sampler(image) is image
    assert calls == []


def test_controlled_sampler_samples_bounded_subset():
    calls = []

    class Recorder:
        def __init__(self, name):
            self.name = name

        def __call__(self, image):
            calls.append(self.name)
            return image

    sampler = _ControlledAugmentationSampler(
        [Recorder(str(index)) for index in range(10)],
        probability=1.0,
        min_transforms=1,
        max_transforms=3,
    )

    sampler(Image.new("RGB", (32, 32)))

    assert 1 <= len(calls) <= 3
    assert len(calls) == len(set(calls))


def test_augmentation_groups_match_table_s1_design():
    geometric = _augmentation_operations_for_suite("geometric", hed_sigma=0.03)
    colour = _augmentation_operations_for_suite("colour", hed_sigma=0.03)
    hed = _augmentation_operations_for_suite("hed", hed_sigma=0.03)
    combined = _augmentation_operations_for_suite("combined", hed_sigma=0.03)

    assert len(geometric) == 6
    assert len(colour) == 5
    assert len(hed) == 1
    assert len(combined) == len(geometric) + len(colour) + len(hed)
