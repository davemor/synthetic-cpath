"""Model loading for patch embedding."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

from .env import ensure_hf_token


CallStyle = Literal["tensor", "hf_pixel_values"]


MODEL_ALIASES = {
    "alexnet": "torchvision:alexnet",
    "convnext_tiny": "torchvision:convnext_tiny",
    "densenet121": "torchvision:densenet121",
    "efficientnet_b0": "torchvision:efficientnet_b0",
    "resnet18": "torchvision:resnet18",
    "resnet50": "torchvision:resnet50",
    "vit_b_16": "torchvision:vit_b_16",
    "uni": "timm:hf-hub:MahmoodLab/UNI",
    "dinov2-small": "hf:facebook/dinov2-small",
    "dinov2-base": "hf:facebook/dinov2-base",
    "dinov2-large": "hf:facebook/dinov2-large",
    "dinov2": "hf:facebook/dinov2-large",
    "phikon": "hf:owkin/phikon",
    "phikon-v2": "hf:owkin/phikon-v2",
    "hibou": "hf:histai/hibou-L",
    "hibou-l": "hf:histai/hibou-L",
    "hibou_L": "hf:histai/hibou-L",
    "lunit-dino": "timm:hf-hub:1aurent/vit_small_patch16_224.lunit_dino",
    "uni2": "timm:hf-hub:MahmoodLab/UNI2-h",
    "uni2_h": "timm:hf-hub:MahmoodLab/UNI2-h",
}

UNI2_H_SOURCE = "hf-hub:MahmoodLab/UNI2-h"
UNI_SOURCE = "hf-hub:MahmoodLab/UNI"


@dataclass(frozen=True)
class ModelSpec:
    backend: str
    source: str


@dataclass
class LoadedEmbeddingModel:
    name: str
    backend: str
    source: str
    model: Any
    preprocess: Callable[[Any], Any]
    device: str
    call_style: CallStyle = "tensor"

    def encode_tensor(self, batch: Any) -> Any:
        if self.call_style == "hf_pixel_values":
            return self.model(pixel_values=batch.to(self.device))
        return self.model(batch.to(self.device))


def available_model_aliases() -> dict[str, str]:
    """Return built-in friendly model aliases."""
    return dict(MODEL_ALIASES)


def resolve_model_spec(model: str) -> ModelSpec:
    """Resolve aliases and infer a backend when the user omits one."""
    resolved = MODEL_ALIASES.get(model, model)
    if resolved.startswith("hf-hub:"):
        return ModelSpec(backend="timm", source=resolved)
    if ":" in resolved:
        backend, source = resolved.split(":", 1)
        if backend in {"tv", "torchvision"}:
            return ModelSpec(backend="torchvision", source=source)
        if backend in {"hf", "huggingface", "transformers"}:
            return ModelSpec(backend="hf", source=source)
        if backend == "timm":
            return ModelSpec(backend="timm", source=source)
        raise ValueError(f"Unknown embedding model backend: {backend}")
    if "/" in resolved:
        return ModelSpec(backend="hf", source=resolved)
    return ModelSpec(backend="torchvision", source=resolved)


def load_embedding_model(
    model: str,
    *,
    device: str = "auto",
    env_file: Path | str = ".env",
    pretrained: bool = True,
) -> LoadedEmbeddingModel:
    """Load an embedding model and preprocessing transform.

    Model strings can be friendly aliases such as ``resnet18`` or explicit
    references like ``torchvision:resnet50``, ``hf:facebook/dinov2-small``, or
    ``timm:hf-hub:1aurent/vit_small_patch16_224.lunit_dino``.
    """
    spec = resolve_model_spec(model)
    ensure_hf_token(env_file)
    selected_device = resolve_device(device)

    if spec.backend == "torchvision":
        bundle = _load_torchvision(spec.source, pretrained=pretrained)
    elif spec.backend == "hf":
        bundle = _load_huggingface(spec.source)
    elif spec.backend == "timm":
        bundle = _load_timm(spec.source, pretrained=pretrained)
    else:
        raise ValueError(f"Unknown embedding model backend: {spec.backend}")

    bundle.name = model
    bundle.backend = spec.backend
    bundle.source = spec.source
    bundle.device = selected_device
    bundle.model.to(selected_device)
    bundle.model.eval()
    return bundle


def resolve_device(device: str) -> str:
    """Resolve ``auto`` to the best available PyTorch device."""
    _require_torch()
    import torch

    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_torchvision(source: str, *, pretrained: bool) -> LoadedEmbeddingModel:
    _require_torchvision()
    import torch.nn as nn
    from torchvision import models, transforms

    weights = None
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    if pretrained:
        weights_enum = models.get_model_weights(source)
        weights = weights_enum.DEFAULT
        preprocess = weights.transforms()

    model = models.get_model(source, weights=weights)
    _strip_classification_head(model, nn)
    return LoadedEmbeddingModel(
        name=source,
        backend="torchvision",
        source=source,
        model=model,
        preprocess=preprocess,
        device="cpu",
    )


def _load_huggingface(source: str) -> LoadedEmbeddingModel:
    _require_transformers()
    from transformers import AutoImageProcessor, AutoModel

    trust_remote_code = source.startswith("histai/hibou")
    processor = AutoImageProcessor.from_pretrained(
        source, trust_remote_code=trust_remote_code
    )
    model = AutoModel.from_pretrained(source, trust_remote_code=trust_remote_code)

    def preprocess(image: Any) -> Any:
        encoded = processor(images=image.convert("RGB"), return_tensors="pt")
        return encoded["pixel_values"].squeeze(0)

    return LoadedEmbeddingModel(
        name=source,
        backend="hf",
        source=source,
        model=model,
        preprocess=preprocess,
        device="cpu",
        call_style="hf_pixel_values",
    )


def _load_timm(source: str, *, pretrained: bool) -> LoadedEmbeddingModel:
    _require_timm()
    import timm

    if source == UNI_SOURCE:
        model = _load_uni(timm, pretrained=pretrained)
    elif source == UNI2_H_SOURCE:
        model = _load_uni2_h(timm, pretrained=pretrained)
    else:
        try:
            model = timm.create_model(
                source, pretrained=pretrained, num_classes=0
            )
        except TypeError:
            model = timm.create_model(source, pretrained=pretrained)

    data_config = timm.data.resolve_model_data_config(model)
    preprocess = timm.data.create_transform(**data_config, is_training=False)

    return LoadedEmbeddingModel(
        name=source,
        backend="timm",
        source=source,
        model=model,
        preprocess=preprocess,
        device="cpu",
    )


def _load_uni2_h(timm: Any, *, pretrained: bool) -> Any:
    import torch

    timm_kwargs = {
        "img_size": 224,
        "patch_size": 14,
        "depth": 24,
        "num_heads": 24,
        "init_values": 1e-5,
        "embed_dim": 1536,
        "mlp_ratio": 2.66667 * 2,
        "num_classes": 0,
        "no_embed_class": True,
        "mlp_layer": timm.layers.SwiGLUPacked,
        "act_layer": torch.nn.SiLU,
        "reg_tokens": 8,
        "dynamic_img_size": True,
    }
    try:
        return timm.create_model(
            UNI2_H_SOURCE, pretrained=pretrained, **timm_kwargs
        )
    except Exception as exc:
        raise RuntimeError(
            "Failed to load UNI2-h from Hugging Face. Check that a valid "
            "HF_TOKEN or HUGGING_FACE_HUB_TOKEN is available and that the "
            "MahmoodLab/UNI2-h gated model terms have been accepted for that "
            "account."
        ) from exc


def _load_uni(timm: Any, *, pretrained: bool) -> Any:
    try:
        return timm.create_model(
            UNI_SOURCE,
            pretrained=pretrained,
            init_values=1e-5,
            num_classes=0,
            dynamic_img_size=True,
        )
    except Exception as exc:
        raise RuntimeError(
            "Failed to load UNI from Hugging Face. Check that a valid "
            "HF_TOKEN or HUGGING_FACE_HUB_TOKEN is available and that the "
            "MahmoodLab/UNI gated model terms have been accepted for that "
            "account."
        ) from exc


def _strip_classification_head(model: Any, nn: Any) -> None:
    """Replace common torchvision classifier heads with identity modules."""
    for attr in ("fc", "head", "heads"):
        if hasattr(model, attr):
            setattr(model, attr, nn.Identity())
            return

    if not hasattr(model, "classifier"):
        return

    classifier = getattr(model, "classifier")
    if isinstance(classifier, nn.Sequential) and len(classifier) > 0:
        classifier[-1] = nn.Identity()
    else:
        setattr(model, "classifier", nn.Identity())


def _require_torch() -> None:
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Patch embedding requires torch. Install the embedding extras with "
            "`uv pip install -e '.[embed]'` or install torch manually."
        ) from exc


def _require_torchvision() -> None:
    _require_torch()
    try:
        import torchvision  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "torchvision models require torchvision. Install with "
            "`uv pip install -e '.[embed]'`."
        ) from exc


def _require_transformers() -> None:
    _require_torch()
    try:
        import transformers  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Hugging Face models require transformers. Install with "
            "`uv pip install -e '.[embed]'`."
        ) from exc


def _require_timm() -> None:
    _require_torch()
    try:
        import timm  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "timm models require timm. Install with "
            "`uv pip install -e '.[embed]'`."
        ) from exc
