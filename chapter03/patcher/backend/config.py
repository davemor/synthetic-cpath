from typing import Optional

from pydantic import BaseModel, Field
import yaml


class DatasetConfig(BaseModel):
    index: str
    labels: str
    slides_dir: Optional[str] = None
    annotations_dir: Optional[str] = None


class AppConfig(BaseModel):
    dataset: DatasetConfig
    patch_size: int
    patch_level: int
    patchsets_dir: str
    thumbnails_dir: str
    patches_dir: str
    annotation_sets_dir: str = "annotations"
    annotators: list[str] = Field(
        default_factory=lambda: ["annotator_1", "annotator_2", "annotator_3"]
    )
    patch_display_size: int
    default_tissue_threshold: float = 0.05
    scrollable_thumbnail_viewport: bool = True


def load_app_config(config_dict: dict) -> AppConfig:
    """
    Load AppConfig from a dictionary.
    """
    return AppConfig(**config_dict)


def load_app_config_from_yaml(yaml_path: str) -> AppConfig:
    """
    Load AppConfig from a YAML file.
    """
    with open(yaml_path, "r") as f:
        config_dict = yaml.safe_load(f)
    return load_app_config(config_dict)
