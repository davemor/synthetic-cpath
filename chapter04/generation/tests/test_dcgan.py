from pathlib import Path

from PIL import Image

from synpath.generation.dcgan import sample_dcgan, train_dcgan
from synpath.manifests import CONDITION_COLUMNS, write_manifest


def write_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), color=color).save(path)


def test_dcgan_checkpoint_and_sampling(tmp_path):
    image_a = tmp_path / "images" / "a.png"
    image_b = tmp_path / "images" / "b.png"
    write_image(image_a, (120, 80, 40))
    write_image(image_b, (40, 80, 120))
    manifest = tmp_path / "condition_manifest.csv"
    write_manifest(
        manifest,
        [
            {
                "image_path": image_a,
                "label": "normal",
                "label_id": 0,
                "cluster_id": "",
                "condition": "normal",
                "condition_id": 0,
                "prompt": "a histopathology patch of normal",
            },
            {
                "image_path": image_b,
                "label": "tumor",
                "label_id": 1,
                "cluster_id": "",
                "condition": "tumor",
                "condition_id": 1,
                "prompt": "a histopathology patch of tumor",
            },
        ],
        CONDITION_COLUMNS,
    )
    cfg = {
        "images": {"generator_size": 16},
        "generation": {
            "dcgan": {
                "image_size": 16,
                "latent_dim": 8,
                "base_channels": 4,
                "batch_size": 2,
                "epochs": 0,
                "sample_batch_size": 2,
                "num_workers": 0,
                "use_data_parallel": False,
            }
        },
        "metrics": {"seed": 7},
    }

    outputs = train_dcgan(
        tmp_path / "run",
        manifest,
        cfg,
        conditional=True,
        seed=7,
    )
    sample_dcgan(
        tmp_path / "run",
        tmp_path / "run" / "samples",
        manifest,
        cfg,
        2,
        seed=7,
    )

    assert outputs[0].name == "dcgan.pt"
    assert (tmp_path / "run" / "samples" / "00000000.png").exists()
    assert (tmp_path / "run" / "samples" / "00000001.png").exists()
