from icaird_cervical_benchmark.config import load_config


def test_default_probe_classifiers_are_from_hproj_generalisation(tmp_path):
    config_path = tmp_path / "config.yml"
    config_path.write_text(
        """
data:
  dataset_root: /tmp/example
probes:
  embeddings: {}
"""
    )

    config = load_config(config_path)

    assert config.probes.classifiers["logistic-regression"].params["C"] == [1, 100]
    assert config.probes.classifiers["mlp"].params["hidden_layer_sizes"] == [[100, 50]]
    assert config.probes.classifiers["xgboost"].search.n_iter == 12


def test_image_augmentation_config_accepts_multiple_suites(tmp_path):
    config_path = tmp_path / "config.yml"
    config_path.write_text(
        """
data:
  dataset_root: /tmp/example
images:
  models: [resnet18]
  augmentation_suites: [geometric, colour, hed, combined]
  augmentation_probability: 0.5
  hed_sigma: 0.02
probes:
  enabled: false
"""
    )

    config = load_config(config_path)

    assert config.images.models == ["resnet18"]
    assert config.images.augmentation_suites == ["geometric", "colour", "hed", "combined"]
    assert config.images.augmentation_probability == 0.5
    assert config.images.hed_sigma == 0.02
