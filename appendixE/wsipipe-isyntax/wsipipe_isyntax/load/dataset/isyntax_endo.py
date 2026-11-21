"""
Loader for the Camelyon 16 dataset.
    - Slides are tiffs read using openslide
    - Annotations are asapxml
    - Output labels for slides are background, normal and tumor

"""

from pathlib import Path
from typing import Dict

from wsipipe.load.annotations import AnnotationSet
from wsipipe.load.annotations.geojson import load_annotations_geojson
from wsipipe.load.datasets.loader import Loader
from wsipipe.load.datasets.registry import register_loader
from wsipipe.load.slides import SlideBase
from wsipipe_isyntax.load.slides import iSynSlide


class EndometrialLoader(Loader):
    @property
    def name(self) -> str:
        return "EndometrialLoader"

    def load_annotations(self, file: Path, label: str) -> AnnotationSet:
        # NOTE: Insufficient is only a slide label not a patch level label
        # Therefore all slides labelled as insufficient should return no patches

        group_labels = {"Malignant":"malignant", "malignant":"malignant",  
                        "Other/benign":"other_benign", "other_benign":"other_benign", "Other":"other_benign", "other":"other_benign", 
                        "Adenocarcinoma": "malignant", "adenocarcinoma":"malignant", 
                        "Sarcoma": "malignant", "sarcoma":"malignant", 
                        "hyperplasia_with_atypia":"malignant", "hyperplasia":"malignant", "Hyperplasia with atypia": "malignant", "Hyperplasia with Atypia": "malignant", "hyperplasia atypia": "malignant", 
                        "CarcinoSarcoma" : "malignant", "carcinosarcoma":"malignant", "Carcinosarcoma": "malignant", 
                        "Hormonal":"other_benign", "hormonal":"other_benign", 
                        "inactive_atrophic":"other_benign", "Inactive/atrophic":"other_benign","inactive":"other_benign", "Inactive/Atrophic":"other_benign", "Innactive/Atrophic":"other_benign",
                        "Menstrual":"other_benign", "menstrual":"other_benign",
                        "Proliferative":"other_benign", "proliferative":"other_benign", 
                        "Secretory":"other_benign", "secretory":"other_benign", 
                        "Cervical Tissue": "other_benign", "Cervical": "other_benign", "Cervix": "other_benign",
                        "Normal EndoCervix": "other_benign", 
                        "polyp": "other_benign", "endometrial polyp": "other_benign", 
                        "Hormonal Change": "other_benign", 
                        "EctoCervix": "other_benign", "Normal EctoCervix": "other_benign",
                        "benign": "other_benign",
                        "normal myometrium": "other_benign"}

        if label == "insufficient":
            # insufficient slides should be excluded in the sampler, so this is a placeholder
            label = "other_benign"

        annotations = load_annotations_geojson(file, group_labels, label) if file else []
        labels_order = ["background",  "other_benign",  "malignant"]

        return AnnotationSet(annotations, self.labels, labels_order, label)

    def load_slide(self, path: Path) -> SlideBase:
        return iSynSlide(path)

    @property
    def labels(self) -> Dict[str, int]:
        return {"background" : 0, "other_benign": 1 , "malignant": 2 }


register_loader(EndometrialLoader)