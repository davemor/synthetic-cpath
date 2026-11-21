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
from wsipipe.load.slides import SlideBase
from wsipipe_isyntax.load.slides import iSynSlide


class CervicalLoader(Loader):
    @property
    def name(self) -> str:
        return "CervicalLoader"

    def load_annotations(self, file: Path) -> AnnotationSet:
        group_labels = {"low_grade": "low_grade", "Low Grade": "low_grade", "Low grade": "low_grade", "low grade" : "low_grade",
            "high_grade": "high_grade", "High Grade": "high_grade", "High grade": "high_grade", "high grade": "high_grade",
            "malignant": "malignant", "Malignant":"malignant", 
            "Normal/inflammation": "normal", "normal/inflammation": "normal", "normal": "normal"}
        annotations = load_annotations(file, group_labels, "normal") if file else []
        labels_order = ["low_grade",  "high_grade",  "malignant", "normal"]
        maxann = "normal_inflammation"
        for ann in annotations:
            annlab = ann.label
            if annlab == "low_grade":
                if maxann == "normal_inflammation":
                    maxann = annlab
            if annlab == "high_grade":
                if maxann == "normal_inflammation":
                    maxann = annlab
                if maxann =="low_grade":
                    maxann = annlab
            if annlab == "malignant":
                maxann = annlab

        if maxann != label:
            print(f'slide label does not match annotations, slide label is {label}, worst annotation is {maxann}')

        return AnnotationSet(annotations, self.labels, labels_order, "normal")

    def load_slide(self, path: Path) -> SlideBase:
        return iSynSlide(path)

    @property
    def labels(self) -> Dict[str, int]:
        return {"background" : 0, "normal": 1, "low_grade": 2 , "high_grade": 3 ,  "malignant": 4}

