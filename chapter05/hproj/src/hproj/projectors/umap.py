from cuml.manifold import UMAP

from hproj.data.feature_space import FeatureSpace
from hproj.projectors.projector import Projector, ProjectorFactory


@ProjectorFactory.register("umap")
class UMAPProjector(Projector):
    def __init__(self, n_components, seed, subsample: int | None = None, **kwargs):
        self.n_components = n_components
        self.seed = seed
        self.kwargs = kwargs
        self.subsample = subsample

    def fit(self, space: FeatureSpace) -> None:
        self.umap = UMAP(
            n_components=self.n_components, random_state=self.seed, **self.kwargs
        )
        if self.subsample is not None and self.subsample < space.num_samples():
            fit_space = space.stratified_sample(self.subsample)
        else:
            fit_space = space
        self.umap.fit(fit_space.features)

    def transform(self, space: FeatureSpace) -> FeatureSpace:
        features = self.umap.transform(space.features)
        return FeatureSpace(features, space.labels)
