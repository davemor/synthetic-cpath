from pathlib import Path


from wsipipe.load.datasets import Loader
from wsipipe.preprocess.patching.patchset import PatchSet, PatchSetting

def load_patchsets_from_directory(patchsets_dir: Path):
    """Loads PatchSets from a directory

    Loads patchsets for a whole dataset stored in subdirectories of patchsets_dir

    Args:
        patchsets_dir (Path): a path to a directory containing subdirectories with PatchSets
    Returns:
        patchset (List[PatchSet]): A list of PatchSets one for each slide
    """
    patchset_dir_list = [x for x in Path(patchsets_dir).iterdir() if x.is_dir()]
    patchset_list = [PatchSet.load(p) for p in patchset_dir_list]
    return patchset_list