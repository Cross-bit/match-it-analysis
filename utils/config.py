from pathlib import Path
import os
import pickle

def find_project_root(marker="img") -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / marker).exists():
            return parent
    raise RuntimeError(f"Project root not found (no {marker}/ folder above {__file__})")

THESIS_PROJECT_ROOT = find_project_root("img") # first directory that contains the img dir
ANALYSIS_PROJECT_ROOT = (
    THESIS_PROJECT_ROOT
    if THESIS_PROJECT_ROOT.name == "analysis"
    else THESIS_PROJECT_ROOT / "analysis"
)

IMG_OUTPUT_PATH = THESIS_PROJECT_ROOT / "img"

CACHE_FILES_DIR = ANALYSIS_PROJECT_ROOT / "cache"
DATASET_ROOT = ANALYSIS_PROJECT_ROOT / "dataset"
MOVIES_DATASET_ROOT = DATASET_ROOT / "movies"
RESTAURANT_DATASET_ROOT = DATASET_ROOT / "restaurants"


def cache_path(*parts: str) -> Path:
    """
    Build a path under shared analysis cache dir.
    Example: cache_path("movies", "init-sampling", "result.pkl")
    """
    return CACHE_FILES_DIR.joinpath(*parts)

HISTOGRAM_EDGECOLOR_1 = "#1F3A93"
HISTOGRAM_COLOR_1 = "#4A90E2"

HISTOGRAM_EDGECOLOR_RED = "#931F1F"
HISTOGRAM_COLOR_RED     = "#E24A4A"

HISTOGRAM_EDGECOLOR_GREEN = "#1F9334"
HISTOGRAM_COLOR_GREEN     = "#4AE275"

AXIS_DESC_SIZE = 16
AXIS_VALS_SIZE = 15
TITLE_SIZE = 20


def load_or_build_pickle(path_or_name: str, builder_fn, *, description: str = "", save: bool = True, force_rebuild: bool = False):
    """
    Load object from cached pickle if exists, otherwise build using builder_fn and optionally save.

    Args:
        path (str): Path to pickle file.
        builder_fn (callable): Function that builds the object if pickle doesn't exist.
        description (str): Optional description to print during load/save.
        save (bool): Whether to save result after building.
        force_rebuild (bool): If True, ignore existing pickle and always rebuild.

    Returns:
        Any: Loaded or built object.
    """

    path = Path(path_or_name)
    if not path.is_absolute():
        path = CACHE_FILES_DIR / path

    if os.path.exists(path) and not force_rebuild:
        with open(path, "rb") as f:
            print(f"✅ Loaded {description or 'object'} from pickle → {os.path.basename(path)}")
            return pickle.load(f)

    # Build new result
    print(f"⚙️  Building {description or 'object'}...")
    result = builder_fn()

    if save:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(result, f)
        print(f"💾 Saved {description or 'object'} to pickle → {os.path.basename(path)}")

    return result


def load_from_pickle(path_or_name: str, *, description: str = ""):
    """
    Load an object from a pickle file.

    Args:
        path_or_name (str): Path to the pickle file.
        description (str): Optional description for logging.

    Returns:
        Any: The loaded object.

    Raises:
        FileNotFoundError: If the pickle file does not exist.
    """
    path = Path(path_or_name)
    if not path.is_absolute():
        path = CACHE_FILES_DIR / path  # nebo uprav na vlastní logiku pro výchozí cestu

    if not path.exists():
        raise FileNotFoundError(f"Pickle file not found: {path}")

    with open(path, "rb") as f:
        print(f"✅ Loaded {description or 'object'} from pickle → {path.name}")
        return pickle.load(f)


def save_to_pickle(obj, path_or_name: str, *, description: str = "", overwrite: bool = True):
    """
    Save an object to a pickle file.

    Args:
        obj (Any): Object to pickle.
        path_or_name (str): Path to the pickle file (absolute or relative to CACHE_FILES_DIR).
        description (str): Optional description for logging.
        overwrite (bool): If False, raises FileExistsError if file already exists.

    Returns:
        pathlib.Path: Path where the object was saved.
    """
    path = Path(path_or_name)
    if not path.is_absolute():
        path = CACHE_FILES_DIR / path  # navázání na tvůj cache dir

    if path.exists() and not overwrite:
        raise FileExistsError(f"File already exists: {path}")

    os.makedirs(path.parent, exist_ok=True)

    with open(path, "wb") as f:
        pickle.dump(obj, f)

    print(f"💾 Saved {description or 'object'} to pickle → {path.name}")
    return path