import os
import urllib
import tarfile
from tensorpotential.calculator.asecalculator import TPCalculator

MODELS_METADATA = {
    "mp-1layer": {
        "name": "train_1.5M_test_75_grace_1layer_v2_7Aug2024",
        "url": "https://ruhr-uni-bochum.sciebo.de/s/pS62iMsFZuFrI5K/download",
    },
    "mp-1layer-shift": {
        "url": "https://ruhr-uni-bochum.sciebo.de/s/ElIUhF6EOH5s8oA/download",
        "name": "train_1.5M_test_75_grace_1L_AB_cont_10Aug24",
    },
    "mp-2layer": {
        "url": "https://ruhr-uni-bochum.sciebo.de/s/TSBVN8P6vn0TbGw/download",
        "name": "train_1.5M_test_75_grace_2layer_8Aug2024",
    },
}

MODELS_NAME_LIST = list(MODELS_METADATA.keys())


def grace_fm(
    model: str,
    pad_neighbors_fraction: float = 0.05,
    pad_atoms_number: int = 1,
    min_dist=None,
) -> TPCalculator:

    assert model in MODELS_NAME_LIST, f"model must be in {MODELS_NAME_LIST}"

    model_path = download_fm(model)

    calc = TPCalculator(
        model=model_path,
        pad_neighbors_fraction=pad_neighbors_fraction,
        pad_atoms_number=pad_atoms_number,
        min_dist=min_dist,
    )
    return calc


def download_fm(model):
    cache_dir = os.path.expanduser("~/.cache/grace")
    model_path = os.path.join(cache_dir, MODELS_METADATA[model]["name"])
    if not os.path.isdir(model_path):
        # download model
        checkpoint_url = MODELS_METADATA[model]["url"]
        os.makedirs(cache_dir, exist_ok=True)
        # download and save to disk
        print(f"Downloading GRACE models from {checkpoint_url!r}")
        # Define the file path
        filename = os.path.join(cache_dir, "tmp.model.tar.gz")

        local_filename, http_msg = urllib.request.urlretrieve(checkpoint_url, filename)
        if "Content-Type: text/html" in http_msg:
            raise RuntimeError(
                f"Model download failed, please check the URL {checkpoint_url}"
            )

        # Unpack the .tar.gz file
        print(f"Unpacking model from {checkpoint_url!r}")
        with tarfile.open(local_filename, "r:gz") as tar:
            tar.extractall(path=cache_dir)
        os.remove(local_filename)
        assert os.path.isdir(model_path), f"Model path {model_path} does not exist"
        print(f"GRACE model downloaded to cache {model_path}")
    else:
        print(f"Using cached GRACE model from {model_path}")
    return model_path


grace_fm.__doc__ = f"""
    model:  (str) One of {", ".join(MODELS_NAME_LIST)}
    extend_fake_neighbors_fraction:  (float) Fraction of atoms to extend (padding for JIT)
    extend_fake_atoms_number:  (int) Number of atoms to extend (padding for JIT)
    min_dist:  (float) Minimum distance. Raise exception if atoms are closer
    
    Returns:  ASE calculator (TPCalculator)
"""

download_fm.__doc__ = f"""
Download foundation model 
    model:  (str) One of {", ".join(MODELS_NAME_LIST)}
        
Returns:
    model_path: (str)
"""
