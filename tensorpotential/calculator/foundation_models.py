from __future__ import annotations

import os
import urllib
import tarfile

import keyword

FOUNDATION_CACHE_DIR = os.path.expanduser("~/.cache/grace")


def to_valid_variable_name(name):
    # Replace invalid characters with underscores
    valid_name = "".join(
        char if char.isalnum() or char == "_" else "_" for char in name
    )

    # Ensure the name starts with a letter or an underscore
    if not valid_name[0].isalpha() and valid_name[0] != "_":
        valid_name = "_" + valid_name

    # Avoid reserved keywords
    if keyword.iskeyword(valid_name):
        valid_name += "_"

    return valid_name


MODELS_METADATA = {
    #######################
    # 1-layer
    #######################
    "MP_GRACE_1L_r6_4Nov2024": {
        "url": "https://ruhr-uni-bochum.sciebo.de/s/zvkS6j3OdaiF48l/download",
        "dirname": "MP-GRACE-1L-r6_4Nov2024",
        "description": "A one-layer local GRACE model parameterized on MPTraj dataset, with fixed 6 Å cutoff.",
        "license": "Academic Software License",
    },
    "MP_GRACE_1L_r6_07Nov2024": {
        "url": "https://ruhr-uni-bochum.sciebo.de/s/d92rGQiRlieY6tJ/download",
        "dirname": "MP_GRACE_1L_r6_07Nov2024",
        "description": "A one-layer local GRACE model parameterized on MPTraj dataset, with fixed 6 Å cutoff.",
        "license": "Academic Software License",
    },
    #######################
    # 2-layer
    #######################3
    "MP_GRACE_2L_r6_11Nov2024": {
        "url": "https://ruhr-uni-bochum.sciebo.de/s/42Ivgi3eaLCynwC/download",
        "dirname": "MP_GRACE_2L_r6_11Nov2024",
        "description": "A two-layer semi-local GRACE model parameterized on MPTraj dataset, with fixed 6 Å cutoff. (version: 11 Nov 2024)",
        "license": "Academic Software License",
    },
    "MP_GRACE_2L_r5_4Nov2024": {
        "url": "https://ruhr-uni-bochum.sciebo.de/s/CJgOzdmGwcjDUPM/download",
        "dirname": "MP-GRACE-2L-r5_4Nov2024",
        "description": """A two-layer semi-local GRACE model parameterized on MPTraj dataset, with fixed 5 Å cutoff."""
        """ Currently the best among GRACE models on MatBench Discovery leaderboard.""",
        "license": "Academic Software License",
    },
    "MP_GRACE_2L_r5_07Nov2024": {
        "url": "https://ruhr-uni-bochum.sciebo.de/s/nWyVO8xC38DJw4z/download",
        "dirname": "MP_GRACE_2L_r5_07Nov2024",
        "description": """A two-layer semi-local GRACE model parameterized on MPTraj dataset, with fixed 5 Å cutoff."""
        """ Currently the best among GRACE models on MatBench Discovery leaderboard.""",
        "license": "Academic Software License",
    },
    ########################################################
    ### Pre-ftited on OMat24, fine tuned on sAlex+MPTraj  ##
    ########################################################
    "GRACE_2L_OAM_28Jan25": {
        "url": "https://ruhr-uni-bochum.sciebo.de/s/4zVTfzxornWfS4T/download",
        "dirname": "GRACE-2L-OAM_28Jan25",
        "description": """A two-layer semi-local GRACE model, pre-fitted on the OMat24 and fine-tuned on sAlex+MPTraj datasets, """
        """with fixed 6 Å cutoff.""",
        "license": "Academic Software License",
    },
    "GRACE-1L-OAM_2Feb25": {
        "url": "https://ruhr-uni-bochum.sciebo.de/s/gFAv8pX2DJbk1kb/download",
        "dirname": "GRACE-1L-OAM_2Feb25",
        "description": """A single-layer local GRACE model, pre-fitted on the OMat24 and fine-tuned on sAlex+MPTraj datasets, """
        """with fixed 6 Å cutoff.""",
        "license": "Academic Software License",
    },
}

# automatically try to import experimental models
try:
    from tensorpotential.experimental.calculator.foundation_models import (
        MODELS_METADATA as MODEL_METADATA_EXPERIMENTAL,
    )

    MODELS_METADATA.update(MODEL_METADATA_EXPERIMENTAL)
except ImportError:
    pass

MODELS_NAME_LIST = list(MODELS_METADATA.keys())


def download_fm(model):
    model_path = os.path.join(FOUNDATION_CACHE_DIR, MODELS_METADATA[model]["dirname"])
    if not os.path.isdir(model_path):
        # download model
        checkpoint_url = MODELS_METADATA[model]["url"]
        os.makedirs(FOUNDATION_CACHE_DIR, exist_ok=True)
        # download and save to disk
        print(f"Downloading GRACE models from {checkpoint_url!r}")
        # Define the file path
        filename = os.path.join(FOUNDATION_CACHE_DIR, "tmp.model.tar.gz")

        local_filename, http_msg = urllib.request.urlretrieve(checkpoint_url, filename)
        if "Content-Type: text/html" in http_msg:
            raise RuntimeError(
                f"Model download failed, please check the URL {checkpoint_url}"
            )

        # Unpack the .tar.gz file
        print(f"Unpacking model from {checkpoint_url!r}")
        with tarfile.open(local_filename, "r:gz") as tar:
            tar.extractall(path=FOUNDATION_CACHE_DIR)
        os.remove(local_filename)
        assert os.path.isdir(model_path), f"Model path {model_path} does not exist"
        print(f"GRACE model downloaded to cache {model_path}")
    else:
        print(f"Using cached GRACE model from {model_path}")
    print(f"Model license: {MODELS_METADATA[model].get('license', 'not provided')}")
    return model_path


def grace_fm(
    model: str,
    pad_neighbors_fraction: float | None = 0.25,
    pad_atoms_number: int | None = 10,
    max_number_reduction_recompilation: int | None = 2,
    min_dist=None,
    **kwargs
):
    from tensorpotential.calculator.asecalculator import TPCalculator

    assert model in MODELS_NAME_LIST, f"model must be in {MODELS_NAME_LIST}"

    model_path = download_fm(model)

    calc = TPCalculator(
        model=model_path,
        pad_neighbors_fraction=pad_neighbors_fraction,
        pad_atoms_number=pad_atoms_number,
        max_number_reduction_recompilation=max_number_reduction_recompilation,
        min_dist=min_dist,
        **kwargs
    )
    return calc


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


class GRACEModels:
    """Namespace class to hold all model names"""

    pass


for model_name in MODELS_NAME_LIST:
    model_name_clean = to_valid_variable_name(model_name)
    setattr(GRACEModels, model_name_clean, model_name)
