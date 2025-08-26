from __future__ import annotations

import os
import urllib
import tarfile

import keyword
from typing import Final

import yaml

# environment variable for GRACE cache. Default will be "~/.cache/grace"
GRACE_CACHE: Final[str] = "GRACE_CACHE"

# path to foundation models in saved_model format
FOUNDATION_CACHE_DIR = os.environ.get(GRACE_CACHE) or os.path.expanduser(
    "~/.cache/grace"
)
# path to foundation models checkpoints (dict-like format)
FOUNDATION_CHECKPOINTS_CACHE_DIR = os.path.join(FOUNDATION_CACHE_DIR, "checkpoints")

# key to model's checkpoint path in metadata
CHECKPOINT_PATH_KEY: Final[str] = "checkpoint_path"

# key to saved model path in metadata
MODEL_PATH_KEY: Final[str] = "path"

# key to saved model's URL in metadata
MODEL_URL_KEY: Final[str] = "url"

# key to model's checkpoint URL in metadata
CHECKPOINT_URL_KEY: Final[str] = "checkpoint_url"

# key to model's description in metadata
DESCRIPTION_KEY: Final[str] = "description"

# key to model's license in metadata
LICENSE_KEY: Final[str] = "license"

# filename for user-defined models metadata
MODELS_REGISTRY_YAML: Final[str] = "models_registry.yaml"

# filename for temporary downloaded archive
TMP_TAR_GZ: Final[str] = "tmp.tar.gz"

MODELS_METADATA = {
    #########################################################
    #  Dataset: Materials Project relaxation trajectories   #
    #########################################################
    "GRACE-1L-MP-r6": {
        MODEL_URL_KEY: "https://ruhr-uni-bochum.sciebo.de/s/d92rGQiRlieY6tJ",
        DESCRIPTION_KEY: "A one-layer local GRACE model parameterized on MPTraj dataset, with fixed 6 A cutoff.",
        LICENSE_KEY: "Academic Software License",
        #         "dirname": "MP-GRACE-1L-r6_4Nov2024",
    },
    "GRACE-2L-MP-r5": {
        MODEL_URL_KEY: "https://ruhr-uni-bochum.sciebo.de/s/nWyVO8xC38DJw4z",
        DESCRIPTION_KEY: """A two-layer semi-local GRACE model parameterized on MPTraj dataset, with fixed 5 A cutoff."""
        """ Currently the best among GRACE models on MatBench Discovery leaderboard.""",
        LICENSE_KEY: "Academic Software License",
        # "dirname": "MP_GRACE_2L_r5_07Nov2024",
    },
    "GRACE-2L-MP-r6": {
        MODEL_URL_KEY: "https://ruhr-uni-bochum.sciebo.de/s/42Ivgi3eaLCynwC",
        DESCRIPTION_KEY: "A two-layer semi-local GRACE model parameterized on MPTraj dataset, with fixed 6 A cutoff. (version: 11 Nov 2024)",
        LICENSE_KEY: "Academic Software License",
        #         "dirname": "MP_GRACE_2L_r6_11Nov2024",
    },
    ########################################################
    ### Pre-fitted on OMat24, fine tuned on sAlex+MPTraj  ##
    ########################################################
    "GRACE-FS-OAM": {
        MODEL_URL_KEY: "https://ruhr-uni-bochum.sciebo.de/s/B5BJLqTQ2t43PdV",
        # "dirname": "GRACE-FS-OAM_28Feb25",
        DESCRIPTION_KEY: """A FS-like single-layer local GRACE model, pre-fitted on the OMat24 and fine-tuned on sAlex+MPTraj datasets, """
        """with fixed 6 A cutoff.""",
        LICENSE_KEY: "Academic Software License",
        CHECKPOINT_URL_KEY: "https://ruhr-uni-bochum.sciebo.de/s/oqXlJdlcHs4oHIB",
    },
    "GRACE-1L-OAM": {
        MODEL_URL_KEY: "https://ruhr-uni-bochum.sciebo.de/s/gFAv8pX2DJbk1kb",
        # "dirname": "GRACE-1L-OAM_2Feb25",
        DESCRIPTION_KEY: """A single-layer local GRACE model, pre-fitted on the OMat24 and fine-tuned on sAlex+MPTraj datasets, """
        """with fixed 6 A cutoff.""",
        LICENSE_KEY: "Academic Software License",
        CHECKPOINT_URL_KEY: "https://ruhr-uni-bochum.sciebo.de/s/6URjfRdV8jRU3xP",
    },
    "GRACE-2L-OAM": {
        MODEL_URL_KEY: "https://ruhr-uni-bochum.sciebo.de/s/4zVTfzxornWfS4T",
        # "dirname": "GRACE-2L-OAM_28Jan25",
        DESCRIPTION_KEY: """A two-layer semi-local GRACE model, pre-fitted on the OMat24 and fine-tuned on sAlex+MPTraj datasets, """
        """with fixed 6 A cutoff.""",
        LICENSE_KEY: "Academic Software License",
        CHECKPOINT_URL_KEY: "https://ruhr-uni-bochum.sciebo.de/s/dD2EuwbZPocFZxo",
    },
    ########################################################
    ### Fitted on OMat24 ##
    ########################################################
    "GRACE-FS-OMAT": {
        MODEL_URL_KEY: "https://ruhr-uni-bochum.sciebo.de/s/qvTU9BLbMlMv9Iu",
        # "dirname": "GRACE-2L-OMAT-3Feb25",
        DESCRIPTION_KEY: """A simple FS-like local GRACE model, fitted on the OMat24 dataset, with fixed 6 A cutoff.""",
        LICENSE_KEY: "Academic Software License",
        CHECKPOINT_URL_KEY: "https://ruhr-uni-bochum.sciebo.de/s/TIpxi7gLvbEu4CK",
    },
    "GRACE-1L-OMAT": {
        MODEL_URL_KEY: "https://ruhr-uni-bochum.sciebo.de/s/DmRvRQeedKomaFU",
        # "dirname": "GRACE-1L-OMAT-30Jan25",
        DESCRIPTION_KEY: """A single-layer local GRACE model, fitted on the OMat24 dataset, with fixed 6 A cutoff.""",
        LICENSE_KEY: "Academic Software License",
        CHECKPOINT_URL_KEY: "https://ruhr-uni-bochum.sciebo.de/s/XeJQgikYkympwGo",
    },
    #######################
    "GRACE-2L-OMAT": {
        MODEL_URL_KEY: "https://ruhr-uni-bochum.sciebo.de/s/vbTYV9Pt4ppKSZ8",
        # "dirname": "GRACE-2L-OMAT-3Feb25",
        DESCRIPTION_KEY: """A two-layer semi-local GRACE model, fitted on the OMat24 dataset, with fixed 6 A cutoff.""",
        LICENSE_KEY: "Academic Software License",
        CHECKPOINT_URL_KEY: "https://ruhr-uni-bochum.sciebo.de/s/rLgF08mdhZJj3OH",
    },
    ########### 1L-medium  ############
    "GRACE-1L-OMAT-medium-base": {
        MODEL_URL_KEY: "https://ruhr-uni-bochum.sciebo.de/s/SoWS9KoQZ5DQ7pD",
        DESCRIPTION_KEY: "A GRACE foundation model: GRACE-1L-OMAT-medium-base.",
        LICENSE_KEY: "Academic Software License",
        CHECKPOINT_URL_KEY: "https://ruhr-uni-bochum.sciebo.de/s/MfEiW7GgKiazFem",
    },
    "GRACE-1L-OMAT-medium-ft-E": {
        MODEL_URL_KEY: "https://ruhr-uni-bochum.sciebo.de/s/g7pqjLAM3X5gQ35",
        DESCRIPTION_KEY: "A GRACE foundation model: GRACE-1L-OMAT-medium-ft-E.",
        LICENSE_KEY: "Academic Software License",
        CHECKPOINT_URL_KEY: "https://ruhr-uni-bochum.sciebo.de/s/2ZxJNgxQmkre3J5",
    },
    "GRACE-1L-OMAT-medium-ft-AM": {
        MODEL_URL_KEY: "https://ruhr-uni-bochum.sciebo.de/s/EMNHjqNR75c9xmH",
        DESCRIPTION_KEY: "A GRACE foundation model: GRACE-1L-OMAT-medium-ft-AM.",
        LICENSE_KEY: "Academic Software License",
        CHECKPOINT_URL_KEY: "https://ruhr-uni-bochum.sciebo.de/s/LtANLYLmyP88it7",
    },
    ########### 1L-large  ############
    "GRACE-1L-OMAT-large-base": {
        MODEL_URL_KEY: "https://ruhr-uni-bochum.sciebo.de/s/4K3ASNkn3nii4a2",
        DESCRIPTION_KEY: "A GRACE foundation model: GRACE-1L-OMAT-large-base.",
        LICENSE_KEY: "Academic Software License",
        CHECKPOINT_URL_KEY: "https://ruhr-uni-bochum.sciebo.de/s/eyBpGWR5me9WxGb",
    },
    "GRACE-1L-OMAT-large-ft-E": {
        MODEL_URL_KEY: "https://ruhr-uni-bochum.sciebo.de/s/pbMELPznrgdRRxk",
        DESCRIPTION_KEY: "A GRACE foundation model: GRACE-1L-OMAT-large-ft-E.",
        LICENSE_KEY: "Academic Software License",
        CHECKPOINT_URL_KEY: "https://ruhr-uni-bochum.sciebo.de/s/Jw7B3pjs39JZDd6",
    },
    "GRACE-1L-OMAT-large-ft-AM": {
        MODEL_URL_KEY: "https://ruhr-uni-bochum.sciebo.de/s/Se2MCSxtDjkD7wC",
        DESCRIPTION_KEY: "A GRACE foundation model: GRACE-1L-OMAT-large-ft-AM.",
        LICENSE_KEY: "Academic Software License",
        CHECKPOINT_URL_KEY: "https://ruhr-uni-bochum.sciebo.de/s/MCMaGmo6bNSLq79",
    },
    ########### 2L-medium  ############
    "GRACE-2L-OMAT-medium-base": {
        MODEL_URL_KEY: "https://ruhr-uni-bochum.sciebo.de/s/NRFH35F66rwsryW",
        DESCRIPTION_KEY: "A GRACE foundation model: GRACE-2L-OMAT-medium-base.",
        LICENSE_KEY: "Academic Software License",
        CHECKPOINT_URL_KEY: "https://ruhr-uni-bochum.sciebo.de/s/GQLWNQb4weje5RD",
    },
    "GRACE-2L-OMAT-medium-ft-E": {
        MODEL_URL_KEY: "https://ruhr-uni-bochum.sciebo.de/s/n9HB7SFxRbNFtSc",
        DESCRIPTION_KEY: "A GRACE foundation model: GRACE-2L-OMAT-medium-ft-E.",
        LICENSE_KEY: "Academic Software License",
        CHECKPOINT_URL_KEY: "https://ruhr-uni-bochum.sciebo.de/s/wKppKBBRmQccByj",
    },
    "GRACE-2L-OMAT-medium-ft-AM": {
        MODEL_URL_KEY: "https://ruhr-uni-bochum.sciebo.de/s/3Y4Jgg67XinajNe",
        DESCRIPTION_KEY: "A GRACE foundation model: GRACE-2L-OMAT-medium-ft-AM.",
        LICENSE_KEY: "Academic Software License",
        CHECKPOINT_URL_KEY: "https://ruhr-uni-bochum.sciebo.de/s/nfaCfrWcRMdqPfs",
    },
    ########### 2L-large  ############
    "GRACE-2L-OMAT-large-base": {
        MODEL_URL_KEY: "https://ruhr-uni-bochum.sciebo.de/s/pKqkYsYWLsB4EGc",
        DESCRIPTION_KEY: "A GRACE foundation model: GRACE-2L-OMAT-large-base.",
        LICENSE_KEY: "Academic Software License",
        CHECKPOINT_URL_KEY: "https://ruhr-uni-bochum.sciebo.de/s/W7KtsjrtKJT6iPS",
    },
    "GRACE-2L-OMAT-large-ft-E": {
        MODEL_URL_KEY: "https://ruhr-uni-bochum.sciebo.de/s/bp9znKKGb7LxyEs",
        DESCRIPTION_KEY: "A GRACE foundation model: GRACE-2L-OMAT-large-ft-E.",
        LICENSE_KEY: "Academic Software License",
        CHECKPOINT_URL_KEY: "https://ruhr-uni-bochum.sciebo.de/s/gFzndA8RnpF83ca",
    },
    "GRACE-2L-OMAT-large-ft-AM": {
        MODEL_URL_KEY: "https://ruhr-uni-bochum.sciebo.de/s/s6YrAAk9aTjQBd8",
        DESCRIPTION_KEY: "A GRACE foundation model: GRACE-2L-OMAT-large-ft-AM.",
        LICENSE_KEY: "Academic Software License",
        CHECKPOINT_URL_KEY: "https://ruhr-uni-bochum.sciebo.de/s/22ZqbGLrg25iWQn",
    },
}


# BACKWARD COMPATIBILITY:
MODELS_ALIASES_DICT = {
    "MP_GRACE_1L_r6_4Nov2024": "GRACE-1L-MP-r6",
    "MP_GRACE_2L_r5_4Nov2024": "GRACE-2L-MP-r5",
    "MP_GRACE_2L_r6_11Nov2024": "GRACE-2L-MP-r6",
    "GRACE-1L-OAM_2Feb25": "GRACE-1L-OAM",
    "GRACE_2L_OAM_28Jan25": "GRACE-2L-OAM",
    # shortname alisases: short name to full name
    "GRACE-2L-OMAT-L": "GRACE-2L-OMAT-large-ft-E",
    "GRACE-2L-OAM-L": "GRACE-2L-OMAT-large-ft-AM",
    "GRACE-2L-OMAT-M": "GRACE-2L-OMAT-medium-ft-E",
    "GRACE-2L-OAM-M": "GRACE-2L-OMAT-medium-ft-AM",
    "GRACE-1L-OMAT-L": "GRACE-1L-OMAT-large-ft-E",
    "GRACE-1L-OAM-L": "GRACE-1L-OMAT-large-ft-AM",
    "GRACE-1L-OMAT-M": "GRACE-1L-OMAT-medium-ft-E",
    "GRACE-1L-OAM-M": "GRACE-1L-OMAT-medium-ft-AM",
}

# loading user-defined models
user_model_registry_yaml_fname = os.path.join(
    FOUNDATION_CACHE_DIR, MODELS_REGISTRY_YAML
)
if os.path.isfile(user_model_registry_yaml_fname):
    with open(user_model_registry_yaml_fname, "r") as f:
        user_models = yaml.safe_load(f)
        # update, but not overwrite with user-defined models
        for name, info in user_models.items():
            if name not in MODELS_METADATA:
                MODELS_METADATA[name] = info


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


def safe_extract(tar, path=".", members=None):
    # Get top-level items (first level of paths)
    top_level_items = {
        os.path.normpath(member.name).split(os.sep)[0]: member
        for member in tar.getmembers()
        if os.sep not in member.name
    }

    # Ensure there's exactly one top-level item
    if len(top_level_items) == 1:
        top_folder_name, member = next(iter(top_level_items.items()))
        if not member.isdir():  # Ensure it's a directory
            raise ValueError(
                f"Expected a directory, but found a file: {top_folder_name}"
            )
    else:
        raise ValueError(
            f"Expected one top-level folder, found: {list(top_level_items.keys())}"
        )

    for member in tar.getmembers():
        if member.name.startswith("/") or ".." in member.name:
            raise ValueError(f"Unsafe path detected: {member.name}")
    tar.extractall(path, members)

    return top_folder_name


# automatically try to import experimental models
try:
    from tensorpotential.experimental.calculator.foundation_models import (
        MODELS_METADATA as MODEL_METADATA_EXPERIMENTAL,
    )

    MODELS_METADATA.update(MODEL_METADATA_EXPERIMENTAL)
except ImportError:
    pass

MODELS_NAME_LIST = list(MODELS_METADATA.keys())


def download_extract_rename(url, model_path):
    model_path = model_path[:-1] if model_path.endswith("/") else model_path
    if not url.endswith("/download"):
        url += "/download"
    root_folder = os.path.dirname(model_path)
    os.makedirs(root_folder, exist_ok=True)
    # download and save to disk
    print(f"Downloading from {url!r}")
    # Define the file path
    filename = os.path.join(root_folder, TMP_TAR_GZ)
    local_filename, http_msg = urllib.request.urlretrieve(url, filename)
    if "Content-Type: text/html" in http_msg:
        raise RuntimeError(f"Download failed, please check the URL {url!r}")
    # Unpack the .tar.gz file
    print(f"Unpacking model from {url!r}")
    with tarfile.open(local_filename, "r:gz") as tar:
        top_folder_name = safe_extract(tar, path=root_folder)
    os.remove(local_filename)
    extracted_model_path = os.path.join(root_folder, top_folder_name)
    if extracted_model_path != model_path:
        os.rename(extracted_model_path, model_path)
    assert os.path.isdir(model_path), f"Path {model_path} does not exist"


def get_or_download_model(model):
    # remap old names into new for backward compat
    model = MODELS_ALIASES_DICT.get(model) or model

    model_metadata = MODELS_METADATA[model]
    model_path = model_metadata.get(MODEL_PATH_KEY) or os.path.join(
        FOUNDATION_CACHE_DIR, model
    )
    if not os.path.isdir(model_path):
        # download model
        url = model_metadata.get(MODEL_URL_KEY)
        if url is None:
            raise ValueError(f"No available URL to download for {model}")
        print(f"Downloading GRACE model")
        download_extract_rename(url, model_path)
        print(f"GRACE model downloaded to {model_path}")
    else:
        print(f"Using cached GRACE model from {model_path}")
    print(f"Model license: {model_metadata.get(LICENSE_KEY, 'not provided')}")
    return model_path


def get_or_download_checkpoint(model):
    model_metadata = MODELS_METADATA[model]
    checkpoint_path = model_metadata.get(CHECKPOINT_PATH_KEY) or os.path.join(
        FOUNDATION_CHECKPOINTS_CACHE_DIR, model
    )
    if not os.path.isdir(checkpoint_path):
        # download checkpoint
        url = model_metadata.get(CHECKPOINT_URL_KEY)
        if url is None:
            raise ValueError(f"No available checkpoint URL for {model}")
        print(f"Downloading GRACE checkpoint")
        download_extract_rename(url, checkpoint_path)
        print(f"GRACE model checkpoint downloaded to {checkpoint_path}")
    else:
        print(f"Using cached GRACE checkpoint from {checkpoint_path}")
    print(f"Model license: {model_metadata.get(LICENSE_KEY, 'not provided')}")
    return checkpoint_path


def grace_fm(
    model: str,
    pad_neighbors_fraction: float | None = 0.05,
    pad_atoms_number: int | None = 10,
    max_number_reduction_recompilation: int | None = 2,
    min_dist=None,
    **kwargs,
):
    from tensorpotential.calculator.asecalculator import TPCalculator

    assert model in MODELS_NAME_LIST, f"model must be in {MODELS_NAME_LIST}"

    model_path = get_or_download_model(model)

    calc = TPCalculator(
        model=model_path,
        pad_neighbors_fraction=pad_neighbors_fraction,
        pad_atoms_number=pad_atoms_number,
        max_number_reduction_recompilation=max_number_reduction_recompilation,
        min_dist=min_dist,
        **kwargs,
    )
    return calc


grace_fm.__doc__ = f"""
    model:  (str) One of {", ".join(MODELS_NAME_LIST)}
    extend_fake_neighbors_fraction:  (float) Fraction of atoms to extend (padding for JIT)
    extend_fake_atoms_number:  (int) Number of atoms to extend (padding for JIT)
    min_dist:  (float) Minimum distance. Raise exception if atoms are closer
    
    Returns:  ASE calculator (TPCalculator)
"""

get_or_download_model.__doc__ = f"""
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
