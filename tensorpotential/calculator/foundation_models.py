from __future__ import annotations

import os
import tarfile

import keyword
from typing import Final

import requests
import yaml
from tqdm import tqdm

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

# key to model's LAMMPS-Kokkos export (kokkos.npz) URL in metadata
KOKKOS_URL_KEY: Final[str] = "kokkos_url"

# key to model's description in metadata
DESCRIPTION_KEY: Final[str] = "description"

# key to model's license in metadata
LICENSE_KEY: Final[str] = "license"

# --- declared capability flags (surfaced by `grace_models list`; verified by
# `grace_models info` against the downloaded artifact). Best-effort: the registry
# describes the PUBLIC artifact, so a flag may lag a re-export — `info` flags drift.
# key to declared weight precision: "fp32" | "fp64" | "mixed"
PRECISION_KEY: Final[str] = "precision"
# key to declared UQ support (SavedModel carries compute_uq + gmm_artifacts.npz)
UQ_KEY: Final[str] = "uq"
# key to declared multi-rank/domain-decomposition support (compute_energy + the
# partitioned fwd/bwd compute functions). 1L/FS models are inherently parallel;
# some legacy 2L exports lack the partitioned signatures (parallel=False).
PARALLEL_KEY: Final[str] = "parallel"

# filename for user-defined models metadata
MODELS_REGISTRY_YAML: Final[str] = "models_registry.yaml"

# filename for temporary downloaded archive
TMP_TAR_GZ: Final[str] = "tmp.tar.gz"

# streaming-download tuning (requests + tqdm progress bar / ETA)
DOWNLOAD_CHUNK_SIZE: Final[int] = 1 << 20  # 1 MiB read chunks
DOWNLOAD_RETRIES: Final[int] = 3  # whole-file retries on transient network errors
DOWNLOAD_TIMEOUT = (10, 60)  # (connect, read) seconds

# ---------------------------------------------------------------------------
# v3 release (uqv6 UQ feature): fp32 is the DEFAULT precision; the full-precision
# build is published non-default as "<name>-fp64". Archives live on immutable HF
# git tags. See pgs/uq/deployment_protocol.md §2b/§4 and deployment_changelog.md.
# ---------------------------------------------------------------------------
HF_REPO_BASE: Final[str] = (
    "https://huggingface.co/AMS-ICAMS-RUB/grace-foundation-models/resolve"
)
UQV6_MODEL_TAG: Final[str] = "model-v3-uq-v6"  # v3: 18 2L/1L SavedModel archives
UQV6_CHECKPOINT_TAG: Final[str] = "checkpoint-v3-uq-v6"  # v3 checkpoint archives
UQV6_KOKKOS_TAG: Final[str] = "kk-v3-uq-v6"  # v3 raw kokkos.npz files

# v4 (uqv6): the first 3L foundation models (fp32-only). Separate immutable tags.
UQV6_V4_MODEL_TAG: Final[str] = "model-v4-uq-v6"
UQV6_V4_CHECKPOINT_TAG: Final[str] = "checkpoint-v4-uq-v6"
UQV6_V4_KOKKOS_TAG: Final[str] = "kk-v4-uq-v6"


def _uqv6_entry(
    name: str,
    description: str,
    kokkos_base: str | None = None,
    *,
    model_tag: str = UQV6_MODEL_TAG,
    checkpoint_tag: str = UQV6_CHECKPOINT_TAG,
    kokkos_tag: str = UQV6_KOKKOS_TAG,
) -> dict:
    """Build a MODELS_METADATA entry for a uqv6 archive on Hugging Face.

    Works for both the fp32 default (``name=<bare>``) and the fp64 variant
    (``name=<bare>-fp64``); only the archive filename differs by the suffix.
    ``kokkos_base`` is the precision-agnostic base name keying the shared
    ``kokkos/<base>-kokkos.npz``. The ``*_tag`` args pin the release: they default
    to the v3 (2L/1L) tags; the 3L release passes the v4 tag family.
    """
    entry = {
        MODEL_URL_KEY: f"{HF_REPO_BASE}/{model_tag}/models/{name}-model.tar.gz",
        DESCRIPTION_KEY: description,
        LICENSE_KEY: "Academic Software License",
        CHECKPOINT_URL_KEY: f"{HF_REPO_BASE}/{checkpoint_tag}/checkpoints/{name}-checkpoint.tar.gz",
        PRECISION_KEY: "fp64" if name.endswith("-fp64") else "fp32",
        UQ_KEY: True,  # uqv6 basis-RP UQ baked into the SavedModel
        PARALLEL_KEY: True,  # re-exported with compute_energy + partitioned fwd/bwd
    }
    if kokkos_base is not None:
        entry[KOKKOS_URL_KEY] = (
            f"{HF_REPO_BASE}/{kokkos_tag}/kokkos/{kokkos_base}-kokkos.npz"
        )
    return entry


MODELS_METADATA = {
    ########################################################
    ### Pre-fitted on OMat24, fine tuned on sAlex+MPTraj  ##
    ########################################################
    "GRACE-FS-OAM": {
        MODEL_URL_KEY: "https://huggingface.co/AMS-ICAMS-RUB/grace-foundation-models/resolve/kk/models/GRACE-FS-OAM-model.tar.gz",
        # "dirname": "GRACE-FS-OAM_28Feb25",
        DESCRIPTION_KEY: """A FS-like single-layer local GRACE model, pre-fitted on the OMat24 and fine-tuned on sAlex+MPTraj datasets, """
        """with fixed 6 A cutoff.""",
        LICENSE_KEY: "Academic Software License",
        CHECKPOINT_URL_KEY: "https://huggingface.co/AMS-ICAMS-RUB/grace-foundation-models/resolve/kk/checkpoints/GRACE-FS-OAM-checkpoint.tar.gz",
        PRECISION_KEY: "fp64",
        UQ_KEY: False,
        PARALLEL_KEY: True,  # local 1L model — inherently domain-decomposable
    },
    "GRACE-1L-OAM": {
        MODEL_URL_KEY: "https://huggingface.co/AMS-ICAMS-RUB/grace-foundation-models/resolve/kk/models/GRACE-1L-OAM-model.tar.gz",
        # "dirname": "GRACE-1L-OAM_2Feb25",
        DESCRIPTION_KEY: """A single-layer local GRACE model, pre-fitted on the OMat24 and fine-tuned on sAlex+MPTraj datasets, """
        """with fixed 6 A cutoff.""",
        LICENSE_KEY: "Academic Software License",
        CHECKPOINT_URL_KEY: "https://huggingface.co/AMS-ICAMS-RUB/grace-foundation-models/resolve/kk/checkpoints/GRACE-1L-OAM-checkpoint.tar.gz",
        PRECISION_KEY: "fp64",
        UQ_KEY: False,
        PARALLEL_KEY: True,  # local 1L model — inherently domain-decomposable
    },
    "GRACE-2L-OAM": {
        MODEL_URL_KEY: "https://huggingface.co/AMS-ICAMS-RUB/grace-foundation-models/resolve/kk/models/GRACE-2L-OAM-model.tar.gz",
        # "dirname": "GRACE-2L-OAM_28Jan25",
        DESCRIPTION_KEY: """A two-layer semi-local GRACE model, pre-fitted on the OMat24 and fine-tuned on sAlex+MPTraj datasets, """
        """with fixed 6 A cutoff.""",
        LICENSE_KEY: "Academic Software License",
        CHECKPOINT_URL_KEY: "https://huggingface.co/AMS-ICAMS-RUB/grace-foundation-models/resolve/kk/checkpoints/GRACE-2L-OAM-checkpoint.tar.gz",
        PRECISION_KEY: "fp64",
        UQ_KEY: False,
        PARALLEL_KEY: False,  # legacy 2L export: no compute_energy / partitioned fwd-bwd
    },
    ########################################################
    ### Fitted on OMat24 ##
    ########################################################
    "GRACE-FS-OMAT": {
        MODEL_URL_KEY: "https://huggingface.co/AMS-ICAMS-RUB/grace-foundation-models/resolve/kk/models/GRACE-FS-OMAT-model.tar.gz",
        # "dirname": "GRACE-2L-OMAT-3Feb25",
        DESCRIPTION_KEY: """A simple FS-like local GRACE model, fitted on the OMat24 dataset, with fixed 6 A cutoff.""",
        LICENSE_KEY: "Academic Software License",
        CHECKPOINT_URL_KEY: "https://huggingface.co/AMS-ICAMS-RUB/grace-foundation-models/resolve/kk/checkpoints/GRACE-FS-OMAT-checkpoint.tar.gz",
        PRECISION_KEY: "fp64",
        UQ_KEY: False,
        PARALLEL_KEY: True,  # local FS model — inherently domain-decomposable
    },
    "GRACE-1L-OMAT": {
        MODEL_URL_KEY: "https://huggingface.co/AMS-ICAMS-RUB/grace-foundation-models/resolve/kk/models/GRACE-1L-OMAT-model.tar.gz",
        # "dirname": "GRACE-1L-OMAT-30Jan25",
        DESCRIPTION_KEY: """A single-layer local GRACE model, fitted on the OMat24 dataset, with fixed 6 A cutoff.""",
        LICENSE_KEY: "Academic Software License",
        CHECKPOINT_URL_KEY: "https://huggingface.co/AMS-ICAMS-RUB/grace-foundation-models/resolve/kk/checkpoints/GRACE-1L-OMAT-checkpoint.tar.gz",
        PRECISION_KEY: "fp64",
        UQ_KEY: False,
        PARALLEL_KEY: True,  # local 1L model — inherently domain-decomposable
    },
    #######################
    "GRACE-2L-OMAT": {
        MODEL_URL_KEY: "https://huggingface.co/AMS-ICAMS-RUB/grace-foundation-models/resolve/kk/models/GRACE-2L-OMAT-model.tar.gz",
        # "dirname": "GRACE-2L-OMAT-3Feb25",
        DESCRIPTION_KEY: """A two-layer semi-local GRACE model, fitted on the OMat24 dataset, with fixed 6 A cutoff.""",
        LICENSE_KEY: "Academic Software License",
        CHECKPOINT_URL_KEY: "https://huggingface.co/AMS-ICAMS-RUB/grace-foundation-models/resolve/kk/checkpoints/GRACE-2L-OMAT-checkpoint.tar.gz",
        PRECISION_KEY: "fp64",
        UQ_KEY: False,
        PARALLEL_KEY: False,  # legacy 2L export: no compute_energy / partitioned fwd-bwd
    },
    ### uqv6 (v3) OMAT 1L/2L medium+large models (12) — fp32 default + "-fp64"
    ### are generated below from _UQV6_MODEL_NAMES via _uqv6_entry().
    #### mixed
    "GRACE-2L-OMAT-large-mx": {
        MODEL_URL_KEY: "https://huggingface.co/AMS-ICAMS-RUB/grace-foundation-models/resolve/kk/models/GRACE-2L-OMAT-large-mx-model.tar.gz",
        DESCRIPTION_KEY: "A GRACE-2L large foundation model with mixed precision, fitted on OMat24.",
        LICENSE_KEY: "Academic Software License",
        CHECKPOINT_URL_KEY: "https://huggingface.co/AMS-ICAMS-RUB/grace-foundation-models/resolve/uq-v1/checkpoints/GRACE-2L-OMAT-large-mx-checkpoint.tar.gz",
        PRECISION_KEY: "mixed",
        UQ_KEY: True,  # uq-v1 era UQ checkpoint (verify with `grace_models info`)
        PARALLEL_KEY: True,
    },
    "GRACE-2L-OMAT-large-mx-ft-AM": {
        MODEL_URL_KEY: "https://huggingface.co/AMS-ICAMS-RUB/grace-foundation-models/resolve/kk/models/GRACE-2L-OMAT-large-mx-ft-AM-model.tar.gz",
        DESCRIPTION_KEY: "A GRACE-2L large foundation model with mixed precision, fitted on OMat24 and fine-tuned on sAlex+MPTraj.",
        LICENSE_KEY: "Academic Software License",
        CHECKPOINT_URL_KEY: "https://huggingface.co/AMS-ICAMS-RUB/grace-foundation-models/resolve/uq-v1/checkpoints/GRACE-2L-OMAT-large-mx-ft-AM-checkpoint.tar.gz",
        PRECISION_KEY: "mixed",
        UQ_KEY: True,  # uq-v1 era UQ checkpoint (verify with `grace_models info`)
        PARALLEL_KEY: True,
    },
    ### uqv6 (v3) SMAX models (6) — fp32 default + "-fp64" are generated
    ### below from _UQV6_MODEL_NAMES via _uqv6_entry().
}

# ---------------------------------------------------------------------------
# v3 (uqv6) release wiring — fp32 default + "<name>-fp64" full precision.
# These 18 models SUPERSEDE any bare-name entries above: the bare name now
# resolves to the fp32 SavedModel/checkpoint (2x faster, 1/2 memory; fp32<->fp64
# gap negligible — gamma max-rel <=5e-5), and a matching "<name>-fp64" full
# precision variant is registered. kokkos is precision-agnostic (one fp64 npz
# serves both). See pgs/uq/deployment_protocol.md §2b.
# ---------------------------------------------------------------------------
_UQV6_MODEL_NAMES: Final[list] = [
    # 1L (8)
    "GRACE-1L-OMAT-medium-base",
    "GRACE-1L-OMAT-medium-ft-E",
    "GRACE-1L-OMAT-medium-ft-AM",
    "GRACE-1L-OMAT-large-base",
    "GRACE-1L-OMAT-large-ft-E",
    "GRACE-1L-OMAT-large-ft-AM",
    "GRACE-1L-SMAX-large",
    "GRACE-1L-SMAX-OMAT-large",
    # 2L (10)
    "GRACE-2L-OMAT-medium-base",
    "GRACE-2L-OMAT-medium-ft-E",
    "GRACE-2L-OMAT-medium-ft-AM",
    "GRACE-2L-OMAT-large-base",
    "GRACE-2L-OMAT-large-ft-E",
    "GRACE-2L-OMAT-large-ft-AM",
    "GRACE-2L-SMAX-large",
    "GRACE-2L-SMAX-medium",
    "GRACE-2L-SMAX-OMAT-large",
    "GRACE-2L-SMAX-OMAT-medium",
]
for _name in _UQV6_MODEL_NAMES:
    MODELS_METADATA[_name] = _uqv6_entry(
        _name,
        f"A GRACE foundation model: {_name} (fp32, default precision).",
        kokkos_base=_name,
    )
    MODELS_METADATA[f"{_name}-fp64"] = _uqv6_entry(
        f"{_name}-fp64",
        f"A GRACE foundation model: {_name}, full fp64 precision (non-default; default is fp32).",
        kokkos_base=_name,
    )

# ---------------------------------------------------------------------------
# v4 (uqv6) release wiring — the first 3-layer (3L) foundation models.
# These are natively fp32 (no fp64 build), so the bare name IS the fp32 payload;
# there is NO "-fp64" variant. LAMMPS kk = pair_style grace/3l/kk[/fp32]
# (MIXED/FP32 only). Pinned to the immutable v4 tag family. See changelog v4.
# ---------------------------------------------------------------------------
_UQV6_V4_MODEL_NAMES: Final[list] = [
    "GRACE-3L-OMAT-large",
    "GRACE-3L-OMAT-large-ft-AM",
]
for _name in _UQV6_V4_MODEL_NAMES:
    MODELS_METADATA[_name] = _uqv6_entry(
        _name,
        f"A 3-layer GRACE foundation model: {_name} (fp32).",
        kokkos_base=_name,
        model_tag=UQV6_V4_MODEL_TAG,
        checkpoint_tag=UQV6_V4_CHECKPOINT_TAG,
        kokkos_tag=UQV6_V4_KOKKOS_TAG,
    )

# Built-in foundation models defined in this module, captured BEFORE the
# user-registry (models_registry.yaml) and experimental additions below. Lets
# `grace_models list` separate these from locally-registered/experimental models.
CORE_MODELS_NAME_LIST: Final[list] = list(MODELS_METADATA.keys())


# BACKWARD COMPATIBILITY:
MODELS_ALIASES_DICT = {
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


def safe_extract(tar_filename, path=".", members=None):
    with tarfile.open(tar_filename, "r:gz") as tar:
        # Get top-level items (first level of paths)
        top_level_items = {
            member.name.split("/")[0]: member
            for member in tar.getmembers()
            if "/" not in member.name.strip("/")
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


def _stream_download(url, dest_path, desc=None):
    """Stream ``url`` to ``dest_path`` with a tqdm progress bar (rate + ETA).

    Uses ``requests`` with large (1 MiB) chunks — far faster than the small-buffer
    single-read urllib path it replaces. Writes to a ``.part`` temp file and only
    renames it into place on success, so an interrupted download never masquerades
    as a complete file. Retries the whole transfer on transient network errors.
    Raises RuntimeError if the response is an HTML page (e.g. a wrong/expired URL)
    rather than a binary file.
    """
    desc = desc or os.path.basename(dest_path)
    tmp_path = dest_path + ".part"
    last_err = None
    for attempt in range(1, DOWNLOAD_RETRIES + 1):
        try:
            with requests.get(
                url, stream=True, timeout=DOWNLOAD_TIMEOUT, allow_redirects=True
            ) as resp:
                resp.raise_for_status()
                if "text/html" in resp.headers.get("Content-Type", ""):
                    raise RuntimeError(
                        f"Download failed (got an HTML page, not a file). "
                        f"Please check the URL {url!r}"
                    )
                total = int(resp.headers.get("Content-Length", 0)) or None
                with open(tmp_path, "wb") as f, tqdm(
                    total=total,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=desc,
                    miniters=1,
                ) as bar:
                    for chunk in resp.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
                        if chunk:
                            f.write(chunk)
                            bar.update(len(chunk))
            os.replace(tmp_path, dest_path)
            return dest_path
        except (requests.RequestException, OSError) as err:
            last_err = err
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            if attempt < DOWNLOAD_RETRIES:
                print(
                    f"  download attempt {attempt}/{DOWNLOAD_RETRIES} failed "
                    f"({err}); retrying..."
                )
    raise RuntimeError(
        f"Download of {url!r} failed after {DOWNLOAD_RETRIES} attempts: {last_err}"
    )


def download_extract_rename(url, model_path):
    model_path = model_path[:-1] if model_path.endswith("/") else model_path
    # sciebo/Nextcloud public-share links require a trailing "/download";
    # direct hosts (e.g. Hugging Face "resolve" URLs) are used as-is.
    if "sciebo" in url and not url.endswith("/download"):
        url += "/download"
    root_folder = os.path.dirname(model_path)
    os.makedirs(root_folder, exist_ok=True)
    # download and save to disk
    print(f"Downloading from {url!r}")
    # Define the file path
    filename = os.path.join(root_folder, TMP_TAR_GZ)
    local_filename = _stream_download(
        url, filename, desc=os.path.basename(model_path) + " (archive)"
    )
    # Unpack the .tar.gz file
    print(
        f"Unpacking model from {url!r} (local file: {local_filename}) to {root_folder}"
    )
    top_folder_name = safe_extract(local_filename, path=root_folder)
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
        print("Downloading GRACE model")
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
        print("Downloading GRACE checkpoint")
        download_extract_rename(url, checkpoint_path)
        print(f"GRACE model checkpoint downloaded to {checkpoint_path}")
    else:
        print(f"Using cached GRACE checkpoint from {checkpoint_path}")
    print(f"Model license: {model_metadata.get(LICENSE_KEY, 'not provided')}")
    return checkpoint_path


def get_or_download_kokkos(model):
    """Download the LAMMPS-Kokkos export ``kokkos.npz`` for ``model`` into its cached
    model dir, returning the local path. On-demand: ASE users never need it, so it is
    NOT pulled with the SavedModel (and the SavedModel is not required here either).

    The kokkos export is precision-agnostic — one fp64 ``kokkos.npz`` serves both the
    fp32 (bare) and fp64 (``-fp64``) variants — so both metadata entries point at the
    same shared ``kokkos/<base>-kokkos.npz``.
    """
    model = MODELS_ALIASES_DICT.get(model) or model
    model_metadata = MODELS_METADATA[model]
    model_path = model_metadata.get(MODEL_PATH_KEY) or os.path.join(
        FOUNDATION_CACHE_DIR, model
    )
    kokkos_path = os.path.join(model_path, "kokkos.npz")
    if os.path.isfile(kokkos_path):
        print(f"Using cached kokkos.npz from {kokkos_path}")
        return kokkos_path
    url = model_metadata.get(KOKKOS_URL_KEY)
    if url is None:
        raise ValueError(f"No available kokkos.npz URL for {model}")
    if "sciebo" in url and not url.endswith("/download"):
        url += "/download"
    os.makedirs(model_path, exist_ok=True)
    print(f"Downloading kokkos.npz from {url!r}")
    _stream_download(url, kokkos_path, desc=f"{model} kokkos.npz")
    print(f"kokkos.npz downloaded to {kokkos_path}")
    return kokkos_path


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
