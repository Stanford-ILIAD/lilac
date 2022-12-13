# LILAC – Online Language Corrections for Robotic Manipulation via Shared Autonomy 

> Open-Source Code Release for 
> *"No, to the Right – Online Language Corrections for Robotic Manipulation via Shared Autonomy"* 

Repository containing scripts for kinesthetic demonstration collection, model definitions (with baselines) in 
[PyTorch](https://pytorch.org/), model training code 
(via [PyTorch-Lightning](https://pytorch-lightning.readthedocs.io/en/latest/)), as well as code for deploying and 
running models on actual robots (e.g., for a demo or user study).

Uses [Anaconda](https://www.anaconda.com/) for maintaining python dependencies & reproducibility, and sane quality 
defaults (`black`, `isort`, `flake8`, `precommit`). Robot control stack for the Franka Emika Panda arm is built 
using [Polymetis](https://facebookresearch.github.io/fairo/polymetis/).

---

## Quickstart

Clones `lilac` to the working directory, then walks through dependency setup using the pinned versions in
`environments/requirements.txt`. If contributing to this repository, please make sure to run `pre-commit install`
before pushing a commit.

We have two sets of installation instructions, mainly for setting up GPU-accelerated PyTorch for training models (this
is by no means necessary; models train within 30 minutes on modern CPUs), as well as for setting up a CPU-only version
for inference (e.g., what runs on our lightweight robot control computer). 

Note: we have written the core of `lilac` as a Python module, with the `setup.py` and `pyproject.toml` providing the 
minimal information to fully replicate the Python environment used for the original work. For further reproducibility,
we define `environments/requirements.txt` with more explicit pinned versions of the remaining dependencies.

### Installation - Linux w/ GPU & CUDA 11.3 - Training

Note: the instructions below assume that `conda` (Miniconda/Anaconda are both fine) is installed and on your path. 
However, feel free to use the environment manager of your choosing!

```bash
git clone https://<anonymized>/lilac
cd lilac

conda create --name lilac python=3.8
conda activate lilac

# Install PyTorch == 1.11.0, Torchvision == 0.12.0, TorchAudio == 0.11.0 w/ CUDA Toolkit 11.3
#   > Any more recent versions will work as well; just edit the pinned versions in `requirements.txt`
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113

# Install remaining dependencies via an editable install (make sure you're in the root of this repository!)
#   > Note: if contributing to this repository: run `pip install -e ".[dev]"` for black / flake8 / isort / pre-commit
pip install -e .
```

### Installation - Linux CPU (Intel NUC) - Robot Control Computer

To use [Polymetis](https://facebookresearch.github.io/fairo/polymetis/installation.html) for the robot control 
stack, it's highly recommended to follow the [conda-based installation instructions here](https://facebookresearch.github.io/fairo/polymetis/installation.html#from-anaconda).
This will ensure the entire PyTorch ecosystem is installed as well.

Then, assuming a conda environment named `lilac`:

```bash
git clone https://<anonymized>/lilac
cd lilac

# Assumes Polymetis + PyTorch Ecosystem has already been installed...
conda activate lilac

# Install remaining dependencies...
#   > Note: if contributing to this repository: run `pip install -e ".[dev]"` for black / flake8 / isort / pre-commit
pip install -e .
```  

---

## Usage & Entry Points

This repository contains all the steps necessary to collect demonstration data for training LILAC models, as well as the
LILA and Language-Conditioned Imitation Learning Baselines, with further code for running online evaluations.

Following the general structure of the method outlined in our paper, we have defined the following four "top-level"
scripts that are meant to be run in order:

1. `python scripts/demonstrate.py <task_id>` - This script walks through collecting kinesthetic demonstrations for a given
    task (e.g., `water-plant`, `clean-trash`, or `towards-shelf` from the paper). Note the format that the demonstrations
    are saved in.
    + After this, make sure to add language annotations following the directions in `data/language`

2. `python scripts/alphas.py` - this script takes the language annotations collected in the prior step, and uses the
   OpenAI GPT-3 API to perform Alpha annotation, as per Section 4.4 of our paper. Note that this requires a (paid)
   OpenAI GPT-3 API key.

3. `python scripts/train.py --model <lilac | lila | imitation> ...` - this standalone script reads the demonstrations 
   collected in Step 1 and trains the various models compared in this work (you will need to pass a list of tasks you
   wish to train on). The default hyperparameters are fairly generalizable.
   
4. `python scripts/evaluate.py --run_directory <run-directory>` - once you've trained a model, this script enables online
    evaluation of the learned policy (if Imitation Learning) or latent actions controller (if LILA/LILAC) on an actual
    robot. 

These 4 steps comprise the full pipeline we use in our work. The user study is conducted as per the description in the
paper, using the `evaluate.py` script.

*Note*: To run any of the real-robot scripts (e.g., `demonstrate.py` and `evaluate.py`), you'll need to run the Robot
and Gripper servers on the Franka Control Computer; default launch scripts for reference can be found in `scripts/bin/`.

---

## Repository Structure

High-level overview of the repository -- besides the entry points described above, the important parts of the codebase
are in `models/` (architecture, optimizer, and training definitions), and in `robot/` (all robot control code).

+ `conf/` - Polymetis configurations for the Franka Emika Panda Arm.
+ `environments/` - Stores any & all information for reproducing the Python environment from the original work; right
                    now all dependencies are serialized in `requirements.txt`, but we'll add support for more versions
                    and platforms as needed. 
+ `lilac/` - Package source code - has *everything* -- utilities for preprocessing, model definitions, training.
    + `models/` - PyTorch-Lightning self-contained modules containing all architecture definition and training 
                  hyperparameters for LILAC, LILA, and Language-Conditioned Imitation Learning.
    + `preprocessing/` - Preprocessing utilities & PyTorch Dataset initialization functions.
    + `robot/` - Isolate Polymetis control code, with subdirectories for defining an OpenAI gym-like interface for 
                 communicating with the physical robot, and utility classes for collecting kinesthetic demonstrations.
+ `Makefile` - Makefile (by default, supports autoformatting, linting, and style checks). 
               Requires `pip install -e ".[dev]"`
+ `.flake8` - Flake8 configuration file (Sane Defaults).
+ `.pre-commit-config.yaml` - Pre-commit configuration file (Sane Defaults).
+ `LICENSE` - All of the LILAC codebase is made available under the MIT License. 
+ `pyproject.toml` - Black and isort configuration file (Sane Defaults).
+ `README.md` - You are here!
+ `setup.py` - Default `setuptools` setup.py until PEP 621 is fully resolved, enabling editable installations.

---

## Questions & Support

We are committed to maintaining this repository for the foreseeable future, across multiple PyTorch and Polymetis 
versions. If a specific platform/library release is not supported, please post an issue to the Github (or feel free to
fork and PR).

For more sensitive queries, please email `skaramcheti@cs.stanford.edu`.
