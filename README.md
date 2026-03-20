# kermut-package

This repository is a fork of the [original repository](https://github.com/petergroth/kermut) 
providing the implementation of 
[Kermut: Composite kernel regression for protein variant effects](https://doi.org/10.48550/arXiv.2407.00002).
It acts as an installable wrapper project that allows users to easily train and evaluate 
Kermut on any protein variant effect dataset, provided that:

- A structure exists from which the variants are derived
- The sequences are all of equal length
- Each sequence has at least one mutation

Additionally, this project adds preferential training support to Kermut. This replaces the 
regression training objective with a pairwise preference prediction objective, matching 
the implementation described in [Preference learning with Gaussian processes](https://doi.org/10.1145/1102351.1102369),
and built using [`botorch`](https://botorch.readthedocs.io/en/latest/models.html#module-botorch.models.pairwise_gp).


## Structure

This codebase tries to preserve as much as possible of the original source code, which 
is stored under `src/kermut/kermut`. `src/kermut/pg_model` acts as a wrapper module, providing a 
single entrypoint for training and evaluation through `src/kermut/pg_model/kermut_run.py`, 
and also contains utilities to provide preferential training support, such as preference 
pair sampling algorithms.

## Installation

```shell
git clone https://github.com/florisvdf/kermut-package.git
cd kermut
uv sync
```

Or directly install the project using your favorite package manager, e.g. pip:

```shell
pip install git+https://github.com/florisvdf/kermut-package
```

[ProteinMPNN](https://github.com/dauparas/ProteinMPNN) must be installed to compute 
structure-conditioned amino acid distributions. This can be done by cloning the repository 
and passing its path to an environment variable named `PROTEINMPNN_DIR`.

## Usage

Kermut can be trained provided a dataframe with a `sequence` column storing sequences, 
a `split` column storing `train` and `test` values and an arbitrarily named column storing 
the values to model stored at `data_dir/<dataset_name>.csv`. 


```console
$ python src/kermut/pg_model/kermut_run.py --help

 Usage: kermut_run.py [OPTIONS]                                                                                                                                                      
                                                                                                                                                                                     
╭─ Options ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ *  --dataset-name                                              TEXT     Name of the dataset. It should match the csv storing the sequence-property pairs. [required]              │
│ *  --target                                                    TEXT     Name of the columns storing the property values to fit. [required]                                        │
│ *  --reference-sequence                                        TEXT     Sequence of the parent protein. [required]                                                                │
│ *  --pdb-file                                                  TEXT     PDB file to extract 3D coordinates from. [required]                                                       │
│ *  --data-dir                                                  TEXT     Directory storing the dataset and to save the precomputed artifacts to. [required]                        │
│ *  --output-path                                               TEXT     Path to save the predictions and metrics to. [required]                                                   │
│    --prepare-artifacts               --no-prepare-artifacts             Whether or not artifacts (kernel inputs) should be computed. [default: prepare-artifacts]                 │
│    --n-steps                                                   INTEGER  Number of optimization steps for fitting the Gaussian Process. [default: 150]                             │
│    --preferential                    --no-preferential                  Train Kermut in preferential mode. [default: no-preferential]                                             │
│    --preference-sampling-strategy                              TEXT     How to sample preference for training Kermut in preferential mode. Currently only uniform sampling is     │
│                                                                         supported. Valid values are 'uniform_{avg_degree}' where avg_degree is the average degree of the          │
│                                                                         resulting preference graph.                                                                               │
│    --device                                                    TEXT     PyTorch backend device [default: cpu]                                                                     │
│    --install-completion                                                 Install completion for the current shell.                                                                 │
│    --show-completion                                                    Show completion for the current shell, to copy it or customize the installation.                          │
│    --help                                                               Show this message and exit.                                                                               │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯```
```

Alternatively, import `main` from `kermut_run.py` in your script:

```python
from kermut.pg_model.kermut_run import main as train


train(
    dataset_name="my_protein",
    target="my_target", 
    reference_sequence="MYREFERENCESEQWENCE",
    pdb_file="path/to/my_structure.pdb",
    data_dir="path/to/my/artifacts", 
    output_path="path/to/my/outputs",
    n_steps=150, 
)
```

This wil train Kermut and evaluate in on both the training and test set. Predictions and metrics
are saved to `path/to/my/outputs`. A demo of this usage can be found under `tests/kermut/pg_model/test_kermut_run.py`, 
which users can run with `pytest`:

```shell
pytest -xvs tests/kermut/pg_model/test_kermut_run.py
```

In addition, users can compute all necessary artifacts at training time, or precompute 
artifacts with `src/kermut/pg_model/scripts/precompute_artifacts.py`:

```console
$ python src/kermut/pg_model/scripts/precompute_artifacts.py --help

 Usage: precompute_artifacts.py [OPTIONS]                                                                                                                                            
                                                                                                                                                                                     
╭─ Options ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ *  --dataset-name              TEXT     Name of the dataset. Used for naming the dataframe containing zero shot scores [required]                                                 │
│ *  --data-dir                  TEXT     Directory storing the dataset and to save the precomputed artifacts to. [required]                                                        │
│ *  --pdb-file                  TEXT     PDB file to extract 3D coordinates from [required]                                                                                        │
│ *  --reference-sequence        TEXT     Reference sequence [required]                                                                                                             │
│    --toks-per-batch            INTEGER  Number of tokens to process per batch [default: 16384]                                                                                    │
│    --device                    TEXT     PyTorch backend device [default: cpu]                                                                                                     │
│    --install-completion                 Install completion for the current shell.                                                                                                 │
│    --show-completion                    Show completion for the current shell, to copy it or customize the installation.                                                          │
│    --help                               Show this message and exit.                                                                                                               │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

Training Kermut without computing artifacts at training time assumes that the artifacts 
are stored as follows:


```console
.
├── conditional_probs
│   ├── <dataset_name>.npy
├── embeddings
│   └── <dataset_name>.h5
├── <dataset_name>.csv
├── structures
│   └── coords
│       └── <dataset_name>.npy
└── zero_shot_fitness_predictions
    └── <dataset_name>.csv

```

## Known limitations
- Currently only a fixed kernel setting matching the model reported in the original publication is available.
- Kermut struggles with highly combinatorial datasets.