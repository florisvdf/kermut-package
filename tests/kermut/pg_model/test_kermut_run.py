from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from kermut.pg_model.kermut_run import main as kermut_run
from kermut.pg_model.utils import load_pdb_structure, dump_structure


def read_fasta(path: str) -> str:
    """Dummy function for only reading a single sequence stored on the second line of the fasta file"""
    with open(path) as fp:
        return fp.readlines()[-1]


def test_kermut_run(
    mini_protein_name,
    mini_dataset_path,
    mini_dataset_target,
    mini_reference_sequence_path,
    mini_structure_path,
):
    df = pd.read_csv(mini_dataset_path)
    mini_reference_sequence = read_fasta(mini_reference_sequence_path)
    mini_structure = load_pdb_structure(mini_structure_path)
    with TemporaryDirectory() as temp_dir:
        dump_structure(str(Path(temp_dir) / f"{mini_protein_name}.pdb"), mini_structure)
        df.to_csv(str(Path(temp_dir) / f"{mini_protein_name}.csv"))
        kermut_run(
            dataset_name=mini_protein_name,
            target=mini_dataset_target,
            data_dir=temp_dir,
            pdb_file=mini_structure_path,
            output_path=temp_dir,
            reference_sequence=mini_reference_sequence,
            prepare_artifacts=True,
            n_steps=2,
            preferential=False,
            device="cpu",
        )
