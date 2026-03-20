import pytest

from kermut.pg_model.constants import PROJECT_ROOT


@pytest.fixture(scope="session")
def mini_protein_name():
    return "mini_protein"


@pytest.fixture(scope="session")
def mini_reference_sequence_path():
    return str(PROJECT_ROOT / "tests/data/mini_protein.fasta")


@pytest.fixture(scope="session")
def mini_dataset_path():
    return str(PROJECT_ROOT / "tests/data/mini_protein.csv")


@pytest.fixture(scope="session")
def mini_dataset_target():
    return "DMS_score"


@pytest.fixture(scope="session")
def mini_structure_path():
    return str(PROJECT_ROOT / "tests/data/mini_protein.pdb")
