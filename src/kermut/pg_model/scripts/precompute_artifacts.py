from pathlib import Path

from loguru import logger
import typer
from typing import Annotated

from kermut.pg_model.scripts.coords import extract_3d_coords
from kermut.pg_model.scripts.conditionals import (
    extract_proteinmpnn_conditional_probabilities,
)
from kermut.pg_model.scripts.process_conditionals import (
    process_probabilities,
)
from kermut.pg_model.scripts.embeddings import (
    extract_esm2_embeddings,
)
from kermut.pg_model.scripts.zero_shot import extract_esm2_zero_shots


app = typer.Typer(
    help="Precompute artifacts for Kermut using the configuration as presented in the paper.",
    add_completion=True,
)


@app.command()
def precompute_artifacts(
    dataset_name: Annotated[
        str,
        typer.Option(
            help="Name of the dataset. Used for naming the dataframe containing zero shot scores"
        ),
    ],
    data_dir: Annotated[
        str,
        typer.Option(
            help="Directory storing the dataset and to save the precomputed artifacts to."
        ),
    ],
    pdb_file: Annotated[
        str, typer.Option(help="PDB file to extract 3D coordinates from")
    ],
    reference_sequence: Annotated[
        str,
        typer.Option(
            help="Reference sequence",
        ),
    ],
    toks_per_batch: Annotated[
        int, typer.Option(help="Number of tokens to process per batch")
    ] = 16384,
    device: Annotated[str, typer.Option(help="PyTorch backend device")] = "cpu",
) -> dict:
    dataset_path = str(Path(data_dir) / f"{dataset_name}.csv")
    path_conf = {
        "coords_dir": Path(data_dir) / "structures/coords",
        "conditional_probs_dir": Path(data_dir) / "conditional_probs",
        "embedding_dir": Path(data_dir) / "embeddings",
        "zero_shot_dir": Path(data_dir) / "zero_shot_fitness_predictions",
    }
    for path_name in path_conf.values():
        path_name.mkdir(parents=True, exist_ok=True)

    logger.info(f"Extracting 3D coordinates")
    extract_3d_coords(
        dataset_name=dataset_name,
        pdb_file=pdb_file,
        coords_dir=str(path_conf["coords_dir"]),
    )
    logger.info(f"Obtaining conditional probabilities from ProteinMPNN")
    extract_proteinmpnn_conditional_probabilities(
        pdb_file=pdb_file,
        dataset_name=dataset_name,
        conditional_probs_dir=str(path_conf["conditional_probs_dir"]),
    )
    logger.info(f"Processing conditional probabilities")
    process_probabilities(
        dataset_name=dataset_name,
        reference_sequence=reference_sequence,
        pdb_file=pdb_file,
        conditional_probs_dir=str(path_conf["conditional_probs_dir"]),
    )
    logger.info(f"Extracting embeddings from ESM2")
    extract_esm2_embeddings(
        data_path=dataset_path,
        dataset_name=dataset_name,
        embedding_dir=str(path_conf["embedding_dir"]),
        toks_per_batch=toks_per_batch,
        device=device,
    )
    logger.info(f"Extracting zeroshot scores from ESM2")
    extract_esm2_zero_shots(
        data_path=dataset_path,
        dataset_name=dataset_name,
        zero_shot_dir=str(path_conf["zero_shot_dir"]),
        reference_sequence=reference_sequence,
        device=device,
    )
    logger.success("Done!")
    return path_conf


if __name__ == "__main__":
    _ = app()
