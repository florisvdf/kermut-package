import shutil
from typing import Tuple

import pandas as pd
import yaml
import json
from pathlib import Path
from loguru import logger

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import r2_score
from Bio.PDB.PDBIO import PDBIO
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Structure import Structure
import torch


def load_pdb_structure(pdb_path: str, structure_id: str) -> Structure:
    parser = PDBParser(PERMISSIVE=1)
    structure = parser.get_structure(structure_id, pdb_path)
    return structure


def dump_structure(path: str, structure: Structure) -> None:
    io = PDBIO()
    io.set_structure(structure)
    io.save(path)


def prepare_hydra_configs(
    hydra_src_config_path: str, hydra_dest_config_path: str, params_to_update: dict
) -> None:
    shutil.copytree(hydra_src_config_path, hydra_dest_config_path, dirs_exist_ok=True)
    with open(Path(hydra_dest_config_path) / "data/paths.yaml", "r") as fp:
        hydra_paths_config = yaml.safe_load(fp)
    with open(Path(hydra_dest_config_path) / "data/dataset.yaml", "r") as fp:
        hydra_datasets_config = yaml.safe_load(fp)
    with open(Path(hydra_dest_config_path) / "benchmark.yaml", "r") as fp:
        hydra_benchmark_config = yaml.safe_load(fp)

    hydra_paths_config["data_dir"] = params_to_update["data_dir"]
    hydra_paths_config["sequence_col"] = "sequence"
    hydra_paths_config["paths"]["embeddings"] = str(
        Path(params_to_update["data_dir"]) / "embeddings"
    )
    hydra_paths_config["paths"]["conditional_probs"] = str(
        Path(params_to_update["data_dir"]) / "conditional_probs"
    )
    hydra_paths_config["paths"]["DMS_input_folder"] = params_to_update["data_dir"]
    hydra_paths_config["paths"]["output_folder"] = params_to_update["output_path"]
    hydra_paths_config["target_col"] = "target"

    # TODO: Setting other Kermut params (types of kernels, etc)
    hydra_benchmark_config["optim"]["n_steps"] = params_to_update["n_steps"]
    hydra_benchmark_config["DMS_id"] = params_to_update["dataset_name"]
    hydra_benchmark_config["target_seq"] = params_to_update["reference_sequence"]
    hydra_benchmark_config["use_gpu"] = params_to_update["use_gpu"]
    hydra_benchmark_config["preferential"] = params_to_update["preferential"]
    hydra_benchmark_config["preference_sampling_strategy"] = params_to_update[
        "preference_sampling_strategy"
    ]

    with open(Path(hydra_dest_config_path) / "data/paths.yaml", "w") as fp:
        yaml.dump(hydra_paths_config, fp)
    with open(Path(hydra_dest_config_path) / "data/dataset.yaml", "w") as fp:
        yaml.dump(hydra_datasets_config, fp)
    with open(Path(hydra_dest_config_path) / "benchmark.yaml", "w") as fp:
        yaml.dump(hydra_benchmark_config, fp)


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return np.mean(np.square((y_true - y_pred)))


def r_square(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    try:
        return r2_score(y_true=y_true, y_pred=y_pred)
    except ValueError:
        return np.nan


def spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(np.unique(y_true)) <= 1 or len(np.unique(y_pred)) <= 1:
        return np.float64(0.0)
    return spearmanr(y_true, y_pred).correlation


def log_and_save_metrics(results: pd.DataFrame, output_dir: str) -> None:
    metrics = {
        "mse": mse,
        "r_square": r_square,
        "spearman": spearman,
    }
    metric_stats = {}
    for split in results["split"].unique():
        split_df = results[results["split"] == split]
        for metric_name, metric in metrics.items():
            value = metric(split_df["y"].values, split_df["y_pred"].values)
            metric_stats[f"{split}_{metric_name}"] = value
            logger.info(f"{split}_{metric_name}: {value:.4f}")
    with open(Path(output_dir) / "metrics.json", "w") as fh:
        json.dump(metric_stats, fh)


def variant_sequence_to_mutations(variant: str, reference: str) -> str:
    return ":".join(
        [
            f"{aa_ref}{pos + 1}{aa_var}"
            for pos, (aa_ref, aa_var) in enumerate(zip(reference, variant))
            if aa_ref != aa_var
        ]
    )


def set_torch_dtype(preferential: bool) -> None:
    torch.set_default_dtype(torch.float32)
    if preferential:
        torch.set_default_dtype(torch.float64)


def assign_mutant_column(df: pd.DataFrame, reference_sequence: str) -> pd.DataFrame:
    if "mutant" not in df.columns:
        logger.info(
            "No mutant column in the dataframe. Creating a copy with mutation information"
        )
        df["mutant"] = df["sequence"].map(
            lambda x: variant_sequence_to_mutations(x, reference_sequence)
        )
    return df


def validate_dataset(df: pd.DataFrame, reference_sequence: str):
    variant_matches_reference = df["sequence"] == reference_sequence
    if variant_matches_reference.any():
        raise ValueError("Dataframe contains variants identical to reference!")


def torch_delete(tensor: torch.Tensor, indices: list) -> torch.Tensor:
    indices_to_delete = torch.tensor(indices)
    mask = torch.ones(tensor.shape[0], dtype=torch.bool)
    mask[indices_to_delete] = False
    return tensor[mask]


def prune_isolated_nodes(
    train_inputs: Tuple[torch.Tensor, ...],
    train_targets: torch.Tensor,
) -> Tuple[Tuple[torch.Tensor, ...], torch.Tensor]:
    """
    Apparently unnecessary. Also this function should update the dataframe, which it doesn't right now.
    """

    def compute_offset(idx, sorted_isolated):
        return sum(1 for isolated in sorted_isolated if isolated < idx)

    full_sample_indices = list(range(len(train_inputs[0])))
    flattened_pairs = [int(idx) for pair in train_targets for idx in pair]
    observed_comparison_indices = set(flattened_pairs)
    isolated_node_indices = list(
        set(full_sample_indices).difference(observed_comparison_indices)
    )
    if len(isolated_node_indices) > 0:
        logger.info(f"Pruning {len(isolated_node_indices)} isolated nodes.")
        sorted_isolated = sorted(isolated_node_indices)
        updated_flattened_pairs = [
            idx - compute_offset(idx, sorted_isolated) for idx in flattened_pairs
        ]
        updated_pairs = torch.tensor(
            [
                (updated_flattened_pairs[i], updated_flattened_pairs[i + 1])
                for i in range(0, len(updated_flattened_pairs), 2)
            ]
        )
        updated_train_inputs = tuple(
            [torch_delete(inp, isolated_node_indices) for inp in train_inputs]
        )
        return updated_train_inputs, updated_pairs
    else:
        logger.info(f"No isolated nodes to prune.")
        return train_inputs, train_targets


def add_pseudo_if_variant_matches_reference(
    df: pd.DataFrame, reference_sequence: str
) -> pd.DataFrame:
    matches_reference = df[df["sequence"] == reference_sequence]
    if len(matches_reference) > 0:
        if reference_sequence[-1] == "G":
            replacement_aa = "A"
        else:
            replacement_aa = "G"
        modified_sequence = reference_sequence[:-1] + replacement_aa
        logger.warning(
            f"Found {len(matches_reference)} sequence(s) matching the reference sequence,"
            f"replacing with pseudo variant: \n {modified_sequence}\n"
            f"(mutating final residue to {replacement_aa})."
        )
        df = df.copy()
        df.loc[df["sequence"] == reference_sequence, "sequence"] = modified_sequence
        return df
    else:
        return df
