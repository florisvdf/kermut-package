from pathlib import Path
from loguru import logger

from typing import Annotated

import typer
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig

from hydra import initialize_config_dir, compose

from kermut.kermut.data import (
    prepare_GP_inputs,
    prepare_GP_kwargs,
    split_inputs,
    standardize,
)
from kermut.kermut.gp import instantiate_gp, optimize_gp, predict
from kermut.pg_model.scripts.precompute_artifacts import precompute_artifacts
from kermut.pg_model.utils import (
    assign_mutant_column,
    set_torch_dtype,
    prepare_hydra_configs,
    log_and_save_metrics,
    add_pseudo_if_variant_matches_reference
)
from kermut.pg_model.pair_sampling import pair_sampling_factory
from kermut.pg_model.constants import HYDRA_CONFIG_PATH, HYDRA_TEMP_CONFIG_PATH


app = typer.Typer(
    help="Train and evaluate Kermut on a dataset.",
    add_completion=True,
)


def _evaluate_dms(cfg: DictConfig) -> None:
    DMS_id = cfg.DMS_id
    target_seq = cfg.target_seq
    device = "cuda" if cfg.use_gpu and torch.cuda.is_available() else "cpu"
    df, y, x_toks, x_embed, x_zero_shot = prepare_GP_inputs(cfg, DMS_id)
    gp_inputs = prepare_GP_kwargs(
        cfg, DMS_id, target_seq, dtype=torch.get_default_dtype()
    )

    df_out = df.copy()
    df_out = df_out.assign(fold=np.nan, y=np.nan, y_pred=np.nan, y_var=np.nan)

    test_fold = (
        -1
    )  # Predict needs test fold, only for fold tracking in results, not necessary here
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    train_idx = (df["split"] == "train").tolist()
    test_idx = (df["split"] == "test").tolist()

    y_train, y_test = split_inputs(train_idx, test_idx, y)
    y_train, y_test = (
        standardize(y_train, y_test) if cfg.data.standardize else (y_train, y_test)
    )
    x_toks_train, x_toks_test = split_inputs(train_idx, test_idx, x_toks)
    x_embed_train, x_embed_test = split_inputs(train_idx, test_idx, x_embed)
    x_zero_shot_train, x_zero_shot_test = split_inputs(train_idx, test_idx, x_zero_shot)

    train_inputs = (x_toks_train, x_embed_train, x_zero_shot_train)
    test_inputs = (x_toks_test, x_embed_test, x_zero_shot_test)
    train_targets = y_train
    test_targets = y_test

    if cfg.preferential:
        torch.set_default_dtype(torch.float64)
        sampler = pair_sampling_factory(
            cfg.preference_sampling_strategy,
            split=df.get("split"),
            batch_labels=df.get("batch_label"),
        )
        train_targets = torch.tensor(
            sampler.sample(train_targets.cpu().numpy()), device=device
        )
        train_inputs = tuple(
            [x.to(device=device, dtype=torch.get_default_dtype()) for x in train_inputs]
        )
        test_inputs = tuple(
            [x.to(device=device, dtype=torch.get_default_dtype()) for x in test_inputs]
        )
    gp, likelihood = instantiate_gp(
        cfg=cfg,
        train_inputs=train_inputs,
        train_targets=train_targets,
        gp_inputs=gp_inputs,
    )

    gp, likelihood = optimize_gp(
        gp=gp,
        likelihood=likelihood,
        train_inputs=train_inputs,
        train_targets=train_targets,
        lr=cfg.optim.lr,
        n_steps=cfg.optim.n_steps,
        progress_bar=cfg.optim.progress_bar,
        preferential=cfg.preferential,
    )

    test_df_out = predict(
        gp=gp,
        likelihood=likelihood,
        test_inputs=test_inputs,
        test_targets=test_targets,
        test_fold=test_fold,
        test_idx=test_idx,
        df_out=df_out,
        preferential=cfg.preferential,
    )

    train_df_out = predict(
        gp=gp,
        likelihood=likelihood,
        test_inputs=train_inputs,
        test_targets=y_train,  # Make sure to not accidentally set preferences
        test_fold=test_fold,
        test_idx=train_idx,
        df_out=df_out,
        preferential=cfg.preferential,
        training_data=True,
    )
    df_out = train_df_out.combine_first(test_df_out)

    out_path = Path(cfg.data.paths.output_folder) / f"{DMS_id}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_path, index=False)


@app.command()
def main(
    dataset_name: Annotated[
        str,
        typer.Option(
            help="Name of the dataset. It should match the csv storing the sequence-property pairs."
        ),
    ],
    target: Annotated[
        str,
        typer.Option(help="Name of the columns storing the property values to fit."),
    ],
    reference_sequence: Annotated[
        str,
        typer.Option(
            help="Sequence of the parent protein.",
        ),
    ],
    pdb_file: Annotated[
        str, typer.Option(help="PDB file to extract 3D coordinates from.")
    ],
    data_dir: Annotated[
        str,
        typer.Option(
            help="Directory storing the dataset and to save the precomputed artifacts to."
        ),
    ],
    output_path: Annotated[
        str,
        typer.Option(help="Path to save the predictions and metrics to."),
    ],
    prepare_artifacts: Annotated[
        bool,
        typer.Option(
            help="Whether or not artifacts (kernel inputs) should be computed."
        ),
    ] = True,
    n_steps: Annotated[
        int,
        typer.Option(
            help="Number of optimization steps for fitting the Gaussian Process."
        ),
    ] = 150,
    preferential: Annotated[
        bool, typer.Option(help="Train Kermut in preferential mode.")
    ] = False,
    preference_sampling_strategy: Annotated[
        str,
        typer.Option(
            help="How to sample preference for training Kermut in preferential mode. "
            "Currently only uniform sampling is supported. Valid values are "
            "'uniform_{avg_degree}' where avg_degree is the average degree of the "
            "resulting preference graph."
        ),
    ] = None,
    device: Annotated[str, typer.Option(help="PyTorch backend device")] = "cpu",
) -> None:
    dataset_path = str(Path(data_dir) / f"{dataset_name}.csv")
    df = pd.read_csv(dataset_path)
    df = assign_mutant_column(df, reference_sequence)
    df.rename(columns={target: "target"}, inplace=True)
    # Add a pseudo mutation to any sequences that match the reference sequence
    pseudo_mask = df["sequence"] == reference_sequence
    df = add_pseudo_if_variant_matches_reference(df, reference_sequence)
    # Drop duplicates, as PKermut can't handle duplicate inputs
    df.drop_duplicates(subset=["sequence"]).to_csv(dataset_path)
    if prepare_artifacts:
        logger.info("PDB file passed, computing necessary artifacts")
        _ = precompute_artifacts(
            dataset_name=dataset_name,
            data_dir=data_dir,
            pdb_file=pdb_file,
            reference_sequence=reference_sequence,
            device=device,
        )
    set_torch_dtype(preferential)

    params_to_update = {
        "dataset_name": dataset_name,
        "data_dir": data_dir,
        "reference_sequence": reference_sequence,
        "output_path": output_path,
        "n_steps": n_steps,
        "use_gpu": True if device == "cuda" else False,
        "preferential": preferential,
        "preference_sampling_strategy": preference_sampling_strategy,
    }
    prepare_hydra_configs(
        str(HYDRA_CONFIG_PATH), str(HYDRA_TEMP_CONFIG_PATH), params_to_update
    )

    with initialize_config_dir(config_dir=str(HYDRA_TEMP_CONFIG_PATH)):
        cfg = compose(config_name="benchmark")
        _evaluate_dms(cfg)

    results = pd.read_csv(Path(output_path) / f"{dataset_name}.csv")
    # Return to original size df again with possible duplicate sequences
    results = df.merge(results[["sequence", "fold", "y", "y_pred", "y_var"]], on="sequence", how="left")
    # Replace pseudo variants with reference
    results.loc[pseudo_mask, "sequence"] = reference_sequence
    logger.debug(results.columns)
    results.rename(columns={"y_var": "y_pred_var"}, inplace=True)
    results[["sequence", "split", "y", "y_pred", "y_pred_var"]].to_csv(
        Path(output_path) / "predictions.csv", index=False
    )
    log_and_save_metrics(results, str(output_path))


if __name__ == "__main__":
    app()
