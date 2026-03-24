"""Train temporal models: Transformer, GRU, and flattened MLP.

These models consume sliding windows of T=6 monthly snapshots.
Training uses all windows; evaluation uses the LAST window per repo
for per-repo C-index comparison with baselines and Xu et al.

Usage:
    python -m depwatch.model_training.train_temporal
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from depwatch.common.features import FeatureVector
from depwatch.model_training.evaluate import EvalResult, evaluate_binary
from depwatch.model_training.temporal_models import (
    AbandonmentGRU,
    AbandonmentMLP,
    AbandonmentTransformer,
)

logger = logging.getLogger(__name__)

WINDOW_SIZE = 6
HORIZONS = (3, 6, 12)
PRIMARY_HORIZON_IDX = 1  # 6-month horizon for main comparison


def prepare_temporal_data(
    df: pd.DataFrame,
    *,
    stride: int = 3,
    seed: int = 42,
) -> dict[str, object]:
    """Build sliding windows split by repo into train/val/test.

    Returns dict with:
    - train/val/test DataLoaders (all windows)
    - test_last_* arrays (last window per repo, for per-repo evaluation)
    """
    feature_cols = [c for c in FeatureVector.feature_names() if c in df.columns]
    logger.info("Using %d features for temporal models", len(feature_cols))

    # Split repos first (70/15/15)
    rng = np.random.default_rng(seed)
    repos = df["repo_name"].unique()
    rng.shuffle(repos)
    n = len(repos)
    n_train = int(n * 0.7)
    n_val = int(n * 0.85)

    train_repos = set(repos[:n_train])
    val_repos = set(repos[n_train:n_val])
    test_repos = set(repos[n_val:])

    # Build windows per split
    def build_windows(
        split_df: pd.DataFrame,
        window_stride: int = stride,
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[str], list[str]]:
        """Build sliding windows, tracking which repo and end-month each belongs to."""
        windows: list[np.ndarray] = []
        labels: list[np.ndarray] = []
        window_repos: list[str] = []
        window_end_months: list[str] = []

        for repo_name, group in split_df.groupby("repo_name"):
            group = group.sort_values("snapshot_month")
            if len(group) < WINDOW_SIZE:
                continue

            is_abandoned = bool(group.iloc[-1]["is_abandoned"])
            abd_date_raw = group.iloc[-1].get("estimated_abandonment_date")

            feat_values = group[feature_cols].values.astype(np.float32)
            snap_months = group["snapshot_month"].values

            last_start = len(group) - WINDOW_SIZE
            starts = list(range(0, last_start + 1, window_stride))
            # Always include the last window
            if starts[-1] != last_start:
                starts.append(last_start)

            for start in starts:
                window = feat_values[start : start + WINDOW_SIZE]
                end_month = str(snap_months[start + WINDOW_SIZE - 1])

                # Compute horizon labels
                horizon_labels = np.zeros(len(HORIZONS), dtype=np.float32)
                if is_abandoned and abd_date_raw is not None:
                    try:
                        end_dt = datetime.fromisoformat(str(end_month)[:10])
                        abd_dt = datetime.fromisoformat(str(abd_date_raw)[:10])
                        months_until = (
                            (abd_dt.year - end_dt.year) * 12 + abd_dt.month - end_dt.month
                        )
                        for hi, h in enumerate(HORIZONS):
                            if 0 <= months_until <= h:
                                horizon_labels[hi] = 1.0
                    except (ValueError, TypeError):
                        pass

                windows.append(window)
                labels.append(horizon_labels)
                window_repos.append(str(repo_name))
                window_end_months.append(end_month)

        return windows, labels, window_repos, window_end_months

    train_df = df[df["repo_name"].isin(train_repos)]
    val_df = df[df["repo_name"].isin(val_repos)]
    test_df = df[df["repo_name"].isin(test_repos)]

    logger.info("Building sliding windows...")
    train_w, train_l, _, _ = build_windows(train_df)
    val_w, val_l, _, _ = build_windows(val_df)
    test_w, test_l, test_w_repos, _ = build_windows(test_df)

    logger.info(
        "Windows: train=%d, val=%d, test=%d",
        len(train_w),
        len(val_w),
        len(test_w),
    )

    # Convert to tensors
    def to_dataset(
        windows: list[np.ndarray],
        labels: list[np.ndarray],
    ) -> TensorDataset:
        x = torch.tensor(np.array(windows), dtype=torch.float32)
        y = torch.tensor(np.array(labels), dtype=torch.float32)
        return TensorDataset(x, y)

    train_ds = to_dataset(train_w, train_l)
    val_ds = to_dataset(val_w, val_l)
    test_ds = to_dataset(test_w, test_l)

    # Create balanced sampler for training: oversample positive windows
    # so each batch has ~50% positive (6-month horizon)
    train_labels_6mo = torch.tensor(
        [la[PRIMARY_HORIZON_IDX] for la in train_l],
        dtype=torch.float32,
    )
    pos_mask = train_labels_6mo == 1.0
    n_pos = pos_mask.sum().item()
    n_neg = len(train_labels_6mo) - n_pos
    sample_weights = torch.where(pos_mask, float(n_neg) / n_pos, 1.0)
    train_sampler = torch.utils.data.WeightedRandomSampler(
        sample_weights.tolist(),
        num_samples=len(sample_weights),
        replacement=True,
    )
    logger.info(
        "Training balance: %d positive (%.1f%%), oversampling to ~50%%",
        int(n_pos),
        n_pos / len(train_labels_6mo) * 100,
    )

    # Build per-repo test data: last window per repo
    last_window_idx: dict[str, int] = {}
    for i, repo in enumerate(test_w_repos):
        last_window_idx[repo] = i  # later windows overwrite earlier ones

    last_indices = list(last_window_idx.values())
    test_last_x = np.array([test_w[i] for i in last_indices])
    test_last_y = np.array([test_l[i] for i in last_indices])
    test_last_repos = [test_w_repos[i] for i in last_indices]

    # Time-to-event for per-repo C-index
    repo_info = (
        test_df.sort_values("snapshot_month")
        .drop_duplicates("repo_name", keep="last")
        .set_index("repo_name")
    )
    test_last_time = []
    test_last_event = []
    snap_max = pd.to_datetime(df["snapshot_month"]).max()

    for repo in test_last_repos:
        if repo in repo_info.index:
            row = repo_info.loc[repo]
            is_abd = bool(row["is_abandoned"])
            if is_abd:
                abd_dt = pd.to_datetime(row["estimated_abandonment_date"])
                snap_dt = pd.to_datetime(row["snapshot_month"])
                months = max(
                    (abd_dt.year - snap_dt.year) * 12 + abd_dt.month - snap_dt.month,
                    1,
                )
                test_last_time.append(float(months))
                test_last_event.append(1)
            else:
                snap_dt = pd.to_datetime(row["snapshot_month"])
                months = max(
                    (snap_max.year - snap_dt.year) * 12 + snap_max.month - snap_dt.month,
                    1,
                )
                test_last_time.append(float(months))
                test_last_event.append(0)
        else:
            test_last_time.append(1.0)
            test_last_event.append(0)

    logger.info(
        "Per-repo test set: %d repos (%.1f%% abandoned)",
        len(test_last_repos),
        np.mean(test_last_event) * 100,
    )

    return {
        "train_loader": DataLoader(train_ds, batch_size=256, sampler=train_sampler),
        "val_loader": DataLoader(val_ds, batch_size=512),
        "test_loader": DataLoader(test_ds, batch_size=512),
        "test_last_x": torch.tensor(test_last_x, dtype=torch.float32),
        "test_last_y": np.array(test_last_y),
        "test_last_time": np.array(test_last_time, dtype=np.float32),
        "test_last_event": np.array(test_last_event),
    }


def train_model(
    model: nn.Module,
    data: dict[str, object],
    *,
    model_name: str,
    epochs: int = 30,
    lr: float = 1e-3,
    patience: int = 5,
) -> EvalResult:
    """Train a temporal model and evaluate per-repo."""
    # Use CPU — MPS has known issues with transformer ops and small models
    # where transfer overhead negates any GPU benefit
    device = torch.device("cpu")
    logger.info("Training %s on %s...", model_name, device)

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_loader: DataLoader[tuple[torch.Tensor, ...]] = data["train_loader"]  # type: ignore[assignment]
    val_loader: DataLoader[tuple[torch.Tensor, ...]] = data["val_loader"]  # type: ignore[assignment]

    # Horizon importance weights: 0.2 * 3mo + 0.3 * 6mo + 0.5 * 12mo
    # Class balance handled by WeightedRandomSampler (batches are ~50/50)
    horizon_weights = torch.tensor([0.2, 0.3, 0.5], device=device)
    criterion = nn.BCEWithLogitsLoss(reduction="none")

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_batches = 0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(x_batch)
            raw_loss = criterion(logits, y_batch)  # (batch, 3)
            loss = (raw_loss * horizon_weights).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            train_batches += 1

        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                logits = model(x_batch)
                raw_loss = criterion(logits, y_batch)
                val_loss += (raw_loss * horizon_weights).mean().item()
                val_batches += 1

        avg_train = train_loss / max(train_batches, 1)
        avg_val = val_loss / max(val_batches, 1)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(
                "  %s epoch %d/%d: train_loss=%.4f val_loss=%.4f",
                model_name,
                epoch + 1,
                epochs,
                avg_train,
                avg_val,
            )

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                logger.info("  Early stopping at epoch %d", epoch + 1)
                break

    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.to(device)

    # Per-repo evaluation on last window
    model.eval()
    test_last_x: torch.Tensor = data["test_last_x"]  # type: ignore[assignment]
    with torch.no_grad():
        logits = model(test_last_x.to(device))
        pred = torch.sigmoid(logits).cpu().numpy()

    # Use 6-month horizon (index 1) for primary metrics
    y_pred_6mo = pred[:, PRIMARY_HORIZON_IDX]
    test_last_y: np.ndarray = data["test_last_y"]  # type: ignore[assignment]
    y_true_6mo = test_last_y[:, PRIMARY_HORIZON_IDX]
    time_test: np.ndarray = data["test_last_time"]  # type: ignore[assignment]
    event_test: np.ndarray = data["test_last_event"]  # type: ignore[assignment]

    result = evaluate_binary(
        f"{model_name} (test, per-repo)",
        y_true_6mo,
        y_pred_6mo,
        time_test,
        event_test,
    )
    logger.info(result.summary())

    # Also log per-horizon AUROC
    from sklearn.metrics import roc_auc_score

    for hi, h in enumerate(HORIZONS):
        y_true_h = test_last_y[:, hi]
        if y_true_h.sum() > 0 and y_true_h.sum() < len(y_true_h):
            auc = roc_auc_score(y_true_h, pred[:, hi])
            logger.info("  %s %d-month AUROC: %.4f", model_name, h, auc)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info("  %s params: %d", model_name, n_params)

    return result


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Train temporal models (Transformer, GRU, MLP)",
    )
    parser.add_argument(
        "--input",
        default="data/training/training_dataset.parquet",
        help="Path to training dataset",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Max training epochs (default: 30)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=3,
        help="Sliding window stride (default: 3, reduces training data 3x)",
    )
    parser.add_argument(
        "--model",
        choices=["all", "transformer", "gru", "mlp"],
        default="all",
        help="Which model to train (default: all)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    logger.info("Loading dataset from %s", args.input)
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pq.read_table(args.input)
    columns = []
    names = []
    for field in table.schema:
        col = table.column(field.name)
        if pa.types.is_date(field.type):
            col = col.cast(pa.string())
        elif pa.types.is_timestamp(field.type) and field.type.tz is not None:
            col = col.cast(pa.timestamp("us"))
        columns.append(col)
        names.append(field.name)
    df = pa.table(columns, names=names).to_pandas()

    logger.info("Dataset: %d rows, %d repos", len(df), df["repo_name"].nunique())

    # Prepare temporal data
    data = prepare_temporal_data(df, stride=args.stride)

    # Normalize features using training set statistics
    train_loader_raw: DataLoader[tuple[torch.Tensor, ...]] = data["train_loader"]  # type: ignore[assignment]
    all_train_x = torch.cat([x for x, _ in train_loader_raw])
    all_train_y = torch.cat([y for _, y in train_loader_raw])
    feat_mean = all_train_x.mean(dim=(0, 1))  # (D,)
    feat_std = all_train_x.std(dim=(0, 1)).clamp(min=1e-6)  # (D,)

    # Normalize and rebuild loaders
    all_train_x = (all_train_x - feat_mean) / feat_std
    train_ds_norm = TensorDataset(all_train_x, all_train_y)

    # Rebuild train sampler on normalized dataset
    train_labels_6mo = all_train_y[:, PRIMARY_HORIZON_IDX]
    pos_mask = train_labels_6mo == 1.0
    n_pos = pos_mask.sum().item()
    n_neg = len(train_labels_6mo) - n_pos
    sample_weights = torch.where(pos_mask, float(n_neg) / max(n_pos, 1), 1.0)
    train_sampler = torch.utils.data.WeightedRandomSampler(
        sample_weights.tolist(),
        num_samples=len(sample_weights),
        replacement=True,
    )
    data["train_loader"] = DataLoader(train_ds_norm, batch_size=256, sampler=train_sampler)

    val_loader_raw: DataLoader[tuple[torch.Tensor, ...]] = data["val_loader"]  # type: ignore[assignment]
    val_x = (torch.cat([x for x, _ in val_loader_raw]) - feat_mean) / feat_std
    val_y = torch.cat([y for _, y in val_loader_raw])
    data["val_loader"] = DataLoader(TensorDataset(val_x, val_y), batch_size=512)

    data["test_last_x"] = (data["test_last_x"] - feat_mean) / feat_std  # type: ignore[operator]

    # Train selected models
    results: list[EvalResult] = []
    models_to_train = ["transformer", "gru", "mlp"] if args.model == "all" else [args.model]

    if "transformer" in models_to_train:
        transformer_result = train_model(
            AbandonmentTransformer(),
            data,
            model_name="Transformer",
            epochs=args.epochs,
        )
        results.append(transformer_result)

    if "gru" in models_to_train:
        gru_result = train_model(
            AbandonmentGRU(),
            data,
            model_name="GRU",
            epochs=args.epochs,
        )
        results.append(gru_result)

    if "mlp" in models_to_train:
        mlp_result = train_model(
            AbandonmentMLP(),
            data,
            model_name="MLP",
            epochs=args.epochs,
        )
        results.append(mlp_result)

    # Summary
    logger.info("")
    logger.info("=" * 80)
    logger.info("TEMPORAL MODEL RESULTS (test set, per-repo, 6-month horizon)")
    logger.info("=" * 80)
    for r in results:
        logger.info(r.summary())
    logger.info("=" * 80)
    logger.info("Baseline comparison: XGBoost AFT C-index = 0.7477")
    logger.info("Xu et al. reference: C-index = 0.846")


if __name__ == "__main__":
    main()
