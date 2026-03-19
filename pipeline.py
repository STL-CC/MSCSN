import json
import os
import time
from datetime import datetime
from typing import Dict

import h5py
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch import nn, optim
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

import datasets
from losses import SupInfoLoss
from utils import CustomDataset, Tee, get_features, import_class, unified_access


def load_hdf5_data(file_path: str):
    """Load shapelet-discovery inputs from an HDF5 file.

    Returns:
        tuple:
            - full_representation: NumPy array of flattened representations.
            - all_dims: torch.Tensor of source dimensions.
            - representation_candidates: list of torch.Tensor candidate shapelets.
    """
    try:
        with h5py.File(file_path, "r") as hdf5_file:
            full_representation = hdf5_file["full_representation"][:]
            all_dims = hdf5_file["all_dims"][:]

            candidates_group = hdf5_file.get("candidates")
            if candidates_group is None:
                representation_candidates = []
                print("Warning: 'candidates' group not found in HDF5.")
            else:
                representation_candidates = [
                    candidates_group[key][()]
                    for key in candidates_group.keys()
                    if key.startswith("candidate_")
                ]

        all_dims = torch.from_numpy(all_dims)
        representation_candidates = [
            torch.from_numpy(array_item) for array_item in representation_candidates
        ]

        print(f"Successfully loaded data from {file_path}")
        print(f"  full_representation shape: {full_representation.shape}")
        print(f"  all_dims shape: {all_dims.shape}")
        print(f"  Number of candidate representations: {len(representation_candidates)}")
        for index, candidate in enumerate(representation_candidates):
            print(f"    candidate_{index} shape: {candidate.shape}")

        return full_representation, all_dims, representation_candidates

    except FileNotFoundError:
        print(f"Error: HDF5 file not found at {file_path}")
        raise
    except KeyError as key_error:
        print(f"Error: Missing key in {file_path}: {key_error}")
        raise
    except Exception as error:
        print(f"An unexpected error occurred while loading HDF5: {error}")
        raise


def _train_stage1_representation_model(
    args,
    train_data: torch.Tensor,
    train_labels: torch.Tensor,
):
    """Train Stage 1 representation model and return model/loss objects."""
    stage1_model_cls = import_class("model", args.stage1_model)
    backbone_cls = import_class("model", args.stage1_backbone)

    # Wrapper models (e.g. EncoderWithClassifier) require a backbone class.
    if args.stage1_model == "EncoderWithClassifier":
        model = stage1_model_cls(args, backbone_cls).to(args.device)
    else:
        try:
            model = stage1_model_cls(args, backbone_cls).to(args.device)
        except TypeError:
            model = stage1_model_cls(args).to(args.device)

    criterion = SupInfoLoss(args)
    optimizer = optim.Adam(model.parameters(), lr=args.stage1_lr)
    amp_enabled = bool(args.stage1_fp16 and str(args.device).startswith("cuda"))
    scaler = GradScaler("cuda", enabled=amp_enabled)

    dataset = TensorDataset(train_data, train_labels)
    loader = DataLoader(dataset, batch_size=args.stage1_batch_size, shuffle=True)

    print(
        f"Stage 1 training started | epochs={args.stage1_epochs}, "
        f"batch_size={args.stage1_batch_size}, amp={amp_enabled}"
    )

    model.train()
    for epoch in range(1, args.stage1_epochs + 1):
        epoch_loss = 0.0
        batch_count = 0

        for signals, labels in loader:
            signals = signals.to(args.device)
            labels = labels.to(args.device)

            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=amp_enabled):
                loss_items = criterion(signals, labels, model)
                total_loss = loss_items[0]

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += float(total_loss.item())
            batch_count += 1

        avg_epoch_loss = epoch_loss / max(1, batch_count)
        print(f"Stage 1 epoch {epoch}/{args.stage1_epochs} | loss={avg_epoch_loss:.6f}")

    return model, criterion


def _export_stage1_hdf5(
    args,
    model,
    criterion,
    train_data: torch.Tensor,
    train_labels: torch.Tensor,
    file_path: str,
):
    """Export Stage 1 representations/candidates into one HDF5 file."""
    dataset = TensorDataset(train_data, train_labels)
    loader = DataLoader(dataset, batch_size=args.stage1_batch_size, shuffle=False)

    full_representation = []
    all_dims = []
    all_labels = []
    representation_candidates = [[] for _ in range(args.slide_num)]

    model.eval()
    with torch.no_grad():
        for signals, labels in loader:
            signals = signals.to(args.device)
            labels = labels.to(args.device)

            _ = criterion(signals, labels, model)
            full_representation.append(criterion.encoded_x.detach().cpu())
            all_dims.append(criterion.dim_labels.detach().cpu())
            all_labels.append(criterion.class_labels.detach().cpu())

            for index in range(args.slide_num):
                representation_candidates[index].append(
                    criterion.rep_candidates[index].detach().cpu()
                )

    full_representation_np = torch.concat(full_representation).numpy().astype(np.float32)
    all_dims_np = torch.concat(all_dims).numpy()
    all_labels_np = torch.concat(all_labels).numpy()
    representation_candidates_np = [
        torch.concat(candidate_list).numpy().astype(np.float32)
        for candidate_list in representation_candidates
    ]

    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with h5py.File(file_path, "w") as hdf5_file:
        hdf5_file.create_dataset(
            "full_representation", data=full_representation_np, compression="gzip"
        )
        hdf5_file.create_dataset("all_dims", data=all_dims_np, compression="gzip")
        hdf5_file.create_dataset("all_labels", data=all_labels_np, compression="gzip")

        candidates_group = hdf5_file.create_group("candidates")
        for index, candidate_np in enumerate(representation_candidates_np):
            candidates_group.create_dataset(
                f"candidate_{index}",
                data=candidate_np,
                compression="gzip",
            )

        hdf5_file.attrs["creation_date"] = datetime.now().isoformat(timespec="seconds")
        hdf5_file.attrs["source"] = "stage1_representation"

    print(f"Stage 1 export complete: {file_path}")


def ensure_stage1_hdf5(args, train_data: torch.Tensor, train_labels: torch.Tensor) -> str:
    """Ensure representation file exists under data/representations.

    Priority:
    1. Reuse data/representations/<dataset>_train.h5 if it exists and not forcing rebuild.
    2. Migrate legacy <dataset>_train.h5 in repository root if present and not forcing rebuild.
    3. Otherwise run Stage 1 training and generate file.
    """
    representations_dir = os.path.join("data", "representations")
    target_path = os.path.join(representations_dir, f"{args.dataset}_train.h5")
    legacy_path = f"{args.dataset}_train.h5"

    if os.path.exists(target_path) and not args.run_stage1:
        print(f"Stage 1 HDF5 found, reuse existing file: {target_path}")
        return target_path

    if os.path.exists(legacy_path) and not args.run_stage1:
        os.makedirs(representations_dir, exist_ok=True)
        os.replace(legacy_path, target_path)
        print(f"Migrated legacy representation file to: {target_path}")
        return target_path

    if os.path.exists(target_path) and args.run_stage1:
        print(f"Stage 1 HDF5 exists but will be rebuilt: {target_path}")

    model, criterion = _train_stage1_representation_model(args, train_data, train_labels)
    _export_stage1_hdf5(args, model, criterion, train_data, train_labels, target_path)
    return target_path


def discover_shapelets(
    full_representation: np.ndarray,
    all_dims: torch.Tensor,
    representation_candidates: list,
    args,
):
    """Discover shapelets using KMeans and nearest-centroid samples."""
    num_shapelets = args.shapelet_num
    print(f"Starting shapelet discovery with {num_shapelets} clusters...")

    if len(full_representation) == 0:
        print("Error: full_representation is empty. Cannot perform KMeans.")
        return [], torch.empty(0)
    if num_shapelets <= 0:
        print("Error: Number of shapelets must be positive.")
        return [], torch.empty(0)
    if num_shapelets > len(full_representation):
        print(
            f"Warning: Requested {num_shapelets} shapelets, but only "
            f"{len(full_representation)} representations available. Reducing clusters."
        )
        num_shapelets = len(full_representation)
        if num_shapelets == 0:
            return [], torch.empty(0)

    if args.cuml:
        from cuml import KMeans as cuKMeans
        from cuml.neighbors import NearestNeighbors

        kmeans = cuKMeans(
            n_clusters=num_shapelets,
            init="k-means++",
            max_iter=1000,
            tol=1e-4,
            random_state=args.random_seed,
        )

        kmeans.fit(full_representation)
        centroids = kmeans.cluster_centers_

        neighbors = NearestNeighbors(n_neighbors=1).fit(full_representation)
        _, centroid_indices = neighbors.kneighbors(centroids)
        centroid_indices = centroid_indices.flatten()
    else:
        from sklearn.cluster import KMeans
        from sklearn.neighbors import NearestNeighbors

        kmeans = KMeans(
            n_clusters=num_shapelets,
            init="k-means++",
            max_iter=1000,
            tol=1e-4,
            random_state=args.random_seed,
        )
        kmeans.fit(full_representation)

        centroids = kmeans.cluster_centers_
        neighbors = NearestNeighbors(n_neighbors=1).fit(full_representation)
        _, centroid_indices = neighbors.kneighbors(centroids)
        centroid_indices = centroid_indices.flatten()

    final_shapelets = []
    shapelet_dims = []
    valid_indices = centroid_indices

    if len(valid_indices) < len(centroid_indices):
        print(
            f"Warning: {len(centroid_indices) - len(valid_indices)} centroid indices "
            "were out of bounds for loaded data."
        )

    for index in valid_indices:
        try:
            selected_shapelet = unified_access(representation_candidates, index)
            if selected_shapelet is not None:
                final_shapelets.append(selected_shapelet)
                shapelet_dims.append(all_dims[index])
            else:
                print(f"Warning: unified_access returned None for index {index}. Skipping.")
        except Exception as error:
            print(f"Error accessing shapelet for index {index}: {error}. Skipping.")

    print(f"Discovered {len(final_shapelets)} shapelets.")
    if not final_shapelets:
        print(
            "Error: No valid shapelets were discovered after filtering. "
            "Check data and unified_access."
        )

    shapelet_dims = torch.stack(shapelet_dims) if shapelet_dims else torch.empty(0)
    return final_shapelets, shapelet_dims


def prepare_dataloaders(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    val_features: torch.Tensor,
    val_labels: torch.Tensor,
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
    args,
):
    """Prepare train/validation/test dataloaders."""
    print("Preparing data for classifier training...")
    train_dataset = CustomDataset(train_features, train_labels)
    val_dataset = CustomDataset(val_features, val_labels)
    test_dataset = CustomDataset(test_features, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")
    print(f"  Batch size: {args.batch_size}")

    return train_loader, val_loader, test_loader


def evaluate_loader(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    criterion: nn.Module = None,
) -> Dict:
    """Compute loss and classification metrics on a dataloader."""
    model.eval()
    running_loss = 0.0
    total = 0
    all_predictions, all_targets = [], []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            if inputs.numel() == 0:
                continue

            outputs = model(inputs)
            predictions = torch.argmax(outputs, dim=1)

            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

            if criterion is not None:
                loss = criterion(outputs, labels)
                running_loss += loss.item() * inputs.size(0)
                total += inputs.size(0)

    metrics = {}
    if criterion is not None:
        avg_loss = running_loss / total if total else float("nan")
        metrics["Loss"] = avg_loss

    if all_targets:
        metrics.update(
            {
                "ACC": accuracy_score(all_targets, all_predictions),
                "F1_macro": f1_score(
                    all_targets,
                    all_predictions,
                    average="macro",
                    zero_division=0,
                ),
                "F1_weighted": f1_score(
                    all_targets,
                    all_predictions,
                    average="weighted",
                    zero_division=0,
                ),
                "Recall_macro": recall_score(
                    all_targets,
                    all_predictions,
                    average="macro",
                    zero_division=0,
                ),
                "Precision_macro": precision_score(
                    all_targets,
                    all_predictions,
                    average="macro",
                    zero_division=0,
                ),
                "Kappa": cohen_kappa_score(all_targets, all_predictions),
                "G_Mean": np.sqrt(
                    np.prod(
                        recall_score(
                            all_targets,
                            all_predictions,
                            average=None,
                            zero_division=0,
                        )
                    )
                ),
                "Confusion_Matrix": confusion_matrix(
                    all_targets, all_predictions
                ).tolist(),
                "Classification_Report": classification_report(
                    all_targets,
                    all_predictions,
                    output_dict=True,
                    zero_division=0,
                ),
            }
        )

    return metrics


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler,
    args,
    writer: SummaryWriter,
    eval_metric_key: str = "ACC",
):
    """Train classifier and save best validation/test metrics."""
    device = args.device
    best_metrics = {eval_metric_key: -float("inf")}
    model_save_path = os.path.join(args.run_dir, "best_model.pth")
    val_metrics_path = os.path.join(args.run_dir, "best_val_metrics.json")

    overall_start = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        model.train()
        optimizer.zero_grad()

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            if inputs.numel() == 0:
                continue

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        scheduler.step()
        current_lr = optimizer.state_dict()["param_groups"][0]["lr"]

        train_metrics = evaluate_loader(model, train_loader, device, criterion)
        val_metrics = evaluate_loader(model, val_loader, device, criterion)

        for key in train_metrics.keys():
            if key in ["Confusion_Matrix", "Classification_Report"]:
                continue
            writer.add_scalars(
                key,
                {
                    "train": train_metrics[key],
                    "val": val_metrics.get(key, float("nan")),
                },
                epoch,
            )
        writer.add_scalar("LR", current_lr, epoch)

        train_score = train_metrics.get(eval_metric_key, float("nan"))
        val_score = val_metrics.get(eval_metric_key, float("nan"))
        train_loss = train_metrics.get("Loss", float("nan"))
        val_loss = val_metrics.get("Loss", float("nan"))
        print(
            f"Epoch {epoch}/{args.epochs} | Loss (train/val): "
            f"{train_loss:.4f}/{val_loss:.4f} | "
            f"{eval_metric_key} (train/val): {train_score:.4f}/{val_score:.4f} | "
            f"LR:{current_lr:.8f} | Time: {time.time() - epoch_start:.2f}s"
        )

        if val_score > best_metrics[eval_metric_key]:
            best_metrics = val_metrics.copy()
            torch.save(model.state_dict(), model_save_path)
            print(
                f" Saved best model at epoch {epoch} "
                f"(Val {eval_metric_key}: {val_score:.4f})."
            )
            with open(val_metrics_path, "w") as file_pointer:
                json.dump(best_metrics, file_pointer, indent=2)

    total_time = time.time() - overall_start
    print(f"Training complete. Best Val {eval_metric_key}: {best_metrics[eval_metric_key]:.4f}")
    print(f"Best validation metrics saved to {val_metrics_path}: {best_metrics}")
    print(
        f"\nTraining complete in {total_time / 60:.2f} minutes "
        f"({total_time:.2f} seconds)."
    )

    if os.path.exists(model_save_path):
        model.load_state_dict(torch.load(model_save_path, map_location=device))
        test_metrics = evaluate_loader(model, test_loader, device, criterion)

        for key, value in test_metrics.items():
            if key in ["Confusion_Matrix", "Classification_Report"]:
                continue
            writer.add_scalar(f"Test/{key}", value)

        test_metrics_path = os.path.join(args.run_dir, "test_metrics.json")
        with open(test_metrics_path, "w") as file_pointer:
            json.dump(test_metrics, file_pointer, indent=2)
        print(f"Test metrics saved to {test_metrics_path}: {test_metrics}")
    else:
        print("Warning: Best model not found; skipping test evaluation.")

    return best_metrics


def stage1_initialize_run(args):
    """Create run directory, logging, argument snapshot, and TensorBoard writer."""
    from datetime import datetime

    timestamp = datetime.now().strftime("%y%m%d-%H:%M")
    run_name = f"{timestamp}-{args.dataset}-{args.model}-{args.shapelet_num}"
    run_dir = os.path.join(args.log_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    args.run_dir = run_dir

    log_path = os.path.join(run_dir, "train.log")
    log_file = open(log_path, "w")

    import sys

    sys.stdout = Tee(sys.stdout, log_file)
    sys.stderr = Tee(sys.stderr, log_file)

    print(f"Logging to {log_path}")

    with open(os.path.join(run_dir, "args.json"), "w") as file_pointer:
        json.dump(vars(args), file_pointer, indent=2)

    writer = SummaryWriter(log_dir=os.path.join(run_dir, "tensorboard"))

    print(f"Run directory: {run_dir}")
    print(f"Device: {args.device} | Dataset: {args.dataset} | Seed: {args.random_seed}")
    return writer


def stage2_load_data(args):
    """Load train/validation/test data from selected dataset loader."""
    dataset_loader = getattr(datasets, f"load_{args.dataset}_dataset")
    (
        train_data,
        train_labels,
        val_data,
        val_labels,
        test_data,
        test_labels,
        class_map,
    ) = dataset_loader(args)
    args.num_classes = len(class_map)

    return train_data, train_labels, val_data, val_labels, test_data, test_labels


def stage3_prepare_representation_file(args, train_data, train_labels):
    """Prepare or reuse Stage 1 HDF5 representation file."""
    if train_data.ndim != 3:
        raise ValueError(f"Stage 1 expects train_data with shape [N, C, L], got {train_data.shape}")

    args.in_channels = int(train_data.shape[1])
    if len(args.alpha_arr) != args.slide_num:
        raise ValueError(
            "alpha_arr length must match slide_num, "
            f"got len(alpha_arr)={len(args.alpha_arr)} and slide_num={args.slide_num}"
        )

    return ensure_stage1_hdf5(args, train_data, train_labels)


def stage4_discover_shapelets(args, hdf5_path):
    """Load HDF5 representations and discover final shapelets."""
    full_representation, all_dims, candidates = load_hdf5_data(hdf5_path)
    final_shapelets, shapelet_dims = discover_shapelets(
        full_representation, all_dims, candidates, args
    )
    return final_shapelets, shapelet_dims


def stage5_extract_features(
    train_data,
    val_data,
    test_data,
    final_shapelets,
    shapelet_dims,
):
    """Extract shapelet-based features for train/validation/test sets."""
    train_features = get_features(train_data, final_shapelets, shapelet_dims)
    val_features = get_features(val_data, final_shapelets, shapelet_dims)
    test_features = get_features(test_data, final_shapelets, shapelet_dims)

    return train_features, val_features, test_features


def stage6_train_classifier(
    args,
    writer,
    train_features,
    train_labels,
    val_features,
    val_labels,
    test_features,
    test_labels,
):
    """Prepare loaders, initialize model, and execute training."""
    train_loader, val_loader, test_loader = prepare_dataloaders(
        train_features,
        train_labels,
        val_features,
        val_labels,
        test_features,
        test_labels,
        args,
    )

    classifier_class = import_class("model", args.model)
    model = classifier_class(args).to(args.device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.Tmax,
        eta_min=args.lr * args.minlrfac,
    )

    return train_model(
        model,
        train_loader,
        val_loader,
        test_loader,
        criterion,
        optimizer,
        scheduler,
        args,
        writer,
        eval_metric_key="F1_macro",
    )