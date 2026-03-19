import argparse
import numpy as np
import matplotlib.pyplot as plt 
import random
import sys
import importlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset

def parse_arguments():
    """Create the command-line parser for ECG classification.

    Returns:
        argparse.ArgumentParser: Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description='ECG classification with shapelet learning'
    )

    # Dataset configuration
    parser.add_argument('--dataset', type=str, default='MITBIH',
                        help='Dataset name (MITBIH/AFDB/CPSC2018)')
    parser.add_argument('--ratio', nargs='+', type=float, default=[0.8,0.1,0.1],
                        help='Train/Val/Test split ratios')
    parser.add_argument("--log_dir", default=r"./log", type=str,
                        help="Log directory path (default: ./log)")


    # Model hyperparameters
    parser.add_argument('--model', type=str, default="DNNClassifier",
                        help='Classifier architecture selection')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Training batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--minlrfac', type=float, default=1e-3,
                        help='Mininum learning rate = args.lr * args.minlrfac')
    parser.add_argument('--Tmax', type=int, default=1200,
                        help='Max LR scheduler')
    parser.add_argument('--epochs', type=int, default=1500,
                        help='Number of training epochs')
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[256, 256] ,
                        help='List of hidden layer dimensions for the classifier')
    # input_dim will be determined dynamically

    # Shapelet configuration
    parser.add_argument('--shapelet_num', type=int, default=100,
                        help='Number of shapelets to discover (KMeans clusters)')

    # System settings
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"],
                        help="Training device")
    parser.add_argument("--random_seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument('--sglead', action='store_true',
                        help='Use single ECG lead')
    parser.add_argument('--cuml', action='store_true',
                        help='Use cuml')

    # Stage 1 (representation learning) configuration
    parser.add_argument('--run_stage1', action='store_true',
                        help='Run Stage 1 representation learning to generate <dataset>_train.h5')
    parser.add_argument('--stage1_epochs', type=int, default=20,
                        help='Epochs for Stage 1 representation training')
    parser.add_argument('--stage1_batch_size', type=int, default=128,
                        help='Batch size for Stage 1 representation training')
    parser.add_argument('--stage1_lr', type=float, default=1e-3,
                        help='Learning rate for Stage 1 optimizer')
    parser.add_argument('--stage1_fp16', action='store_true',
                        help='Enable AMP in Stage 1 training')
    parser.add_argument('--stage1_model', type=str, default='TransTCNEncoderWithClassifier',
                        help='Stage 1 architecture class name in model package')
    parser.add_argument('--stage1_backbone', type=str, default='TransTCNEncoder',
                        help='Backbone encoder class for wrapper models such as EncoderWithClassifier')
    parser.add_argument('--slide_num', type=int, default=3,
                        help='Number of sliding-window views in Stage 1')
    parser.add_argument('--alpha_arr', nargs='+', type=float, default=[0.8, 0.5, 0.3],
                        help='Window-length ratios for each Stage 1 sliding-window view')
    parser.add_argument('--loss_weight', nargs='+', type=float,
                        default=[0.1, 0.1, 10.0, 0.1, 100.0, 0.1, 0.1],
                        help='Loss weights: class_all class_sep ce entropy cca rind_all rind_sep')
    parser.add_argument('--in_channels', type=int, default=1,
                        help='Input channels for Stage 1 encoder')
    parser.add_argument('--channels', type=int, default=32,
                        help='Backbone channels for Stage 1 encoder')
    parser.add_argument('--depth', type=int, default=3,
                        help='Encoder depth for Stage 1 model')
    parser.add_argument('--reduced_size', type=int, default=64,
                        help='Adaptive pooled feature size for Stage 1 encoder')
    parser.add_argument('--out_channels', type=int, default=64,
                        help='Output embedding size for Stage 1 encoder')
    parser.add_argument('--kernel_size', type=int, default=5,
                        help='Kernel size for Stage 1 temporal convolution block')

    return parser

def setup_seed(seed=42):
    """Set random seeds for reproducibility.
    
    Args:
        seed (int): Random seed value (default: 42)
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

def import_class(module_path: str, class_name: str) -> object:
    """Dynamically import a class from a module.
    
    Args:
        module_path (str): Dot-separated module path (e.g. "package.module")
        class_name (str): Target class name to import
        
    Returns:
        object: Imported class object
        
    Raises:
        ImportError: If module or class not found
    """
    try:
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except (ModuleNotFoundError, AttributeError) as e:
        raise ImportError(f"Failed to import {class_name} from {module_path}") from e


def get_step_size(sequence_length: int) -> int:
    """Compute sliding-window step size from sequence length."""
    return max(1, int(sequence_length / 20))


def slide_mts_general(X, labels, alpha=0.5, return_labels=False, return_rind=False):
    """Create sliding-window samples from multivariate time series.

    Args:
        X: Tensor shaped [num_samples, num_variates, seq_len].
        labels: Tensor shaped [num_samples].
        alpha: New window length ratio in (0, 1].
        return_labels: Whether to return window-level labels.
        return_rind: Whether to return sample-dimension group indices.
    """
    if not 0 < alpha <= 1:
        raise ValueError("alpha must be in (0, 1]")

    num_samples, num_variates, old_length = X.size()
    new_length = int(old_length * alpha)
    device = X.device
    step = get_step_size(old_length)

    windows = X.unfold(dimension=2, size=new_length, step=step)
    num_windows = windows.size(2)

    flat_windows = windows.permute(0, 2, 1, 3).contiguous().view(-1, new_length)

    if not return_labels:
        return flat_windows

    dim_labels = torch.arange(num_variates, device=device).repeat(num_samples * num_windows)
    class_labels = labels.repeat_interleave(num_variates * num_windows)

    if return_rind:
        sample_index = torch.arange(num_samples, device=device).view(num_samples, 1, 1)
        dim_index = torch.arange(num_variates, device=device).view(1, 1, num_variates)
        rind = (sample_index * num_variates + dim_index).expand(-1, num_windows, -1).reshape(-1)
        return flat_windows, dim_labels, class_labels, rind

    return flat_windows, dim_labels, class_labels


def compute_self_entropy(logits):
    """Compute mean self-entropy of class probabilities."""
    probs = torch.softmax(logits, dim=1)
    log_probs = torch.log(probs + 1e-10)
    entropy = -torch.sum(probs * log_probs, dim=1)
    return entropy.mean()


def compute_cluster_centers(class_labels, encoded_features):
    """Compute class-wise feature centroids."""
    unique_labels = torch.unique(class_labels)
    n_classes = len(unique_labels)
    n_features = encoded_features.shape[1]

    centers = torch.zeros((n_classes, n_features), device=encoded_features.device)
    for index, label in enumerate(unique_labels):
        mask = class_labels == label
        if torch.sum(mask) == 0:
            raise ValueError(f"Class {label} has no samples.")
        centers[index] = torch.mean(encoded_features[mask], dim=0)
    return centers


def cluster_center_alignment(centers_list):
    """Compute pairwise MSE alignment loss across center lists."""
    loss = 0.0
    num_views = len(centers_list)
    for i in range(num_views):
        for j in range(i + 1, num_views):
            loss += F.mse_loss(centers_list[i], centers_list[j])
    return loss / (num_views * (num_views - 1) / 2)

def unified_access(arrays: list[np.ndarray], index: int) -> np.ndarray:
    """Access elements across multiple arrays with unified indexing.
    
    Args:
        arrays (list): List of numpy arrays with compatible shapes
        index (int): Global index across all arrays
        
    Returns:
        np.ndarray: Element from corresponding array
        
    Raises:
        IndexError: When index exceeds total elements
    """
    cumulative_sizes = np.cumsum([arr.shape[0] for arr in arrays])
    
    if index >= cumulative_sizes[-1]:
        raise IndexError(f"Index {index} out of range (max {cumulative_sizes[-1]-1})")
        
    arr_idx = np.searchsorted(cumulative_sizes, index, side='right')
    local_idx = index - (cumulative_sizes[arr_idx-1] if arr_idx > 0 else 0)
    
    return arrays[arr_idx][local_idx]

def plot_shapelets(
    candidates: list[torch.Tensor], 
    channels: list[int], 
    sorted_indices: list[int]
) -> plt.Figure:
    """Visualize shapelets in grid layout.
    
    Args:
        candidates (list): List of shapelet tensors
        channels (list): Corresponding ECG lead channels
        sorted_indices (list): Display order indices
        
    Returns:
        plt.Figure: Matplotlib figure object
    """
    num_shapelets = len(sorted_indices)
    rows = (num_shapelets + 3) // 4  # 4 columns per row
    
    fig, axes = plt.subplots(rows, 4, figsize=(20, 5*rows))
    axes = axes.flat
    
    # Plot each shapelet
    for i, idx in enumerate(sorted_indices):
        ax = axes[i]
        shapelet = candidates[idx].cpu().numpy() if isinstance(candidates[idx], torch.Tensor) else candidates[idx]
        
        ax.plot(shapelet)
        ax.set_title(f"Channel {channels[idx]}\nLength: {len(shapelet)}, Rank: {i}")
        ax.grid(True)
    
    # Hide empty subplots
    for j in range(num_shapelets, rows*4):
        axes[j].axis('off')
    
    plt.tight_layout()
    return fig

def get_features(train_data, final_shapelets, shapelet_dims):
    """Extract shapelet-based features from ECG data.
    
    Args:
        train_data (Tensor): Raw ECG data [num_samples, num_dims, seq_len]
        final_shapelets (list): Selected shapelet tensors
        shapelet_dims (Tensor): Dimension indices for each shapelet
        
    Returns:
        ndarray: Feature matrix [num_samples, num_shapelets]
    """
    batch_size = 512
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = device.type == "cuda"
    
    # Move shapelets to the target device once.
    final_shapelets_gpu = [s.to(device) for s in final_shapelets]
    
    features = []
    for j, shapelet in enumerate(final_shapelets_gpu):
        dim = int(shapelet_dims[j])
        window_size = len(shapelet)
        
        # Build sliding windows over each time series.
        ts_data = train_data[:, dim, :]
        dataset = TensorDataset(ts_data)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        batch_features = []
        for batch in loader:
            batch_ts = batch[0].to(device)
            
            # Compute minimum distance from each window to one shapelet.
            windows = batch_ts.unfold(1, window_size, 1)  # [B, num_windows, w]
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                distances = torch.cdist(windows.float(), 
                                      shapelet.view(1, 1, -1).float(), 
                                      p=2)
                min_dist, _ = torch.min(distances, dim=1)
            
            # Release per-batch tensors to reduce peak memory.
            batch_features.append(min_dist.cpu())
            del batch_ts, windows, distances
            torch.cuda.empty_cache()
        
        features.append(torch.cat(batch_features))
    
    return torch.stack(features, dim=1).numpy().squeeze()

class CustomDataset(Dataset):
    """PyTorch dataset for ECG feature-label pairs
    
    Args:
        features (array-like): Feature matrix
        labels (array-like): Target labels
    """
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class Tee:
    """Write the same output message to multiple streams."""
    def __init__(self, *streams):
        self.streams = streams
    def write(self, msg):
        for s in self.streams:
            s.write(msg)
    def flush(self):
        for s in self.streams:
            s.flush()