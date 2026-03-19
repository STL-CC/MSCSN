import wfdb
import numpy as np
import pywt
import os
import torch
import h5py
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

def split_dataset(labels, ratios):
    """Stratified split dataset into train/val/test subsets
    
    Args:
        labels (np.ndarray): Array of data labels for stratification
        ratios (list): Split proportions (sum ≤ 1)
        
    Returns:
        list: Indices arrays for each subset
    """
    assert sum(ratios) <= 1, "Sum of ratios cannot exceed 1"
    original_size = len(labels)
    remaining_idx = np.arange(original_size)
    split_indices = []

    for ratio in ratios[:-1]:
        # Calculate required sample size
        global_test_size = int(original_size * ratio)
        if global_test_size == 0:
            split_indices.append(np.array([]))
            continue

        # Perform stratified split
        current_test_ratio = global_test_size / len(remaining_idx)
        split = StratifiedShuffleSplit(n_splits=1, test_size=current_test_ratio)
        
        for _, test_idx in split.split(remaining_idx, labels[remaining_idx]):
            split_indices.append(remaining_idx[test_idx])
            remaining_idx = np.setdiff1d(remaining_idx, remaining_idx[test_idx])
    
    split_indices.append(remaining_idx)
    return split_indices

def print_label_stats(all_labels, class_map):
    """Display class distribution statistics
    
    Args:
        all_labels (np.ndarray): Integer label array
        class_map (dict): Mapping from indices to class names
    """
    total = len(all_labels)
    unique, counts = np.unique(all_labels, return_counts=True)
    count_dict = dict(zip(unique, counts))
    
    # Build statistics table
    stats = []
    for idx in sorted(class_map.keys()):
        class_name = class_map[idx]
        count = count_dict.get(idx, 0)
        percent = count / total * 100 if total > 0 else 0
        stats.append((class_name, idx, count, percent))

    # Print formatted table
    print("\nClass Statistics:")
    print("| Class | Index | Count  | Percent |")
    print("|-------|-------|--------|---------|")
    for class_name, idx, count, percent in stats:
        print(f"| {class_name:<5} | {idx:<5} | {count:<6} | {percent:>6.2f}% |")
    
    print(f"\nTotal samples: {total}")
    if len(unique) != len(class_map):
        missing = set(class_map.keys()) - set(unique)
        print(f"Missing classes: {[class_map[i] for i in sorted(missing)]}")

def _wavelet_denoise(signal):
    """Apply wavelet denoising to single-lead ECG signal
    
    Args:
        signal (np.ndarray): Raw ECG signal
        
    Returns:
        np.ndarray: Denoised signal
    """
    coeffs = pywt.wavedec(signal, 'db5', level=9)
    threshold = np.median(np.abs(coeffs[-1]))/0.6745 * np.sqrt(2*np.log(len(signal)))
    
    # Threshold detail coefficients
    for i in range(1, len(coeffs)-2):
        coeffs[i] = pywt.threshold(coeffs[i], threshold)
        
    return pywt.waverec(coeffs, 'db5')

def load_AFDB_dataset(args):
    """Load and preprocess AFDB dataset
    
    Args:
        args: Configuration parameters
        
    Returns:
        tuple: (train_data, val_data, test_data, class_map)
    """
    data_dir = './data/AFDB'
    ecg_classes = ['N', 'A']
    class_map = {cls:i for i, cls in enumerate(ecg_classes)}
    class_map_r = {i:cls for i, cls in enumerate(ecg_classes)}
    
    # Load numpy arrays
    X = np.expand_dims(np.load(os.path.join(data_dir, 'data.npy')), 
                     axis=1).astype(np.float32).squeeze()
    y = np.load(os.path.join(data_dir, 'labels.npy'))
    indices = split_dataset(y, args.ratio)
    
    # Initialize data scalers
    scalers = {}
    datasets = {}
    for i, split in enumerate(['train', 'val', 'test']):
        datasets[split] = (X[indices[i]], y[indices[i]])
    
    # Calculate normalization parameters
    if datasets['train'][0].size > 0:
        for lead in range(datasets['train'][0].shape[1]):
            scaler = StandardScaler()
            scaler.fit(datasets['train'][0][:, lead])
            scalers[lead] = scaler
    else:
        scalers = {lead: StandardScaler() for lead in range(2)}

    # Create tensor datasets
    final_data = {}
    for split in datasets:
        X, y = datasets[split]
        if X.size == 0:
            X_tensor = torch.zeros((0, 2, 300), dtype=torch.float32)
            y_tensor = torch.zeros(0, dtype=torch.long)
        else:
            X_normalized = X.copy()
            for lead in range(X_normalized.shape[1]):
                X_normalized[:, lead] = scalers[lead].transform(X[:, lead])
            X_tensor = torch.from_numpy(X_normalized.astype(np.float32))
            if args.sglead:
                X_tensor = X_tensor[:,0,:].unsqueeze(1)
            y_tensor = torch.from_numpy(y)
        
        final_data[split] = (X_tensor, y_tensor)
        print(f"\n{split.capitalize()} set: {final_data[split][0].shape}")
        print_label_stats(datasets[split][1], class_map_r)
    
    return final_data['train'] + final_data['val'] + final_data['test'] + (class_map_r,)

def load_MITBIH_dataset(args):
    """Load and preprocess MIT-BIH arrhythmia database
    
    Args:
        args: Configuration parameters
        
    Returns:
        tuple: (train_data, val_data, test_data, class_map)
    """
    data_folder = './data/MITBIH/files/'
    target_leads = {'MLII', 'V1'}
    
    # Class definitions
    ecg_classes = ['N', 'A', 'V', 'L', 'R']
    class_map = {cls:i for i, cls in enumerate(ecg_classes)}
    class_map_r = {i:cls for i, cls in enumerate(ecg_classes)}
    
    valid_records = ['232', '115', '208', '203', '214', '212', '213', 
                   '207', '210', '228', '202', '116', '119', '221', 
                   '231', '209', '101', '220', '233', '230', '106', 
                   '108', '111', '112', '200', '105', '113', '122', 
                   '109', '205', '219', '118', '201', '223', '215', 
                   '121', '107', '217', '222', '234']

    X, y = [], []

    # Process ECG records
    for num in valid_records:
        try:
            # Read and validate ECG leads
            header = wfdb.rdheader(os.path.join(data_folder, num))
            if not set(header.sig_name) >= target_leads:
                print(f"Skipping {num}: Invalid lead configuration")
                continue
                
            # Extract target leads
            ml_idx = header.sig_name.index('MLII')
            v1_idx = header.sig_name.index('V1')
            record = wfdb.rdrecord(os.path.join(data_folder, num), 
                                  channels=[ml_idx, v1_idx])
            signals = record.p_signal.T
            
            # Apply denoising
            signals = np.array([_wavelet_denoise(sig) for sig in signals])
            
            # Extract heartbeats
            annotation = wfdb.rdann(os.path.join(data_folder, num), 'atr')
            for pos, sym in zip(annotation.sample, annotation.symbol):
                if sym not in class_map or pos < 120 or pos+240 > signals.shape[1]:
                    continue
                X.append(signals[:, pos-120:pos+240])
                y.append(class_map[sym])
                
        except Exception as e:
            print(f"Error processing {num}: {str(e)}")
            continue

    # Prepare dataset splits
    X = np.array(X).astype(np.float32)
    y = np.array(y)
    indices = split_dataset(y, args.ratio)
    
    # Initialize data scalers
    scalers = {}
    datasets = {}
    for i, split in enumerate(['train', 'val', 'test']):
        datasets[split] = (X[indices[i]], y[indices[i]])
    
    # Create tensor datasets
    final_data = {}
    for split in datasets:
        X, y = datasets[split]
        if X.size == 0:
            X_tensor = torch.zeros((0, 2, X.shape[-1]), dtype=torch.float32)
            y_tensor = torch.zeros(0, dtype=torch.long)
        else:
            X_normalized = X.copy()
            for lead in range(X_normalized.shape[1]):
                X_normalized[:, lead] = StandardScaler().fit_transform(X[:, lead])
            X_tensor = torch.from_numpy(X_normalized.astype(np.float32))
            y_tensor = torch.from_numpy(y)
        
        final_data[split] = (X_tensor, y_tensor)
        print(f"\n{split.capitalize()} set: {final_data[split][0].shape}")
        print_label_stats(datasets[split][1], class_map_r)
    
    return final_data['train'] + final_data['val'] + final_data['test'] + (class_map_r,)
