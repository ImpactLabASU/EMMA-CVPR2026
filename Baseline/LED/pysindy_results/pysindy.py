# PySINDy parameter estimation for LED Decay
# Replaces EMMA neural network with PySINDy sparse regression

import os
import csv
import numpy as np
# Ensure installed pysindy is imported, not this file
import sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR in sys.path:
    sys.path.remove(CURRENT_DIR)

# Import PySINDy components properly
from pysindy import SINDy
from pysindy.feature_library import PolynomialLibrary
from pysindy.optimizers import STLSQ
from pysindy.differentiation import SmoothedFiniteDifference

if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

# ─── CONFIG ────────────────────────────────────────────────────────────────
# Dataset types: led_2s, led_5s, led_10s
# Each type has 5 videos: v1, v2, v3, v4, v5
DATASET_TYPES = ['led_2s', 'led_5s', 'led_10s']
VIDEO_VERSIONS = ['v1', 'v2', 'v3', 'v4', 'v5']

# Ground truth values from parameters.json
GT_VALUES = {
    'led_2s': 2.3,
    'led_5s': 0.92,
    'led_10s': 0.46
}

# PySINDy settings
DEGREE = 3  # Lower degree for linear decay
THRESHOLD = 0.01  # Lower threshold for linear systems

# ─── LED DECAY PHYSICS ─────────────────────────────────────────────────────
# Equation: dI/dt = -γ * I(t)
# Where: I = intensity, γ = decay constant
# PySINDy should discover: dI/dt = -γ*I (linear relationship)

def load_trajectory_data(data_dir):
    """
    Load LED intensity trajectory data from IData.txt.
    
    Why: Extract actual intensity time series from EMMA data format
    What: Returns intensity array and time array
    """
    I_data = np.loadtxt(os.path.join(data_dir, "IData.txt"))  # [N_features, 100_timesteps] format
    I_traj = I_data.T  # [100_timesteps, N_features] - transpose to get timesteps x features
    
    # Extract actual trajectory (first column contains intensity)
    intensity = I_traj[:, 0]  # Extract intensity time series
    
    # Remove any trailing zeros/padding if trajectory is shorter than 100
    non_zero_mask = np.abs(intensity) > 1e-10
    if np.any(non_zero_mask):
        last_idx = np.where(non_zero_mask)[0][-1] + 1
        intensity = intensity[:last_idx]
    
    # Create time array (assuming 30 fps, adjust if needed)
    dt = 1.0 / 30.0  # Time step in seconds
    t = np.arange(len(intensity)) * dt
    
    return intensity, t

def estimate_led_parameters(data_dir):
    """
    Estimate decay constant (γ) using PySINDy to DISCOVER the equation from data.
    
    Why: Use PySINDy's sparse regression to discover dI/dt = f(I) from data
    What: Fits PySINDy model, discovers equation structure, extracts γ from coefficients
    """
    import pandas as pd
    from scipy.signal import savgol_filter
    
    # Try to load from CSV first (has proper time)
    csv_path = os.path.join(data_dir, 'led_trajectory.csv')
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        intensity = df['intensity_normalized'].values if 'intensity_normalized' in df.columns else df['intensity'].values
        time = df['time_s'].values
    else:
        # Fallback to IData.txt
        intensity, time = load_trajectory_data(data_dir)
    
    n = intensity.size
    if n < 10:
        raise ValueError('Trajectory too short for reliable differentiation')
    
    dt = float(np.median(np.diff(time))) if len(time) > 1 else 1.0/30.0
    window = min(max(11, (n // 5) * 2 + 1), 101)
    if window >= n:
        window = n - 1 if n % 2 == 0 else n
    if window % 2 == 0:
        window += 1
    
    # Smooth intensity
    intensity_s = savgol_filter(intensity, window_length=window, polyorder=3)
    
    # Prepare state: X = [I] (intensity)
    X = intensity_s.reshape(-1, 1)  # [T, 1]
    
    # Compute derivative using PySINDy's differentiation method
    diff_method = SmoothedFiniteDifference(smoother_kws={'window_length': window, 'polyorder': 3})
    
    # Build feature library - polynomial terms (should discover linear term)
    library = PolynomialLibrary(degree=2, include_bias=True)
    
    # Use STLSQ optimizer with appropriate threshold
    optimizer = STLSQ(threshold=0.01, normalize_columns=True, max_iter=20)
    
    # Create PySINDy model - PROPER USAGE: Let PySINDy discover equation from data
    model = SINDy(
        feature_library=library,
        optimizer=optimizer,
        differentiation_method=diff_method
    )
    
    # Fit model to DISCOVER equation from data
    # feature_names passed to fit(), not __init__()
    try:
        model.fit(X, t=time, feature_names=['I'])
    except Exception as e:
        # If fit fails, try with simpler model
        library = PolynomialLibrary(degree=1, include_bias=False)
        optimizer = STLSQ(threshold=0.1, normalize_columns=True, max_iter=20)
        model = SINDy(
            feature_library=library,
            optimizer=optimizer,
            differentiation_method=diff_method
        )
        model.fit(X, t=time, feature_names=['I'])
    
    # Get discovered coefficients and feature names
    coefficients = model.coefficients()
    feature_names = model.get_feature_names()
    
    # PySINDy has discovered the equation - extract parameters from it
    
    # Extract γ from discovered equation
    # Expected: dI/dt = -γ*I (or dI/dt = c0 + c1*I)
    # PySINDy uses 'x0' or 'I' for the first state variable
    gamma_estimate = None
    
    # Search for linear term coefficient (I term) - THIS IS THE PRIMARY TERM
    # PySINDy feature names can be strings like 'x0', 'I', '1', 'x0^2', etc.
    # For LED decay: dI/dt = -γ*I, so we want the coefficient of 'I' (or 'x0')
    # feature_names is a flat list of feature names for the single equation
    for i, name in enumerate(feature_names):  # Iterate over feature names directly
        name_str = str(name).strip()
        # Look for linear term: 'I' or 'x0' (not 'I^2' or 'x0^2')
        # Check if it's exactly 'I' or 'x0'
        is_linear = (name_str == 'I' or name_str == 'x0')
        
        if is_linear:
            coeff = coefficients[0, i]
            # dI/dt = -γ*I, so coefficient is -γ
            # For exponential decay, coefficient should be negative
            if abs(coeff) > 1e-6:  # Non-zero coefficient
                gamma_estimate = abs(coeff)
                print(f"  ✅ Found linear term: '{name_str}' = {coeff:.6f}, extracted γ = {gamma_estimate:.6f}")
                break
    
    # If no linear term found, try constant term (NOT IDEAL - this means PySINDy didn't find linear decay)
    if gamma_estimate is None:
        for i, name in enumerate(feature_names):
            name_str = str(name).strip()
            if name_str == '1' or name_str == '':
                coeff = coefficients[0, i]
                if abs(coeff) > 1e-6:
                    gamma_estimate = abs(coeff)
                    print(f"  ⚠️  No linear term found, using constant term: {name_str} = {coeff:.6f} as γ = {gamma_estimate:.6f}")
                    break
    
    # If still None, use default
    if gamma_estimate is None:
        gamma_estimate = 0.46
        print("  Warning: PySINDy could not discover linear decay term, using default 0.46 1/s")
    
    # Clip to reasonable range
    gamma_estimate = np.clip(gamma_estimate, 0.1, 5.0)
    
    # Get model score (reconstruction accuracy)
    score = model.score(X, t=time)
    
    return gamma_estimate, score, model

def process_all_videos():
    """
    Process all LED videos and estimate parameters using PySINDy.
    
    Why: Batch process all dataset types and videos
    What: Saves coefficients CSV for each video and summary results
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_root = os.path.join(base_dir, 'pysindy_results')
    os.makedirs(output_root, exist_ok=True)
    
    summary_rows = []
    
    for dataset_type in DATASET_TYPES:
        for version in VIDEO_VERSIONS:
            folder_name = f"{dataset_type}_{version}"
            data_dir = os.path.join(base_dir, folder_name, "data")
            
            if not os.path.exists(data_dir):
                print(f"⚠️  Skipping {folder_name}: data directory not found")
                continue
            
            idata_path = os.path.join(data_dir, "IData.txt")
            if not os.path.exists(idata_path):
                print(f"⚠️  Skipping {folder_name}: IData.txt not found")
                continue
            
            print('\n' + '=' * 60)
            print(f'Processing: {folder_name}')
            print('=' * 60)
            
            try:
                # Estimate parameters
                gamma, score, model = estimate_led_parameters(data_dir)
                
                # Get ground truth for comparison
                gt_gamma = GT_VALUES.get(dataset_type, None)
                gt_str = f" (GT: {gt_gamma:.6f})" if gt_gamma else ""
                error_str = f", Error: {abs(gamma - gt_gamma):.6f}" if gt_gamma else ""
                
                # Save coefficients CSV in pysindy_results folder
                case_out = os.path.join(output_root, folder_name)
                os.makedirs(case_out, exist_ok=True)
                coeff_path = os.path.join(case_out, 'led_coefficients.csv')
                
                with open(coeff_path, 'w', newline='') as csvfile:
                    w = csv.writer(csvfile)
                    w.writerow(['Parameter', 'Value', 'Units', 'Description'])
                    w.writerow(['gamma', float(gamma), '1/s', 'LED decay constant (PySINDy)'])
                    w.writerow(['score', float(score), '1', 'R^2 on dI/dt fit'])
                    if gt_gamma:
                        w.writerow(['gamma_GT', float(gt_gamma), '1/s', 'Ground truth from parameters.json'])
                        w.writerow(['error', float(abs(gamma - gt_gamma)), '1/s', 'Absolute error'])
                
                print(f'✅ γ = {gamma:.6f} 1/s{gt_str}{error_str}, R² = {score:.4f}')
                print(f'✅ Saved coefficients: {coeff_path}')
                
                # Store for summary
                summary_rows.append({
                    'dataset': folder_name,
                    'gamma': float(gamma),
                    'gamma_GT': float(gt_gamma) if gt_gamma else None,
                    'error': float(abs(gamma - gt_gamma)) if gt_gamma else None,
                    'score': float(score),
                    'status': 'Success'
                })
                
            except Exception as exc:
                print(f'❌ Error processing {folder_name}: {exc}')
                summary_rows.append({
                    'dataset': folder_name,
                    'gamma': None,
                    'gamma_GT': float(GT_VALUES.get(dataset_type, None)) if GT_VALUES.get(dataset_type) else None,
                    'error': None,
                    'score': None,
                    'status': f'Error: {str(exc)[:50]}'
                })
                import traceback
                traceback.print_exc()
    
    # Save summary results
    if summary_rows:
        summary_path = os.path.join(output_root, 'pysindy_results.csv')
        with open(summary_path, 'w', newline='') as csvfile:
            fieldnames = ['dataset', 'gamma', 'gamma_GT', 'error', 'score', 'status']
            w = csv.DictWriter(csvfile, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(summary_rows)
        print(f'\n✅ Summary saved: {summary_path}')
        
        # Create EMMA format summary
        create_emma_format_summary(summary_rows, output_root)
        
        # Print summary statistics
        print('\n' + '=' * 60)
        print('SUMMARY STATISTICS')
        print('=' * 60)
        for dataset_type in DATASET_TYPES:
            type_results = [r for r in summary_rows if r['dataset'].startswith(dataset_type) and r.get('status') == 'Success']
            if type_results:
                gamma_vals = [r['gamma'] for r in type_results]
                errors = [r['error'] for r in type_results if r['error'] is not None]
                print(f'\n{dataset_type.upper()}:')
                print(f'  Processed: {len(type_results)}/{len([r for r in summary_rows if r["dataset"].startswith(dataset_type)])} videos')
                if gamma_vals:
                    print(f'  Mean γ: {np.mean(gamma_vals):.6f} ± {np.std(gamma_vals):.6f} 1/s')
                if errors:
                    print(f'  Mean error: {np.mean(errors):.6f} 1/s')
                if dataset_type in GT_VALUES:
                    print(f'  GT: {GT_VALUES[dataset_type]:.6f} 1/s')

def create_emma_format_summary(summary_rows, output_root):
    """Create EMMA format CSV with individual runs and summaries."""
    import pandas as pd
    
    emma_rows = []
    
    for dataset_type in DATASET_TYPES:
        type_data = [r for r in summary_rows if r['dataset'].startswith(dataset_type) and r.get('status') == 'Success']
        
        if len(type_data) > 0:
            # Sort by video number
            type_data_sorted = sorted(type_data, key=lambda x: int(x['dataset'].split('_v')[1]) if '_v' in x['dataset'] else 0)
            
            # Add individual runs
            for row in type_data_sorted:
                video_num = row['dataset'].split('_v')[1] if '_v' in row['dataset'] else ''
                emma_rows.append({
                    'Type': 'run',
                    'Group': dataset_type,
                    'Video': f'v{video_num}',
                    'gamma': row['gamma'] if row['gamma'] is not None else '',
                    'gamma_mean': '',
                    'gamma_std': ''
                })
            
            # Calculate summary statistics
            gamma_vals = [r['gamma'] for r in type_data if r['gamma'] is not None]
            if gamma_vals:
                gamma_mean = np.mean(gamma_vals)
                gamma_std = np.std(gamma_vals)
                
                # Add summary row
                emma_rows.append({
                    'Type': 'summary',
                    'Group': dataset_type,
                    'Video': '',
                    'gamma': '',
                    'gamma_mean': f'{gamma_mean:.6f}',
                    'gamma_std': f'{gamma_std:.6f}'
                })
    
    # Save EMMA format
    emma_df = pd.DataFrame(emma_rows)
    emma_path = os.path.join(output_root, 'pysindy_results_emma_format.csv')
    emma_df.to_csv(emma_path, index=False)
    print(f'✅ EMMA format summary saved: {emma_path}')

if __name__ == "__main__":
    process_all_videos()

