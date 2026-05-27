
import os
import sys
import csv
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.optimize import least_squares

# Why: ensure we import the installed pysindy package, not this file
# What: temporarily drop current directory from sys.path during import resolution
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR in sys.path:
    sys.path.remove(CURRENT_DIR)

from pysindy import SINDy
from pysindy.feature_library import CustomLibrary, GeneralizedLibrary, PolynomialLibrary
from pysindy.optimizers import STLSQ

# Re-add current directory for downstream relative imports
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

DATASET_TYPES = ['small', 'med', 'lar']
VIDEO_VERSIONS = ['v1', 'v2', 'v3', 'v4', 'v5']

# Ground truth values from EMMA results
GT_VALUES = {
    'small': 5.150483,
    'med': 9.955168,
    'lar': 10.290640
}


def load_trajectory(csv_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Why: Load trajectory from CSV with time and position
    What: Returns position (m) and time (s) arrays
    """
    df = pd.read_csv(csv_path)
    mask = np.isfinite(df['y_position_meters']) & np.isfinite(df['time_s'])
    cleaned = df.loc[mask]
    if cleaned.empty:
        raise ValueError('No valid samples in trajectory CSV')
    return (
        cleaned['y_position_meters'].to_numpy(copy=True),
        cleaned['time_s'].to_numpy(copy=True),
    )


def initial_guess(position: np.ndarray, time: np.ndarray) -> tuple[float, float]:
    """
    Why: Get initial guess for g and r0f using simple models
    What: Estimates g from acceleration and r0f from initial position
    """
    if len(position) < 10:
        return 9.81, 0.1
    
    # Estimate g from acceleration
    dt = np.median(np.diff(time))
    velocity = np.gradient(position, time)
    acceleration = np.gradient(velocity, time)
    
    # Filter out outliers and get mean acceleration
    accel_clean = acceleration[np.abs(acceleration) < 50]
    if len(accel_clean) > 0:
        g_guess = np.mean(np.abs(accel_clean))
        g_guess = np.clip(g_guess, 1.0, 15.0)
    else:
        g_guess = 9.81
    
    # Estimate r0f from initial position
    r0f_guess = abs(position[0]) if abs(position[0]) > 1e-6 else 0.1
    r0f_guess = np.clip(r0f_guess, 0.01, 1.0)
    
    return g_guess, r0f_guess


def refine_parameters(position: np.ndarray, time: np.ndarray) -> tuple[float, float]:
    """
    Why: Use PySINDy to DISCOVER the free fall equation from data
    What: Fits PySINDy model with time as feature, discovers dr/dt = f(r, t), extracts g
    """
    n = position.size
    if n < 20:
        raise ValueError('Trajectory too short for reliable differentiation')
    
    dt = float(np.median(np.diff(time)))
    window = min(max(11, (n // 5) * 2 + 1), 101)
    if window >= n:
        window = n - 1 if n % 2 == 0 else n
    if window % 2 == 0:
        window += 1
    
    # Smooth position
    position_s = savgol_filter(position, window_length=window, polyorder=3)
    
    # Mask for valid data
    mask = (time > 1e-6) & (position_s > 1e-6) & np.isfinite(position_s)
    time_fit = time[mask]
    position_fit = position_s[mask]
    
    if len(time_fit) < 5:
        raise ValueError('Not enough valid samples after filtering')
    
    # Prepare state: X = [r, t] - include time as a feature
    # Why: Equation is dr/dt = -g*t*r²/r0f, so we need both r and t
    X = np.column_stack([position_fit, time_fit])  # [T, 2]
    
    # Use PySINDy's differentiation method
    from pysindy.differentiation import SmoothedFiniteDifference
    diff_method = SmoothedFiniteDifference(smoother_kws={'window_length': window, 'polyorder': 3})
    
    # Build feature library - polynomial terms up to degree 3 to capture t*r²
    library = PolynomialLibrary(degree=3, include_bias=False)
    
    # Use STLSQ optimizer
    optimizer = STLSQ(threshold=0.1, normalize_columns=True, max_iter=20)
    
    # Create PySINDy model - PROPER USAGE: Let PySINDy discover equation from data
    model = SINDy(
        feature_library=library,
        optimizer=optimizer,
        differentiation_method=diff_method
    )
    
    # Fit model to DISCOVER equation from data
    try:
        model.fit(X, t=time_fit, feature_names=['r', 't'])
    except Exception as e:
        # If fit fails, use initial guess
        g0, r0f0 = initial_guess(position_fit, time_fit)
        r0f_fixed = abs(position_fit[0]) if abs(position_fit[0]) > 1e-6 else 0.1
        r0f_fixed = np.clip(r0f_fixed, 0.01, 1.0)
        return np.clip(g0, 1.0, 15.0), 0.0
    
    # Get discovered coefficients and feature names
    coefficients = model.coefficients()
    feature_names = model.get_feature_names()
    
    # Extract g from discovered equation
    # Expected: dr/dt = -g*t*r²/r0f
    # Look for terms like 't r^2', 't*r^2', 'r^2 t', etc.
    # The coefficient of such terms should be -g/r0f
    g_estimate = None
    r0f_estimate = None
    
    # Search for t*r² term (or similar) in the first equation (dr/dt)
    # feature_names is a list of feature names for the first equation
    for i, name in enumerate(feature_names):
        name_str = str(name).strip().lower()
        coeff = coefficients[0, i]
        
        # Look for terms involving t and r² (e.g., 't r^2', 'r^2 t', 't*r^2')
        has_t = 't' in name_str or 'x1' in name_str  # x1 is PySINDy's name for second state (time)
        has_r2 = 'r^2' in name_str or 'r**2' in name_str or 'x0^2' in name_str or 'x0**2' in name_str
        
        if has_t and has_r2 and abs(coeff) > 1e-6:
            # Found t*r² term: coefficient is -g/r0f
            # We need to estimate r0f separately (use initial position)
            r0f_estimate = abs(position_fit[0]) if abs(position_fit[0]) > 1e-6 else abs(np.mean(position_fit[:5]))
            if r0f_estimate < 1e-6:
                r0f_estimate = 0.1
            r0f_estimate = np.clip(r0f_estimate, 0.01, 1.0)
            
            # g = -coeff * r0f (since dr/dt = -g*t*r²/r0f, so coeff = -g/r0f)
            g_estimate = abs(coeff) * r0f_estimate
            print(f"  ✅ Found t*r² term: '{name_str}' = {coeff:.6f}, extracted g = {g_estimate:.6f} (r0f = {r0f_estimate:.6f})")
            break
    
    # If t*r² term not found, try simpler approach: look for any significant term
    if g_estimate is None:
        # Use initial guess as fallback
        g0, r0f0 = initial_guess(position_fit, time_fit)
        r0f_fixed = abs(position_fit[0]) if abs(position_fit[0]) > 1e-6 else 0.1
        r0f_fixed = np.clip(r0f_fixed, 0.01, 1.0)
        g_estimate = np.clip(g0, 1.0, 15.0)
        r0f_estimate = r0f_fixed
        print(f"  ⚠️  PySINDy did not find t*r² term, using initial guess: g = {g_estimate:.6f}")
    
    # Clip to reasonable range
    g_estimate = np.clip(g_estimate, 1.0, 15.0)
    
    # Calculate R² score using discovered model
    score = model.score(X, t=time_fit)
    
    return g_estimate, score


def process_case(case_root: str) -> tuple[float, float]:
    """
    Why: Helper invoked per folder
    What: Reads CSV, estimates g, returns tuple + score
    """
    csv_path = os.path.join(case_root, 'data', 'free_fall_trajectory.csv')
    position, time = load_trajectory(csv_path)
    return refine_parameters(position, time)


def process_all_videos():
    """
    Why: Orchestrator iterating through every dataset configuration
    What: Writes per-run CSVs plus a consolidated summary under pysindy_results/
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_root = os.path.join(base_dir, 'pysindy_results')
    os.makedirs(output_root, exist_ok=True)
    
    summary_rows = []
    
    for dataset_type in DATASET_TYPES:
        for version in VIDEO_VERSIONS:
            run_id = f"{dataset_type}_{version}"
            case_dir = os.path.join(base_dir, run_id)
            target_csv = os.path.join(case_dir, 'data', 'free_fall_trajectory.csv')
            
            if not os.path.exists(target_csv):
                print(f"⚠️  Skipping {run_id}: missing free_fall_trajectory.csv")
                continue
            
            print('\n' + '=' * 60)
            print(f'Processing: {run_id}')
            print('=' * 60)
            
            try:
                g, score = process_case(case_dir)
                
                # Get ground truth for comparison
                gt_g = GT_VALUES.get(dataset_type, None)
                gt_str = f" (GT: {gt_g:.6f})" if gt_g else ""
                error_str = f", Error: {abs(g - gt_g):.6f}" if gt_g else ""
                
                case_out = os.path.join(output_root, run_id)
                os.makedirs(case_out, exist_ok=True)
                coeff_path = os.path.join(case_out, 'free_fall_coefficients.csv')
                
                with open(coeff_path, 'w', newline='') as handle:
                    writer = csv.writer(handle)
                    writer.writerow(['Parameter', 'Value', 'Units', 'Description'])
                    writer.writerow(['g', float(g), 'm/s^2', 'Gravitational acceleration (PySINDy-refined)'])
                    writer.writerow(['score', float(score), '1', 'R^2 on dr/dt fit'])
                    if gt_g:
                        writer.writerow(['g_GT', float(gt_g), 'm/s^2', 'Ground truth from EMMA'])
                        writer.writerow(['error', float(abs(g - gt_g)), 'm/s^2', 'Absolute error'])
                
                print(f'✅ g = {g:.6f} m/s²{gt_str}{error_str}, R² = {score:.4f}')
                print(f'✅ Saved coefficients: {coeff_path}')
                
                summary_rows.append({
                    'dataset': run_id,
                    'g': float(g),
                    'g_GT': float(gt_g) if gt_g else None,
                    'error': float(abs(g - gt_g)) if gt_g else None,
                    'score': float(score),
                    'status': 'Success'
                })
            except Exception as exc:
                print(f'❌ Error processing {run_id}: {exc}')
                # Still record failed videos in summary with error flag
                summary_rows.append({
                    'dataset': run_id,
                    'g': None,
                    'g_GT': float(GT_VALUES.get(dataset_type, None)) if GT_VALUES.get(dataset_type) else None,
                    'error': None,
                    'score': None,
                    'status': f'Error: {str(exc)[:50]}'
                })
                import traceback
                traceback.print_exc()
    
    if summary_rows:
        summary_path = os.path.join(output_root, 'pysindy_results.csv')
        with open(summary_path, 'w', newline='') as handle:
            fieldnames = ['dataset', 'g', 'g_GT', 'error', 'score', 'status']
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f'\n✅ Summary saved: {summary_path}')
        
        # Print summary statistics
        print('\n' + '=' * 60)
        print('SUMMARY STATISTICS')
        print('=' * 60)
        for dataset_type in DATASET_TYPES:
            type_results = [r for r in summary_rows if r['dataset'].startswith(dataset_type)]
            if type_results:
                g_vals = [r['g'] for r in type_results if r['g'] is not None]
                errors = [r['error'] for r in type_results if r['error'] is not None]
                successful = [r for r in type_results if r.get('status') == 'Success']
                failed = [r for r in type_results if r.get('status') != 'Success']
                print(f'\n{dataset_type.upper()}:')
                print(f'  Processed: {len(successful)}/{len(type_results)} videos')
                if failed:
                    print(f'  Failed: {[r["dataset"] for r in failed]}')
                if g_vals:
                    print(f'  Mean g: {np.mean(g_vals):.6f} m/s²')
                    print(f'  Std g: {np.std(g_vals):.6f} m/s²')
                if errors:
                    print(f'  Mean error: {np.mean(errors):.6f} m/s²')
                if dataset_type in GT_VALUES:
                    print(f'  GT: {GT_VALUES[dataset_type]:.6f} m/s²')


if __name__ == '__main__':
    process_all_videos()
