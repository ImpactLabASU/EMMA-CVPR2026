
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
from pysindy.feature_library import PolynomialLibrary, FourierLibrary, GeneralizedLibrary
from pysindy.optimizers import STLSQ

# Re-add current directory for downstream relative imports or debugging helpers
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

G = 9.81  # gravitational constant used across refinements
DATASET_TYPES = ['45', '90', '150']
VIDEO_VERSIONS = ['v1', 'v2', 'v3', 'v4', 'v5']


def load_trajectory(csv_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Why: all pendulum preprocessing exports a rich CSV; we lean on it for clean signals
    What: returns theta (rad), omega (rad/s), and timestamp arrays (seconds)
    """
    df = pd.read_csv(csv_path)
    mask = np.isfinite(df['theta_rad']) & np.isfinite(df['omega_rad_s'])
    cleaned = df.loc[mask]
    if cleaned.empty:
        raise ValueError('No valid samples in trajectory CSV')
    return (
        cleaned['theta_rad'].to_numpy(copy=True),
        cleaned['omega_rad_s'].to_numpy(copy=True),
        cleaned['time_s'].to_numpy(copy=True),
    )


def initial_guess(theta: np.ndarray, omega: np.ndarray, dt: float) -> tuple[float, float]:
    """
    Why: PySINDy gives fast coarse structure; we use it to seed the constrained optimizer
    What: fits a sparse model constrained to {theta, omega, sin(theta)} features
    """
    # Build low-order libs so SINDy keeps the equation minimal
    poly = PolynomialLibrary(degree=1, include_bias=False, include_interaction=False)
    fourier = FourierLibrary(n_frequencies=1, include_cos=False)
    library = GeneralizedLibrary([poly, fourier])
    optimizer = STLSQ(threshold=1.0, normalize_columns=True)
    model = SINDy(feature_library=library, optimizer=optimizer)
    X = np.column_stack([theta, omega])
    model.fit(X, t=dt)
    coeffs = model.coefficients()
    features = model.get_feature_names()

    tau_guess = 0.1
    L_guess = 0.8
    # Parse the omega equation: index 1 corresponds to dω/dt
    for idx, name in enumerate(features[1]):
        if name == 'x1':
            tau_guess = min(max(abs(coeffs[1, idx]), 0.01), 1.5)
        if 'sin' in name and 'x0' in name and coeffs[1, idx] != 0.0:
            L_guess = np.clip(G / abs(coeffs[1, idx]), 0.2, 3.5)
    return tau_guess, L_guess


def refine_parameters(theta: np.ndarray, omega: np.ndarray, time: np.ndarray) -> tuple[float, float, float]:
    """
    Why: Use PySINDy to DISCOVER the pendulum equation from data
    What: Fits PySINDy model, discovers dω/dt = f(θ, ω), extracts τ and L from coefficients
    """
    n = theta.size
    if n < 21:
        raise ValueError('Trajectory too short for reliable differentiation')

    dt = float(np.median(np.diff(time)))
    window = min(max(11, (n // 5) * 2 + 1), 121)
    if window >= n:
        window = n - 1 if n % 2 == 0 else n
    if window % 2 == 0:
        window += 1

    # Smooth data
    theta_s = savgol_filter(theta, window_length=window, polyorder=3)
    omega_s = savgol_filter(omega, window_length=window, polyorder=3)

    # Mask for valid data
    mask = np.abs(theta_s) > 0.02  # discard near-linear region
    theta_fit = theta_s[mask]
    omega_fit = omega_s[mask]
    time_fit = time[mask]
    
    if theta_fit.size < 10:
        raise ValueError('Not enough informative samples after filtering')

    # Prepare state: X = [θ, ω]
    X = np.column_stack([theta_fit, omega_fit])  # [T, 2]

    # Use PySINDy's differentiation method
    from pysindy.differentiation import SmoothedFiniteDifference
    diff_method = SmoothedFiniteDifference(smoother_kws={'window_length': window, 'polyorder': 3})

    # Build feature library - polynomial + Fourier to capture sin(θ) term
    poly = PolynomialLibrary(degree=2, include_bias=False)
    fourier = FourierLibrary(n_frequencies=1, include_cos=False)
    library = GeneralizedLibrary([poly, fourier])

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
        model.fit(X, t=time_fit, feature_names=['theta', 'omega'])
    except Exception as e:
        # If fit fails, use initial guess
        tau0, L0 = initial_guess(theta_fit, omega_fit, dt)
        return tau0, L0, 0.0

    # Get discovered coefficients and feature names
    coefficients = model.coefficients()
    feature_names = model.get_feature_names()

    # PySINDy has discovered the equations - extract parameters from them
    # For multi-equation systems, get_feature_names() returns feature names for all equations
    # We need to get the feature names for the second equation (dω/dt)
    
    # Extract τ and L from discovered equation
    # Expected: dω/dt = -τ*ω - (g/L)*sin(θ)
    # Look for:
    # - ω term coefficient = -τ (in the dω/dt equation, row 1)
    # - sin(θ) term coefficient = -g/L
    tau_estimate = None
    L_estimate = None

    # Get feature names - for GeneralizedLibrary, this returns a list of feature names
    # We need to iterate through all features and check coefficients in row 1 (dω/dt equation)
    # The feature library generates features like: x0, x1, x0^2, x0*x1, x1^2, sin(1 x0), sin(1 x1)
    # Where x0 = theta, x1 = omega
    
    # Try to get feature names properly - might need to access library directly
    try:
        # Get all feature names from the library
        # For GeneralizedLibrary, we need to get features from each sub-library
        all_features = []
        if hasattr(library, 'libraries'):
            for lib in library.libraries:
                if hasattr(lib, 'get_feature_names'):
                    lib_features = lib.get_feature_names(input_features=['theta', 'omega'])
                    all_features.extend(lib_features)
        else:
            # Fallback: use model's feature names
            all_features = model.get_feature_names()
        
        # If all_features is still not right, parse from the printed equation
        # Or iterate through coefficients and match patterns
        omega_coeffs = coefficients[1, :]  # Second equation (dω/dt)
        
        # Search through coefficients - need to match with feature indices
        # For now, let's use a simpler approach: look at the printed equation
        # Or iterate through all possible feature patterns
        
        # Try accessing feature names through the library's transform
        X_sample = X[:1, :]  # Sample to get feature names
        try:
            theta_features = library.transform(X_sample)
            # Get feature names from library
            if hasattr(library, 'get_feature_names'):
                feature_list = library.get_feature_names(input_features=['theta', 'omega'])
            else:
                # Manual feature list based on PolynomialLibrary(degree=2) + FourierLibrary
                feature_list = library.get_feature_names(input_features=['theta', 'omega'])
        except:
            # Manual feature list based on PolynomialLibrary(degree=2) + FourierLibrary(n_frequencies=1)
            # Order: poly terms (theta, omega, theta^2, theta*omega, omega^2) + Fourier terms (sin(1 theta), sin(1 omega))
            feature_list = ['theta', 'omega', 'theta^2', 'theta omega', 'omega^2', 'sin(1 theta)', 'sin(1 omega)']
        
        # Now search through features for the dω/dt equation
        for i, feat_name in enumerate(feature_list):
            if i >= len(omega_coeffs):
                break
            coeff = omega_coeffs[i]
            name_str = str(feat_name).strip().lower()
            
            # Look for ω term - coefficient is -τ
            if name_str == 'omega' or name_str == 'x1' or (name_str.startswith('omega') and '^' not in name_str and 'sin' not in name_str):
                if abs(coeff) > 1e-6:
                    tau_estimate = abs(coeff)
                    print(f"  ✅ Found ω term: '{feat_name}' = {coeff:.6f}, extracted τ = {tau_estimate:.6f}")
            
            # Look for sin(θ) term - coefficient is -g/L
            if 'sin' in name_str and ('theta' in name_str or 'x0' in name_str):
                if abs(coeff) > 1e-6:
                    # coeff = -g/L, so L = g / abs(coeff)
                    L_estimate = G / abs(coeff)
                    L_estimate = np.clip(L_estimate, 0.2, 3.5)
                    print(f"  ✅ Found sin(θ) term: '{feat_name}' = {coeff:.6f}, extracted L = {L_estimate:.6f}")
                    
    except Exception as e:
        print(f"  ⚠️  Error extracting features: {e}")

    # If terms not found, use initial guess as fallback
    if tau_estimate is None or L_estimate is None:
        tau0, L0 = initial_guess(theta_fit, omega_fit, dt)
        if tau_estimate is None:
            tau_estimate = tau0
            print(f"  ⚠️  PySINDy did not find ω term, using initial guess: τ = {tau_estimate:.6f}")
        if L_estimate is None:
            L_estimate = L0
            print(f"  ⚠️  PySINDy did not find sin(θ) term, using initial guess: L = {L_estimate:.6f}")

    # Clip to reasonable ranges
    tau_estimate = np.clip(tau_estimate, 0.0, 1.5)
    L_estimate = np.clip(L_estimate, 0.2, 3.5)

    # Calculate R² score using discovered model
    score = model.score(X, t=time_fit)

    return tau_estimate, L_estimate, score


def process_case(case_root: str) -> tuple[float, float, float]:
    """
    Why: helper invoked per folder
    What: reads CSV, estimates tau & L, returns tuple + score
    """
    csv_path = os.path.join(case_root, 'data', 'pendulum_trajectory.csv')
    theta, omega, time = load_trajectory(csv_path)
    return refine_parameters(theta, omega, time)


def process_all_videos():
    """
    Why: orchestrator iterating through every dataset configuration
    What: writes per-run CSVs plus a consolidated summary under pysindy_results/
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_root = os.path.join(base_dir, 'pysindy_results')
    os.makedirs(output_root, exist_ok=True)

    summary_rows = []

    for dataset in DATASET_TYPES:
        for version in VIDEO_VERSIONS:
            run_id = f'{dataset}_{version}'
            case_dir = os.path.join(base_dir, run_id)
            target_csv = os.path.join(case_dir, 'data', 'pendulum_trajectory.csv')
            if not os.path.exists(target_csv):
                print(f"⚠️  Skipping {run_id}: missing pendulum_trajectory.csv")
                continue

            print('\n' + '=' * 60)
            print(f'Processing: {run_id}')
            print('=' * 60)
            try:
                tau, length, score = process_case(case_dir)
                case_out = os.path.join(output_root, run_id)
                os.makedirs(case_out, exist_ok=True)
                coeff_path = os.path.join(case_out, 'pendulum_coefficients.csv')
                with open(coeff_path, 'w', newline='') as handle:
                    writer = csv.writer(handle)
                    writer.writerow(['Parameter', 'Value', 'Units', 'Description'])
                    writer.writerow(['L', float(length), 'm', 'Pendulum length (PySINDy-refined)'])
                    writer.writerow(['tau', float(tau), '1/s', 'Damping coefficient (PySINDy-refined)'])
                    writer.writerow(['score', float(score), '1', 'R^2 on ω̇ fit'])
                print(f'✅ L = {length:.4f} m, τ = {tau:.4f} 1/s, R² = {score:.4f}')
                print(f'✅ Saved coefficients: {coeff_path}')
                summary_rows.append({'dataset': run_id, 'L': float(length), 'tau': float(tau), 'score': float(score)})
            except Exception as exc:
                print(f'❌ Error processing {run_id}: {exc}')

    if summary_rows:
        summary_path = os.path.join(output_root, 'pysindy_results.csv')
        with open(summary_path, 'w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=['dataset', 'L', 'tau', 'score'])
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f'\n✅ Summary saved: {summary_path}')


if __name__ == '__main__':
    process_all_videos()
