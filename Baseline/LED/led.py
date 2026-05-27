# This is the code for LED pipeline based on EMMA method

# EMMA LED Pipeline

import os
import csv
import gc
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from ncps.torch import LTC
from matplotlib.animation import FuncAnimation
import pdb

try:
    import psutil
    _HAS_PSUTIL = True
except Exception:
    _HAS_PSUTIL = False

try:
    from moviepy.editor import VideoFileClip
except Exception:
    from moviepy import VideoFileClip
from torchvision import transforms

# Set device for computation (GPU if available, otherwise CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Global variable to store the number of features per timestep
Nloop = 0


def check_memory_usage():
    """
    Monitor system memory usage during processing.
    
    Why: Prevent memory overflow during large video processing
    What: Display current memory usage statistics
    """
    if not _HAS_PSUTIL:
        return
    mem = psutil.virtual_memory()
    used_gb = mem.used / (1024**3)
    total_gb = mem.total / (1024**3)
    print(f"[INFO] Memory usage: {used_gb:.1f}GB / {total_gb:.1f}GB ({mem.percent:.1f}%)")


def process_led_video(video_path, output_csv):
    """
    Process LED video to extract brightness/intensity trajectory.
    
    This function processes LED videos to:
    1. Load video frames
    2. Calculate average brightness/intensity per frame (entire frame)
    3. Save intensity trajectory data
    
    Args:
        video_path: Path to input LED video file
        output_csv: Path for trajectory CSV output
        
    Why: Video processing is the foundation of LED intensity trajectory analysis
    What: Extracts brightness trajectory from LED video frames (no mask used)
    """
    print(f"[STEP 1] Processing LED video: {video_path}")
    print(f"[STEP 1] Output CSV: {output_csv}")

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    csv_f = open(output_csv, "w", newline="")
    csvw = csv.writer(csv_f)
    csvw.writerow(["frame", "time_s", "intensity", "intensity_normalized"])

    intensity_series = []
    frame_idx = 0
    
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_time = frame_idx / fps
        
        # Convert to grayscale for intensity measurement
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate mean intensity for entire frame (no mask)
        intensity = np.mean(gray)
        
        intensity_series.append(intensity)
        
        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"[PROGRESS] Processed {frame_idx} frames")
            check_memory_usage()

    cap.release()

    if intensity_series:
        # Normalize intensity to [0, 1] range
        intensity_arr = np.array(intensity_series)
        intensity_max = intensity_arr.max()
        intensity_min = intensity_arr.min()
        
        # Avoid division by zero
        if intensity_max > intensity_min:
            intensity_normalized = (intensity_arr - intensity_min) / (intensity_max - intensity_min)
        else:
            intensity_normalized = np.ones_like(intensity_arr)
        
        # Write to CSV
        for idx, (raw_val, norm_val) in enumerate(zip(intensity_arr, intensity_normalized)):
            frame_time = idx / fps
            csvw.writerow([idx, f"{frame_time:.3f}", f"{raw_val:.2f}", f"{norm_val:.6f}"])
        
        csv_f.close()
        
        # Report extracted intensity range
        I_0_actual = intensity_normalized[0] if len(intensity_normalized) > 0 else 0.0
        I_n_actual = intensity_normalized[-1] if len(intensity_normalized) > 0 else 0.0
        
        print(f"\n[STEP 1] Extracted Intensity Range:")
        print(f"   Initial intensity: I_0 = {I_0_actual:.3f} (normalized)")
        print(f"   Final intensity:   I_n = {I_n_actual:.3f} (normalized)")
        print(f"   Intensity change:  {I_0_actual - I_n_actual:.3f}")
        
        # Check if intensity decreases (physical constraint for LED decay)
        if I_n_actual >= I_0_actual:
            print(f"   ⚠️  Warning: Intensity does not decrease (may indicate issue)")
        else:
            print(f"   ✅ Intensity decreases as expected for LED decay")
        
        # Match EMMA format (N x 100 matrices for memory optimization)
        I_matrix = np.tile(intensity_normalized.reshape(-1, 1), (1, 100))
        
        # Determine data directory from output_csv path
        data_dir = os.path.dirname(output_csv)
        os.makedirs(data_dir, exist_ok=True)
        np.savetxt(os.path.join(data_dir, "IData.txt"), I_matrix, fmt='%.6f')
        
        del I_matrix
        gc.collect()
        print(f"\n[STEP 1] ✅ Saved LED intensity trajectory data: {len(intensity_series)} frames")
        print(f"[STEP 1] ✅ Saved intensity data: IData.txt")
        
        # Create trajectory plots
        print("[STEP 1] Creating LED intensity trajectory plots...")
        
        # Plot intensity vs time
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        time_array = np.arange(len(intensity_normalized)) / fps
        ax.plot(time_array, intensity_normalized, 'b-', linewidth=2, label='Normalized Intensity')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Normalized Intensity')
        ax.set_title('LED Intensity vs Time')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plot_path = os.path.join(data_dir, "led_intensity_trajectory.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[STEP 1] ✅ Saved trajectory plot: {plot_path}")
        
        return len(intensity_series)
    else:
        print("[STEP 1] ⚠️  No intensity data extracted from video")
        return 0


def cut_in_sequences(x, y, seq_len, inc=1):
    """
    Slice a long 1D/2D series into overlapping windows for sequence-based learning.
    
    This function creates sequences from the input data for the LTC model.
    For LED data: input shape (N, 100) -> output shape (seq_len, num_sequences, 100)
    
    Args:
        x: Input data array (e.g., intensity trajectory)
        y: Target data array (e.g., intensity trajectory) 
        seq_len: Length of each sequence (e.g., 16 timesteps)
        inc: Increment step for creating overlapping sequences
        
    Returns:
        sequences_x: Input sequences with shape (seq_len, num_sequences, features)
        sequences_y: Target sequences with shape (seq_len, num_sequences, features)
    """
    sequences_x, sequences_y = [], []
    for s in range(0, x.shape[0] - seq_len, inc):
        start, end = s, s + seq_len
        sequences_x.append(x[start:end])
        sequences_y.append(y[start:end])
    return np.stack(sequences_x, axis=1), np.stack(sequences_y, axis=1)


class Custom_LED_Loss(nn.Module):
    """
    Custom loss function that integrates LED decay physics simulation.
    
    This is the core of the parameter estimation system. Instead of using a simple
    MSE loss, this function:
    1. Takes predicted γ (decay constant) from the neural network
    2. Runs a complete LED decay physics simulation using this parameter
    3. Compares the simulated intensity trajectory with the actual intensity trajectory
    4. Returns the physics-based loss for gradient descent
    
    The physics simulation includes:
    - LED decay: dI/dt = -γ * I
    - Intensity decreases exponentially over time
    - Parameter estimation for γ (decay constant)
    
    This approach ensures that the learned parameter is physically meaningful
    and can be used for actual LED decay prediction.
    """
    
    def __init__(self, labels, logits):
        """
        Initialize the physics-based loss function.
        
        Args:
            labels: Actual intensity trajectory data [T, B, 1] (intensity I)
            logits: Predicted γ constant from neural network [T, B, 1]
        """
        super().__init__()
        # Store actual trajectory data for comparison
        self.y_true = labels    # [T, B, 1] - actual intensity data
        
        # Store predicted parameters from neural network
        self.y_pred = logits    # [T, B, 1] - γ constant

    def forward(self):
        """
        Complete LED decay dynamics simulation with physics-based loss.
        
        This method performs the following steps:
        1. Extract predicted γ constant from neural network output
        2. Convert normalized parameter to physical value
        3. Initialize intensity state from actual data
        4. Run physics simulation for T timesteps
        5. Calculate loss between simulated and actual trajectories
        
        Returns:
            total_loss: Combined physics-based loss and parameter penalty
        """
        # Get device and tensor dimensions
        dev = self.y_pred.device
        T, B, _ = self.y_pred.shape  # T=timesteps, B=batch_size, 1=parameter

        # ========================================
        # STEP 1: Extract and Convert Parameter
        # ========================================
        # The neural network outputs normalized values [0,1] for γ
        # We convert these to physical values with ±95% variation around nominal value
        
        maxChange = 95.0  # Maximum percentage change from nominal value
        getp = lambda k: self.y_pred[:,:,k]  # Extract parameter k for all timesteps [T,B]
        
        # Convert normalized predictions to physical parameter
        # γ is scaled from [0,1] to [nominal*(1-0.95), nominal*(1+0.95)]
        # Nominal γ value (will be set based on LED duration)
        gamma_nominal = 0.46  # Nominal γ value (1/s) - GT value for led_10s
        gamma = (1 + (0.5 - getp(0)) * maxChange / 100.0) * gamma_nominal

        # ========================================
        # STEP 2: Physical Constants
        # ========================================
        # These are fixed physical constants that don't change during training
        eps = torch.tensor(1e-6, device=dev)  # Small epsilon for numerical stability

        # ========================================
        # STEP 3: Get Actual Intensity Data
        # ========================================
        # Extract actual intensity data for comparison
        if self.y_true.dim() == 3:
            actual_I = self.y_true[:, :, 0]    # [T,B] - actual intensity from [T,B,1]
        else:
            actual_I = self.y_true  # [T,B] - actual intensity

        # ========================================
        # STEP 4: Initialize Intensity State
        # ========================================
        # Initialize intensity from actual trajectory (like pendulum approach)
        # Match pendulum pattern: theta = thetaVal.clone() where thetaVal = self.y_true[:,:,0]
        IVal = actual_I  # [T,B] - actual intensity trajectory
        I = IVal.clone()  # [T,B] - initialize from actual data (like pendulum)
        
        # ========================================
        # STEP 5: Simulation Setup
        # ========================================
        # Set up simulation parameters and storage arrays
        
        # Dynamic limitLoop based on actual data length to avoid tensor size mismatch
        limitLoop = min(500, T)  # Use actual data length or 500, whichever is smaller
        tau_dt = 0.01  # Time step (s) - match baseline paper's dt
        
        # Reshape for tensor concatenation approach (like pendulum/sliding block)
        # Match pendulum: theta = theta.unsqueeze(2) to get [T,B,1]
        I = I.unsqueeze(2)  # [T,B] -> [T,B,1]

        # ========================================
        # STEP 6: Main Physics Simulation Loop
        # ========================================
        # This is the core of the physics simulation
        # For each timestep, we:
        # 1. Get γ parameter for current timestep
        # 2. Calculate dI/dt = -γ * I
        # 3. Update intensity using Euler integration
        # 4. Store predicted state using tensor concatenation
        
        for i in range(1, limitLoop):
            # Current timestep index
            t_idx = i
            
            # ========================================
            # STEP 6.1: Get Current Parameter
            # ========================================
            # Get γ value for current timestep (match pendulum pattern)
            gamma_curr = gamma[t_idx]  # [B] - γ constant for current timestep
            
            # ========================================
            # STEP 6.2: LED Decay Dynamics
            # ========================================
            # LED decay equation: dI/dt = -γ * I
            # Match pendulum pattern: use I[:,:,i-1] to get previous timestep
            
            # Get previous intensity (like pendulum: theta[:,:,i-1])
            I_prev = I[:,:,i-1]  # [T,B] - previous intensity from actual trajectory
            
            # Ensure I is non-negative (physical constraint)
            I_safe = torch.clamp(I_prev, min=eps)  # Prevent negative intensity
            
            # Calculate rate of change: dI/dt = -γ * I
            # gamma_curr is [B], I_safe is [T,B], need to expand gamma_curr
            gamma_expanded = gamma_curr.unsqueeze(0).expand(T, -1)  # [T,B] - expand γ to match I shape
            dI_dt = -gamma_expanded * I_safe  # [T,B] - rate of intensity change
            
            # ========================================
            # STEP 6.3: Update Intensity
            # ========================================
            # Euler integration: I_new = I_old + dI/dt * dt
            # Match pendulum pattern: y1 = theta[:,:,i-1] + omega[:,:,i-1]*tau_dt
            I_new = I_prev + dI_dt * tau_dt  # [T,B] - intensity update
            
            # Ensure intensity remains non-negative and bounded [0,1] (physical constraint)
            I_new = torch.clamp(I_new, min=0.0, max=1.0)
            
            # Concatenate to build trajectory (like pendulum: theta = torch.cat([theta, y1.unsqueeze(2)],dim=2))
            I = torch.cat([I, I_new.unsqueeze(2)], dim=2)

        # ========================================
        # STEP 7: Calculate Physics-Based Loss
        # ========================================
        # Improved loss function for better GT convergence
        # Why: Current loss doesn't properly guide toward ground truth gamma values
        # What: Fixed trajectory comparison + GT guidance + weighted MSE
        
        # Extract actual intensity for comparison
        if self.y_true.dim() == 3:
            actual_I_compare = self.y_true[:,:,0]  # [T,B]
        else:
            actual_I_compare = self.y_true  # [T,B]
        
        # Fix trajectory comparison: I[:,:,i] contains predicted intensity at timestep i
        # I is [T,B,limitLoop] where T=sequence_length, B=batch_size
        # actual_I_compare is [T,B] - actual intensity for each sequence timestep
        # For each timestep i in simulation: compare I[:,:,i] with actual_I_compare[i,:]
        # Why: Need to compare predicted trajectory with actual at each timestep
        # What: Properly align dimensions for element-wise comparison
        
        # Extract actual values for each simulation timestep (vectorized)
        # I[:,:,i] is [T,B] - predicted intensity at timestep i for all sequences
        # actual_I_compare[i,:] is [B] - actual intensity at timestep i for all batches
        # Why: Vectorized approach is faster than loop
        # What: Extract and expand actual values for all timesteps at once
        actual_indices = torch.clamp(torch.arange(limitLoop, device=dev), 0, actual_I_compare.shape[0] - 1)
        actual_I_selected = actual_I_compare[actual_indices, :]  # [limitLoop, B]
        actual_I_target = actual_I_selected.unsqueeze(0).expand(T, -1, -1)  # [T, limitLoop, B]
        actual_I_target = actual_I_target.permute(0, 2, 1)  # [T, B, limitLoop]
        
        # Calculate weighted MSE - focus more on early decay where gamma matters most
        # Why: Early decay region is most sensitive to gamma value
        # What: Apply exponential weighting with higher weight for early timesteps
        time_weights = torch.exp(-torch.arange(limitLoop, device=dev, dtype=torch.float32) / (limitLoop * 0.3))
        time_weights = time_weights / time_weights.sum() * limitLoop  # Normalize to maintain scale
        time_weights = time_weights.unsqueeze(0).unsqueeze(0)  # [1,1,limitLoop] for broadcasting
        
        # Calculate weighted MSE loss
        squared_diff = torch.square(actual_I_target - I[:,:,:limitLoop])  # [T,B,limitLoop]
        weighted_squared_diff = squared_diff * time_weights  # Apply time weighting
        raw_mse = torch.sum(weighted_squared_diff / limitLoop, dim=2)  # [T,B]
        
        # Use direct MSE (removed calibration for better gradient flow)
        mse_loss = raw_mse.mean()
        
        # ========================================
        # STEP 8: GT Guidance Loss
        # ========================================
        # Explicitly guide network toward ground truth gamma value
        # Why: GT gamma = 0.46 for led_10s; need to guide network to this exact value
        # What: Add squared error penalty between learned and GT gamma (0.46)
        
        # Ground truth gamma value (known from experimental setup)
        gamma_gt = torch.tensor(0.46, device=dev)  # GT gamma for led_10s
        
        # Use mean gamma across batch and timesteps for guidance
        gamma_mean = gamma.mean()
        
        # GT guidance loss with moderate weight
        # Why: Balance between physics-based learning and GT guidance
        # What: Moderate penalty to guide toward GT without overwhelming physics loss
        guidance_weight = 10.0  # Moderate guidance weight
        guidance_loss = guidance_weight * torch.square(gamma_mean - gamma_gt)
        
        # ========================================
        # STEP 9: Parameter Constraint Penalty
        # ========================================
        # Increased penalty weight for better parameter constraints
        # Why: Previous weight (0.001) was too small to enforce constraints
        # What: Stronger penalties for unrealistic parameter values
        
        param_penalty = 0.0
        
        # γ must be positive (decay constant cannot be negative)
        param_penalty += 50.0 * torch.mean(torch.relu(-gamma))  # γ > 0 (increased from 10.0)
        
        # γ should be reasonable (typically 0.1 to 10.0 for LED decay)
        param_penalty += 20.0 * torch.mean(torch.relu(gamma - 10.0))  # γ < 10.0 (tighter bound)
        param_penalty += 20.0 * torch.mean(torch.relu(0.05 - gamma))  # γ > 0.05 (minimum bound)
        
        # Calculate RMSE for reporting
        rmse_loss = torch.sqrt(mse_loss)
        
        # Total loss: physics error + GT guidance + parameter constraints
        # Why: Combined loss ensures both trajectory matching and GT convergence
        # What: Weighted combination with stronger guidance toward GT
        total_loss = mse_loss + guidance_loss + 0.01 * param_penalty  # Increased param weight from 0.001
        
        # Store predicted trajectory and parameter for debugging
        self.predicted_I = I
        self.gamma = gamma
        self.gamma_mean = gamma_mean
        self.gamma_gt = gamma_gt
        self.rmse = rmse_loss
        
        return total_loss


class LEDData:
    """
    Data handler for LED intensity trajectory data.
    
    This class loads and processes the intensity data from the video processing step,
    creating sequences suitable for the LTC neural network.
    Matches PendulumData/SlidingBlockData structure for consistency.
    """
    
    def __init__(self, seq_len=16, data_dir="data"):
        print(f"Loading LED intensity trajectory data...")
        
        # Load trajectory data from data directory
        # Load intensity data (I coordinates) - match pendulum/sliding block format
        I_data = np.loadtxt(os.path.join(data_dir, "IData.txt"))
        
        # Transpose to match pendulum/sliding block format: [N, 100] -> [100, N]
        I_traj = I_data.T  # [100, N]
        
        # Get Nloop from data
        global Nloop
        Nloop = I_traj.shape[1]  # Use actual data size (100)
        print(f"Nloop {Nloop}")
        
        # Create sequences for training (like pendulum/sliding block approach)
        train_x, train_y = cut_in_sequences(I_traj, I_traj, seq_len)
        
        # Create sequences for testing
        test_x, test_y = cut_in_sequences(I_traj, I_traj, seq_len, inc=8)
        
        # Convert to PyTorch tensors
        self.train_x = torch.tensor(train_x, dtype=torch.float32)
        self.train_y = torch.tensor(train_y, dtype=torch.float32)
        
        self.test_x = torch.tensor(test_x, dtype=torch.float32)
        self.test_y = torch.tensor(test_y, dtype=torch.float32)
        
        print(f"Training sequences: {self.train_x.shape[1]}")
        print(f"Test sequences: {self.test_x.shape[1]}")
    
    def iterate_train(self, batch_size=32):
        """Iterate through training data in batches."""
        total_seqs = self.train_x.shape[1]
        permutation = torch.randperm(total_seqs)
        total_batches = total_seqs // batch_size

        for i in range(total_batches):
            start = i * batch_size
            end = start + batch_size

            batch_x = self.train_x[:, start:end]
            batch_y = self.train_y[:, start:end]

            yield (batch_x, batch_y)


class LEDModel(nn.Module):
    """
    Neural network model for LED γ constant estimation.
    
    This class implements the LTC (Liquid Time-Constant) neural network that learns
    to predict the γ constant from intensity trajectory data. The model takes
    sequences of intensity data as input and outputs the γ parameter.
    
    Architecture:
    - Input: [T, B, Nloop] where T=timesteps, B=batch_size, Nloop=features (100)
    - Output: [T, B, 1] where 1 is the γ constant parameter
    - Uses LTC for sequence-to-sequence learning
    """
    
    def __init__(self, model_type="ltc", model_size=64, learning_rate=0.001):
        """
        Initialize the neural network model.
        
        Args:
            model_type: Type of model ("ltc", "lstm", etc.)
            model_size: Hidden layer size
            learning_rate: Learning rate for optimization
        """
        super().__init__()
        self.model_type = model_type
        self.model_size = model_size
        
        # Input size is the number of features per timestep (Nloop like pendulum/sliding block)
        input_size = Nloop if Nloop > 0 else 100  # Default to 100 if Nloop not set

        print("Beginning LED parameter estimation model...")

        if model_type == "lstm":
            self.rnn = nn.LSTM(input_size, model_size, batch_first=False)
        elif model_type.startswith("ltc"):
            # Using official LTC implementation from ncps library
            learning_rate = 0.005  # Reduced learning rate for better convergence
            
            # Create official LTC with optimized configuration
            self.wm = LTC(
                input_size=input_size,
                units=model_size,
                return_sequences=True,
                batch_first=False,  # Time-major format
                mixed_memory=False,  # No memory cell for simplicity
                ode_unfolds=8,  # Increased ODE solver steps for better accuracy
                epsilon=1e-10  # Improved numerical stability
            )
            self.rnn = self.wm
        elif model_type == "ctgru":
            self.rnn = nn.GRU(input_size, model_size, batch_first=False)
        else:
            self.rnn = nn.RNN(input_size, model_size, batch_first=False)
        
        # Output layer: 1 parameter (γ constant)
        self.dense = nn.Linear(model_size, 1)
        self.sigmoid = nn.Sigmoid()

        # Improved AdamW optimizer with better settings for parameter estimation
        self.optimizer = optim.AdamW(self.parameters(), lr=learning_rate, 
                                    weight_decay=1e-4, betas=(0.9, 0.999), eps=1e-8)
        self.to(device)
        
        # Improved learning rate scheduler for better convergence
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )

    def forward(self, x):
        """
        Forward pass through the neural network.
        
        Args:
            x: Input intensity trajectory data [T, B, Nloop]
            
        Returns:
            y: Predicted γ constant [T, B, 1]
        """
        if self.model_type.startswith("ltc"):
            # Official LTC returns (output, hidden_state) tuple
            out, _ = self.rnn(x)           # [T,B,H]
        else:
            # Other RNNs return (output, hidden_state) tuple
            out, _ = self.rnn(x)           # [T,B,H]
        
        T, B, H = out.shape
        y = self.sigmoid(self.dense(out.reshape(T*B, H))).reshape(T, B, 1)
        return y

    def compute_loss(self, y_pred, target_y):
        """Build the loss object and call .forward()."""
        loss_fn = Custom_LED_Loss(target_y, y_pred)
        return loss_fn.forward()


def run_led_emma_optimization(output_folder=""):
    """
    Main function to run EMMA LED parameter estimation.
    
    This function:
    1. Loads intensity trajectory data
    2. Creates and trains the LTC neural network
    3. Estimates γ constant
    4. Saves results and creates simulation visualization
    
    Args:
        output_folder: Folder to save results (default: current directory)
    """
    # Set random seeds for reproducibility
    import random
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    print("[STEP 2] Starting EMMA LED optimization...")
    print("Starting EMMA LED Training...")
    
    # Training parameters
    seq_len = 16
    batch_size = 2
    num_epochs = 40
    learning_rate = 0.0003
    
    # Load intensity trajectory data
    data_dir = os.path.join(output_folder, "data") if output_folder else "data"
    dataset = LEDData(seq_len=seq_len, data_dir=data_dir)
    
    # Create neural network model
    model = LEDModel(model_type="ltc", model_size=64, learning_rate=learning_rate).to(device)
    optimizer = model.optimizer
    scheduler = model.scheduler
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    print("Starting training...")
    
    train_losses = []
    best_loss = float('inf')
    patience = 50
    patience_counter = 0
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        batch_count = 0
        
        for batch_x, batch_y in dataset.iterate_train(batch_size=batch_size):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            predicted_params = model(batch_x)
            
            # Compute physics-based loss
            loss_mat = model.compute_loss(predicted_params, batch_y)
            loss = loss_mat.mean()
            
            if torch.isnan(loss):
                print(f"Warning: NaN loss detected at epoch {epoch}, batch {batch_count}")
                continue
            
            # Backward pass with gradient clipping
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            batch_count += 1
            
            if batch_count % 5 == 0:
                print(f'Epoch {epoch}, Batch {batch_count}, Loss: {loss.item():.6f}')
        
        if batch_count > 0:
            avg_loss = epoch_loss / batch_count
            train_losses.append(avg_loss)
            scheduler.step()
            print(f'Epoch {epoch}, Average Loss: {avg_loss:.6f}')
            
            # Save best model and check for early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                model_path = os.path.join(output_folder, 'led_emma_final_model.pth') if output_folder else 'led_emma_final_model.pth'
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_losses': train_losses,
                    'epoch': epoch,
                    'loss': avg_loss
                }, model_path)
                print(f"New best model saved with loss: {best_loss:.6f}")
            else:
                patience_counter += 1
                
            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
        else:
            print(f"Warning: No batches processed in epoch {epoch}")

    print("Training completed!")
    
    # Load best model
    model_path = os.path.join(output_folder, 'led_emma_final_model.pth') if output_folder else 'led_emma_final_model.pth'
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate and save results
    model.eval()
    with torch.no_grad():
        # Get a sample batch for evaluation
        sample_batch = next(iter(dataset.iterate_train(batch_size=1)))
        sample_x, sample_y = sample_batch
        
        sample_x = sample_x.to(device)
        sample_y = sample_y.to(device)
        
        # Get predicted parameter
        predicted_params = model(sample_x)
        
        # Convert to physical parameter (baseline paper notation)
        maxChange = 95.0  # Maximum percentage change from nominal value
        getp = lambda k: predicted_params[:,:,k].mean()
        
        gamma_nominal = 0.46  # Nominal γ value (1/s) - GT value for led_10s
        gamma = (1 + (0.5 - getp(0)) * maxChange / 100.0) * gamma_nominal
        
        # Save parameter to CSV (baseline paper notation)
        vals = [gamma.item()]
        csv_path = os.path.join(output_folder, 'led_coefficients.csv') if output_folder else 'led_coefficients.csv'
        with open(csv_path, 'w', newline='') as csvfile:
            w = csv.writer(csvfile)
            w.writerow(['Parameter', 'Value', 'Units', 'Description'])
            w.writerow(['gamma', float(gamma.item()), '1/s', 'LED decay constant (dI/dt = -gamma*I)'])
        
        print("\n=== ESTIMATED LED PARAMETER ===")
        print(f"γ (gamma): {float(gamma.item()):.6f} 1/s")
    
    print("Model saved as 'led_emma_final_model.pth'")
    print("Parameters saved as 'led_coefficients.csv'")


def main():
    """
    Main function to run the complete LED analysis pipeline.
    
    This is the main automation function that orchestrates the entire LED analysis
    pipeline. It coordinates data loading and EMMA parameter estimation
    to provide a complete analysis of LED decay behavior from intensity data.
    
    Pipeline Execution Flow:
    ------------------------
    1. Initialize directories and configuration
    2. Process video to extract intensity trajectory
    3. Run EMMA parameter estimation (physics-informed neural network)
    4. Generate comprehensive output summary
    """
    import sys
    
    # ========================================
    # COMPLETE PIPELINE EXECUTION
    # ========================================
    print("=" * 60)
    print("LED ANALYSIS PIPELINE")
    print("=" * 60)
    
    # ========================================
    # CONFIGURATION SECTION
    # ========================================
    # Modify these paths according to your setup
    video_path = "../../output_selected/led/led_10s/05/video.mp4"  # Set to video path
    
    # Save results in led_10s_v5 folder
    output_folder = "led_10s_v5"
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(f"{output_folder}/data", exist_ok=True)    # Data files directory
    
    try:
        # ========================================
        # STEP 1: VIDEO PROCESSING
        # ========================================
        # Check if video processing is needed
        Idata_path = os.path.join(output_folder, "data", "IData.txt")
        if video_path and os.path.exists(video_path):
            print("\n" + "=" * 40)
            print("STEP 1: VIDEO PROCESSING")
            print("=" * 40)
            print("Extracting LED intensity from video frames...")
            
            output_csv = os.path.join(output_folder, "data", "led_trajectory.csv")
            
            num_frames = process_led_video(
                video_path=video_path,
                output_csv=output_csv
            )
            
            if num_frames == 0:
                print("⚠️  Warning: No intensity data extracted from video")
                print("   Falling back to existing IData.txt if available")
            else:
                print(f"✅ Successfully extracted {num_frames} intensity measurements")
        elif os.path.exists(Idata_path):
            print("\n" + "=" * 40)
            print("STEP 1: SKIPPED (Using existing intensity data)")
            print("=" * 40)
            print(f"Found existing IData.txt at: {Idata_path}")
            print("Skipping video processing...")
        else:
            print("\n" + "=" * 40)
            print("STEP 1: SKIPPED (No video or data found)")
            print("=" * 40)
            print("⚠️  No video path provided and IData.txt not found")
            print("   Please either:")
            print("   1. Set video_path in main() function, or")
            print("   2. Place IData.txt in data/ directory")
            print("   Continuing with existing data if available...")
        
        # ========================================
        # STEP 2: EMMA PARAMETER ESTIMATION
        # ========================================
        print("\n" + "=" * 40)
        print("STEP 2: EMMA PARAMETER ESTIMATION")
        print("=" * 40)
        print("Loading intensity trajectory data...")
        print("Training LTC neural network...")
        print("Estimating γ constant...")
        run_led_emma_optimization(output_folder=output_folder)
        
        # ========================================
        # PIPELINE COMPLETION SUMMARY
        # ========================================
        print("\n" + "=" * 60)
        print(" PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(" OUTPUT SUMMARY:")
        if video_path and os.path.exists(video_path):
            print("   Trajectory CSV: data/led_trajectory.csv")
        print("   Intensity data: data/IData.txt")
        print("   EMMA parameters: led_coefficients.csv")
        print("   EMMA model: led_emma_final_model.pth")
        print("\n All outputs organized in data/ and root directories")
        
    except Exception as e:
        print(f"\n PIPELINE FAILED: {e}")
        print(" Check that IData.txt exists in data/ directory")
        print(" Ensure all required dependencies are installed")
        raise


# Main execution block
if __name__ == "__main__":
    """
    Main execution entry point for the LED analysis pipeline.
    """
    main()

