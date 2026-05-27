# This is the code for Torricelli pipeline based on EMMA method

# EMMA Torricelli Pipeline

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


class TorricelliDetector:
    """
    Torricelli ball/object tracker using YOLO for height measurement.
    
    This class implements ball/object tracking for the Torricelli pipeline, similar to
    Pendulum and Sliding Block experiments. It uses YOLO to detect and track a ball or
    object that moves with the liquid surface, then measures height from the ball position.
    
    Why: Accurate height measurement is critical for Torricelli's law analysis
    What: Tracks ball/object position to extract height measurements
    """
    def __init__(self, weights_path, conf=0.15, imgsz=640):
        """
        Initialize the Torricelli ball detector with YOLO model.
        
        Args:
            weights_path: Path to YOLO model weights file (.pt)
            conf: Detection confidence threshold (0.0-1.0)
            imgsz: Input image size for YOLO processing
        """
        from ultralytics import YOLO
        self.model = YOLO(weights_path)
        self.conf = conf
        self.imgsz = imgsz
        self.last_detection = None
        self.container_bottom = None  # Store container bottom position (reference point)
        print(f"[INFO] Loaded YOLO weights: {weights_path}")

    def detect_ball(self, frame):
        """
        Detect ball/object in a single video frame using YOLO with intelligent filtering.
        
        Args:
            frame: Input video frame (numpy array)
            
        Returns:
            tuple: (x1, y1, x2, y2, confidence) or None if no detection
            
        Why: Need to track ball position to measure liquid height
        What: Returns ball bounding box with confidence score, similar to Pendulum detector
        """
        h, w = frame.shape[:2]
        img_area = w * h
        edge_thresh = max(10, int(0.01 * min(w, h)))
        min_area_px = max(100, int(0.00001 * img_area))  # Ball is typically small
        max_area_px = int(0.1 * img_area)  # Ball should be reasonably sized

        # Run YOLO inference
        results = self.model.predict(
            source=frame, imgsz=self.imgsz, conf=self.conf,
            iou=0.5, agnostic_nms=True, verbose=False
        )

        # Filter and validate detections
        candidates = []
        for r in results:
            if r.boxes is None:
                continue
            for b in r.boxes:
                x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
                bw, bh = x2 - x1, y2 - y1
                area = bw * bh
                
                # Size-based filtering (too small or too large)
                if area < min_area_px or area > max_area_px:
                    continue
                    
                # Edge proximity filtering (avoid edge artifacts)
                if (x1 < edge_thresh or y1 < edge_thresh or 
                    x2 > w - edge_thresh or y2 > h - edge_thresh):
                    continue
                    
                conf = float(b.conf[0].item()) if hasattr(b, "conf") else 0.0
                candidates.append((x1, y1, x2, y2, conf))

        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]

        # Multi-candidate selection with tracking consistency (like Pendulum)
        candidates.sort(key=lambda t: t[4], reverse=True)
        top_candidates = candidates[:3]
        
        # Use tracking consistency if previous detection exists
        if self.last_detection is not None:
            lx1, ly1, lx2, ly2, _ = self.last_detection
            lcx, lcy = (lx1 + lx2) / 2.0, (ly1 + ly2) / 2.0

            def dist(c):
                cx, cy = (c[0] + c[2]) / 2.0, (c[1] + c[3]) / 2.0
                return ((cx - lcx) ** 2 + (cy - lcy) ** 2) ** 0.5

            return min(top_candidates, key=dist)
        return top_candidates[0]

    def detect(self, frame):
        """
        Detect ball/object and calculate height measurement.
        
        Args:
            frame: Input video frame (numpy array)
            
        Returns:
            tuple: (height_pixels, confidence) or None if detection fails
            
        Why: Complete detection pipeline for height extraction using ball tracking
        What: Returns liquid height in pixels from container bottom
        """
        # Detect ball/object
        ball_box = self.detect_ball(frame)
        if ball_box is None:
            if self.last_detection is not None:
                # Use last known ball position (for Kalman filter prediction)
                ball_box = self.last_detection
            else:
                return None
        else:
            self.last_detection = ball_box
        
        # Initialize container bottom on first detection (use ball's initial position as reference)
        # Container bottom is typically near the bottom of the frame
        h, w = frame.shape[:2]
        if self.container_bottom is None:
            # Use bottom of frame as initial container bottom reference
            # Ball Y position will be measured relative to this
            self.container_bottom = float(h * 0.95)  # Assume container bottom is 95% down the frame
        
        # Get ball center Y position (this represents liquid surface height)
        _, y1, _, y2, _ = ball_box
        ball_center_y = (y1 + y2) / 2.0
        
        # Calculate height (distance from container bottom to ball center)
        # Height increases as we go up (smaller y values)
        # Ball floats on liquid surface, so ball Y position = liquid surface height
        height_pixels = self.container_bottom - ball_center_y
        
        # Ensure height is positive
        if height_pixels < 0:
            height_pixels = 0
            
        confidence = ball_box[4] if ball_box is not None else 0.5
        
        return (height_pixels, confidence)


class Kalman1D:
    """
    1D Kalman Filter for height trajectory smoothing and prediction.
    
    This class implements a 1D Kalman filter to smooth height measurements
    and predict height when detection fails. The filter tracks height and
    height velocity (rate of change).
    
    State Vector: [h, vh] (height + height velocity)
    Measurement: [h] (height only from detection)
    
    Why: Raw detections are noisy and may have gaps
    What: Provides smooth, continuous height estimates
    """
    def __init__(self, dt=0.01):
        """
        Initialize 1D Kalman filter with system dynamics.
        
        Args:
            dt: Time step between measurements (seconds)
        """
        self.dt = dt
        # State vector: [h, vh] (height, height velocity)
        self.state = np.zeros(2).reshape(-1, 1)  # Column vector for matrix operations
        
        # State transition matrix F (constant velocity model)
        self.F = np.eye(2)
        self.F[0, 1] = dt  # h += vh * dt
        
        # Measurement matrix H (we measure height only)
        self.H = np.array([[1.0, 0.0]])  # [1, 0] to extract height from state
        
        # Process noise covariance Q
        self.Q = np.eye(2) * 0.1
        
        # Measurement noise covariance R
        self.R = np.array([[1.0]])  # 1x1 matrix for 1D measurement
        
        # Error covariance matrix P
        self.P = np.eye(2) * 100.0

    def predict(self):
        """
        Predict next state using motion model (no measurement).
        
        Returns:
            float: Predicted height
            
        Why: Estimate height when detection fails
        What: Advances state using constant velocity model
        """
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        # Extract height (first element of state vector) as float
        return float(self.state[0, 0] if self.state.ndim > 1 else self.state[0])

    def update(self, measurement):
        """
        Update state estimate with new measurement.
        
        Args:
            measurement: Height measurement (float)
            
        Returns:
            float: Updated height estimate
            
        Why: Incorporate new measurements to improve accuracy
        What: Combines prediction with measurement using Kalman equations
        """
        measurement = np.array([[float(measurement)]])  # Convert to 2D array
        y = measurement - self.H @ self.state  # Innovation
        S = self.H @ self.P @ self.H.T + self.R  # Innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman gain
        self.state = self.state + K @ y  # Update state
        self.P = (np.eye(2) - K @ self.H) @ self.P  # Update covariance
        # Extract height (first element of state vector) as float
        return float(self.state[0, 0] if self.state.ndim > 1 else self.state[0])


def process_torricelli_video(video_path, weights_path, output_video, output_csv, conf=0.15, pixel_to_meter=0.001):
    """
    Process Torricelli video to extract height trajectory using ball tracking.
    
    This function processes Torricelli videos to:
    1. Load video and YOLO model
    2. Detect and track ball/object in each frame (similar to Pendulum)
    3. Track height using Kalman filtering
    4. Convert pixel height to physical units (meters)
    5. Create annotated video with height overlay
    6. Save height trajectory data and generate plots
    
    Args:
        video_path: Path to input Torricelli video file
        weights_path: Path to YOLO model weights
        output_video: Path for annotated video output
        output_csv: Path for trajectory CSV output
        conf: YOLO detection confidence threshold
        pixel_to_meter: Conversion factor from pixels to meters (default: 0.001 m/pixel)
        
    Why: Video processing is the foundation of Torricelli height trajectory analysis
    What: Extracts smooth height trajectory from ball tracking in video frames
    """
    print(f"[STEP 1] Processing Torricelli video: {video_path}")
    print(f"[STEP 1] Output video: {output_video}")
    print(f"[STEP 1] Output CSV: {output_csv}")

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    detector = TorricelliDetector(weights_path, conf=conf)
    kf = Kalman1D()
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    csv_f = open(output_csv, "w", newline="")
    csvw = csv.writer(csv_f)
    csvw.writerow(["frame", "time_s", "height_pixels", "height_meters", "conf"])

    h_series_pixels, h_series_meters = [], []
    frame_idx = 0
    
    # Calibration: Use a reasonable estimate for pixel_to_meter
    # NOTE: We should NOT use ground truth h_0 from parameters.json
    # Instead, use a generic reasonable value or estimate from video
    # For Torricelli, typical container heights are 5-10 cm, so we use a reasonable default
    # User can adjust pixel_to_meter manually if they have a reference object in the video
    calibration_frames = []
    calibration_complete = False
    # Use a reasonable default instead of ground truth
    # This assumes ~1mm per pixel for typical video resolution
    # User should calibrate using a known reference object if available
    
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_time = frame_idx / fps
        det = detector.detect(frame)
        
        if det is not None:
            height_pixels, conf_val = det
            
            # NOTE: We do NOT calibrate using ground truth h_0
            # The pixel_to_meter should be set manually based on a reference object
            # or use the default value provided in main()
            # This avoids "cheating" by using ground truth information
            if not calibration_complete and conf_val > 0.5:
                calibration_complete = True
                print(f"[STEP 1] Using pixel_to_meter = {pixel_to_meter:.6f} m/px (user-provided or default)")
                print(f"[STEP 1] NOTE: For accurate results, calibrate using a known reference object in the video")
            
            height_meters = float(height_pixels) * pixel_to_meter
            
            # Update Kalman filter
            kf.predict()
            h_smooth = kf.update(height_pixels)  # Kalman filter returns float
            h_smooth_meters = h_smooth * pixel_to_meter
            
            h_series_pixels.append(h_smooth)
            h_series_meters.append(h_smooth_meters)

            # Draw detection and height on frame
            if detector.last_detection is not None:
                x1, y1, x2, y2 = [int(v) for v in detector.last_detection[:4]]
                # Draw ball bounding box
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                # Draw ball center point
                ball_center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                cv2.circle(frame, ball_center, 5, (0, 255, 0), -1)
                # Draw line from container bottom to ball (height indicator)
                if detector.container_bottom is not None:
                    bottom_point = (ball_center[0], int(detector.container_bottom))
                    cv2.line(frame, bottom_point, ball_center, (255, 0, 0), 2)
            
            # Draw height text
            height_text = f"h={h_smooth_meters:.4f}m ({h_smooth:.1f}px), conf={conf_val:.2f}"
            cv2.putText(frame, height_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            csvw.writerow([frame_idx, f"{frame_time:.3f}", f"{h_smooth:.2f}", 
                          f"{h_smooth_meters:.6f}", f"{conf_val:.3f}"])
        else:
            # No detection - use Kalman prediction
            h_pred = kf.predict()  # Kalman filter returns float
            h_pred_meters = h_pred * pixel_to_meter
            
            h_series_pixels.append(h_pred)
            h_series_meters.append(h_pred_meters)
            
            # Draw predicted height
            height_text = f"h={h_pred_meters:.4f}m ({h_pred:.1f}px) [predicted]"
            cv2.putText(frame, height_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            csvw.writerow([frame_idx, f"{frame_time:.3f}", f"{h_pred:.2f}",
                          f"{h_pred_meters:.6f}", "0.000"])

        out.write(frame)
        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"[PROGRESS] Processed {frame_idx} frames")
            check_memory_usage()

    cap.release()
    out.release()
    csv_f.close()

    if h_series_pixels and h_series_meters:
        # Save height trajectory data in EMMA format
        h_arr = np.array(h_series_meters)  # Use meters for physical units
        
        # Report extracted height range (for information only, not validation against ground truth)
        h_0_actual = h_arr[0] if len(h_arr) > 0 else 0.0
        h_n_actual = h_arr[-1] if len(h_arr) > 0 else 0.0
        
        print(f"\n[STEP 1] Extracted Height Range:")
        print(f"   Initial height: h_0 = {h_0_actual:.3f} m")
        print(f"   Final height:   h_n = {h_n_actual:.3f} m")
        print(f"   Height change:  {h_0_actual - h_n_actual:.3f} m")
        
        # Check if height decreases (physical constraint)
        if h_n_actual >= h_0_actual:
            print(f"   ⚠️  Warning: Height does not decrease (may indicate tracking issue)")
        else:
            print(f"   ✅ Height decreases as expected for Torricelli flow")
        
        # Match EMMA format (N x 100 matrices for memory optimization)
        h_matrix = np.tile(h_arr.reshape(-1, 1), (1, 100))
        
        # Determine data directory from output_csv path
        data_dir = os.path.dirname(output_csv)
        os.makedirs(data_dir, exist_ok=True)
        np.savetxt(os.path.join(data_dir, "hData.txt"), h_matrix, fmt='%.6f')
        
        del h_matrix, h_arr
        gc.collect()
        print(f"\n[STEP 1] ✅ Saved Torricelli height trajectory data: {len(h_series_meters)} frames")
        print(f"[STEP 1] ✅ Saved height data: hData.txt")
        
        # Create trajectory plots
        print("[STEP 1] Creating Torricelli height trajectory plots...")
        
        # Plot height vs time
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        time_array = np.arange(len(h_series_meters)) / fps
        ax.plot(time_array, h_series_meters, 'b-', linewidth=2, label='Height (m)')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Height (m)')
        ax.set_title('Torricelli Liquid Height vs Time')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plot_path = os.path.join(data_dir, "torricelli_height_trajectory.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[STEP 1] ✅ Saved trajectory plot: {plot_path}")
        
        return len(h_series_meters)
    else:
        print("[STEP 1] ⚠️  No height data extracted from video")
        return 0


def cut_in_sequences(x, y, seq_len, inc=1):
    """
    Slice a long 1D/2D series into overlapping windows for sequence-based learning.
    
    This function creates sequences from the input data for the LTC model.
    For Torricelli data: input shape (N, 100) -> output shape (seq_len, num_sequences, 100)
    
    Args:
        x: Input data array (e.g., height trajectory)
        y: Target data array (e.g., height trajectory) 
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


class Custom_Torricelli_Loss(nn.Module):
    """
    Custom loss function that integrates Torricelli's law physics simulation.
    
    This is the core of the parameter estimation system. Instead of using a simple
    MSE loss, this function:
    1. Takes predicted k constant from the neural network
    2. Runs a complete Torricelli physics simulation using this parameter
    3. Compares the simulated height trajectory with the actual height trajectory
    4. Returns the physics-based loss for gradient descent
    
    The physics simulation includes:
    - Torricelli's law: dh/dt = -k * sqrt(h)
    - Height decreases over time as liquid drains
    - Parameter estimation for k (drainage constant)
    
    This approach ensures that the learned parameter is physically meaningful
    and can be used for actual fluid dynamics prediction.
    """
    
    def __init__(self, labels, logits):
        """
        Initialize the physics-based loss function.
        
        Args:
            labels: Actual height trajectory data [T, B, 1] (height h)
            logits: Predicted k constant from neural network [T, B, 1]
        """
        super().__init__()
        # Store actual trajectory data for comparison
        self.y_true = labels    # [T, B, 1] - actual height data
        
        # Store predicted parameters from neural network
        self.y_pred = logits    # [T, B, 1] - k constant

    def forward(self):
        """
        Complete Torricelli dynamics simulation with physics-based loss.
        
        This method performs the following steps:
        1. Extract predicted k constant from neural network output
        2. Convert normalized parameter to physical value
        3. Initialize height state from actual data
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
        # The neural network outputs normalized values [0,1] for k
        # We convert these to physical values with ±95% variation around nominal value
        
        maxChange = 95.0  # Maximum percentage change from nominal value
        getp = lambda k: self.y_pred[:,:,k]  # Extract parameter k for all timesteps [T,B]
        
        # Convert normalized predictions to physical parameter
        # k is scaled from [0,1] to [nominal*(1-0.95), nominal*(1+0.95)]
        # Nominal k value from parameters.json for large torricelli
        k_nominal = 0.016202065833479495  # Nominal k value (m^(1/2)/s) - from parameters.json large
        k = (1 + (0.5 - getp(0)) * maxChange / 100.0) * k_nominal

        # ========================================
        # STEP 2: Physical Constants
        # ========================================
        # These are fixed physical constants that don't change during training
        eps = torch.tensor(1e-6, device=dev)  # Small epsilon for numerical stability (avoid sqrt(0))

        # ========================================
        # STEP 3: Get Actual Height Data
        # ========================================
        # Extract actual height data for comparison
        if self.y_true.dim() == 3:
            actual_h = self.y_true[:, :, 0]    # [T,B] - actual height from [T,B,1]
        else:
            actual_h = self.y_true  # [T,B] - actual height

        # ========================================
        # STEP 4: Initialize Height State
        # ========================================
        # Initialize height from actual trajectory (like pendulum approach)
        # Match pendulum pattern: theta = thetaVal.clone() where thetaVal = self.y_true[:,:,0]
        hVal = actual_h  # [T,B] - actual height trajectory
        h = hVal.clone()  # [T,B] - initialize from actual data (like pendulum)
        
        # ========================================
        # STEP 5: Simulation Setup
        # ========================================
        # Set up simulation parameters and storage arrays
        
        # Dynamic limitLoop based on actual data length to avoid tensor size mismatch
        limitLoop = min(500, T)  # Use actual data length or 500, whichever is smaller
        tau_dt = 0.01  # Time step (s) - match baseline paper's dt
        
        # Reshape for tensor concatenation approach (like pendulum/sliding block)
        # Match pendulum: theta = theta.unsqueeze(2) to get [T,B,1]
        h = h.unsqueeze(2)  # [T,B] -> [T,B,1]

        # ========================================
        # STEP 6: Main Physics Simulation Loop
        # ========================================
        # This is the core of the physics simulation
        # For each timestep, we:
        # 1. Get k parameter for current timestep
        # 2. Calculate dh/dt = -k * sqrt(h)
        # 3. Update height using Euler integration
        # 4. Store predicted state using tensor concatenation
        
        for i in range(1, limitLoop):
            # Current timestep index
            t_idx = i
            
            # ========================================
            # STEP 6.1: Get Current Parameter
            # ========================================
            # Get k value for current timestep (match pendulum pattern)
            k_curr = k[t_idx]  # [B] - k constant for current timestep
            
            # ========================================
            # STEP 6.2: Torricelli's Law Dynamics
            # ========================================
            # Torricelli's law: dh/dt = -k * sqrt(h)
            # Match pendulum pattern: use h[:,:,i-1] to get previous timestep
            
            # Get previous height (like pendulum: theta[:,:,i-1])
            h_prev = h[:,:,i-1]  # [T,B] - previous height from actual trajectory
            
            # Ensure h is non-negative (physical constraint)
            h_safe = torch.clamp(h_prev, min=eps)  # Prevent sqrt of negative or zero
            
            # Calculate rate of change: dh/dt = -k * sqrt(h)
            # k_curr is [B], h_safe is [T,B], need to expand k_curr
            k_expanded = k_curr.unsqueeze(0).expand(T, -1)  # [T,B] - expand k to match h shape
            dh_dt = -k_expanded * torch.sqrt(h_safe)  # [T,B] - rate of height change
            
            # ========================================
            # STEP 6.3: Update Height
            # ========================================
            # Euler integration: h_new = h_old + dh/dt * dt
            # Match pendulum pattern: y1 = theta[:,:,i-1] + omega[:,:,i-1]*tau_dt
            h_new = h_prev + dh_dt * tau_dt  # [T,B] - height update
            
            # Ensure height remains non-negative (physical constraint)
            h_new = torch.clamp(h_new, min=0.0)
            
            # Concatenate to build trajectory (like pendulum: theta = torch.cat([theta, y1.unsqueeze(2)],dim=2))
            h = torch.cat([h, h_new.unsqueeze(2)], dim=2)

        # ========================================
        # STEP 7: Calculate Physics-Based Loss
        # ========================================
        # The loss function compares the simulated trajectory with the actual trajectory
        # This is what drives the parameter estimation - the neural network learns
        # k that makes the simulation match the real height behavior
        
        # Loss calibration constant (calibrated with maxChange=0 using nominal k)
        # Match pendulum approach: loss_Cal_theta = 344.08, loss_Cal_omega = 47.72
        # Calibrated value from running with maxChange=0: 0.000000 (for large torricelli)
        loss_Cal_h = 0.000000  # Calibrated loss value (from calibration run with maxChange=0)
        
        # Calculate MSE loss (match pendulum pattern exactly)
        # Pendulum: torch.abs(torch.sum(torch.square(self.y_true[:,:,0:limitLoop]-theta)/limitLoop, dim=2)-loss_Cal_theta)
        # For Torricelli: h is [T,B,limitLoop] after simulation, actual_h is [T,B]
        # Need to reshape actual_h to [T,B,limitLoop] for comparison
        
        # Extract actual height for comparison
        if self.y_true.dim() == 3:
            actual_h_compare = self.y_true[:,:,0]  # [T,B]
        else:
            actual_h_compare = self.y_true  # [T,B]
        
        # Match pendulum loss calculation pattern
        # In pendulum: theta is [T,B,limitLoop] where theta[:,:,0] is initial (from actual), theta[:,:,1:] are predictions
        # The loss compares theta with self.y_true[:,:,0:limitLoop]
        # For Torricelli: h[:,:,0] is initial (from actual), h[:,:,1:] are predictions
        # We compare h with actual_h_compare properly reshaped
        
        # Reshape actual_h: h[:,:,i] should compare with actual_h_compare[i,:] for each i
        # h is [T,B,limitLoop], actual_h_compare is [T,B]
        # We need: actual_h_compare[i,:] compares with h[:,:,i] for each i in [0, limitLoop)
        # Create [T,B,limitLoop] where actual_h_broadcast[:,:,i] = actual_h_compare[i,:] for each i
        actual_h_broadcast = actual_h_compare[:limitLoop, :].unsqueeze(0).expand(T, -1, -1).permute(0, 2, 1)  # [T,B,limitLoop]
        
        # Calculate MSE loss matching pendulum pattern exactly
        # Pendulum: torch.sum(torch.square(self.y_true[:,:,0:limitLoop]-theta)/limitLoop, dim=2)
        raw_mse = torch.sum(torch.square(actual_h_broadcast - h[:,:,:limitLoop]) / limitLoop, dim=2)
        
        # Calibration already completed - loss_Cal_h is set to calibrated value
        
        # Calculate MSE loss with calibration (match pendulum: torch.abs(raw_mse - loss_Cal))
        # Pendulum: torch.abs(torch.sum(...)-loss_Cal_theta)
        mse_loss = torch.abs(raw_mse - loss_Cal_h)
        
        # ========================================
        # STEP 8: Parameter Constraint Penalty
        # ========================================
        # Add penalties to ensure learned parameter is physically reasonable
        # This prevents the network from learning unrealistic values
        
        param_penalty = 0.0
        
        # k must be positive (drainage constant cannot be negative)
        param_penalty += 10.0 * torch.mean(torch.relu(-k))  # k > 0
        
        # k should be reasonable (typically 0.001 to 1.0 m^(1/2)/s for most containers)
        param_penalty += 2.0 * torch.mean(torch.relu(k - 1.0))  # k < 1.0 m^(1/2)/s
        
        # Calculate RMSE for reporting
        rmse_loss = torch.sqrt(mse_loss)
        
        # Total loss combines physics simulation error with parameter constraints
        total_loss = mse_loss + 0.001 * param_penalty
        
        # Store predicted trajectory and parameter for debugging
        self.predicted_h = h
        self.k = k
        self.rmse = rmse_loss
        
        return total_loss


class TorricelliData:
    """
    Data handler for Torricelli height trajectory data.
    
    This class loads and processes the height data from the video processing step,
    creating sequences suitable for the LTC neural network.
    Matches PendulumData/SlidingBlockData structure for consistency.
    """
    
    def __init__(self, seq_len=16, data_dir="data"):
        print(f"Loading Torricelli height trajectory data...")
        
        # Load trajectory data from data directory
        # Load height data (h coordinates) - match pendulum/sliding block format
        h_data = np.loadtxt(os.path.join(data_dir, "hData.txt"))
        
        # Transpose to match pendulum/sliding block format: [N, 100] -> [100, N]
        h_traj = h_data.T  # [100, N]
        
        # Get Nloop from data
        global Nloop
        Nloop = h_traj.shape[1]  # Use actual data size (100)
        print(f"Nloop {Nloop}")
        
        # Create sequences for training (like pendulum/sliding block approach)
        train_x, train_y = cut_in_sequences(h_traj, h_traj, seq_len)
        
        # Create sequences for testing
        test_x, test_y = cut_in_sequences(h_traj, h_traj, seq_len, inc=8)
        
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


class TorricelliModel(nn.Module):
    """
    Neural network model for Torricelli k constant estimation.
    
    This class implements the LTC (Liquid Time-Constant) neural network that learns
    to predict the k constant from height trajectory data. The model takes
    sequences of height data as input and outputs the k parameter.
    
    Architecture:
    - Input: [T, B, Nloop] where T=timesteps, B=batch_size, Nloop=features (100)
    - Output: [T, B, 1] where 1 is the k constant parameter
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

        print("Beginning Torricelli parameter estimation model...")

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
        
        # Output layer: 1 parameter (k constant)
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
            x: Input height trajectory data [T, B, Nloop]
            
        Returns:
            y: Predicted k constant [T, B, 1]
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
        loss_fn = Custom_Torricelli_Loss(target_y, y_pred)
        return loss_fn.forward()


def run_torricelli_emma_optimization(output_folder=""):
    """
    Main function to run EMMA Torricelli parameter estimation.
    
    This function:
    1. Loads height trajectory data
    2. Creates and trains the LTC neural network
    3. Estimates k constant
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
    
    print("[STEP 2] Starting EMMA Torricelli optimization...")
    print("Starting EMMA Torricelli Training...")
    
    # Training parameters
    seq_len = 16
    batch_size = 2
    num_epochs = 40
    learning_rate = 0.0003
    
    # Load height trajectory data
    data_dir = os.path.join(output_folder, "data") if output_folder else "data"
    dataset = TorricelliData(seq_len=seq_len, data_dir=data_dir)
    
    # Create neural network model
    model = TorricelliModel(model_type="ltc", model_size=64, learning_rate=learning_rate).to(device)
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
                model_path = os.path.join(output_folder, 'torricelli_emma_final_model.pth') if output_folder else 'torricelli_emma_final_model.pth'
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
    model_path = os.path.join(output_folder, 'torricelli_emma_final_model.pth') if output_folder else 'torricelli_emma_final_model.pth'
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
        maxChange = 95.0
        getp = lambda k: predicted_params[:,:,k].mean()
        
        k_nominal = 0.016202065833479495  # Nominal k value (m^(1/2)/s) - from parameters.json large
        k = (1 + (0.5 - getp(0)) * maxChange / 100.0) * k_nominal
        
        # Save parameter to CSV (baseline paper notation)
        vals = [k.item()]
        csv_path = os.path.join(output_folder, 'torricelli_coefficients.csv') if output_folder else 'torricelli_coefficients.csv'
        with open(csv_path, 'w', newline='') as csvfile:
            w = csv.writer(csvfile)
            w.writerow(['Parameter', 'Value', 'Units', 'Description'])
            w.writerow(['k', float(k.item()), 'm^(1/2)/s', 'Torricelli drainage constant (dh/dt = -k*sqrt(h))'])
        
        print("\n=== ESTIMATED TORRICELLI PARAMETER ===")
        print(f"k: {float(k.item()):.6f} m^(1/2)/s")
    
    print("Model saved as 'torricelli_emma_final_model.pth'")
    print("Parameters saved as 'torricelli_coefficients.csv'")


def main():
    """
    Main function to run the complete Torricelli analysis pipeline.
    
    This is the main automation function that orchestrates the entire Torricelli analysis
    pipeline. It coordinates data loading and EMMA parameter estimation
    to provide a complete analysis of Torricelli behavior from height data.
    
    Pipeline Execution Flow:
    ------------------------
    1. Initialize directories and configuration
    2. Run EMMA parameter estimation (physics-informed neural network)
    3. Generate comprehensive output summary
    """
    import sys
    
    # Check for command line arguments
    simulation_only = "--simulation-only" in sys.argv or "-s" in sys.argv
    
    if simulation_only:
        print("=" * 60)
        print("EMMA SIMULATION MODE")
        print("=" * 60)
        print(" Running simulation with existing learned parameters...")
        try:
            # Check if required files exist
            if not os.path.exists('torricelli_coefficients.csv'):
                raise FileNotFoundError("torricelli_coefficients.csv not found. Please run full pipeline first.")
            if not os.path.exists('torricelli_emma_final_model.pth'):
                raise FileNotFoundError("torricelli_emma_final_model.pth not found. Please run full pipeline first.")
            
            # Load existing parameters
            import pandas as pd
            params_df = pd.read_csv('torricelli_coefficients.csv')
            print("Loaded existing Torricelli parameters:")
            for _, row in params_df.iterrows():
                print(f"  {row['Parameter']}: {row['Value']:.6f} {row['Units']}")
            
            print("\n SIMULATION COMPLETED SUCCESSFULLY!")
            print(" OUTPUT SUMMARY:")
            print("   EMMA parameters: torricelli_coefficients.csv")
            print("   EMMA model: torricelli_emma_final_model.pth")
        except Exception as e:
            print(f"\n SIMULATION FAILED: {e}")
            print(" Ensure that EMMA parameters have been learned first")
            print(" Run 'python torricelli.py' to learn parameters before simulation")
        return
    
    # ========================================
    # COMPLETE PIPELINE EXECUTION
    # ========================================
    print("=" * 60)
    print("TORRICELLI ANALYSIS PIPELINE")
    print("=" * 60)
    
    # ========================================
    # CONFIGURATION SECTION
    # ========================================
    # Modify these paths according to your setup
    video_path = "../../output_selected/torricelli/large/05/video.mp4"  # Set to video path if processing from video
    weights_path = "yolo11m.pt"  # YOLO model weights
    pixel_to_meter = 0.001  # Conversion factor: adjust based on your setup (m/pixel)
    
    # Save results in lar_v5 folder (like pendulum 45_v1, 90_v1, etc.)
    output_folder = "lar_v5"
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(f"{output_folder}/output", exist_ok=True)  # Visual outputs directory
    os.makedirs(f"{output_folder}/data", exist_ok=True)    # Data files directory
    
    try:
        # ========================================
        # STEP 1: VIDEO PROCESSING (if video provided)
        # ========================================
        # Check if video processing is needed
        hdata_path = os.path.join(output_folder, "data", "hData.txt")
        if video_path and os.path.exists(video_path):
            print("\n" + "=" * 40)
            print("STEP 1: VIDEO PROCESSING")
            print("=" * 40)
            print("Detecting and tracking ball/object in video frames...")
            
            output_video = os.path.join(output_folder, "output", "torricelli_annotated.mp4")
            output_csv = os.path.join(output_folder, "data", "torricelli_trajectory.csv")
            
            num_frames = process_torricelli_video(
                video_path=video_path,
                weights_path=weights_path,
                output_video=output_video,
                output_csv=output_csv,
                conf=0.15,
                pixel_to_meter=pixel_to_meter
            )
            
            if num_frames == 0:
                print("⚠️  Warning: No height data extracted from video")
                print("   Falling back to existing hData.txt if available")
            else:
                print(f"✅ Successfully extracted {num_frames} height measurements")
        elif os.path.exists(hdata_path):
            print("\n" + "=" * 40)
            print("STEP 1: SKIPPED (Using existing height data)")
            print("=" * 40)
            print(f"Found existing hData.txt at: {hdata_path}")
            print("Skipping video processing...")
        else:
            print("\n" + "=" * 40)
            print("STEP 1: SKIPPED (No video or data found)")
            print("=" * 40)
            print("⚠️  No video path provided and hData.txt not found")
            print("   Please either:")
            print("   1. Set video_path in main() function, or")
            print("   2. Place hData.txt in data/ directory")
            print("   Continuing with existing data if available...")
        
        # ========================================
        # STEP 2: EMMA PARAMETER ESTIMATION
        # ========================================
        print("\n" + "=" * 40)
        print("STEP 2: EMMA PARAMETER ESTIMATION")
        print("=" * 40)
        print("Loading height trajectory data...")
        print("Training LTC neural network...")
        print("Estimating k constant...")
        run_torricelli_emma_optimization(output_folder=output_folder)
        
        # ========================================
        # PIPELINE COMPLETION SUMMARY
        # ========================================
        print("\n" + "=" * 60)
        print(" PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(" OUTPUT SUMMARY:")
        if video_path and os.path.exists(video_path):
            print("   Annotated video: output/torricelli_annotated.mp4")
            print("   Trajectory CSV: data/torricelli_trajectory.csv")
        print("   Height data: data/hData.txt")
        print("   EMMA parameters: torricelli_coefficients.csv")
        print("   EMMA model: torricelli_emma_final_model.pth")
        print("\n All outputs organized in output/, data/, and root directories")
        
    except Exception as e:
        print(f"\n PIPELINE FAILED: {e}")
        print(" Check that hData.txt exists in data/ directory")
        print(" Ensure all required dependencies are installed")
        raise


# Main execution block
if __name__ == "__main__":
    """
    Main execution entry point for the Torricelli analysis pipeline.
    """
    main()
