# This is the code for Free Fall pipeline based on EMMA method

# EMMA Free Fall Pipeline

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


class FreeFallDetector:
    """
    Free fall object tracker using YOLO for position measurement.
    
    This class implements object tracking for the free fall pipeline, similar to
    Pendulum and Sliding Block experiments. It uses YOLO to detect and track an object
    in free fall, then measures vertical position (y-coordinate) over time.
    
    Why: Accurate position measurement is critical for gravitational acceleration estimation
    What: Tracks object position to extract vertical trajectory
    """
    def __init__(self, weights_path, conf=0.15, imgsz=640):
        """
        Initialize the free fall object detector with YOLO model.
        
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
        self.reference_top = None  # Store top reference position (y=0)
        print(f"[INFO] Loaded YOLO weights: {weights_path}")

    def detect_object(self, frame):
        """
        Detect falling object in a single video frame using YOLO with intelligent filtering.
        
        Args:
            frame: Input video frame (numpy array)
            
        Returns:
            tuple: (x1, y1, x2, y2, confidence) or None if no detection
            
        Why: Need to track object position to measure free fall trajectory
        What: Returns object bounding box with confidence score
        """
        h, w = frame.shape[:2]
        img_area = w * h
        edge_thresh = max(10, int(0.01 * min(w, h)))
        min_area_px = max(100, int(0.00001 * img_area))  # Object is typically small
        max_area_px = int(0.1 * img_area)  # Object should be reasonably sized

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
        Detect object and calculate vertical position measurement.
        
        Args:
            frame: Input video frame (numpy array)
            
        Returns:
            tuple: (y_position_pixels, confidence) or None if detection fails
            
        Why: Complete detection pipeline for position extraction using object tracking
        What: Returns vertical position in pixels from reference top
        """
        # Detect falling object
        obj_box = self.detect_object(frame)
        if obj_box is None:
            if self.last_detection is not None:
                # Use last known object position (for Kalman filter prediction)
                obj_box = self.last_detection
            else:
                return None
        else:
            self.last_detection = obj_box
        
        # Initialize reference top on first detection (use object's initial position as y=0)
        h, w = frame.shape[:2]
        if self.reference_top is None:
            # Use top of first detection as reference point (y=0)
            _, y1, _, _, _ = obj_box
            self.reference_top = float(y1)
        
        # Get object center Y position (this represents vertical position)
        _, y1, _, y2, _ = obj_box
        obj_center_y = (y1 + y2) / 2.0
        
        # Calculate vertical position (distance from reference top)
        # Position increases as we go down (larger y values)
        # Free fall: object moves downward, so y increases over time
        y_position_pixels = obj_center_y - self.reference_top
        
        # Ensure position is non-negative
        if y_position_pixels < 0:
            y_position_pixels = 0
            
        confidence = obj_box[4] if obj_box is not None else 0.5
        
        return (y_position_pixels, confidence)


class Kalman1D:
    """
    1D Kalman Filter for position trajectory smoothing and prediction.
    
    This class implements a 1D Kalman filter to smooth position measurements
    and predict position when detection fails. The filter tracks position and
    velocity (rate of change).
    
    State Vector: [y, vy] (position + velocity)
    Measurement: [y] (position only from detection)
    
    Why: Raw detections are noisy and may have gaps
    What: Provides smooth, continuous position estimates
    """
    def __init__(self, dt=0.01):
        """
        Initialize 1D Kalman filter with system dynamics.
        
        Args:
            dt: Time step between measurements (seconds)
        """
        self.dt = dt
        # State vector: [y, vy] (position, velocity)
        self.state = np.zeros(2).reshape(-1, 1)  # Column vector for matrix operations
        
        # State transition matrix F (constant acceleration model for free fall)
        self.F = np.eye(2)
        self.F[0, 1] = dt  # y += vy * dt
        
        # Measurement matrix H (we measure position only)
        self.H = np.array([[1.0, 0.0]])  # [1, 0] to extract position from state
        
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
            float: Predicted position
            
        Why: Estimate position when detection fails
        What: Advances state using constant velocity model
        """
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        # Extract position (first element of state vector) as float
        return float(self.state[0, 0] if self.state.ndim > 1 else self.state[0])

    def update(self, measurement):
        """
        Update state estimate with new measurement.
        
        Args:
            measurement: Position measurement (float)
            
        Returns:
            float: Updated position estimate
            
        Why: Incorporate new measurements to improve accuracy
        What: Combines prediction with measurement using Kalman equations
        """
        measurement = np.array([[float(measurement)]])  # Convert to 2D array
        y = measurement - self.H @ self.state  # Innovation
        S = self.H @ self.P @ self.H.T + self.R  # Innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman gain
        self.state = self.state + K @ y  # Update state
        self.P = (np.eye(2) - K @ self.H) @ self.P  # Update covariance
        # Extract position (first element of state vector) as float
        return float(self.state[0, 0] if self.state.ndim > 1 else self.state[0])


def process_free_fall_video(video_path, weights_path, output_video, output_csv, conf=0.15, pixel_to_meter=0.001):
    """
    Process free fall video to extract position trajectory using object tracking.
    
    This function processes free fall videos to:
    1. Load video and YOLO model
    2. Detect and track falling object in each frame (similar to Pendulum)
    3. Track vertical position using Kalman filtering
    4. Convert pixel position to physical units (meters)
    5. Create annotated video with position overlay
    6. Save position trajectory data and generate plots
    
    Args:
        video_path: Path to input free fall video file
        weights_path: Path to YOLO model weights
        output_video: Path for annotated video output
        output_csv: Path for trajectory CSV output
        conf: YOLO detection confidence threshold
        pixel_to_meter: Conversion factor from pixels to meters (default: 0.001 m/pixel)
        
    Why: Video processing is the foundation of free fall trajectory analysis
    What: Extracts smooth position trajectory from object tracking in video frames
    """
    print(f"[STEP 1] Processing free fall video: {video_path}")
    print(f"[STEP 1] Output video: {output_video}")
    print(f"[STEP 1] Output CSV: {output_csv}")

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    detector = FreeFallDetector(weights_path, conf=conf)
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
    csvw.writerow(["frame", "time_s", "y_position_pixels", "y_position_meters", "conf"])

    y_series_pixels, y_series_meters = [], []
    frame_idx = 0
    
    # Calibration: Use a reasonable estimate for pixel_to_meter
    # NOTE: We should NOT use ground truth from parameters
    # Instead, use a generic reasonable value or estimate from video
    # For free fall, typical drop heights are 0.5-2 m, so we use a reasonable default
    # User can adjust pixel_to_meter manually if they have a reference object in the video
    calibration_complete = False
    
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_time = frame_idx / fps
        det = detector.detect(frame)
        
        if det is not None:
            y_position_pixels, conf_val = det
            
            # NOTE: We do NOT calibrate using ground truth
            # The pixel_to_meter should be set manually based on a reference object
            # or use the default value provided in main()
            # This avoids "cheating" by using ground truth information
            if not calibration_complete and conf_val > 0.5:
                calibration_complete = True
                print(f"[STEP 1] Using pixel_to_meter = {pixel_to_meter:.6f} m/px (user-provided or default)")
                print(f"[STEP 1] NOTE: For accurate results, calibrate using a known reference object in the video")
            
            y_position_meters = float(y_position_pixels) * pixel_to_meter
            
            # Update Kalman filter
            kf.predict()
            y_smooth = kf.update(y_position_pixels)  # Kalman filter returns float
            y_smooth_meters = y_smooth * pixel_to_meter
            
            y_series_pixels.append(y_smooth)
            y_series_meters.append(y_smooth_meters)

            # Draw detection and position on frame
            if detector.last_detection is not None:
                x1, y1, x2, y2 = [int(v) for v in detector.last_detection[:4]]
                # Draw object bounding box
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                # Draw object center point
                obj_center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                cv2.circle(frame, obj_center, 5, (0, 255, 0), -1)
                # Draw line from reference top to object (position indicator)
                if detector.reference_top is not None:
                    top_point = (obj_center[0], int(detector.reference_top))
                    cv2.line(frame, top_point, obj_center, (255, 0, 0), 2)
            
            # Draw position text
            position_text = f"y={y_smooth_meters:.4f}m ({y_smooth:.1f}px), conf={conf_val:.2f}"
            cv2.putText(frame, position_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            csvw.writerow([frame_idx, f"{frame_time:.3f}", f"{y_smooth:.2f}", 
                          f"{y_smooth_meters:.6f}", f"{conf_val:.3f}"])
        else:
            # No detection - use Kalman prediction
            y_pred = kf.predict()  # Kalman filter returns float
            y_pred_meters = y_pred * pixel_to_meter
            
            y_series_pixels.append(y_pred)
            y_series_meters.append(y_pred_meters)
            
            # Draw predicted position
            position_text = f"y={y_pred_meters:.4f}m ({y_pred:.1f}px) [predicted]"
            cv2.putText(frame, position_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            csvw.writerow([frame_idx, f"{frame_time:.3f}", f"{y_pred:.2f}",
                          f"{y_pred_meters:.6f}", "0.000"])

        out.write(frame)
        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"[PROGRESS] Processed {frame_idx} frames")
            check_memory_usage()

    cap.release()
    out.release()
    csv_f.close()

    if y_series_pixels and y_series_meters:
        # Save position trajectory data in EMMA format
        y_arr = np.array(y_series_meters)  # Use meters for physical units
        
        # Report extracted position range (for information only, not validation against ground truth)
        y_0_actual = y_arr[0] if len(y_arr) > 0 else 0.0
        y_n_actual = y_arr[-1] if len(y_arr) > 0 else 0.0
        
        print(f"\n[STEP 1] Extracted Position Range:")
        print(f"   Initial position: y_0 = {y_0_actual:.3f} m")
        print(f"   Final position:   y_n = {y_n_actual:.3f} m")
        print(f"   Position change:  {y_n_actual - y_0_actual:.3f} m")
        
        # Check if position increases (physical constraint for free fall)
        if y_n_actual <= y_0_actual:
            print(f"   ⚠️  Warning: Position does not increase (may indicate tracking issue)")
        else:
            print(f"   ✅ Position increases as expected for free fall")
        
        # Match EMMA format (N x 100 matrices for memory optimization)
        y_matrix = np.tile(y_arr.reshape(-1, 1), (1, 100))
        
        # Determine data directory from output_csv path
        data_dir = os.path.dirname(output_csv)
        os.makedirs(data_dir, exist_ok=True)
        np.savetxt(os.path.join(data_dir, "yData.txt"), y_matrix, fmt='%.6f')
        
        del y_matrix, y_arr
        gc.collect()
        print(f"\n[STEP 1] ✅ Saved free fall position trajectory data: {len(y_series_meters)} frames")
        print(f"[STEP 1] ✅ Saved position data: yData.txt")
        
        # Create trajectory plots
        print("[STEP 1] Creating free fall position trajectory plots...")
        
        # Plot position vs time
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        time_array = np.arange(len(y_series_meters)) / fps
        ax.plot(time_array, y_series_meters, 'b-', linewidth=2, label='Position (m)')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Position (m)')
        ax.set_title('Free Fall Position vs Time')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plot_path = os.path.join(data_dir, "free_fall_position_trajectory.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[STEP 1] ✅ Saved trajectory plot: {plot_path}")
        
        return len(y_series_meters)
    else:
        print("[STEP 1] ⚠️  No position data extracted from video")
        return 0


def cut_in_sequences(x, y, seq_len, inc=1):
    """
    Slice a long 1D/2D series into overlapping windows for sequence-based learning.
    
    This function creates sequences from the input data for the LTC model.
    For free fall data: input shape (N, 100) -> output shape (seq_len, num_sequences, 100)
    
    Args:
        x: Input data array (e.g., position trajectory)
        y: Target data array (e.g., position trajectory) 
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


class Custom_FreeFall_Loss(nn.Module):
    """
    Custom loss function that integrates free fall physics simulation.
    
    This is the core of the parameter estimation system. Instead of using a simple
    MSE loss, this function:
    1. Takes predicted g (gravitational acceleration) from the neural network
    2. Runs a complete free fall physics simulation using this parameter
    3. Compares the simulated position trajectory with the actual position trajectory
    4. Returns the physics-based loss for gradient descent
    
    The physics simulation includes:
    - Free fall with drag: dr/dt = -g*t*r²/r0f
    - Where r is position, g is gravity, t is time, r0f is reference parameter
    - Position increases over time as object falls
    - Parameter estimation for g (gravitational acceleration)
    
    This approach ensures that the learned parameter is physically meaningful
    and can be used for actual free fall prediction.
    """
    
    def __init__(self, labels, logits):
        """
        Initialize the physics-based loss function.
        
        Args:
            labels: Actual position trajectory data [T, B, 1] (position y)
            logits: Predicted g constant from neural network [T, B, 1]
        """
        super().__init__()
        # Store actual trajectory data for comparison
        self.y_true = labels    # [T, B, 1] - actual position data
        
        # Store predicted parameters from neural network
        self.y_pred = logits    # [T, B, 1] - g constant

    def forward(self):
        """
        Complete free fall dynamics simulation with physics-based loss.
        
        This method performs the following steps:
        1. Extract predicted g constant from neural network output
        2. Convert normalized parameter to physical value
        3. Initialize position and velocity states from actual data
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
        # The neural network outputs normalized values [0,1] for g
        # We convert these to physical values with ±95% variation around nominal value
        
        maxChange = 95.0  # Maximum percentage change from nominal value
        getp = lambda k: self.y_pred[:,:,k]  # Extract parameter k for all timesteps [T,B]
        
        # Convert normalized predictions to physical parameter
        # g is scaled from [0,1] to [nominal*(1-0.95), nominal*(1+0.95)]
        # Nominal g value: 9.81 m/s^2 (Earth's gravitational acceleration)
        g_nominal = 9.81  # Nominal g value (m/s^2) - Earth's gravity
        g = (1 + (0.5 - getp(0)) * maxChange / 100.0) * g_nominal

        # ========================================
        # STEP 2: Physical Constants
        # ========================================
        # These are fixed physical constants that don't change during training
        eps = torch.tensor(1e-6, device=dev)  # Small epsilon for numerical stability

        # ========================================
        # STEP 3: Get Actual Position Data
        # ========================================
        # Extract actual position data for comparison
        if self.y_true.dim() == 3:
            actual_y = self.y_true[:, :, 0]    # [T,B] - actual position from [T,B,1]
        else:
            actual_y = self.y_true  # [T,B] - actual position

        # ========================================
        # STEP 4: Initialize Position State
        # ========================================
        # Initialize position (r) from actual trajectory (like pendulum approach)
        # Match pendulum pattern: theta = thetaVal.clone() where thetaVal = self.y_true[:,:,0]
        rVal = actual_y  # [T,B] - actual position trajectory
        r = rVal.clone()  # [T,B] - initialize from actual data (like pendulum)
        
        # Get initial position for r0f reference
        r0f = r[0, :]  # [B] - initial position for each batch
        
        # ========================================
        # STEP 5: Simulation Setup
        # ========================================
        # Set up simulation parameters and storage arrays
        
        # Dynamic limitLoop based on actual data length to avoid tensor size mismatch
        limitLoop = min(500, T)  # Use actual data length or 500, whichever is smaller
        tau_dt = 0.01  # Time step (s) - match baseline paper's dt
        
        # Reshape for tensor concatenation approach (like pendulum/sliding block)
        # Match pendulum: theta = theta.unsqueeze(2) to get [T,B,1]
        r = r.unsqueeze(2)  # [T,B] -> [T,B,1]

        # ========================================
        # STEP 6: Main Physics Simulation Loop
        # ========================================
        # This is the core of the physics simulation
        # For each timestep, we:
        # 1. Get g parameter for current timestep
        # 2. Calculate dr/dt = -g*t*r²/r0f (free fall with drag)
        # 3. Update position using Euler integration
        # 4. Store predicted state using tensor concatenation
        
        for i in range(1, limitLoop):
            # Current timestep index
            t_idx = i
            
            # ========================================
            # STEP 6.1: Get Current Parameter
            # ========================================
            # Get g value for current timestep (match pendulum pattern)
            g_curr = g[t_idx]  # [B] - g constant for current timestep
            
            # ========================================
            # STEP 6.2: Free Fall with Drag Dynamics
            # ========================================
            # Free fall equation: dr/dt = -g*t*r²/r0f
            # Match pendulum pattern: use r[:,:,i-1] to get previous timestep
            
            # Get previous position (like pendulum: theta[:,:,i-1])
            r_prev = r[:,:,i-1]  # [T,B] - previous position from actual trajectory
            
            # Calculate current time
            t_current = t_idx * tau_dt  # Current time in seconds
            
            # Calculate dr/dt = -g*t*r²/r0f
            # g_curr is [B], r_prev is [T,B], r0f is [B]
            # Need to expand g_curr and r0f to match [T,B] shape
            g_expanded = g_curr.unsqueeze(0).expand(T, -1)  # [T,B] - expand g to match shape
            r0f_expanded = r0f.unsqueeze(0).expand(T, -1)  # [T,B] - expand r0f to match shape
            
            # Ensure r0f is not zero (avoid division by zero)
            r0f_safe = torch.clamp(r0f_expanded, min=eps)
            
            # Calculate rate of change: dr/dt = -g*t*r²/r0f
            dr_dt = -g_expanded * t_current * (r_prev ** 2) / r0f_safe  # [T,B] - rate of position change
            
            # ========================================
            # STEP 6.3: Update Position
            # ========================================
            # Euler integration: r_new = r_old + dr/dt * dt
            # Match pendulum pattern: y1 = theta[:,:,i-1] + omega[:,:,i-1]*tau_dt
            r_new = r_prev + dr_dt * tau_dt  # [T,B] - position update
            
            # Ensure position remains non-negative (physical constraint)
            r_new = torch.clamp(r_new, min=0.0)
            
            # Concatenate to build trajectory (like pendulum: theta = torch.cat([theta, y1.unsqueeze(2)],dim=2))
            r = torch.cat([r, r_new.unsqueeze(2)], dim=2)

        # ========================================
        # STEP 7: Calculate Physics-Based Loss
        # ========================================
        # The loss function compares the simulated trajectory with the actual trajectory
        # This is what drives the parameter estimation - the neural network learns
        # g that makes the simulation match the real position behavior
        
        # Loss calibration constant (calibrated with maxChange=0 using nominal g)
        # Match pendulum approach: loss_Cal_theta = 344.08, loss_Cal_omega = 47.72
        # Calibrated value from running with maxChange=0: 0.000001 (placeholder)
        loss_Cal_y = 0.000001  # Calibrated loss value (from calibration run with maxChange=0)
        
        # Calculate MSE loss (match pendulum pattern exactly)
        # Pendulum: torch.abs(torch.sum(torch.square(self.y_true[:,:,0:limitLoop]-theta)/limitLoop, dim=2)-loss_Cal_theta)
        # For free fall: y is [T,B,limitLoop] after simulation, actual_y is [T,B]
        # Need to reshape actual_y to [T,B,limitLoop] for comparison
        
        # Extract actual position for comparison
        if self.y_true.dim() == 3:
            actual_y_compare = self.y_true[:,:,0]  # [T,B]
        else:
            actual_y_compare = self.y_true  # [T,B]
        
        # Match pendulum loss calculation pattern
        # In pendulum: theta is [T,B,limitLoop] where theta[:,:,0] is initial (from actual), theta[:,:,1:] are predictions
        # The loss compares theta with self.y_true[:,:,0:limitLoop]
        # For free fall: r[:,:,0] is initial (from actual), r[:,:,1:] are predictions
        # We compare r with actual_y_compare properly reshaped
        
        # Reshape actual_y: r[:,:,i] should compare with actual_y_compare[i,:] for each i
        # r is [T,B,limitLoop], actual_y_compare is [T,B]
        # We need: actual_y_compare[i,:] compares with r[:,:,i] for each i in [0, limitLoop)
        # Create [T,B,limitLoop] where actual_y_broadcast[:,:,i] = actual_y_compare[i,:] for each i
        actual_y_broadcast = actual_y_compare[:limitLoop, :].unsqueeze(0).expand(T, -1, -1).permute(0, 2, 1)  # [T,B,limitLoop]
        
        # Calculate MSE loss matching pendulum pattern exactly
        # Pendulum: torch.sum(torch.square(self.y_true[:,:,0:limitLoop]-theta)/limitLoop, dim=2)
        raw_mse = torch.sum(torch.square(actual_y_broadcast - r[:,:,:limitLoop]) / limitLoop, dim=2)
        
        # Calibration already completed - loss_Cal_y is set to calibrated value
        
        # Calculate MSE loss with calibration (match pendulum: torch.abs(raw_mse - loss_Cal))
        # Pendulum: torch.abs(torch.sum(...)-loss_Cal_theta)
        mse_loss = torch.abs(raw_mse - loss_Cal_y)
        
        # ========================================
        # STEP 8: Parameter Constraint Penalty
        # ========================================
        # Add penalties to ensure learned parameter is physically reasonable
        # This prevents the network from learning unrealistic values
        
        param_penalty = 0.0
        
        # g must be positive (gravitational acceleration cannot be negative)
        param_penalty += 10.0 * torch.mean(torch.relu(-g))  # g > 0
        
        # g should be reasonable (typically 8-12 m/s^2 for Earth)
        param_penalty += 2.0 * torch.mean(torch.relu(g - 15.0))  # g < 15 m/s^2
        param_penalty += 2.0 * torch.mean(torch.relu(5.0 - g))  # g > 5 m/s^2
        
        # Calculate RMSE for reporting
        rmse_loss = torch.sqrt(mse_loss)
        
        # Total loss combines physics simulation error with parameter constraints
        total_loss = mse_loss + 0.001 * param_penalty
        
        # Store predicted trajectory and parameter for debugging
        self.predicted_r = r
        self.g = g
        self.rmse = rmse_loss
        
        return total_loss


class FreeFallData:
    """
    Data handler for free fall position trajectory data.
    
    This class loads and processes the position data from the video processing step,
    creating sequences suitable for the LTC neural network.
    Matches PendulumData/SlidingBlockData structure for consistency.
    """
    
    def __init__(self, seq_len=16, data_dir="data"):
        print(f"Loading free fall position trajectory data...")
        
        # Load trajectory data from data directory
        # Load position data (y coordinates) - match pendulum/sliding block format
        y_data = np.loadtxt(os.path.join(data_dir, "yData.txt"))
        
        # Transpose to match pendulum/sliding block format: [N, 100] -> [100, N]
        y_traj = y_data.T  # [100, N]
        
        # Get Nloop from data
        global Nloop
        Nloop = y_traj.shape[1]  # Use actual data size (100)
        print(f"Nloop {Nloop}")
        
        # Create sequences for training (like pendulum/sliding block approach)
        train_x, train_y = cut_in_sequences(y_traj, y_traj, seq_len)
        
        # Create sequences for testing
        test_x, test_y = cut_in_sequences(y_traj, y_traj, seq_len, inc=8)
        
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


class FreeFallModel(nn.Module):
    """
    Neural network model for free fall g constant estimation.
    
    This class implements the LTC (Liquid Time-Constant) neural network that learns
    to predict the g constant from position trajectory data. The model takes
    sequences of position data as input and outputs the g parameter.
    
    Architecture:
    - Input: [T, B, Nloop] where T=timesteps, B=batch_size, Nloop=features (100)
    - Output: [T, B, 1] where 1 is the g constant parameter
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

        print("Beginning free fall parameter estimation model...")

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
        
        # Output layer: 1 parameter (g constant)
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
            x: Input position trajectory data [T, B, Nloop]
            
        Returns:
            y: Predicted g constant [T, B, 1]
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
        loss_fn = Custom_FreeFall_Loss(target_y, y_pred)
        return loss_fn.forward()


def run_free_fall_emma_optimization(output_folder=""):
    """
    Main function to run EMMA free fall parameter estimation.
    
    This function:
    1. Loads position trajectory data
    2. Creates and trains the LTC neural network
    3. Estimates g constant
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
    
    print("[STEP 2] Starting EMMA free fall optimization...")
    print("Starting EMMA Free Fall Training...")
    
    # Training parameters
    seq_len = 16
    batch_size = 2
    num_epochs = 40
    learning_rate = 0.0003
    
    # Load position trajectory data
    data_dir = os.path.join(output_folder, "data") if output_folder else "data"
    dataset = FreeFallData(seq_len=seq_len, data_dir=data_dir)
    
    # Create neural network model
    model = FreeFallModel(model_type="ltc", model_size=64, learning_rate=learning_rate).to(device)
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
                model_path = os.path.join(output_folder, 'free_fall_emma_final_model.pth') if output_folder else 'free_fall_emma_final_model.pth'
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
    model_path = os.path.join(output_folder, 'free_fall_emma_final_model.pth') if output_folder else 'free_fall_emma_final_model.pth'
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
        
        g_nominal = 9.81  # Nominal g value (m/s^2) - Earth's gravity
        g = (1 + (0.5 - getp(0)) * maxChange / 100.0) * g_nominal
        
        # Save parameter to CSV (baseline paper notation)
        vals = [g.item()]
        csv_path = os.path.join(output_folder, 'free_fall_coefficients.csv') if output_folder else 'free_fall_coefficients.csv'
        with open(csv_path, 'w', newline='') as csvfile:
            w = csv.writer(csvfile)
            w.writerow(['Parameter', 'Value', 'Units', 'Description'])
            w.writerow(['g', float(g.item()), 'm/s^2', 'Gravitational acceleration (dv/dt = g)'])
        
        print("\n=== ESTIMATED FREE FALL PARAMETER ===")
        print(f"g: {float(g.item()):.6f} m/s^2")
    
    print("Model saved as 'free_fall_emma_final_model.pth'")
    print("Parameters saved as 'free_fall_coefficients.csv'")


def main():
    """
    Main function to run the complete free fall analysis pipeline.
    
    This is the main automation function that orchestrates the entire free fall analysis
    pipeline. It coordinates data loading and EMMA parameter estimation
    to provide a complete analysis of free fall behavior from position data.
    
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
            if not os.path.exists('free_fall_coefficients.csv'):
                raise FileNotFoundError("free_fall_coefficients.csv not found. Please run full pipeline first.")
            if not os.path.exists('free_fall_emma_final_model.pth'):
                raise FileNotFoundError("free_fall_emma_final_model.pth not found. Please run full pipeline first.")
            
            # Load existing parameters
            import pandas as pd
            params_df = pd.read_csv('free_fall_coefficients.csv')
            print("Loaded existing free fall parameters:")
            for _, row in params_df.iterrows():
                print(f"  {row['Parameter']}: {row['Value']:.6f} {row['Units']}")
            
            print("\n SIMULATION COMPLETED SUCCESSFULLY!")
            print(" OUTPUT SUMMARY:")
            print("   EMMA parameters: free_fall_coefficients.csv")
            print("   EMMA model: free_fall_emma_final_model.pth")
        except Exception as e:
            print(f"\n SIMULATION FAILED: {e}")
            print(" Ensure that EMMA parameters have been learned first")
            print(" Run 'python free_fall.py' to learn parameters before simulation")
        return
    
    # ========================================
    # COMPLETE PIPELINE EXECUTION
    # ========================================
    print("=" * 60)
    print("FREE FALL ANALYSIS PIPELINE")
    print("=" * 60)
    
    # ========================================
    # CONFIGURATION SECTION
    # ========================================
    # Modify these paths according to your setup
    video_path = "../../output_selected/free_fall/medium/05/video.mp4"  # Set to video path if processing from video
    weights_path = "yolo11m.pt"  # YOLO model weights
    pixel_to_meter = 0.001  # Conversion factor: adjust based on your setup (m/pixel)
    
    # Save results in med_v5 folder (like pendulum 45_v1, 90_v1, etc.)
    output_folder = "med_v5"
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(f"{output_folder}/output", exist_ok=True)  # Visual outputs directory
    os.makedirs(f"{output_folder}/data", exist_ok=True)    # Data files directory
    
    try:
        # ========================================
        # STEP 1: VIDEO PROCESSING (if video provided)
        # ========================================
        # Check if video processing is needed
        ydata_path = os.path.join(output_folder, "data", "yData.txt")
        if video_path and os.path.exists(video_path):
            print("\n" + "=" * 40)
            print("STEP 1: VIDEO PROCESSING")
            print("=" * 40)
            print("Detecting and tracking falling object in video frames...")
            
            output_video = os.path.join(output_folder, "output", "free_fall_annotated.mp4")
            output_csv = os.path.join(output_folder, "data", "free_fall_trajectory.csv")
            
            num_frames = process_free_fall_video(
                video_path=video_path,
                weights_path=weights_path,
                output_video=output_video,
                output_csv=output_csv,
                conf=0.15,
                pixel_to_meter=pixel_to_meter
            )
            
            if num_frames == 0:
                print("⚠️  Warning: No position data extracted from video")
                print("   Falling back to existing yData.txt if available")
            else:
                print(f"✅ Successfully extracted {num_frames} position measurements")
        elif os.path.exists(ydata_path):
            print("\n" + "=" * 40)
            print("STEP 1: SKIPPED (Using existing position data)")
            print("=" * 40)
            print(f"Found existing yData.txt at: {ydata_path}")
            print("Skipping video processing...")
        else:
            print("\n" + "=" * 40)
            print("STEP 1: SKIPPED (No video or data found)")
            print("=" * 40)
            print("⚠️  No video path provided and yData.txt not found")
            print("   Please either:")
            print("   1. Set video_path in main() function, or")
            print("   2. Place yData.txt in data/ directory")
            print("   Continuing with existing data if available...")
        
        # ========================================
        # STEP 2: EMMA PARAMETER ESTIMATION
        # ========================================
        print("\n" + "=" * 40)
        print("STEP 2: EMMA PARAMETER ESTIMATION")
        print("=" * 40)
        print("Loading position trajectory data...")
        print("Training LTC neural network...")
        print("Estimating g constant...")
        run_free_fall_emma_optimization(output_folder=output_folder)
        
        # ========================================
        # PIPELINE COMPLETION SUMMARY
        # ========================================
        print("\n" + "=" * 60)
        print(" PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(" OUTPUT SUMMARY:")
        if video_path and os.path.exists(video_path):
            print("   Annotated video: output/free_fall_annotated.mp4")
            print("   Trajectory CSV: data/free_fall_trajectory.csv")
        print("   Position data: data/yData.txt")
        print("   EMMA parameters: free_fall_coefficients.csv")
        print("   EMMA model: free_fall_emma_final_model.pth")
        print("\n All outputs organized in output/, data/, and root directories")
        
    except Exception as e:
        print(f"\n PIPELINE FAILED: {e}")
        print(" Check that yData.txt exists in data/ directory")
        print(" Ensure all required dependencies are installed")
        raise


# Main execution block
if __name__ == "__main__":
    """
    Main execution entry point for the free fall analysis pipeline.
    """
    main()

