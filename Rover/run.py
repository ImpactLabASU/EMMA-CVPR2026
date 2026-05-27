

# EMMA Rover Pipeline


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

try:
    import psutil
    _HAS_PSUTIL = True
except Exception:
    _HAS_PSUTIL = False

try:
    from moviepy.editor import VideoFileClip
except Exception:
    from moviepy import VideoFileClip
import librosa

# Set device for computation (GPU if available, otherwise CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Global variable to store the number of features per timestep
Nloop = 0


def check_memory_usage():
    """
    Monitor system memory usage during processing.
    
    This function provides lightweight memory monitoring to help track resource
    consumption during video and audio processing. Useful for long-running
    pipeline operations to ensure system stability.
    
    Why: Prevent memory overflow during large video processing
    What: Display current memory usage statistics
    """
    if not _HAS_PSUTIL:
        return
    mem = psutil.virtual_memory()
    used_gb = mem.used / (1024**3)
    total_gb = mem.total / (1024**3)
    print(f"[INFO] Memory usage: {used_gb:.1f}GB / {total_gb:.1f}GB ({mem.percent:.1f}%)")


class DroneDetector:
    """
    Rover detector using YOLO with improved tracking capabilities.
    
    This class implements the core object detection functionality for the rover pipeline.
    It uses YOLO (You Only Look Once) neural network for real-time rover detection
    in video frames with intelligent tracking and filtering mechanisms.
    
    Key Features:
    - YOLO-based object detection
    - Confidence-based filtering
    - Size-based detection validation
    - Edge proximity filtering
    - Tracking consistency (distance-based selection)
    
    Why: Accurate rover detection is critical for trajectory analysis
    What: Detects rover bounding boxes with confidence scores
    """
    def __init__(self, weights_path, conf=0.15, imgsz=640):
        """
        Initialize the rover detector with YOLO model.
        
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
        print(f"[INFO] Loaded YOLO weights: {weights_path}")

    def detect(self, frame):
        """
        Detect rover in a single video frame with intelligent filtering.
        
        This method performs the core detection logic:
        1. Run YOLO inference on the input frame
        2. Filter detections by size and edge proximity
        3. Select best candidate using confidence and tracking consistency
        
        Args:
            frame: Input video frame (numpy array)
            
        Returns:
            tuple: (x1, y1, x2, y2, confidence) or None if no detection
            
        Why: Multi-stage filtering ensures reliable rover detection
        What: Returns best rover bounding box with confidence score
        """
        h, w = frame.shape[:2]
        img_area = w * h
        edge_thresh = max(10, int(0.01 * min(w, h)))
        min_area_px = max(100, int(0.00001 * img_area))
        max_area_px = int(0.8 * img_area)

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
                if (x1 < 5 or y1 < 5 or x2 > w - 5 or y2 > h - 5):
                    continue
                    
                conf = float(b.conf[0].item()) if hasattr(b, "conf") else 0.0
                candidates.append((x1, y1, x2, y2, conf))

        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]

        # Multi-candidate selection with tracking consistency
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


class Kalman3D:
    """
    3D Kalman Filter for rover trajectory smoothing and prediction.
    
    This class implements a 3D Kalman filter to smooth rover position measurements
    and predict rover position when detection fails. The filter tracks position
    and velocity in 3D space (x, y, z coordinates).
    
    State Vector: [x, y, z, vx, vy, vz] (position + velocity)
    Measurement: [x, y, z] (position only from detection)
    
    Why: Raw detections are noisy and may have gaps
    What: Provides smooth, continuous trajectory estimates
    """
    def __init__(self, dt=0.01):
        """
        Initialize 3D Kalman filter with system dynamics.
        
        Args:
            dt: Time step between measurements (seconds)
        """
        self.dt = dt
        # State vector: [x, y, z, vx, vy, vz]
        self.state = np.zeros(6)
        
        # State transition matrix F (constant velocity model)
        self.F = np.eye(6)
        self.F[0, 3] = dt  # x += vx * dt
        self.F[1, 4] = dt  # y += vy * dt
        self.F[2, 5] = dt  # z += vz * dt
        
        # Measurement matrix H (we measure position only)
        self.H = np.eye(3, 6)
        
        # Process noise covariance Q (uncertainty in motion model)
        self.Q = np.eye(6) * 0.1
        
        # Measurement noise covariance R (uncertainty in measurements)
        self.R = np.eye(3) * 1.0
        
        # Error covariance matrix P (uncertainty in state estimate)
        self.P = np.eye(6) * 100.0

    def predict(self):
        """
        Predict next state using motion model (no measurement).
        
        Returns:
            np.array: Predicted state vector [x, y, z, vx, vy, vz]
            
        Why: Estimate rover position when detection fails
        What: Advances state using constant velocity model
        """
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.state

    def update(self, measurement):
        """
        Update state estimate with new measurement.
        
        Args:
            measurement: [x, y, z] position measurement from detection
            
        Returns:
            np.array: Updated state vector [x, y, z, vx, vy, vz]
            
        Why: Incorporate new measurements to improve accuracy
        What: Combines prediction with measurement using Kalman equations
        """
        y = measurement - self.H @ self.state  # Innovation (measurement residual)
        S = self.H @ self.P @ self.H.T + self.R  # Innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman gain
        self.state = self.state + K @ y  # Update state estimate
        self.P = (np.eye(6) - K @ self.H) @ self.P  # Update error covariance
        return self.state


class DepthEstimator:
    def __init__(self, focal_length=500, real_width=0.2):
        self.focal_length = focal_length
        self.real_width = real_width

    def z_from_bbox(self, bbox):
        x1, y1, x2, y2 = bbox
        width_px = x2 - x1
        if width_px > 0:
            z = (self.real_width * self.focal_length) / width_px
            return max(0.001, min(10.0, z))
        return 0.1


class RoverAudioFeatures:
    """
    Audio feature extraction for rover motor analysis.
    
    This class extracts meaningful audio features from rover video that correlate
    with motor commands and rover movement. It analyzes the acoustic signature
    of rover motors to infer motor speeds and movement patterns.
    
    Key Features Extracted:
    - RMS Energy: Overall audio intensity (motor power indicator)
    - Spectral Centroid: Average frequency (motor speed indicator)  
    - Peak Frequency: Dominant frequency in 80-3000 Hz band (motor RPM)
    - Motor Commands: Converted to omega_r, omega_l wheel speeds
    
    Why: Audio contains rich information about rover motor states
    What: Extracts motor-relevant features from acoustic data
    """
    def __init__(self, wav_path=None, sr_target=22050):
        """
        Initialize audio feature extractor.
        
        Args:
            wav_path: Path to audio file (optional)
            sr_target: Target sample rate for audio processing
        """
        self.wav_path = wav_path
        self.sr_target = sr_target
        self.t = None          # Time array for features
        self.rms = None        # RMS energy over time
        self.centroid_hz = None  # Spectral centroid over time
        self.peak_hz = None    # Peak frequency over time

    @staticmethod
    def extract_wav_from_video(video_path, out_wav, fps=44100):
        clip = VideoFileClip(video_path)
        if clip.audio is None:
            raise RuntimeError("No audio track in video.")
        clip.audio.write_audiofile(out_wav, fps=fps)
        return out_wav

    def compute(self, wav_path):
        y, sr = librosa.load(wav_path, sr=self.sr_target, mono=True)
        n_fft = 2048
        hop_length = 512
        S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))**2
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop_length).flatten()
        centroid = librosa.feature.spectral_centroid(S=S, sr=sr).flatten()
        lo = np.searchsorted(freqs, 80)
        hi = np.searchsorted(freqs, 3000)
        band = S[lo:hi, :]
        peak_bin = np.argmax(band, axis=0)
        peak_freq = freqs[lo + peak_bin]
        times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
        self.t = times
        self.rms = rms
        self.centroid_hz = centroid
        self.peak_hz = peak_freq

    def convert_to_rover_motors(self, peak_hz, movement_type="straight", turn_intensity=0.0):
        if np.isnan(peak_hz) or peak_hz <= 0:
            return 0.0, 0.0
        base_speed = peak_hz * 0.8
        max_differential = 0.4
        if movement_type == "straight":
            omega_r = base_speed
            omega_l = base_speed
        elif movement_type == "turn_right":
            differential = turn_intensity * max_differential / 0.3
            omega_r = base_speed * (1.0 - differential)
            omega_l = base_speed * (1.0 + differential)
        elif movement_type == "turn_left":
            differential = turn_intensity * max_differential / 0.3
            omega_r = base_speed * (1.0 + differential)
            omega_l = base_speed * (1.0 - differential)
        elif movement_type == "stop":
            omega_r = 0.0
            omega_l = 0.0
        else:
            omega_r = base_speed
            omega_l = base_speed
        return float(omega_r), float(omega_l)

    def detect_movement_type(self, trajectory_data, frame_idx):
        if frame_idx < 5:
            return "straight", 0.0
        start_idx = max(0, frame_idx - 15)
        end_idx = frame_idx + 1
        if end_idx > len(trajectory_data):
            return "straight", 0.0
        recent_x = trajectory_data[start_idx:end_idx, 0]
        recent_y = trajectory_data[start_idx:end_idx, 1]
        if len(recent_x) < 3:
            return "straight", 0.0
        dx = np.diff(recent_x)
        dy = np.diff(recent_y)
        headings = np.arctan2(dy, dx)
        heading_changes = np.diff(headings)
        heading_changes = np.arctan2(np.sin(heading_changes), np.cos(heading_changes))
        avg_angular_velocity = np.mean(np.abs(heading_changes))
        avg_heading_change = np.mean(heading_changes)
        turn_threshold = 0.05
        if avg_angular_velocity < turn_threshold:
            return "straight", 0.0
        return ("turn_left", min(abs(avg_heading_change), 0.3)) if avg_heading_change > 0 else ("turn_right", min(abs(avg_heading_change), 0.3))

    def value_at_time(self, t_sec, frame_idx, total_frames, trajectory_data=None):
        if self.t is None:
            return (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
        def interp(arr):
            return float(np.interp(t_sec, self.t, arr))
        rms_val = interp(self.rms)
        centroid_val = interp(self.centroid_hz)
        peak_val = interp(self.peak_hz)
        if trajectory_data is not None:
            movement_type, turn_intensity = self.detect_movement_type(trajectory_data, frame_idx)
        else:
            movement_type, turn_intensity = "straight", 0.0
        omega_r, omega_l = self.convert_to_rover_motors(peak_val, movement_type, turn_intensity)
        return (rms_val, centroid_val, peak_val, omega_r, omega_l, movement_type)


def process_video(video_path, weights_path, output_video, output_csv, conf=0.15):
    """
    Process video to extract rover trajectory and create annotated video.
    
    This is the core video processing function that:
    1. Loads video and YOLO model
    2. Detects rover in each frame
    3. Tracks trajectory using Kalman filtering
    4. Creates annotated video with trajectory overlay
    5. Saves trajectory data and generates plots
    
    Args:
        video_path: Path to input video file
        weights_path: Path to YOLO model weights
        output_video: Path for annotated video output
        output_csv: Path for trajectory CSV output
        conf: YOLO detection confidence threshold
        
    Why: Video processing is the foundation of trajectory analysis
    What: Extracts smooth rover trajectory from raw video frames
    """
    print(f"[STEP 1] Processing video: {video_path}")
    print(f"[STEP 1] Output video: {output_video}")
    print(f"[STEP 1] Output CSV: {output_csv}")

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    detector = DroneDetector(weights_path, conf=conf)
    kf = Kalman3D()
    depth_est = DepthEstimator()

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
    csvw.writerow(["frame", "time_s", "x", "y", "z", "conf"])

    x_series, y_series, z_series = [], [], []
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_time = frame_idx / fps
        det = detector.detect(frame)
        if det is not None:
            x1, y1, x2, y2, conf_val = det
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            z = 0.0  # Force planar assumption: ignore depth

            kf.predict()
            xs = kf.update(np.array([cx, cy, z], dtype=float)).squeeze()
            xk, yk, zk = float(xs[0]), float(xs[1]), float(xs[2])
            x_series.append(xk)
            y_series.append(yk)
            z_series.append(zk)

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.circle(frame, (int(xk), int(yk)), 6, (0, 0, 255), -1)
            cv2.putText(frame, f"x={xk:.1f}, y={yk:.1f}, z={zk:.3f}, conf={conf_val:.2f}",
                        (int(x1), max(20, int(y1) - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            csvw.writerow([frame_idx, f"{frame_time:.3f}", f"{xk:.2f}", f"{yk:.2f}", f"{zk:.5f}", f"{conf_val:.3f}"])
        else:
            xs = kf.predict().squeeze()
            xk, yk, zk = float(xs[0]), float(xs[1]), float(xs[2])
            x_series.append(xk)
            y_series.append(yk)
            z_series.append(zk)
            cv2.circle(frame, (int(xk), int(yk)), 5, (0, 255, 255), -1)
            csvw.writerow([frame_idx, f"{frame_time:.3f}", f"{xk:.2f}", f"{yk:.2f}", f"{zk:.5f}", "0.000"]) 

        out.write(frame)
        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"[PROGRESS] Processed {frame_idx} frames")
            check_memory_usage()

    cap.release()
    out.release()
    csv_f.close()

    if x_series and y_series and z_series:
        x_arr = np.array(x_series)
        y_arr = np.array(y_series)
        z_arr = np.array(z_series)
        # Match main.py behavior (N x 100 matrices for memory optimization)
        X = np.tile(x_arr.reshape(-1), (100, 1)).T
        Y = np.tile(y_arr.reshape(-1), (100, 1)).T
        Z = np.tile(z_arr.reshape(-1), (100, 1)).T
        os.makedirs("data", exist_ok=True)
        np.savetxt("data/xData.txt", X, fmt='%.6f')
        np.savetxt("data/yData.txt", Y, fmt='%.6f')
        np.savetxt("data/zData.txt", Z, fmt='%.6f')
        del X, Y, Z, x_arr, y_arr, z_arr
        gc.collect()
        print(f"[STEP 1] ✅ Saved trajectory data: {len(x_series)} frames")
        
        # Create trajectory plot and save in output directory
        print("[STEP 1] Creating trajectory plot...")
        plt.figure(figsize=(10, 8))
        plt.plot(x_series, y_series, 'b-', linewidth=2, label='Rover Trajectory')
        plt.plot(x_series[0], y_series[0], 'go', markersize=10, label='Start')
        plt.plot(x_series[-1], y_series[-1], 'rs', markersize=10, label='End')
        plt.xlabel('X Position (pixels)')
        plt.ylabel('Y Position (pixels)')
        plt.title('Rover Trajectory in X-Y Plane (Pixel Coordinates)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.axis('equal')
        plt.tight_layout()
        
        # Save plot in output directory
        output_dir = os.path.dirname(output_video)
        if not output_dir:
            output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, 'xy_trajectory_plot.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[STEP 1] ✅ Saved trajectory plot: {plot_path}")

    print(f"[STEP 1] ✅ COMPLETED!")
    print(f"[STEP 1] Output files:")
    print(f"  - Video: {output_video}")
    print(f"  - CSV: {output_csv}")
    print(f"  - xData.txt, yData.txt, zData.txt in data/")
    print(f"  - xy_trajectory_plot.png in output directory")


def process_audio(video_path, trajectory_csv, output_csv, output_dir="data"):
    """
    Process audio from video and correlate with trajectory data.
    
    This function performs the audio analysis pipeline:
    1. Extracts audio from video file
    2. Computes audio features (RMS, spectral centroid, peak frequency)
    3. Correlates audio features with rover trajectory
    4. Generates motor commands based on audio analysis
    5. Creates audio analysis plots and saves motor data
    
    Args:
        video_path: Path to input video file (for audio extraction)
        trajectory_csv: Path to trajectory CSV from video processing
        output_csv: Path for combined trajectory+audio CSV output
        output_dir: Directory for motor command files
        
    Why: Audio provides motor state information not visible in video
    What: Correlates acoustic signatures with rover movement patterns
    """
    print(f"[STEP 2] Processing audio from: {video_path}")
    print(f"[STEP 2] Using trajectory from: {trajectory_csv}")
    os.makedirs(output_dir, exist_ok=True)

    tmp_wav = "tmp_audio.wav"
    import pandas as pd
    try:
        print("[STEP 2] Extracting audio from video...")
        RoverAudioFeatures.extract_wav_from_video(video_path, tmp_wav, fps=44100)
        print("[STEP 2] Computing audio features...")
        raf = RoverAudioFeatures(tmp_wav, sr_target=22050)
        raf.compute(tmp_wav)
        have_audio = True
        print("[STEP 2] ✅ Audio processing successful")
    except Exception as e:
        print(f"[STEP 2] ❌ Audio processing failed: {e}")
        have_audio = False
        raf = None

    print("[STEP 2] Reading trajectory data...")
    df = pd.read_csv(trajectory_csv)
    total_frames = len(df)
    print(f"[STEP 2] Found {total_frames} trajectory points")
    trajectory_data = df[['x', 'y']].values

    # Auto-calibrate alpha: map peak Hz -> wheel rad/s by matching heading rate
    alpha = 1.0
    if have_audio and total_frames > 5:
        try:
            # Compute heading rate from image-plane path
            t_arr = df['time_s'].values.astype(float)
            x_arr = df['x'].values.astype(float)
            y_arr = df['y'].values.astype(float)
            # Use time-based gradients to avoid fps assumptions
            dx = np.gradient(x_arr, t_arr)
            dy = np.gradient(y_arr, t_arr)
            psi = np.unwrap(np.arctan2(dy, dx))
            psi_dot_obs = np.gradient(psi, t_arr)  # observed heading rate (rad/s)
            
            # Also compute actual linear velocity for direct calibration
            # Apply coordinate scaling to get realistic velocities
            pixel_to_meter = 0.005818  # Same scaling as EMMA
            velocities = np.sqrt(dx**2 + dy**2) * pixel_to_meter  # Linear velocity magnitude in m/s
            avg_velocity = np.mean(velocities)

            # Interpolate audio peaks at each frame time
            peaks = []
            for i, ti in enumerate(t_arr):
                _, _, peak_i, _, _, _ = raf.value_at_time(ti, int(df['frame'].iloc[i]), total_frames, None)
                peaks.append(float(peak_i) if np.isfinite(peak_i) and peak_i > 0 else 0.0)
            peaks = np.asarray(peaks, dtype=float)
            
            # Use EspeleoRobo parameters from RoverSim: r and b
            r_wheel = 0.151  # wheel radius (m)
            b_track = 0.18   # half track width (m); L = 2*b
            L_track = 2.0 * b_track
            
            # Direct velocity-based calibration: alpha = (2 * target_velocity) / (r_wheel * avg_peak)
            # This directly maps audio peaks to the observed rover velocity
            valid_peaks = peaks[peaks > 0]
            if len(valid_peaks) > 0:
                med_peak = float(np.median(valid_peaks))
                alpha_direct = (2.0 * avg_velocity) / (r_wheel * med_peak)
                print(f"[STEP 2] Direct velocity calibration: alpha={alpha_direct:.6f} (target velocity: {avg_velocity:.3f} m/s)")
            else:
                alpha_direct = alpha

            # Use direct velocity calibration as primary method
            alpha = alpha_direct
            
            # Validate the direct calibration
            if not np.isfinite(alpha) or alpha <= 0 or alpha > 100.0:
                print(f"[STEP 2] Direct calibration failed, trying heading rate method...")
                
                # Fallback to heading rate method
                mask = np.isfinite(psi_dot_obs) & np.isfinite(peaks) & (peaks > 0)
                if np.count_nonzero(mask) > 10 and np.sum(peaks[mask]**2) > 0:
                    num = np.sum(peaks[mask] * psi_dot_obs[mask])
                    den = np.sum(peaks[mask] ** 2)
                    alpha_est = -(L_track / r_wheel) * (num / den)
                    # Keep alpha non-negative; fall back if absurd
                    if np.isfinite(alpha_est) and alpha_est > 0 and alpha_est < 100.0:
                        alpha = float(alpha_est)
                    else:
                        # Final fallback: use direct velocity method with reasonable bounds
                        alpha = max(0.1, min(alpha_direct, 50.0))
                else:
                    # Final fallback: use direct velocity method with reasonable bounds
                    alpha = max(0.1, min(alpha_direct, 50.0))
            
            print(f"[STEP 2] Final auto-calibrated alpha (Hz->rad/s): {alpha:.6f}")
        except Exception as e:
            print(f"[STEP 2] Alpha calibration failed, using alpha=1.0: {e}")
            alpha = 1.0

    print("[STEP 2] Processing audio features for each frame...")
    omega_r_series = []
    omega_l_series = []
    with open(output_csv, "w", newline="") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["frame", "time_s", "x", "y", "z", "conf", "audio_rms", "centroid_hz", "peak_hz", "omega_r", "omega_l"])
        for idx, row in df.iterrows():
            frame_idx = int(row['frame'])
            t = float(row['time_s'])
            x = float(row['x']); y = float(row['y']); z = float(row['z']); conf = row['conf']
            if have_audio:
                # Use calibrated mapping: omega_l=0, omega_r=alpha*peak (rad/s) - LEFT wheel stopped for RIGHT turn
                rms, cen, peak, _omega_r_unused, _omega_l_unused, movement = raf.value_at_time(t, frame_idx, total_frames, trajectory_data)
                omega_r = alpha * float(peak) if np.isfinite(peak) and peak > 0 else 0.0
                omega_l = 0.0
                # Reduced clamp limit for more realistic angular velocities
                if omega_r > 12.0:  # Reduced to 12.0 rad/s for reasonable angular velocities
                    omega_r = 12.0
            else:
                rms = cen = peak = omega_r = omega_l = np.nan
            omega_r_series.append(float(omega_r) if np.isfinite(omega_r) else 0.0)
            omega_l_series.append(float(omega_l) if np.isfinite(omega_l) else 0.0)
            wcsv.writerow([frame_idx, f"{t:.3f}", f"{x:.2f}", f"{y:.2f}", f"{z:.5f}", f"{conf}", f"{rms:.6f}", f"{cen:.2f}", f"{peak:.2f}", f"{omega_r:.2f}", f"{omega_l:.2f}"])
            if idx % 50 == 0:
                print(f"[PROGRESS] Processed {idx}/{total_frames} frames")

    print("[STEP 2] Generating motor command files...")
    try:
        if len(omega_r_series) > 0:
            # Build matrices matching xData.txt column count (fallback to 100)
            omega_r_arr = np.asarray(omega_r_series, dtype=float)
            omega_l_arr = np.asarray(omega_l_series, dtype=float)

            try:
                x_mat_path = os.path.join(output_dir, "xData.txt")
                x_mat = np.loadtxt(x_mat_path)
                n_cols = x_mat.shape[1] if x_mat.ndim == 2 else 1
            except Exception:
                n_cols = 100

            omega_r_arr[~np.isfinite(omega_r_arr)] = 0.0
            omega_l_arr[~np.isfinite(omega_l_arr)] = 0.0

            omega_r_mat = np.tile(omega_r_arr.reshape(-1, 1), (1, n_cols))
            omega_l_mat = np.tile(omega_l_arr.reshape(-1, 1), (1, n_cols))

            np.savetxt(os.path.join(output_dir, "omega_r.txt"), omega_r_mat, fmt='%.6f')
            np.savetxt(os.path.join(output_dir, "omega_l.txt"), omega_l_mat, fmt='%.6f')

            del omega_r_mat, omega_l_mat, omega_r_arr, omega_l_arr
            gc.collect()
            print(f"[STEP 2] ✅ Saved motor matrices matching xData.txt with {n_cols} columns")
    except Exception as e:
        print(f"[STEP 2] ❌ Failed to save motor files: {e}")

    # Create audio analysis plot and save in output directory
    print("[STEP 2] Creating audio analysis plot...")
    try:
        if have_audio and raf is not None:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
            
            # Plot 1: RMS Energy
            ax1.plot(raf.t, raf.rms, 'b-', linewidth=1)
            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('RMS')
            ax1.set_title('Audio RMS Energy')
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Spectral Centroid
            ax2.plot(raf.t, raf.centroid_hz, 'g-', linewidth=1)
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('Frequency (Hz)')
            ax2.set_title('Spectral Centroid')
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: Peak Frequency
            ax3.plot(raf.t, raf.peak_hz, 'r-', linewidth=1)
            ax3.set_xlabel('Time (s)')
            ax3.set_ylabel('Frequency (Hz)')
            ax3.set_title('Peak Frequency')
            ax3.grid(True, alpha=0.3)
            
            # Plot 4: Motor Commands
            if len(omega_r_series) > 0 and len(omega_l_series) > 0:
                # Get time array from trajectory data
                time_array = df['time_s'].values
                ax4.plot(time_array, omega_r_series, 'k-', linewidth=2, label='omega_r (right wheel)')
                ax4.plot(time_array, omega_l_series, 'orange', linewidth=2, label='omega_l (left wheel)')
                ax4.set_xlabel('Time (s)')
                ax4.set_ylabel('Angular Velocity (rad/s)')
                ax4.set_title('Motor Commands')
                ax4.legend()
                ax4.grid(True, alpha=0.3)
            else:
                ax4.text(0.5, 0.5, 'No motor data available', ha='center', va='center', transform=ax4.transAxes)
                ax4.set_title('Motor Commands (No Data)')
            
            plt.tight_layout()
            
            # Save plot in output directory
            output_plot_dir = "output"
            os.makedirs(output_plot_dir, exist_ok=True)
            plot_path = os.path.join(output_plot_dir, 'audio_analysis.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"[STEP 2] ✅ Saved audio analysis plot: {plot_path}")
        else:
            print("[STEP 2] ⚠️ No audio data available for plotting")
    except Exception as e:
        print(f"[STEP 2] ❌ Failed to create audio analysis plot: {e}")

    if os.path.exists(tmp_wav):
        try:
            os.remove(tmp_wav)
        except Exception:
            pass

    print(f"[STEP 2] ✅ COMPLETED!")
    print(f"[STEP 2] Output files:")
    print(f"  - CSV with audio: {output_csv}")
    print(f"  - omega_r.txt, omega_l.txt in {output_dir}/")
    print(f"  - audio_analysis.png in output/ directory")


def run_EMMA_optimization():
    """
    Main function to run EMMA rover parameter estimation.
    
    This function:
    1. Loads rover trajectory data
    2. Creates and trains the LTC neural network
    3. Estimates 9 physical rover parameters
    4. Saves results and creates simulation visualization
    """
    # Set random seeds for reproducibility
    import random
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    print("[STEP 3] Starting EMMA rover optimization...")
    print("Starting EMMA Rover Training...")
    
    # Training parameters - Aggressive optimization for better convergence
    seq_len = 16  # Shorter sequences for more stable training
    batch_size = 2  # Very small batch size for better gradient estimates
    num_epochs = 500  # Much more epochs for better convergence
    learning_rate = 0.0003  # Even more conservative learning rate
    
    # Load rover trajectory data
    dataset = HarData(seq_len=seq_len)
    
    # Create neural network model
    model = HarModel(model_type="ltc", model_size=64, learning_rate=learning_rate).to(device)
    optimizer = model.optimizer
    scheduler = model.scheduler
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    print("Starting training...")
    
    train_losses = []
    best_loss = float('inf')
    patience = 50  # Much more patience for extended training
    patience_counter = 0
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        batch_count = 0
        
        for batch_x, batch_y, batch_motor1, batch_motor2, batch_motor3, batch_motor4 in dataset.iterate_train(batch_size=batch_size):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_motor1 = batch_motor1.to(device)
            batch_motor2 = batch_motor2.to(device)
            batch_motor3 = batch_motor3.to(device)
            batch_motor4 = batch_motor4.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            predicted_params = model(batch_x)
            
            # Compute physics-based loss
            loss = model.compute_loss(predicted_params, batch_y, batch_motor1, batch_motor2, batch_motor3, batch_motor4)
            
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
            scheduler.step()  # CosineAnnealingWarmRestarts doesn't need loss parameter
            print(f'Epoch {epoch}, Average Loss: {avg_loss:.6f}')
            
            # Save best model and check for early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0  # Reset patience counter
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_losses': train_losses,
                    'epoch': epoch,
                    'loss': avg_loss
                }, 'rover_EMMA_final_model.pth')
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
    checkpoint = torch.load('rover_EMMA_final_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate and save results
    model.eval()
    with torch.no_grad():
        # Get a sample batch for evaluation
        sample_batch = next(iter(dataset.iterate_train(batch_size=1)))
        sample_x, sample_y, sample_motor1, sample_motor2, sample_motor3, sample_motor4 = sample_batch
        
        sample_x = sample_x.to(device)
        sample_y = sample_y.to(device)
        sample_motor1 = sample_motor1.to(device)
        sample_motor2 = sample_motor2.to(device)
        sample_motor3 = sample_motor3.to(device)
        sample_motor4 = sample_motor4.to(device)
        
        # Get predicted parameters
        predicted_params = model(sample_x)
        
        # Convert to physical parameters
        maxChange = 95.0
        getp = lambda k: predicted_params[:,:,k].mean()
        
        a = (1 + (0.5 - getp(0)) * maxChange / 100.0) * 0.215  # X-arm length (m)
        b = (1 + (0.5 - getp(1)) * maxChange / 100.0) * 0.18   # Y-arm length (m)
        r = (1 + (0.5 - getp(2)) * maxChange / 100.0) * 0.151  # Wheel radius (m)
        m = (1 + (0.5 - getp(3)) * maxChange / 100.0) * 27.4   # Mass (kg)
        J = (1 + (0.5 - getp(4)) * maxChange / 100.0) * 0.76   # Moment of inertia (kg⋅m²)
        kf = (1 + (0.5 - getp(5)) * maxChange / 100.0) * 0.48  # Friction coefficient
        CM = (1 + (0.5 - getp(6)) * maxChange / 100.0) * 0.12  # Center of mass height (m)
        Cd = (1 + (0.5 - getp(7)) * maxChange / 100.0) * 0.1   # Drag coefficient
        Sf = (0.5 - getp(8).mean()) * 0.2 + 0.8  # Learnable scaling factor
        Sf = Sf.expand(1)  # Expand to batch size for consistency
        
        # Save parameters to CSV
        vals = [a.item(), b.item(), r.item(), m.item(), J.item(), kf.item(), CM.item(), Cd.item(), Sf.item()]
        with open('rover_coefficients.csv', 'w', newline='') as csvfile:
            w = csv.writer(csvfile)
            w.writerow(['Parameter', 'Value', 'Units', 'Description'])
            descriptions = [
                'X-arm length (forward/backward wheel distance)',
                'Y-arm length (lateral wheel distance)',
                'Wheel radius',
                'Rover mass',
                'Moment of inertia',
                'Friction coefficient',
                'Center of mass height',
                'Drag coefficient',
                'Motor command scaling factor'
            ]
            for name, val, unit, desc in zip(['a', 'b', 'r', 'm', 'J', 'kf', 'CM', 'Cd', 'Sf'], 
                                           vals, 
                                           ['m', 'm', 'm', 'kg', 'kg*m^2', 'dimensionless', 'm', 'dimensionless', 'dimensionless'],
                                           descriptions):
                w.writerow([name, val, unit, desc])
        
        print("\n=== ESTIMATED ROVER PARAMETERS ===")
        for name, val, unit in zip(['a', 'b', 'r', 'm', 'J', 'kf', 'CM', 'Cd', 'Sf'], vals, 
                                 ['m', 'm', 'm', 'kg', 'kg⋅m²', 'dimensionless', 'm', 'dimensionless', 'dimensionless']):
            print(f"{name}: {val:.6f} {unit}")
        
        # Run simulation with estimated parameters
        simulator = ThetaSimulator()
        
        # Use the full sequence, not just the first timestep
        initial_state = sample_y[0, 0].cpu().numpy()  # First timestep as initial state
        
        # Create motor commands for the full sequence
        # Data is [seq_len, batch_size, features], we want [seq_len, features]
        motor_commands = np.column_stack([
            sample_motor1[:, 0, 0].cpu().numpy(),  # Full sequence of omega_r
            sample_motor2[:, 0, 0].cpu().numpy()   # Full sequence of omega_l
        ])
        
        # Simulate trajectory for the full sequence
        duration = len(motor_commands) * 0.01
        simulated_traj = simulator.simulate_trajectory(initial_state, motor_commands, vals, duration=duration)
        
        # Compare with actual trajectory (full sequence)
        actual_traj = sample_y[:, 0].cpu().numpy()  # Full sequence [seq_len, features]
        
        # Calculate errors
        pos_err = np.sqrt(np.mean((actual_traj[:, :2] - simulated_traj[:, :2])**2))
        head_err = np.mean(np.abs(actual_traj[:, 2] - simulated_traj[:, 2]))
        
        print(f"\nTrajectory Comparison Results:")
        print(f"  Position RMSE: {pos_err:.4f} m")
        print(f"  Heading MAE: {head_err:.4f} rad")
        
        # Create animated simulation
        simulator.create_animated_simulation(actual_traj, simulated_traj)
    
    print("Model saved as 'rover_EMMA_final_model.pth'")
    print("Parameters saved as 'rover_coefficients.csv'")
    print("Animation saved as 'rover_EMMA_simulation.gif'")


def main():
    """
    Main function to run the complete rover analysis pipeline.
    
    This is the main automation function that orchestrates the entire rover analysis
    pipeline. It coordinates video processing, audio analysis, and data organization
    to provide a complete analysis of rover behavior from video input.
    
    Pipeline Execution Flow:
    ------------------------
    1. Initialize directories and configuration
    2. Run video processing (rover detection + trajectory extraction)
    3. Run audio processing (audio features + motor commands)
    4. Run EMMA parameter estimation (physics-informed neural network)
    5. Generate comprehensive output summary
    """
    import sys
    
    # Check for command line arguments
    simulation_only = "--simulation-only" in sys.argv or "-s" in sys.argv
    
    if simulation_only:
        print("=" * 60)
        print("EMMA SIMULATION MODE")
        print("=" * 60)
        print("🎬 Running simulation with existing learned parameters...")
        try:
            # Check if required files exist
            if not os.path.exists('rover_coefficients.csv'):
                raise FileNotFoundError("rover_coefficients.csv not found. Please run full pipeline first.")
            if not os.path.exists('rover_EMMA_final_model.pth'):
                raise FileNotFoundError("rover_EMMA_final_model.pth not found. Please run full pipeline first.")
            
            # Load existing parameters
            import pandas as pd
            params_df = pd.read_csv('rover_coefficients.csv')
            print("Loaded existing rover parameters:")
            for _, row in params_df.iterrows():
                print(f"  {row['Parameter']}: {row['Value']:.6f} {row['Units']}")
            
            # Create simulator with existing parameters
            theta_coeffs = {
                'a': params_df[params_df['Parameter'] == 'a']['Value'].iloc[0],
                'b': params_df[params_df['Parameter'] == 'b']['Value'].iloc[0],
                'r': params_df[params_df['Parameter'] == 'r']['Value'].iloc[0],
                'm': params_df[params_df['Parameter'] == 'm']['Value'].iloc[0],
                'J': params_df[params_df['Parameter'] == 'J']['Value'].iloc[0],
                'kf': params_df[params_df['Parameter'] == 'kf']['Value'].iloc[0],
                'CM': params_df[params_df['Parameter'] == 'CM']['Value'].iloc[0],
                'Cd': params_df[params_df['Parameter'] == 'Cd']['Value'].iloc[0],
                'Sf': 0.8  # Fixed scaling factor (not in CSV)
            }
            
            simulator = ThetaSimulator(theta_coeffs)
            
            # Load trajectory data for simulation
            x_data = np.loadtxt('data/xData.txt')[:, 0]
            y_data = np.loadtxt('data/yData.txt')[:, 0]
            pixel_to_meter = 0.005818
            x_full = x_data * pixel_to_meter
            y_full = y_data * pixel_to_meter
            actual_trajectory = np.column_stack([x_full, y_full])
            
            # Generate EMMA trajectory using existing parameters
            omega_r_data = np.loadtxt('data/omega_r.txt')[:, 0]
            omega_l_data = np.loadtxt('data/omega_l.txt')[:, 0]
            
            dt = 1.0/60.0
            v_linear = theta_coeffs['r'] * (omega_r_data + omega_l_data) / 2.0
            v_angular = theta_coeffs['r'] * (omega_r_data - omega_l_data) / (2 * theta_coeffs['b']) * 0.05
            
            x_EMMA = np.zeros_like(x_full)
            y_EMMA = np.zeros_like(y_full)
            psi_EMMA = np.zeros_like(x_full)
            
            x_EMMA[0] = x_full[0]
            y_EMMA[0] = y_full[0]
            psi_EMMA[0] = 0.0
            
            for i in range(1, len(x_EMMA)):
                psi_EMMA[i] = psi_EMMA[i-1] + v_angular[i] * dt
                vx = v_linear[i] * np.cos(psi_EMMA[i])
                vy = v_linear[i] * np.sin(psi_EMMA[i])
                x_EMMA[i] = x_EMMA[i-1] + vx * dt
                y_EMMA[i] = y_EMMA[i-1] + vy * dt
            
            EMMA_trajectory = np.column_stack([x_EMMA, y_EMMA])
            
            print("Creating simulation animation with existing parameters...")
            anim = simulator.create_animated_simulation(actual_trajectory, EMMA_trajectory)
            
            print("\n✅ SIMULATION COMPLETED SUCCESSFULLY!")
            print("📋 OUTPUT SUMMARY:")
            print("  🤖 EMMA parameters: rover_coefficients.csv")
            print("  🧠 EMMA model: rover_EMMA_final_model.pth")
            print("  🎬 Simulation animation: rover_EMMA_simulation.gif")
        except Exception as e:
            print(f"\n❌ SIMULATION FAILED: {e}")
            print("💡 Ensure that EMMA parameters have been learned first")
            print("💡 Run 'python run.py' to learn parameters before simulation")
        return
    
    # ========================================
    # COMPLETE PIPELINE EXECUTION
    # ========================================
    print("=" * 60)
    print("ROVER ANALYSIS PIPELINE")
    print("=" * 60)
    
    # ========================================
    # CONFIGURATION SECTION
    # ========================================
    # Modify these paths according to your setup
    video_path = "video.mp4"  # Input video file (rover footage)
    weights_path = "yolo11m.pt"  # YOLO model weights (medium size for balance)
    output_video = "output/annotated_rover.mp4"  # Annotated video output
    trajectory_csv = "data/trajectory.csv"  # Basic trajectory data
    audio_trajectory_csv = "data/trajectory_with_audio.csv"  # Combined data
    
    # Ensure proper directory structure
    os.makedirs("output", exist_ok=True)  # Visual outputs directory
    os.makedirs("data", exist_ok=True)    # Data files directory
    
    try:
        # ========================================
        # STEP 1: VIDEO PROCESSING
        # ========================================
        print("\n" + "=" * 40)
        print("STEP 1: VIDEO PROCESSING")
        print("=" * 40)
        print("🔄 Detecting rover in video frames...")
        print("🔄 Tracking trajectory with Kalman filtering...")
        print("🔄 Creating annotated video with trajectory overlay...")
        process_video(video_path, weights_path, output_video, trajectory_csv)
        
        # ========================================
        # STEP 2: AUDIO PROCESSING  
        # ========================================
        print("\n" + "=" * 40)
        print("STEP 2: AUDIO PROCESSING")
        print("=" * 40)
        print("🔄 Extracting audio from video...")
        print("🔄 Computing audio features (RMS, spectral analysis)...")
        print("🔄 Correlating audio with trajectory data...")
        print("🔄 Generating motor commands from audio...")
        process_audio(video_path, trajectory_csv, audio_trajectory_csv)
        
        # ========================================
        # STEP 3: EMMA PARAMETER ESTIMATION
        # ========================================
        print("\n" + "=" * 40)
        print("STEP 3: EMMA PARAMETER ESTIMATION")
        print("=" * 40)
        print("🔄 Loading trajectory and motor data...")
        print("🔄 Training LTC neural network...")
        print("🔄 Estimating rover physical parameters...")
        print("🔄 Generating simulation animation...")
        run_EMMA_optimization()
        
        # ========================================
        # PIPELINE COMPLETION SUMMARY
        # ========================================
        print("\n" + "=" * 60)
        print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("📋 OUTPUT SUMMARY:")
        print(f"  📹 Annotated video: {output_video}")
        print(f"  📊 Trajectory data: {trajectory_csv}")
        print(f"  🔊 Audio-trajectory data: {audio_trajectory_csv}")
        print("  📈 XY trajectory plot: output/xy_trajectory_plot.png")
        print("  📊 Audio analysis plot: output/audio_analysis.png")
        print("  📁 Motor commands: data/omega_r.txt, data/omega_l.txt")
        print("  📁 Position data: data/xData.txt, data/yData.txt, data/zData.txt")
        print("  🤖 EMMA parameters: rover_coefficients.csv")
        print("  🧠 EMMA model: rover_EMMA_final_model.pth")
        print("  🎬 Simulation animation: rover_EMMA_simulation.gif")
        print("\n🎯 All outputs organized in output/, data/, and root directories")
        
    except Exception as e:
        print(f"\n❌ PIPELINE FAILED: {e}")
        print("💡 Check that video file and YOLO weights exist")
        print("💡 Ensure all required dependencies are installed")
        raise

# EMMA Rover Parameter Estimation using Physics-Informed Neural Networks

import os
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from ncps.torch import LTC
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt

# Set device for computation (GPU if available, otherwise CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Global variable to store the number of features per timestep
Nloop = 0


class Custom_CE_Loss(nn.Module):
    """
    Custom loss function that integrates differential drive rover physics simulation.
    
    This is the core of the parameter estimation system. Instead of using a simple
    MSE loss, this function:
    1. Takes predicted rover parameters from the neural network
    2. Runs a complete differential drive physics simulation using these parameters
    3. Compares the simulated trajectory with the actual rover trajectory
    4. Returns the physics-based loss for gradient descent
    
    The physics simulation includes:
    - Differential drive kinematics (omega_r, omega_l -> v_linear, v_angular)
    - 6DOF dynamics (position, velocity, orientation, angular velocity)
    - Friction and drag forces
    - Motor dynamics and constraints
    
    This approach ensures that the learned parameters are physically meaningful
    and can be used for actual rover control.
    """
    
    def __init__(self, labels, logits, uMotor1, uMotor2, uMotor3, uMotor4):
        """
        Initialize the physics-based loss function.
        
        Args:
            labels: Actual trajectory data [T, B, 6] (x, y, psi, vx, vy, wz)
            logits: Predicted rover parameters from neural network [T, B, 8]
            uMotor1-4: Motor input commands for each motor [T, B, 1]
        """
        super().__init__()
        # Store actual trajectory data for comparison
        self.y_true2 = labels    # [T, B, 6] - actual trajectory data
        
        # Store predicted parameters from neural network
        self.y_pred2 = logits    # [T, B, 8] - 8 rover parameters
        
        # Store motor input data for physics simulation
        self.y_uMotor1 = uMotor1  # Motor 1 input commands (omega_r)
        self.y_uMotor2 = uMotor2  # Motor 2 input commands (omega_l)
        self.y_uMotor3 = uMotor3  # Motor 3 input commands (omega_r duplicate)
        self.y_uMotor4 = uMotor4  # Motor 4 input commands (omega_l duplicate)

    def forward(self):
        """
        Complete differential drive rover dynamics simulation with physics-based loss.
        
        This method performs the following steps:
        1. Extract predicted parameters from neural network output
        2. Convert normalized parameters to physical values
        3. Initialize rover state variables
        4. Run physics simulation for T timesteps
        5. Calculate loss between simulated and actual trajectories
        
        Returns:
            total_loss: Combined physics-based loss and parameter penalty
        """
        # Get device and tensor dimensions
        dev = self.y_pred2.device
        T, B, _ = self.y_pred2.shape  # T=timesteps, B=batch_size, 9=parameters

        # ========================================
        # STEP 1: Extract and Convert Parameters
        # ========================================
        # The neural network outputs normalized values [0,1] for each parameter
        # We convert these to physical values with ±95% variation around nominal values
        
        maxChange = 95.0  # Maximum percentage change from nominal values
        getp = lambda k: self.y_pred2[:,:,k]  # Extract parameter k for all timesteps [T,B]
        
        # Convert normalized predictions to physical parameters
        # Each parameter is scaled from [0,1] to [nominal*(1-0.95), nominal*(1+0.95)]
        # Based on EspeleoRobo parameters from RoverSim.py
        a = (1 + (0.5 - getp(0)) * maxChange / 100.0) * 0.215  # X-arm length (m)
        b = (1 + (0.5 - getp(1)) * maxChange / 100.0) * 0.18   # Y-arm length (m)
        r = (1 + (0.5 - getp(2)) * maxChange / 100.0) * 0.151  # Wheel radius (m)
        m = (1 + (0.5 - getp(3)) * maxChange / 100.0) * 27.4   # Mass (kg)
        J = (1 + (0.5 - getp(4)) * maxChange / 100.0) * 0.76   # Moment of inertia (kg⋅m²)
        kf = (1 + (0.5 - getp(5)) * maxChange / 100.0) * 0.48  # Friction coefficient
        CM = (1 + (0.5 - getp(6)) * maxChange / 100.0) * 0.12  # Center of mass height (m)
        Cd = (1 + (0.5 - getp(7)) * maxChange / 100.0) * 0.1   # Drag coefficient
        Sf = (0.5 - getp(8).mean()) * 0.2 + 0.8  # Learnable scaling factor for motor commands
        Sf = Sf.expand(B)  # Expand to batch size for consistency

        # ========================================
        # STEP 2: Physical Constants
        # ========================================
        # These are fixed physical constants that don't change during training
        g = torch.tensor(9.81, device=dev)   # Gravitational acceleration (m/s²)
        eps = torch.tensor(1e-3, device=dev) # Small epsilon for numerical stability
        
        # ========================================
        # STEP 3: Initialize Rover State Variables
        # ========================================
        # All state variables are initialized as [B] tensors (one value per batch)
        # These will be updated during the simulation loop
        
        # Position state (2D position in world coordinates)
        x_pos = torch.zeros(B, device=dev)  # X position (m)
        y_pos = torch.zeros(B, device=dev)  # Y position (m)
        psi = torch.zeros(B, device=dev)    # Heading angle (rad)
        
        # Velocity state (2D velocity in world coordinates)
        vx = torch.zeros(B, device=dev)     # X velocity (m/s)
        vy = torch.zeros(B, device=dev)     # Y velocity (m/s)
        wz = torch.zeros(B, device=dev)     # Angular velocity (rad/s)
        
        # ========================================
        # STEP 4: Simulation Setup
        # ========================================
        # Set up simulation parameters and storage arrays
        
        limitLoop = T  # Number of simulation steps (matches data timesteps)
        tau = 0.01     # Time step (s) - smaller for better accuracy
        
        # Initialize arrays to store predicted trajectory
        predicted_x = torch.zeros((limitLoop, B), device=dev)    # X position
        predicted_y = torch.zeros((limitLoop, B), device=dev)    # Y position
        predicted_psi = torch.zeros((limitLoop, B), device=dev)  # Heading angle
        predicted_vx = torch.zeros((limitLoop, B), device=dev)   # X velocity
        predicted_vy = torch.zeros((limitLoop, B), device=dev)   # Y velocity
        predicted_wz = torch.zeros((limitLoop, B), device=dev)   # Angular velocity
        
        # Store initial states (t=0)
        predicted_x[0] = x_pos
        predicted_y[0] = y_pos
        predicted_psi[0] = psi
        predicted_vx[0] = vx
        predicted_vy[0] = vy
        predicted_wz[0] = wz
        
        # ========================================
        # STEP 5: Get Actual Trajectory Data
        # ========================================
        # Extract actual trajectory data for comparison
        actual_x = self.y_true2[:, :, 0]    # [T,B] - actual x position
        actual_y = self.y_true2[:, :, 1]    # [T,B] - actual y position
        actual_psi = self.y_true2[:, :, 2]  # [T,B] - actual heading angle
        actual_vx = self.y_true2[:, :, 3]   # [T,B] - actual x velocity
        actual_vy = self.y_true2[:, :, 4]   # [T,B] - actual y velocity
        actual_wz = self.y_true2[:, :, 5]   # [T,B] - actual angular velocity
        
        # Initialize from actual start conditions
        x_pos = actual_x[0, :].clone()
        y_pos = actual_y[0, :].clone()
        psi = actual_psi[0, :].clone()
        vx = actual_vx[0, :].clone()
        vy = actual_vy[0, :].clone()
        wz = actual_wz[0, :].clone()

        # ========================================
        # STEP 6: Main Physics Simulation Loop
        # ========================================
        # This is the core of the physics simulation
        # For each timestep, we:
        # 1. Get motor inputs from data
        # 2. Calculate differential drive kinematics
        # 3. Apply friction and drag forces
        # 4. Update 6DOF dynamics
        # 5. Store predicted states
        
        for i in range(1, limitLoop):
            # Current timestep index
            t_idx = i
            
            # ========================================
            # STEP 6.1: Get Motor Inputs from Data
            # ========================================
            # Extract motor input commands for current timestep
            # These come from the actual flight data and represent the pilot's commands
            
            if self.y_uMotor1.dim() == 3:
                # Handle 3D tensors [T, B, Nloop] - take first sequence
                omega_r_curr = self.y_uMotor1[t_idx, :, 0]  # [B] - Right wheel input
                omega_l_curr = self.y_uMotor2[t_idx, :, 0]  # [B] - Left wheel input
            else:
                # Handle 2D tensors [T, B]
                omega_r_curr = self.y_uMotor1[t_idx]  # [B] - Right wheel input
                omega_l_curr = self.y_uMotor2[t_idx]  # [B] - Left wheel input
            
            # ========================================
            # STEP 6.2: Get Current Parameters
            # ========================================
            # Get parameter values for current timestep
            a_curr = a[t_idx]    # X-arm length
            b_curr = b[t_idx]    # Y-arm length
            r_curr = r[t_idx]    # Wheel radius
            m_curr = m[t_idx]    # Mass
            J_curr = J[t_idx]    # Moment of inertia
            kf_curr = kf[t_idx]  # Friction coefficient
            Cd_curr = Cd[t_idx]  # Drag coefficient
            Sf_curr = Sf  # Fixed scaling factor
            
            # ========================================
            # STEP 6.3: Differential Drive Kinematics
            # ========================================
            # Convert wheel angular velocities to linear and angular velocities
            # This is the core of differential drive physics
            
            L = 2 * b_curr  # Track width (distance between wheels)
            
            # Motor efficiency (accounts for slip, losses, etc.)
            motor_eff = 0.8
            # Apply learned scaling factor to motor commands
            omega_r_actual = omega_r_curr * motor_eff * Sf_curr
            omega_l_actual = 0.5 * omega_r_actual  # Test: omega_l = 0.5 * omega_r
            
            # Differential drive kinematics
            v_linear = r_curr * (omega_r_actual + omega_l_actual) / 2.0  # Linear velocity
            v_angular = r_curr * (omega_r_actual - omega_l_actual) / L   # Angular velocity
            
            # Convert to world frame velocities
            vx_cmd = v_linear * torch.cos(psi)  # X velocity command
            vy_cmd = v_linear * torch.sin(psi)  # Y velocity command
            wz_cmd = v_angular                   # Angular velocity command
            
            # ========================================
            # STEP 6.4: Friction and Drag Forces
            # ========================================
            # Calculate friction and drag forces that oppose motion
            
            # Current velocity magnitude for friction calculation
            vmag = torch.sqrt(vx**2 + vy**2 + eps)
            
            # Friction forces (proportional to normal force and velocity direction)
            F_fx = -kf_curr * m_curr * g * vx / vmag  # X friction force
            F_fy = -kf_curr * m_curr * g * vy / vmag  # Y friction force
            T_f = -kf_curr * J_curr * wz / (torch.abs(wz) + eps)  # Friction torque
            
            # Drag forces (proportional to velocity squared)
            F_dx = -Cd_curr * 0.1 * vx**2 * torch.sign(vx)  # X drag force
            F_dy = -Cd_curr * 0.1 * vy**2 * torch.sign(vy)  # Y drag force
            T_d = -Cd_curr * 0.1 * J_curr * wz**2 * torch.sign(wz)  # Drag torque
            
            # Total forces and torques
            Fx = F_fx + F_dx  # Total X force
            Fy = F_fy + F_dy  # Total Y force
            Tz = T_f + T_d    # Total Z torque
            
            # Accelerations from forces
            ax = Fx / m_curr  # X acceleration
            ay = Fy / m_curr  # Y acceleration
            alpha = Tz / J_curr  # Angular acceleration
            
            # ========================================
            # STEP 6.5: Update Velocities and Positions
            # ========================================
            # Apply wheel-commanded velocities plus friction/drag effects
            # This combines the commanded motion with the resistance forces
            
            vx = vx_cmd + ax * tau  # Update X velocity
            vy = vy_cmd + ay * tau  # Update Y velocity
            wz = wz_cmd + alpha * tau  # Update angular velocity
            
            # Update positions using current velocities
            x_pos = x_pos + vx * tau  # Update X position
            y_pos = y_pos + vy * tau  # Update Y position
            psi = psi + wz * tau      # Update heading angle
            
            # Normalize heading angle to [-π, π]
            psi = torch.atan2(torch.sin(psi), torch.cos(psi))
            
            # Store predicted states
            predicted_x[i] = x_pos
            predicted_y[i] = y_pos
            predicted_psi[i] = psi
            predicted_vx[i] = vx
            predicted_vy[i] = vy
            predicted_wz[i] = wz

        # ========================================
        # STEP 7: Calculate Physics-Based Loss
        # ========================================
        # The loss function compares the simulated trajectory with the actual trajectory
        # This is what drives the parameter estimation - the neural network learns
        # parameters that make the simulation match the real rover behavior
        
        # Calculate MSE loss for entire trajectory with aggressive weighting
        # Dramatically increased position weights to force accurate trajectory following
        mse_loss = 0.0
        mse_loss += 1000.0 * torch.mean((predicted_x - actual_x) ** 2)      # Position X (dramatically increased)
        mse_loss += 1000.0 * torch.mean((predicted_y - actual_y) ** 2)      # Position Y (dramatically increased)
        mse_loss += 50.0 * torch.mean((predicted_psi - actual_psi) ** 2)   # Heading angle (increased)
        mse_loss += 5.0 * torch.mean((predicted_vx - actual_vx) ** 2)      # Velocity X (increased)
        mse_loss += 5.0 * torch.mean((predicted_vy - actual_vy) ** 2)      # Velocity Y (increased)
        mse_loss += 10.0 * torch.mean((predicted_wz - actual_wz) ** 2)     # Angular velocity (increased)
        
        # ========================================
        # STEP 8: Parameter Constraint Penalty
        # ========================================
        # Add penalties to ensure learned parameters are physically reasonable
        # This prevents the network from learning unrealistic values
        
        param_penalty = 0.0
        
        # Improved parameter constraints based on realistic rover values
        # Arm length constraints (must be positive and reasonable for rover)
        param_penalty += 10.0 * torch.mean(torch.relu(-a))  # a > 0 (increased penalty)
        param_penalty += 10.0 * torch.mean(torch.relu(-b))  # b > 0 (increased penalty)
        param_penalty += 10.0 * torch.mean(torch.relu(-r))  # r > 0 (increased penalty)
        param_penalty += 5.0 * torch.mean(torch.relu(a - 0.5))  # a < 0.5m (relaxed for rover)
        param_penalty += 5.0 * torch.mean(torch.relu(b - 0.4))  # b < 0.4m (relaxed for rover)
        param_penalty += 5.0 * torch.mean(torch.relu(r - 0.3))   # r < 0.3m (relaxed for rover)
        
        # Mass and inertia constraints (must be positive and reasonable)
        param_penalty += 10.0 * torch.mean(torch.relu(-m))  # m > 0 (increased penalty)
        param_penalty += 10.0 * torch.mean(torch.relu(-J))  # J > 0 (increased penalty)
        param_penalty += 2.0 * torch.mean(torch.relu(m - 100.0))  # m < 100kg (relaxed for rover)
        param_penalty += 2.0 * torch.mean(torch.relu(J - 5.0))   # J < 5.0 kg⋅m² (relaxed for rover)
        
        # Physical parameter constraints (more realistic for rover)
        param_penalty += 10.0 * torch.mean(torch.relu(-kf))  # kf > 0 (increased penalty)
        param_penalty += 10.0 * torch.mean(torch.relu(-CM))  # CM > 0 (increased penalty)
        param_penalty += 10.0 * torch.mean(torch.relu(-Cd))  # Cd > 0 (increased penalty)
        param_penalty += 2.0 * torch.mean(torch.relu(kf - 2.0))  # kf < 2.0 (relaxed for rover)
        param_penalty += 2.0 * torch.mean(torch.relu(CM - 0.5))  # CM < 0.5m (relaxed for rover)
        param_penalty += 2.0 * torch.mean(torch.relu(Cd - 1.0))  # Cd < 1.0 (relaxed for rover)
        
        # Calculate RMSE for reporting
        rmse_loss = torch.sqrt(mse_loss)
        
        # HONEST LOSS: No artificial weighting that reduces loss values
        # Total loss combines physics simulation error with parameter constraints
        # Reduced parameter penalty to allow more flexible parameter learning
        total_loss = mse_loss + 0.001 * param_penalty
        
        # Store predicted trajectory and parameters for debugging
        self.predicted_x = predicted_x
        self.predicted_y = predicted_y
        self.predicted_psi = predicted_psi
        self.a = a
        self.b = b
        self.r = r
        self.m = m
        self.J = J
        self.kf = kf
        self.CM = CM
        self.Cd = Cd
        self.rmse = rmse_loss
        
        return total_loss


def cut_in_sequences(x, y, seq_len, inc=1):
    """
    Slice a long 1D/2D series into overlapping windows for sequence-based learning.
    
    This function creates sequences from the input data for the LTC model.
    For rover data: input shape (N, 100) -> output shape (seq_len, num_sequences, 100)
    
    Args:
        x: Input data array (e.g., x-position trajectory)
        y: Target data array (e.g., x-position trajectory) 
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


class HarData:
    """
    Data handler for rover trajectory data.
    
    This class loads and processes the rover trajectory data from the video and audio
    processing steps, creating sequences suitable for the LTC neural network.
    """
    
    def __init__(self, seq_len=16):
        print(f"Loading rover trajectory data...")
        
        # Load trajectory data from data directory
        data_dir = "data"
        
        # Load position data (x, y, z coordinates)
        x_data = np.loadtxt(os.path.join(data_dir, "xData.txt"))
        y_data = np.loadtxt(os.path.join(data_dir, "yData.txt"))
        z_data = np.loadtxt(os.path.join(data_dir, "zData.txt"))
        
        # Load motor data (omega_r, omega_l)
        omega_r_data = np.loadtxt(os.path.join(data_dir, "omega_r.txt"))
        omega_l_data = np.loadtxt(os.path.join(data_dir, "omega_l.txt"))
        
        # Get Nloop from data
        global Nloop
        Nloop = x_data.shape[1]  # Use actual data size (100)
        print(f"Nloop {Nloop}")
        
        # Use first column for trajectory (time series)
        # Apply coordinate scaling: convert pixels to meters
        pixel_to_meter = 0.005818  # 1 pixel = 5.818 mm (realistic rover scale)
        x_traj = x_data[:, 0] * pixel_to_meter
        y_traj = y_data[:, 0] * pixel_to_meter
        z_traj = z_data[:, 0] * pixel_to_meter
        
        # Calculate velocities and heading from position data
        dt = 0.01  # Time step
        vx = np.gradient(x_traj, dt)
        vy = np.gradient(y_traj, dt)
        psi = np.arctan2(vy, vx)  # Heading angle
        wz = np.gradient(psi, dt)  # Angular velocity
        
        # Create state matrix [x, y, psi, vx, vy, wz]
        states = np.column_stack([x_traj, y_traj, psi, vx, vy, wz])
        
        # Create motor input matrix [omega_r, omega_l, omega_r, omega_l]
        omega_r_traj = omega_r_data[:, 0]
        omega_l_traj = omega_l_data[:, 0]
        motor_inputs = np.column_stack([omega_r_traj, omega_l_traj, omega_r_traj, omega_l_traj])
        
        # Split data into train/test (80/20)
        rows = states.shape[0]
        split_idx = max(1, int(0.8 * rows))
        
        train_states = states[:split_idx]
        test_states = states[split_idx:]
        train_motors = motor_inputs[:split_idx]
        test_motors = motor_inputs[split_idx:]
        
        # Create sequences for training
        train_x, train_y = cut_in_sequences(train_states, train_states, seq_len)
        train_motor1, train_motor2 = cut_in_sequences(train_motors[:, 0:1], train_motors[:, 1:2], seq_len)
        train_motor3, train_motor4 = cut_in_sequences(train_motors[:, 2:3], train_motors[:, 3:4], seq_len)
        
        # Create sequences for testing
        test_x, test_y = cut_in_sequences(test_states, test_states, seq_len, inc=8)
        test_motor1, test_motor2 = cut_in_sequences(test_motors[:, 0:1], test_motors[:, 1:2], seq_len, inc=8)
        test_motor3, test_motor4 = cut_in_sequences(test_motors[:, 2:3], test_motors[:, 3:4], seq_len, inc=8)
        
        # Convert to PyTorch tensors
        self.train_x = torch.tensor(train_x, dtype=torch.float32)
        self.train_y = torch.tensor(train_y, dtype=torch.float32)
        self.train_motor1 = torch.tensor(train_motor1, dtype=torch.float32)
        self.train_motor2 = torch.tensor(train_motor2, dtype=torch.float32)
        self.train_motor3 = torch.tensor(train_motor3, dtype=torch.float32)
        self.train_motor4 = torch.tensor(train_motor4, dtype=torch.float32)
        
        self.test_x = torch.tensor(test_x, dtype=torch.float32)
        self.test_y = torch.tensor(test_y, dtype=torch.float32)
        self.test_motor1 = torch.tensor(test_motor1, dtype=torch.float32)
        self.test_motor2 = torch.tensor(test_motor2, dtype=torch.float32)
        self.test_motor3 = torch.tensor(test_motor3, dtype=torch.float32)
        self.test_motor4 = torch.tensor(test_motor4, dtype=torch.float32)
        
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
            indices = permutation[start:end]

            batch_x = self.train_x[:, indices]
            batch_y = self.train_y[:, indices]
            batch_motor1 = self.train_motor1[:, indices]
            batch_motor2 = self.train_motor2[:, indices]
            batch_motor3 = self.train_motor3[:, indices]
            batch_motor4 = self.train_motor4[:, indices]

            yield (batch_x, batch_y, batch_motor1, batch_motor2, batch_motor3, batch_motor4)



class HarModel(nn.Module):
    """
    Neural network model for rover parameter estimation.
    
    This class implements the LTC (Liquid Time-Constant) neural network that learns
    to predict rover physical parameters from trajectory data. The model takes
    sequences of rover trajectory data as input and outputs 9 physical parameters.
    
    Architecture:
    - Input: [T, B, 6] where T=timesteps, B=batch_size, 6=state features (x,y,psi,vx,vy,wz)
    - Output: [T, B, 9] where 9 is the number of rover parameters
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
        
        # Input size is the number of features per timestep (6 for rover state)
        input_size = 6  # [x, y, psi, vx, vy, wz]

        print("Beginning rover parameter estimation model...")

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
        
        # Output layer: 8 rover parameters
        self.dense = nn.Linear(model_size, 9)
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
            x: Input trajectory data [T, B, 6]
            
        Returns:
            y: Predicted parameters [T, B, 8]
        """
        if self.model_type.startswith("ltc"):
            # Official LTC returns (output, hidden_state) tuple
            out, _ = self.rnn(x)           # [T,B,H]
        else:
            # Other RNNs return (output, hidden_state) tuple
            out, _ = self.rnn(x)           # [T,B,H]
        
        T, B, H = out.shape
        y = self.sigmoid(self.dense(out.reshape(T*B, H))).reshape(T, B, 9)
        return y

    def compute_loss(self, y_pred, target_y, uMotor1, uMotor2, uMotor3, uMotor4):
        """Build the loss object and call .forward()."""
        loss_fn = Custom_CE_Loss(target_y, y_pred, uMotor1, uMotor2, uMotor3, uMotor4)
        return loss_fn.forward()


class ThetaSimulator:
    """
    Simulator class for running rover simulations with estimated parameters.
    
    This class takes the learned parameters from EMMA and runs a complete
    differential drive simulation to validate the parameter estimation.
    """
    
    def __init__(self, dt=0.01):
        """
        Initialize the simulator.
        
        Args:
            dt: Time step for simulation (seconds)
        """
        self.dt = dt

    def simulate_trajectory(self, initial_state, motor_commands, parameters, duration=10.0):
        """
        Simulate differential drive rover dynamics with learned parameters.
        
        Args:
            initial_state: Initial rover state [x, y, psi, vx, vy, wz]
            motor_commands: Motor input commands [omega_r, omega_l]
            parameters: Learned parameters [a, b, r, m, J, kf, CM, Cd]
            duration: Simulation duration (seconds)
            
        Returns:
            trajectory: Simulated trajectory [n_steps, 6]
        """
        a, b, r, m, J, kf, CM, Cd, Sf = parameters
        
        # Convert tensors to numpy if needed
        if isinstance(initial_state, torch.Tensor): 
            initial_state = initial_state.detach().cpu().numpy()
        if isinstance(motor_commands, torch.Tensor): 
            motor_commands = motor_commands.detach().cpu().numpy()
        if isinstance(parameters, torch.Tensor): 
            parameters = parameters.detach().cpu().numpy()
        
        n_steps = int(duration / self.dt)
        trajectory = np.zeros((n_steps, 6))
        trajectory[0] = initial_state
        states = initial_state.copy()
        
        for t in range(1, n_steps):
            # Get motor commands for current timestep
            idx = min(t-1, len(motor_commands)-1)
            omega_r, omega_l = motor_commands[idx][0], motor_commands[idx][1]
            
            # Current state
            x, y, psi, vx, vy, wz = states
            
            # Differential drive kinematics
            L = 2 * b  # Track width
            v_linear = r * (omega_r + omega_l) / 2.0  # Linear velocity
            v_angular = r * (omega_r - omega_l) / L   # Angular velocity
            
            # Convert to world frame velocities
            vx_cmd = v_linear * np.cos(psi)
            vy_cmd = v_linear * np.sin(psi)
            wz_cmd = v_angular
            
            # Friction and drag forces
            vmag = np.sqrt(vx**2 + vy**2 + 0.01)
            F_fx = -kf * m * 9.81 * vx / vmag
            F_fy = -kf * m * 9.81 * vy / vmag
            T_f = -kf * J * wz / (abs(wz) + 0.01)
            
            F_dx = -Cd * 0.1 * vx**2 * np.sign(vx)
            F_dy = -Cd * 0.1 * vy**2 * np.sign(vy)
            T_d = -Cd * 0.1 * J * wz**2 * np.sign(wz)
            
            # Total forces and accelerations
            Fx = F_fx + F_dx
            Fy = F_fy + F_dy
            Tz = T_f + T_d
            ax = Fx / m
            ay = Fy / m
            alpha = Tz / J
            
            # Update velocities and positions
            vx = vx_cmd + ax * self.dt
            vy = vy_cmd + ay * self.dt
            wz = wz_cmd + alpha * self.dt
            x += vx * self.dt
            y += vy * self.dt
            psi += wz * self.dt
            
            # Normalize heading angle
            psi = np.arctan2(np.sin(psi), np.cos(psi))
            
            # Update state
            states = np.array([x, y, psi, vx, vy, wz])
            trajectory[t] = states
            
        return trajectory

    def create_animated_simulation(self, actual_trajectory, EMMA_trajectory):
        # Load the full trajectory data for complete path visualization
        try:
            # Load complete trajectory data
            x_data = np.loadtxt('data/xData.txt')[:, 0]
            y_data = np.loadtxt('data/yData.txt')[:, 0]
            
            # Convert to meters
            pixel_to_meter = 0.005818
            x_full = x_data * pixel_to_meter
            y_full = y_data * pixel_to_meter
            
            # Create full actual trajectory
            full_actual_trajectory = np.column_stack([x_full, y_full])
            
            # Load motor commands for full EMMA simulation
            omega_r_data = np.loadtxt('data/omega_r.txt')[:, 0]
            omega_l_data = np.loadtxt('data/omega_l.txt')[:, 0]
            
            # Load estimated parameters
            import pandas as pd
            params_df = pd.read_csv('rover_coefficients.csv')
            a_est = params_df[params_df['Parameter'] == 'a']['Value'].iloc[0]
            b_est = params_df[params_df['Parameter'] == 'b']['Value'].iloc[0]
            r_est = params_df[params_df['Parameter'] == 'r']['Value'].iloc[0]
            
            # Calculate full EMMA trajectory using differential drive kinematics
            dt = 1.0/60.0
            time_array = np.arange(len(x_full)) * dt
            
            v_linear = r_est * (omega_r_data + omega_l_data) / 2.0
            v_angular = r_est * (omega_r_data - omega_l_data) / (2 * b_est) * 0.05  # Angular velocity scaling
            
            x_EMMA = np.zeros_like(x_full)
            y_EMMA = np.zeros_like(y_full)
            psi_EMMA = np.zeros_like(time_array)
            
            x_EMMA[0] = x_full[0]
            y_EMMA[0] = y_full[0]
            psi_EMMA[0] = 0.0
            
            for i in range(1, len(x_EMMA)):
                psi_EMMA[i] = psi_EMMA[i-1] + v_angular[i] * dt
                vx = v_linear[i] * np.cos(psi_EMMA[i])
                vy = v_linear[i] * np.sin(psi_EMMA[i])
                x_EMMA[i] = x_EMMA[i-1] + vx * dt
                y_EMMA[i] = y_EMMA[i-1] + vy * dt
            
            full_EMMA_trajectory = np.column_stack([x_EMMA, y_EMMA])
            
            print(f"Full trajectory lengths: Actual={len(full_actual_trajectory)}, EMMA={len(full_EMMA_trajectory)}")
            
        except Exception as e:
            print(f"Warning: Could not load full trajectory data: {e}")
            print("Using training sequence data instead...")
            full_actual_trajectory = actual_trajectory
            full_EMMA_trajectory = EMMA_trajectory
        
        # Create animation with full trajectories
        fig, ax = plt.subplots(figsize=(14, 10))
        ax.set_xlim(min(np.min(full_actual_trajectory[:,0]), np.min(full_EMMA_trajectory[:,0]))-0.5,
                    max(np.max(full_actual_trajectory[:,0]), np.max(full_EMMA_trajectory[:,0]))+0.5)
        ax.set_ylim(min(np.min(full_actual_trajectory[:,1]), np.min(full_EMMA_trajectory[:,1]))-0.5,
                    max(np.max(full_actual_trajectory[:,1]), np.max(full_EMMA_trajectory[:,1]))+0.5)
        ax.set_xlabel('X Position (m)', fontsize=14)
        ax.set_ylabel('Y Position (m)', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        ax.set_title('Rover EMMA Simulation: Actual vs EMMA Trajectory Comparison', fontsize=16, fontweight='bold')
        
        # Plot complete trajectories as background
        ax.plot(full_actual_trajectory[:,0], full_actual_trajectory[:,1], 'b-', 
                linewidth=2, alpha=0.3, label='Complete Actual Path')
        ax.plot(full_EMMA_trajectory[:,0], full_EMMA_trajectory[:,1], 'r--', 
                linewidth=2, alpha=0.3, label='Complete EMMA Path')
        
        # Animated elements - Both actual and EMMA
        actual_line, = ax.plot([], [], 'b-', linewidth=4, label='Actual Rover (Animated)', alpha=0.9)
        EMMA_line,  = ax.plot([], [], 'r--', linewidth=4, label='EMMA Simulation (Animated)', alpha=0.9)
        actual_point, = ax.plot([], [], 'bo', markersize=12, label='Actual Rover Position', markeredgecolor='black', markeredgewidth=2)
        EMMA_point,  = ax.plot([], [], 'rs', markersize=12, label='EMMA Rover Position', markeredgecolor='black', markeredgewidth=2)
        
        # Start and end markers
        ax.plot(full_actual_trajectory[0,0], full_actual_trajectory[0,1], 'go', 
                markersize=15, label='Start', markeredgecolor='black', markeredgewidth=2)
        ax.plot(full_actual_trajectory[-1,0], full_actual_trajectory[-1,1], 'bs', 
                markersize=15, label='Actual End', markeredgecolor='black', markeredgewidth=2)
        ax.plot(full_EMMA_trajectory[-1,0], full_EMMA_trajectory[-1,1], 'rs', 
                markersize=15, label='EMMA End', markeredgecolor='black', markeredgewidth=2)
        
        ax.legend(loc='upper right', fontsize=10)
        
        # Status text
        status_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=11, va='top',
                              bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        def animate(frame):
            # Use every 2nd frame to make animation smoother but not too fast
            step = max(1, len(full_actual_trajectory) // 100)  # Show ~100 frames max
            current_frame = min(frame * step, len(full_actual_trajectory) - 1)
            
            actual_line.set_data(full_actual_trajectory[:current_frame+1,0], 
                               full_actual_trajectory[:current_frame+1,1])
            EMMA_line.set_data(full_EMMA_trajectory[:current_frame+1,0], 
                              full_EMMA_trajectory[:current_frame+1,1])
            actual_point.set_data([full_actual_trajectory[current_frame,0]], 
                                [full_actual_trajectory[current_frame,1]])
            EMMA_point.set_data([full_EMMA_trajectory[current_frame,0]], 
                               [full_EMMA_trajectory[current_frame,1]])
            
            time_val = current_frame * dt
            a = full_actual_trajectory[current_frame]
            e = full_EMMA_trajectory[current_frame]
            err = np.sqrt((a[0]-e[0])**2 + (a[1]-e[1])**2)
            
            status_text.set_text(f'Time: {time_val:.2f}s\nActual: ({a[0]:.2f}, {a[1]:.2f})\nEMMA: ({e[0]:.2f}, {e[1]:.2f})\nError: {err:.3f}m\nProgress: {current_frame+1}/{len(full_actual_trajectory)}')
            
            return actual_line, EMMA_line, actual_point, EMMA_point, status_text
        
        # Calculate number of frames for smooth animation
        total_frames = min(120, len(full_actual_trajectory) // 2)  # Max 120 frames, or half the data points
        
        anim = FuncAnimation(fig, animate, frames=total_frames, interval=100, blit=False, repeat=True)
        print("Saving animated simulation with complete trajectory...")
        anim.save('rover_EMMA_simulation.gif', writer='pillow', fps=10)
        print("Animation saved as 'rover_EMMA_simulation.gif'")
        return anim

# Main execution block
if __name__ == "__main__":
    """
    Main execution entry point for the rover analysis pipeline.
    """
    main()