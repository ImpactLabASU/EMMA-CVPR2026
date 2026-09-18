# EMMA Pendulum Pipeline (calibration-corrected)
#
# Changes vs. the original:
#   1. limitLoop indexes the FRAME dimension (y_true.shape[2]), not seq_len.
#   2. Symplectic (semi-implicit) Euler integrator -> stable over long horizons.
#   3. Loss is parameterized: maxChange, calibration C (loss_Cal_theta/omega),
#      and GT/nominal (L_nominal, tau_nominal) are arguments, not hardcoded.
#   4. calibrate_C(): one no-backprop pass at maxChange=0 (params pinned at GT)
#      extracts loss_Cal_theta / loss_Cal_omega for THIS length + horizon + dt.
#   5. Guidance loss removed (it injected the answer). Recovery is tested instead
#      via an optional perturbed init.
#
# Usage per length: set L_nominal (e.g. 0.45 / 0.90 / 1.50), keep tau_dt equal
# to 1/fps of the source video, and let the pipeline calibrate then train.
 
import os
import csv
import gc
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from ncps.torch import LTC
 
try:
    import psutil
    _HAS_PSUTIL = True
except Exception:
    _HAS_PSUTIL = False
 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
 
Nloop = 0
 
 
def check_memory_usage():
    if not _HAS_PSUTIL:
        return
    mem = psutil.virtual_memory()
    used_gb = mem.used / (1024 ** 3)
    total_gb = mem.total / (1024 ** 3)
    print(f"[INFO] Memory usage: {used_gb:.1f}GB / {total_gb:.1f}GB ({mem.percent:.1f}%)")
 
 
# ============================================================================
# STEP 1: VIDEO PROCESSING (detection + tracking + angular conversion)
# ============================================================================
 
class PendulumDetector:
    def __init__(self, weights_path, conf=0.15, imgsz=640):
        from ultralytics import YOLO
        self.model = YOLO(weights_path)
        self.conf = conf
        self.imgsz = imgsz
        self.last_detection = None
        print(f"[INFO] Loaded YOLO weights: {weights_path}")
 
    def detect(self, frame):
        h, w = frame.shape[:2]
        img_area = w * h
        min_area_px = max(100, int(0.00001 * img_area))
        max_area_px = int(0.1 * img_area)
 
        results = self.model.predict(
            source=frame, imgsz=self.imgsz, conf=self.conf,
            iou=0.5, agnostic_nms=True, verbose=False
        )
 
        candidates = []
        for r in results:
            if r.boxes is None:
                continue
            for b in r.boxes:
                x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
                area = (x2 - x1) * (y2 - y1)
                if area < min_area_px or area > max_area_px:
                    continue
                if x1 < 5 or y1 < 5 or x2 > w - 5 or y2 > h - 5:
                    continue
                conf = float(b.conf[0].item()) if hasattr(b, "conf") else 0.0
                candidates.append((x1, y1, x2, y2, conf))
 
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]
 
        candidates.sort(key=lambda t: t[4], reverse=True)
        top_candidates = candidates[:3]
 
        if self.last_detection is not None:
            lx1, ly1, lx2, ly2, _ = self.last_detection
            lcx, lcy = (lx1 + lx2) / 2.0, (ly1 + ly2) / 2.0
 
            def dist(c):
                cx, cy = (c[0] + c[2]) / 2.0, (c[1] + c[3]) / 2.0
                return ((cx - lcx) ** 2 + (cy - lcy) ** 2) ** 0.5
 
            return min(top_candidates, key=dist)
        return top_candidates[0]
 
 
class Kalman2D:
    def __init__(self, dt=0.01):
        self.dt = dt
        self.state = np.zeros(4)
        self.F = np.eye(4)
        self.F[0, 2] = dt
        self.F[1, 3] = dt
        self.H = np.eye(2, 4)
        self.Q = np.eye(4) * 0.1
        self.R = np.eye(2) * 1.0
        self.P = np.eye(4) * 100.0
 
    def predict(self):
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.state
 
    def update(self, measurement):
        y = measurement - self.H @ self.state
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.state = self.state + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        return self.state
 
 
class PendulumCoordinateConverter:
    def __init__(self, pivot_point, pixel_to_meter=0.001):
        self.pivot_x = pivot_point[0]
        self.pivot_y = pivot_point[1]
        self.pixel_to_meter = pixel_to_meter
        self.angle_history = []
        self.length_history = []
        self.max_history = 10
 
    def pixel_to_angle(self, bob_x, bob_y):
        dx = bob_x - self.pivot_x
        dy = bob_y - self.pivot_y
        theta = np.arctan2(dx, dy)
        self.angle_history.append(theta)
        if len(self.angle_history) > self.max_history:
            self.angle_history.pop(0)
        if len(self.angle_history) >= 3:
            weights = np.linspace(0.5, 1.0, len(self.angle_history))
            weights = weights / np.sum(weights)
            theta = np.average(self.angle_history, weights=weights)
        return theta
 
 
def process_pendulum_video(video_path, weights_path, output_video, output_csv, conf=0.15):
    print(f"[STEP 1] Processing pendulum video: {video_path}")
 
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    detector = PendulumDetector(weights_path, conf=conf)
    kf = Kalman2D()
 
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
    csvw.writerow(["frame", "time_s", "x_pixel", "y_pixel", "z_pixel",
                   "theta_rad", "omega_rad_s", "conf"])
 
    pivot_point = (width // 2, height // 8)
    coord_converter = PendulumCoordinateConverter(pivot_point)
 
    x_series, y_series, z_series, theta_series, omega_series = [], [], [], [], []
    frame_idx = 0
    dt = 1.0 / fps
 
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_time = frame_idx / fps
        det = detector.detect(frame)
 
        if det is not None:
            x1, y1, x2, y2, conf_val = det
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            kf.predict()
            xs = kf.update(np.array([cx, cy], dtype=float)).squeeze()
            xk, yk = float(xs[0]), float(xs[1])
            conf_out = conf_val
        else:
            xs = kf.predict().squeeze()
            xk, yk = float(xs[0]), float(xs[1])
            conf_out = 0.0
 
        theta = coord_converter.pixel_to_angle(xk, yk)
        dx, dy = xk - pivot_point[0], yk - pivot_point[1]
        length_pixels = np.sqrt(dx ** 2 + dy ** 2)
        z_estimate = length_pixels * 0.1
 
        if frame_idx > 0 and len(theta_series) > 0:
            omega = (theta - theta_series[-1]) / dt
        else:
            omega = 0.0
 
        x_series.append(xk); y_series.append(yk); z_series.append(z_estimate)
        theta_series.append(theta); omega_series.append(omega)
 
        if det is not None:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.circle(frame, (int(xk), int(yk)), 6, (0, 0, 255), -1)
        else:
            cv2.circle(frame, (int(xk), int(yk)), 5, (0, 255, 255), -1)
        cv2.circle(frame, (int(pivot_point[0]), int(pivot_point[1])), 4, (255, 0, 0), -1)
 
        csvw.writerow([frame_idx, f"{frame_time:.3f}", f"{xk:.2f}", f"{yk:.2f}",
                       f"{z_estimate:.2f}", f"{theta:.5f}", f"{omega:.5f}", f"{conf_out:.3f}"])
        out.write(frame)
        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"[PROGRESS] Processed {frame_idx} frames")
            check_memory_usage()
 
    cap.release(); out.release(); csv_f.close()
 
    if theta_series:
        theta_arr = np.array(theta_series)
        omega_arr = np.array(omega_series)
        x_arr, y_arr, z_arr = np.array(x_series), np.array(y_series), np.array(z_series)
 
        data_dir = os.path.dirname(output_csv)
        os.makedirs(data_dir, exist_ok=True)
 
        # Nx100 tiled format kept for pipeline compatibility.
        def tile(a): return np.tile(a.reshape(-1, 1), (1, 100))
        np.savetxt(os.path.join(data_dir, "thetaData.txt"), tile(theta_arr), fmt='%.6f')
        np.savetxt(os.path.join(data_dir, "omegaData.txt"), tile(omega_arr), fmt='%.6f')
        np.savetxt(os.path.join(data_dir, "xData.txt"), tile(x_arr), fmt='%.6f')
        np.savetxt(os.path.join(data_dir, "yData.txt"), tile(y_arr), fmt='%.6f')
        np.savetxt(os.path.join(data_dir, "zData.txt"), tile(z_arr), fmt='%.6f')
        gc.collect()
        print(f"[STEP 1] Saved trajectory data: {len(theta_series)} frames (fps={fps:.3f}, dt={dt:.5f})")
 
        output_dir = os.path.dirname(output_video) or "output"
        os.makedirs(output_dir, exist_ok=True)
        t = np.arange(len(theta_series)) / fps
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        ax1.plot(t, theta_series, 'b-', lw=2); ax1.set_title('Theta vs Time')
        ax1.set_xlabel('Time (s)'); ax1.set_ylabel('theta (rad)'); ax1.grid(True, alpha=0.3)
        ax2.plot(t, omega_series, 'r-', lw=2); ax2.set_title('Omega vs Time')
        ax2.set_xlabel('Time (s)'); ax2.set_ylabel('omega (rad/s)'); ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'pendulum_trajectory_plot.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("[STEP 1] COMPLETED")
    return fps
 
 
# ============================================================================
# STEP 2: EMMA PARAMETER ESTIMATION
# ============================================================================
 
class Custom_Pendulum_Loss(nn.Module):
    """
    Calibration-based (recovery) loss.
 
    loss = |E_theta(p) - C_theta| + |E_omega(p) - C_omega| + 0.001 * penalty
 
    where E_* are the per-(T,B) trajectory errors between the measured signal and
    a symplectic-Euler simulation driven by the network's predicted parameters.
    C_theta, C_omega are the residuals the GT parameters produce (obtained from a
    maxChange=0 calibration pass). At GT, loss -> 0.
    """
 
    def __init__(self, labels, logits, omega,
                 maxChange=95.0, loss_Cal_theta=200.0, loss_Cal_omega=40.0,
                 L_nominal=1.50, tau_nominal=0.05, tau_dt=0.03):
        super().__init__()
        self.y_true = labels       # [T, B, F]  F = number of frames
        self.y_pred = logits       # [T, B, 4]
        self.y_omega = omega       # [T, B, F]
        self.maxChange = maxChange
        self.loss_Cal_theta = loss_Cal_theta
        self.loss_Cal_omega = loss_Cal_omega
        self.L_nominal = L_nominal
        self.tau_nominal = tau_nominal
        self.tau_dt = tau_dt
 
    def forward(self):
        dev = self.y_pred.device
        getp = lambda k: self.y_pred[:, :, k]                 # [T,B]
 
        # Normalized [0,1] outputs -> physical params centered on nominal(=GT).
        # maxChange=0 pins params exactly at nominal regardless of network output.
        alpha = (1 + (0.5 - getp(0)) * self.maxChange / 100.0) * self.L_nominal    # length L  [T,B]
        beta = (1 + (0.5 - getp(1)) * self.maxChange / 100.0) * self.tau_nominal   # damping tau [T,B]
        L = alpha
        tau = beta
 
        g = torch.tensor(9.81, device=dev)
        dt = self.tau_dt
 
        # Horizon indexes the FRAME dimension, capped at 500.
        limitLoop = min(500, self.y_true.shape[2])
 
        # Initial condition = measured frame 0.
        theta = self.y_true[:, :, 0:1].clone()      # [T,B,1]
        omega = self.y_omega[:, :, 0:1].clone()     # [T,B,1]
 
        # Symplectic (semi-implicit) Euler: update omega first, use it for theta.
        for i in range(1, limitLoop):
            th = theta[:, :, i - 1]
            om = omega[:, :, i - 1]
            om_new = om + (-tau * om - (g / L.clamp(min=1e-4)) * torch.sin(th)) * dt
            th_new = th + om_new * dt
            theta = torch.cat([theta, th_new.unsqueeze(2)], dim=2)
            omega = torch.cat([omega, om_new.unsqueeze(2)], dim=2)
 
        base_theta = torch.sum(torch.square(self.y_true[:, :, 0:limitLoop] - theta) / limitLoop, dim=2)
        base_omega = torch.sum(torch.square(self.y_omega[:, :, 0:limitLoop] - omega) / limitLoop, dim=2)
        self.base_theta = base_theta      # exposed for calibration
        self.base_omega = base_omega
 
        mse_loss = (torch.abs(base_theta - self.loss_Cal_theta)
                    + torch.abs(base_omega - self.loss_Cal_omega))

        #mse_loss = torch.abs(base_theta) + torch.abs(base_omega)
 
        # Physical-plausibility penalties (no GT guidance term).
        penalty = 0.0
        penalty += 10.0 * torch.mean(torch.relu(-alpha))
        penalty += 10.0 * torch.mean(torch.relu(-beta))
        penalty += 2.0 * torch.mean(torch.relu(alpha - 3.0))
        penalty += 2.0 * torch.mean(torch.relu(beta - 1.0))
 
        self.L = L
        self.tau = tau
        self.rmse = torch.sqrt(mse_loss + 1e-12)
        return mse_loss + 0.001 * penalty
 
 
def cut_in_sequences(x, y, seq_len, inc=1):
    sx, sy = [], []
    for s in range(0, x.shape[0] - seq_len, inc):
        sx.append(x[s:s + seq_len]); sy.append(y[s:s + seq_len])
    return np.stack(sx, axis=1), np.stack(sy, axis=1)
 
 
class PendulumData:
    def __init__(self, seq_len=16, data_dir="data"):
        print("Loading pendulum trajectory data...")
        theta_data = np.loadtxt(os.path.join(data_dir, "thetaData.txt"))
        omega_data = np.loadtxt(os.path.join(data_dir, "omegaData.txt"))
        theta_traj = theta_data.T      # [100, N]
        omega_traj = omega_data.T
        global Nloop
        Nloop = theta_traj.shape[1]    # N frames
        print(f"Nloop {Nloop}")
 
        train_x, train_y = cut_in_sequences(theta_traj, theta_traj, seq_len)
        train_omega, train_omega_y = cut_in_sequences(omega_traj, omega_traj, seq_len)
 
        self.train_x = torch.tensor(train_x, dtype=torch.float32)
        self.train_y = torch.tensor(train_y, dtype=torch.float32)
        self.train_omega = torch.tensor(train_omega, dtype=torch.float32)
        self.train_omega_y = torch.tensor(train_omega_y, dtype=torch.float32)
        print(f"Training sequences: {self.train_x.shape[1]}")
 
    def iterate_train(self, batch_size=32):
        total = self.train_x.shape[1]
        for i in range(total // batch_size):
            s, e = i * batch_size, i * batch_size + batch_size
            yield (self.train_x[:, s:e], self.train_y[:, s:e],
                   self.train_omega[:, s:e], self.train_omega_y[:, s:e])
 
 
class PendulumModel(nn.Module):
    def __init__(self, model_type="ltc", model_size=64, learning_rate=0.005, perturb_init=0.0):
        super().__init__()
        self.model_type = model_type
        self.model_size = model_size
        input_size = Nloop if Nloop > 0 else 100
 
        if model_type.startswith("ltc"):
            self.wm = LTC(input_size=input_size, units=model_size, return_sequences=True,
                          batch_first=False, mixed_memory=False, ode_unfolds=6, epsilon=1e-8)
            self.rnn = self.wm
        elif model_type == "lstm":
            self.rnn = nn.LSTM(input_size, model_size, batch_first=False)
        else:
            self.rnn = nn.RNN(input_size, model_size, batch_first=False)
 
        self.dense = nn.Linear(model_size, 4)
        self.sigmoid = nn.Sigmoid()
 
        # Recovery test: bias the final layer so params start OFF-GT.
        # perturb_init != 0 shifts sigmoid output away from 0.5.
        if perturb_init != 0.0:
            with torch.no_grad():
                self.dense.bias.fill_(float(perturb_init))
 
        self.optimizer = optim.AdamW(self.parameters(), lr=learning_rate,
                                     weight_decay=1e-4, betas=(0.9, 0.999), eps=1e-8)
        self.to(device)
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2, eta_min=1e-6)
 
    def forward(self, x):
        out, _ = self.rnn(x)
        T, B, H = out.shape
        return self.sigmoid(self.dense(out.reshape(T * B, H))).reshape(T, B, 4)
 
    def compute_loss(self, y_pred, target_y, omega, **kw):
        self.loss_fn = Custom_Pendulum_Loss(target_y, y_pred, omega, **kw)
        return self.loss_fn.forward()
 
 
@torch.no_grad()
def calibrate_C(dataset, model, L_nominal, tau_nominal, tau_dt, batch_size=2):
    """
    Extract loss_Cal_theta / loss_Cal_omega = residual at GT params, using the
    SAME integrator / dt / horizon as training. maxChange=0 pins params at GT,
    so the network output is irrelevant here.
    """
    model.eval()
    Et = Ev = 0.0
    n = 0
    for bx, by, bo, boy in dataset.iterate_train(batch_size=batch_size):
        bx, by, bo = bx.to(device), by.to(device), bo.to(device)
        pred = model(bx)
        model.compute_loss(pred, by, bo, maxChange=0.0,
                           loss_Cal_theta=0.0, loss_Cal_omega=0.0,
                           L_nominal=L_nominal, tau_nominal=tau_nominal, tau_dt=tau_dt)
        Et += model.loss_fn.base_theta.mean().item()
        Ev += model.loss_fn.base_omega.mean().item()
        n += 1
    C_theta, C_omega = Et / n, Ev / n
    print(f"[CALIBRATE] loss_Cal_theta = {C_theta:.6f}   loss_Cal_omega = {C_omega:.6f}"
          f"   (sum = {C_theta + C_omega:.6f})")
    return C_theta, C_omega
 
 
def run_pendulum_emma_optimization(output_folder="", L_nominal=1.50, tau_nominal=0.05,
                                   maxChange=95.0, tau_dt=0.03, perturb_init=0.0,
                                   seq_len=16, batch_size=2, num_epochs=40):
    import random
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
 
    print("[STEP 2] EMMA pendulum optimization")
    print(f"[STEP 2] L_nominal={L_nominal}  tau_nominal={tau_nominal}  "
          f"maxChange={maxChange}  tau_dt={tau_dt}  perturb_init={perturb_init}")
 
    data_dir = os.path.join(output_folder, "data") if output_folder else "data"
    dataset = PendulumData(seq_len=seq_len, data_dir=data_dir)
 
    model = PendulumModel(model_type="ltc", model_size=64,
                          perturb_init=perturb_init).to(device)
 
    # --- 1) Calibrate C at GT (independent of any init perturbation) ---
    C_theta, C_omega = calibrate_C(dataset, model, L_nominal, tau_nominal, tau_dt, batch_size)
 
    optimizer, scheduler = model.optimizer, model.scheduler
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
 
    # --- 2) Train with calibrated C, no guidance ---
    best_loss = float('inf'); patience, patience_counter = 50, 0
    model_path = os.path.join(output_folder, 'pendulum_emma_final_model.pth') if output_folder \
        else 'pendulum_emma_final_model.pth'
 
    for epoch in range(num_epochs):
        model.train()
        epoch_loss, bt, bv, nb = 0.0, 0.0, 0.0, 0
        for bx, byy, bo, boy in dataset.iterate_train(batch_size=batch_size):
            bx, byy, bo = bx.to(device), byy.to(device), bo.to(device)
            optimizer.zero_grad()
            pred = model(bx)
            loss = model.compute_loss(pred, byy, bo, maxChange=maxChange,
                                      loss_Cal_theta=C_theta, loss_Cal_omega=C_omega,
                                      L_nominal=L_nominal, tau_nominal=tau_nominal,
                                      tau_dt=tau_dt).mean()
            if torch.isnan(loss):
                print(f"NaN loss at epoch {epoch}"); continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
            bt += model.loss_fn.base_theta.mean().item()
            bv += model.loss_fn.base_omega.mean().item()
            nb += 1
 
        if nb == 0:
            print(f"No batches at epoch {epoch}"); continue
        avg = epoch_loss / nb
        scheduler.step()
        print(f"Epoch {epoch:3d}  loss={avg:.6f}  E_theta={bt/nb:.4f} (C={C_theta:.4f})  "
              f"E_omega={bv/nb:.4f} (C={C_omega:.4f})")
 
        if avg < best_loss:
            best_loss = avg; patience_counter = 0
            torch.save({'model_state_dict': model.state_dict(),
                        'C_theta': C_theta, 'C_omega': C_omega,
                        'epoch': epoch, 'loss': avg}, model_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}"); break
 
    # --- 3) Report estimate + recovery error ---
    model.load_state_dict(torch.load(model_path, map_location=device)['model_state_dict'])
    model.eval()
    with torch.no_grad():
        sx, sy, so, soy = next(iter(dataset.iterate_train(batch_size=1)))
        pred = model(sx.to(device))
        getp = lambda k: pred[:, :, k].mean()
        alpha = (1 + (0.5 - getp(0)) * maxChange / 100.0) * L_nominal
        beta = (1 + (0.5 - getp(1)) * maxChange / 100.0) * tau_nominal
 
        csv_path = os.path.join(output_folder, 'pendulum_coefficients.csv') if output_folder \
            else 'pendulum_coefficients.csv'
        with open(csv_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['Parameter', 'Value', 'GT', 'AbsError', 'Units'])
            w.writerow(['alpha (L)', f"{alpha.item():.6f}", f"{L_nominal:.6f}",
                        f"{abs(alpha.item() - L_nominal):.6f}", 'm'])
            w.writerow(['beta (tau)', f"{beta.item():.6f}", f"{tau_nominal:.6f}",
                        f"{abs(beta.item() - tau_nominal):.6f}", '1/s'])
 
        print("\n=== ESTIMATED PENDULUM PARAMETERS ===")
        print(f"alpha (L):  {alpha.item():.6f} m   (GT {L_nominal:.4f}, "
              f"err {abs(alpha.item() - L_nominal):.6f}, "
              f"{100*abs(alpha.item()-L_nominal)/L_nominal:.2f}%)")
        print(f"beta (tau): {beta.item():.6f} 1/s (GT {tau_nominal:.4f}, "
              f"err {abs(beta.item() - tau_nominal):.6f})")
    return {'L': alpha.item(), 'tau': beta.item(), 'C_theta': C_theta, 'C_omega': C_omega}
 
 
def main():
    import sys
    import argparse
 
    parser = argparse.ArgumentParser()
    parser.add_argument("--length-cm", type=int, default=150,
                        help="Pendulum length in cm (45/90/150). L_nominal = length/100.")
    parser.add_argument("--tau-dt", type=float, default=0.03,
                        help="Sim timestep; MUST equal 1/fps of the source videos.")
    parser.add_argument("--perturb-init", type=float, default=0.0,
                        help="Final-layer bias; nonzero starts params off-GT (recovery test).")
    parser.add_argument("--skip-video", action="store_true",
                        help="Reuse existing data/*.txt, skip STEP 1.")
    # Run-level GPU sharding: launch one process per GPU with a distinct shard.
    parser.add_argument("--num-shards", type=int, default=1,
                        help="Total parallel processes (e.g. number of GPUs).")
    parser.add_argument("--shard", type=int, default=0,
                        help="This process's shard index in [0, num-shards).")
    args = parser.parse_args()
 
    LENGTH_CM = args.length_cm
    L_nominal = LENGTH_CM / 100.0
    tau_nominal = 0.05
    tau_dt = args.tau_dt
    maxChange = 95.0
    perturb_init = args.perturb_init
    weights_path = "yolo11m.pt"
    run_video = not args.skip_video
 
    all_configs = [
        (f"../../data/output_selected/pendulum/pendulum_{LENGTH_CM}/0{i}/video.mp4",
         f"{LENGTH_CM}_v{i}")
        for i in range(1, 6)
    ]
    # Each process handles every num-shards-th config -> disjoint, no coordination.
    video_configs = all_configs[args.shard::args.num_shards]
    print(f"[SHARD {args.shard}/{args.num_shards}] handling {len(video_configs)} run(s) "
          f"on device {device}: {[c[1] for c in video_configs]}")
 
    for video_path, out_folder in video_configs:
        print("\n" + "#" * 70)
        print(f"Pendulum {LENGTH_CM}cm  video={video_path}  out={out_folder}")
        print("#" * 70)
        os.makedirs(f"{out_folder}/output", exist_ok=True)
        os.makedirs(f"{out_folder}/data", exist_ok=True)
        try:
            if run_video:
                fps = process_pendulum_video(
                    video_path, weights_path,
                    f"{out_folder}/output/annotated_pendulum.mp4",
                    f"{out_folder}/data/pendulum_trajectory.csv")
                print(f"[NOTE] Set tau_dt to 1/fps = {1.0/fps:.5f} for consistent calibration.")
            run_pendulum_emma_optimization(
                output_folder=out_folder, L_nominal=L_nominal, tau_nominal=tau_nominal,
                maxChange=maxChange, tau_dt=tau_dt, perturb_init=perturb_init)
            print(f"PIPELINE COMPLETED for {out_folder}")
        except Exception as e:
            print(f"PIPELINE FAILED for {video_path}: {e}")
            continue
 
 
if __name__ == "__main__":
    main()
