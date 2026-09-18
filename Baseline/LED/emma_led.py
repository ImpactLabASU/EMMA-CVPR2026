# EMMA LED Pipeline (calibration-corrected)
#
# Changes vs. the original (mirrors the pendulum fixes):
#   1. limitLoop indexes the FRAME dimension (y_true.shape[2]), not seq_len.
#   2. gamma is broadcast [T,B] across the horizon (removed gamma[t_idx], which
#      indexed the size-16 sequence axis and crashed for any horizon > 16).
#   3. Target is y_true[:,:,0:limitLoop] (frame axis), not the sequence-axis
#      reconstruction the original used.
#   4. Loss is parameterized: maxChange, calibration C (loss_Cal_I), and
#      gamma_nominal(=GT) are arguments, not hardcoded.
#   5. calibrate_C(): one no-backprop pass at maxChange=0 (gamma pinned at GT)
#      extracts loss_Cal_I for THIS gamma + horizon + dt.
#   6. Guide loss removed (it injected the answer). Recovery is tested instead
#      via an optional perturbed init.
#
# LED decay is first-order (dI/dt = -gamma*I): forward Euler is stable/monotone,
# so no symplectic change is needed. Set tau_dt = 1/fps of the source video.
 
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
    print(f"[INFO] Memory usage: {mem.used/(1024**3):.1f}GB / "
          f"{mem.total/(1024**3):.1f}GB ({mem.percent:.1f}%)")
 
 
# ============================================================================
# STEP 1: VIDEO PROCESSING (mean-frame intensity -> normalized trajectory)
# ============================================================================
 
def process_led_video(video_path, output_csv):
    print(f"[STEP 1] Processing LED video: {video_path}")
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
 
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
 
    fps = cap.get(cv2.CAP_PROP_FPS)
    csv_f = open(output_csv, "w", newline="")
    csvw = csv.writer(csv_f)
    csvw.writerow(["frame", "time_s", "intensity", "intensity_normalized"])
 
    intensity_series = []
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        intensity_series.append(float(np.mean(gray)))
        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"[PROGRESS] Processed {frame_idx} frames")
            check_memory_usage()
    cap.release()
 
    if not intensity_series:
        print("[STEP 1] No intensity data extracted"); csv_f.close(); return 0, fps
 
    arr = np.array(intensity_series)
    lo, hi = arr.min(), arr.max()
    norm = (arr - lo) / (hi - lo) if hi > lo else np.ones_like(arr)
 
    for idx, (raw, nv) in enumerate(zip(arr, norm)):
        csvw.writerow([idx, f"{idx/fps:.3f}", f"{raw:.2f}", f"{nv:.6f}"])
    csv_f.close()
 
    print(f"[STEP 1] I_0={norm[0]:.3f}  I_n={norm[-1]:.3f}  drop={norm[0]-norm[-1]:.3f}")
    if norm[-1] >= norm[0]:
        print("[STEP 1] WARNING: intensity does not decrease")
 
    data_dir = os.path.dirname(output_csv)
    os.makedirs(data_dir, exist_ok=True)
    np.savetxt(os.path.join(data_dir, "IData.txt"),
               np.tile(norm.reshape(-1, 1), (1, 100)), fmt='%.6f')
    gc.collect()
    print(f"[STEP 1] Saved IData.txt: {len(arr)} frames (fps={fps:.3f}, dt={1.0/fps:.5f})")
 
    t = np.arange(len(norm)) / fps
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.plot(t, norm, 'b-', lw=2)
    ax.set_xlabel('Time (s)'); ax.set_ylabel('Normalized Intensity')
    ax.set_title('LED Intensity vs Time'); ax.grid(True, alpha=0.3)
    plt.savefig(os.path.join(data_dir, "led_intensity_trajectory.png"), dpi=150, bbox_inches='tight')
    plt.close()
    return len(arr), fps
 
 
# ============================================================================
# STEP 2: EMMA PARAMETER ESTIMATION
# ============================================================================
 
def cut_in_sequences(x, y, seq_len, inc=1):
    sx, sy = [], []
    for s in range(0, x.shape[0] - seq_len, inc):
        sx.append(x[s:s + seq_len]); sy.append(y[s:s + seq_len])
    return np.stack(sx, axis=1), np.stack(sy, axis=1)
 
 
class Custom_LED_Loss(nn.Module):
    """
    Calibration-based (recovery) loss for LED decay.
 
    loss = |E_I(p) - C_I| + 0.01 * penalty
 
    E_I is the per-(T,B) trajectory error between measured intensity and a
    forward-Euler simulation of dI/dt = -gamma*I driven by the network's gamma.
    C_I is the residual GT gamma produces (from a maxChange=0 calibration pass).
    At GT, loss -> 0. No guide loss.
    """
 
    def __init__(self, labels, logits, maxChange=95.0, loss_Cal_I=0.0,
                 gamma_nominal=0.46, tau_dt=0.01):
        super().__init__()
        self.y_true = labels        # [T, B, F]  F = frames
        self.y_pred = logits        # [T, B, 1]
        self.maxChange = maxChange
        self.loss_Cal_I = loss_Cal_I
        self.gamma_nominal = gamma_nominal
        self.tau_dt = tau_dt
 
    def forward(self):
        dev = self.y_pred.device
        eps = 1e-6
        getp = lambda k: self.y_pred[:, :, k]                 # [T,B]
 
        # maxChange=0 pins gamma at nominal(=GT) regardless of network output.
        gamma = (1 + (0.5 - getp(0)) * self.maxChange / 100.0) * self.gamma_nominal  # [T,B]
 
        limitLoop = min(500, self.y_true.shape[2])
        dt = self.tau_dt
 
        I = self.y_true[:, :, 0:1].clone()      # [T,B,1]  init = measured frame 0
        for i in range(1, limitLoop):
            I_prev = I[:, :, i - 1]
            I_safe = torch.clamp(I_prev, min=eps)
            I_new = I_prev + (-gamma * I_safe) * dt          # gamma broadcast [T,B]
            I_new = torch.clamp(I_new, min=0.0, max=1.0)
            I = torch.cat([I, I_new.unsqueeze(2)], dim=2)
 
        base_I = torch.sum(torch.square(self.y_true[:, :, 0:limitLoop] - I) / limitLoop, dim=2)
        self.base_I = base_I                                 # exposed for calibration
 
        mse_loss = torch.abs(base_I - self.loss_Cal_I)
 
        penalty = 0.0
        penalty += 50.0 * torch.mean(torch.relu(-gamma))     # gamma > 0
        penalty += 20.0 * torch.mean(torch.relu(gamma - 10.0))
        penalty += 20.0 * torch.mean(torch.relu(0.01 - gamma))
 
        self.gamma = gamma
        self.rmse = torch.sqrt(mse_loss + 1e-12)
        return mse_loss + 0.01 * penalty
 
 
class LEDData:
    def __init__(self, seq_len=16, data_dir="data"):
        print("Loading LED intensity trajectory data...")
        I_data = np.loadtxt(os.path.join(data_dir, "IData.txt"))
        I_traj = I_data.T      # [100, N]
        global Nloop
        Nloop = I_traj.shape[1]
        print(f"Nloop {Nloop}")
        train_x, train_y = cut_in_sequences(I_traj, I_traj, seq_len)
        self.train_x = torch.tensor(train_x, dtype=torch.float32)
        self.train_y = torch.tensor(train_y, dtype=torch.float32)
        print(f"Training sequences: {self.train_x.shape[1]}")
 
    def iterate_train(self, batch_size=32):
        total = self.train_x.shape[1]
        for i in range(total // batch_size):
            s, e = i * batch_size, i * batch_size + batch_size
            yield (self.train_x[:, s:e], self.train_y[:, s:e])
 
 
class LEDModel(nn.Module):
    def __init__(self, model_type="ltc", model_size=64, learning_rate=0.005, perturb_init=0.0):
        super().__init__()
        self.model_type = model_type
        input_size = Nloop if Nloop > 0 else 100
 
        if model_type.startswith("ltc"):
            self.wm = LTC(input_size=input_size, units=model_size, return_sequences=True,
                          batch_first=False, mixed_memory=False, ode_unfolds=8, epsilon=1e-10)
            self.rnn = self.wm
        elif model_type == "lstm":
            self.rnn = nn.LSTM(input_size, model_size, batch_first=False)
        else:
            self.rnn = nn.RNN(input_size, model_size, batch_first=False)
 
        self.dense = nn.Linear(model_size, 1)
        self.sigmoid = nn.Sigmoid()
 
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
        return self.sigmoid(self.dense(out.reshape(T * B, H))).reshape(T, B, 1)
 
    def compute_loss(self, y_pred, target_y, **kw):
        self.loss_fn = Custom_LED_Loss(target_y, y_pred, **kw)
        return self.loss_fn.forward()
 
 
@torch.no_grad()
def calibrate_C(dataset, model, gamma_nominal, tau_dt, batch_size=2):
    """Extract loss_Cal_I = residual at GT gamma, same integrator/dt/horizon as training."""
    model.eval()
    Ei, n = 0.0, 0
    for bx, by in dataset.iterate_train(batch_size=batch_size):
        bx, by = bx.to(device), by.to(device)
        pred = model(bx)
        model.compute_loss(pred, by, maxChange=0.0, loss_Cal_I=0.0,
                           gamma_nominal=gamma_nominal, tau_dt=tau_dt)
        Ei += model.loss_fn.base_I.mean().item()
        n += 1
    C_I = Ei / n
    print(f"[CALIBRATE] loss_Cal_I = {C_I:.6f}")
    return C_I
 
 
def run_led_emma_optimization(output_folder="", gamma_nominal=0.46, maxChange=95.0,
                              tau_dt=0.01, perturb_init=0.0,
                              seq_len=16, batch_size=2, num_epochs=40):
    import random
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
 
    print("[STEP 2] EMMA LED optimization")
    print(f"[STEP 2] gamma_nominal={gamma_nominal}  maxChange={maxChange}  "
          f"tau_dt={tau_dt}  perturb_init={perturb_init}")
 
    data_dir = os.path.join(output_folder, "data") if output_folder else "data"
    dataset = LEDData(seq_len=seq_len, data_dir=data_dir)
    model = LEDModel(model_type="ltc", model_size=64, perturb_init=perturb_init).to(device)
 
    # --- 1) Calibrate C at GT ---
    C_I = calibrate_C(dataset, model, gamma_nominal, tau_dt, batch_size)
 
    optimizer, scheduler = model.optimizer, model.scheduler
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
 
    # --- 2) Train with calibrated C, no guidance ---
    best_loss = float('inf'); patience, patience_counter = 50, 0
    model_path = os.path.join(output_folder, 'led_emma_final_model.pth') if output_folder \
        else 'led_emma_final_model.pth'
 
    for epoch in range(num_epochs):
        model.train()
        epoch_loss, bi, nb = 0.0, 0.0, 0
        for bx, by in dataset.iterate_train(batch_size=batch_size):
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            pred = model(bx)
            loss = model.compute_loss(pred, by, maxChange=maxChange, loss_Cal_I=C_I,
                                      gamma_nominal=gamma_nominal, tau_dt=tau_dt).mean()
            if torch.isnan(loss):
                print(f"NaN loss at epoch {epoch}"); continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
            bi += model.loss_fn.base_I.mean().item()
            nb += 1
 
        if nb == 0:
            print(f"No batches at epoch {epoch}"); continue
        avg = epoch_loss / nb
        scheduler.step()
        print(f"Epoch {epoch:3d}  loss={avg:.6f}  E_I={bi/nb:.6f} (C={C_I:.6f})")
 
        if avg < best_loss:
            best_loss = avg; patience_counter = 0
            torch.save({'model_state_dict': model.state_dict(), 'C_I': C_I,
                        'epoch': epoch, 'loss': avg}, model_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}"); break
 
    # --- 3) Report estimate + recovery error ---
    model.load_state_dict(torch.load(model_path, map_location=device)['model_state_dict'])
    model.eval()
    with torch.no_grad():
        sx, sy = next(iter(dataset.iterate_train(batch_size=1)))
        pred = model(sx.to(device))
        getp = lambda k: pred[:, :, k].mean()
        gamma = (1 + (0.5 - getp(0)) * maxChange / 100.0) * gamma_nominal
 
        csv_path = os.path.join(output_folder, 'led_coefficients.csv') if output_folder \
            else 'led_coefficients.csv'
        with open(csv_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['Parameter', 'Value', 'GT', 'AbsError', 'Units'])
            w.writerow(['gamma', f"{gamma.item():.6f}", f"{gamma_nominal:.6f}",
                        f"{abs(gamma.item() - gamma_nominal):.6f}", '1/s'])
 
        print("\n=== ESTIMATED LED PARAMETER ===")
        print(f"gamma: {gamma.item():.6f} 1/s   (GT {gamma_nominal:.4f}, "
              f"err {abs(gamma.item() - gamma_nominal):.6f}, "
              f"{100*abs(gamma.item()-gamma_nominal)/gamma_nominal:.2f}%)")
    return {'gamma': gamma.item(), 'C_I': C_I}
 
 
def main():
    import argparse
 
    parser = argparse.ArgumentParser()
    parser.add_argument("--gamma-nominal", type=float, default=0.46,
                        help="GT decay constant for this configuration (e.g. led_10s -> 0.46).")
    parser.add_argument("--tau-dt", type=float, default=0.01,
                        help="Sim timestep; set to 1/fps of the source video.")
    parser.add_argument("--maxChange", type=float, default=95.0)
    parser.add_argument("--perturb-init", type=float, default=0.0,
                        help="Final-layer bias; nonzero starts gamma off-GT (recovery test).")
    parser.add_argument("--config", type=str, default="led_10s",
                        help="Config name used for the source path and output folders.")
    parser.add_argument("--skip-video", action="store_true")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard", type=int, default=0)
    args = parser.parse_args()
 
    weights_cfg = args.config
    all_configs = [
        (f"../../data/output_selected/led/{weights_cfg}/0{i}/video.mp4", f"{weights_cfg}_v{i}")
        for i in range(1, 6)
    ]
    video_configs = all_configs[args.shard::args.num_shards]
    print(f"[SHARD {args.shard}/{args.num_shards}] on {device}: {[c[1] for c in video_configs]}")
 
    for video_path, out_folder in video_configs:
        print("\n" + "#" * 70)
        print(f"LED {weights_cfg}  video={video_path}  out={out_folder}")
        print("#" * 70)
        os.makedirs(f"{out_folder}/data", exist_ok=True)
        try:
            Idata_path = os.path.join(out_folder, "data", "IData.txt")
            if (not args.skip_video) and os.path.exists(video_path):
                n, fps = process_led_video(video_path, os.path.join(out_folder, "data", "led_trajectory.csv"))
                if n:
                    print(f"[NOTE] Set --tau-dt to 1/fps = {1.0/fps:.5f} for consistent calibration.")
            elif not os.path.exists(Idata_path):
                print(f"No video and no IData.txt for {out_folder}; skipping."); continue
 
            run_led_emma_optimization(
                output_folder=out_folder, gamma_nominal=args.gamma_nominal,
                maxChange=args.maxChange, tau_dt=args.tau_dt, perturb_init=args.perturb_init)
            print(f"PIPELINE COMPLETED for {out_folder}")
        except Exception as e:
            print(f"PIPELINE FAILED for {video_path}: {e}")
            continue
 
 
if __name__ == "__main__":
    main()
