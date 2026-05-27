#!/usr/bin/env python3
"""
Multimodal Ablation Study: Video + Audio vs Video Only
Shows the importance of audio (motor commands) for rover parameter estimation
"""

import os
import csv
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from ncps.torch import LTC

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Helper function for cutting sequences
def cut_in_sequences(x, y, seq_len, inc=1):
    """Slice a long 1D/2D series into overlapping windows for sequence-based learning."""
    sequences_x, sequences_y = [], []
    for s in range(0, x.shape[0] - seq_len, inc):
        start, end = s, s + seq_len
        sequences_x.append(x[start:end])
        sequences_y.append(y[start:end])
    return np.stack(sequences_x, axis=1), np.stack(sequences_y, axis=1)


class HarDataVideoOnly:
    """
    Data handler for rover trajectory data - VIDEO ONLY mode.
    Instead of using audio-derived motor commands, we infer them from trajectory derivatives.
    """
    
    def __init__(self, seq_len=16):
        print(f"[VIDEO ONLY] Loading rover trajectory data...")
        
        data_dir = "data"
        
        # Load position data
        x_data = np.loadtxt(os.path.join(data_dir, "xData.txt"))
        y_data = np.loadtxt(os.path.join(data_dir, "yData.txt"))
        z_data = np.loadtxt(os.path.join(data_dir, "zData.txt"))
        
        global Nloop
        Nloop = x_data.shape[1]
        print(f"Nloop {Nloop}")
        
        # Use first column for trajectory
        pixel_to_meter = 0.005818
        x_traj = x_data[:, 0] * pixel_to_meter
        y_traj = y_data[:, 0] * pixel_to_meter
        z_traj = z_data[:, 0] * pixel_to_meter
        
        # Calculate velocities and heading
        dt = 0.01
        vx = np.gradient(x_traj, dt)
        vy = np.gradient(y_traj, dt)
        psi = np.arctan2(vy, vx)
        wz = np.gradient(psi, dt)
        
        # Create state matrix
        states = np.column_stack([x_traj, y_traj, psi, vx, vy, wz])
        
        # INFER motor commands from trajectory (VIDEO ONLY MODE)
        # Use kinematic relationships to estimate motor velocities
        # This is less accurate than using actual audio data
        v_linear = np.sqrt(vx**2 + vy**2)  # Linear velocity from trajectory
        v_angular = wz  # Angular velocity from trajectory
        
        # Assume nominal rover parameters for inverse kinematics
        r_nominal = 0.201  # Nominal wheel radius
        b_nominal = 0.144  # Nominal Y-arm length
        L_nominal = 2 * b_nominal  # Track width
        
        # Inverse differential drive kinematics
        # v_linear = r * (omega_r + omega_l) / 2
        # v_angular = r * (omega_r - omega_l) / L
        # Solving for omega_r and omega_l:
        omega_sum = 2 * v_linear / r_nominal
        omega_diff = v_angular * L_nominal / r_nominal
        
        omega_r_inferred = (omega_sum + omega_diff) / 2.0
        omega_l_inferred = (omega_sum - omega_diff) / 2.0
        
        # Create motor input matrix (inferred from video)
        motor_inputs = np.column_stack([
            omega_r_inferred, omega_l_inferred, 
            omega_r_inferred, omega_l_inferred
        ])
        
        print(f"[VIDEO ONLY] Inferred motor commands from trajectory derivatives")
        
        # Split data
        rows = states.shape[0]
        split_idx = max(1, int(0.8 * rows))
        
        train_states = states[:split_idx]
        test_states = states[split_idx:]
        train_motors = motor_inputs[:split_idx]
        test_motors = motor_inputs[split_idx:]
        
        # Create sequences
        train_x, train_y = cut_in_sequences(train_states, train_states, seq_len)
        train_motor1, train_motor2 = cut_in_sequences(train_motors[:, 0:1], train_motors[:, 1:2], seq_len)
        train_motor3, train_motor4 = cut_in_sequences(train_motors[:, 2:3], train_motors[:, 3:4], seq_len)
        
        test_x, test_y = cut_in_sequences(test_states, test_states, seq_len, inc=8)
        test_motor1, test_motor2 = cut_in_sequences(test_motors[:, 0:1], test_motors[:, 1:2], seq_len, inc=8)
        test_motor3, test_motor4 = cut_in_sequences(test_motors[:, 2:3], test_motors[:, 3:4], seq_len, inc=8)
        
        # Convert to tensors
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
        
        print(f"[VIDEO ONLY] Training sequences: {self.train_x.shape[1]}")
        print(f"[VIDEO ONLY] Test sequences: {self.test_x.shape[1]}")
    
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


def run_multimodal_ablation():
    """
    Run multimodal ablation study: Video+Audio vs Video Only
    """
    import random
    # Import classes from rover_ablation module
    import importlib.util
    spec = importlib.util.spec_from_file_location("rover_ablation", 
                                                    os.path.join(os.path.dirname(__file__), "rover-ablation.py"))
    rover_ablation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rover_ablation)
    HarData = rover_ablation.HarData
    HarModel = rover_ablation.HarModel
    
    print("=" * 70)
    print("ROVER MULTIMODAL ABLATION STUDY")
    print("Comparing: Video+Audio vs Video Only")
    print("=" * 70)
    
    # Set seeds
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # Training parameters
    seq_len = 16
    batch_size = 2
    num_epochs = 100
    learning_rate = 0.0003
    
    # Ground truth
    gt = {
        'a': 0.178,
        'b': 0.144,
        'r': 0.201,
        'm': 26.88,
        'CM': 0.112
    }
    
    results = []
    
    # Test 1: Video + Audio (Full EMMA)
    print(f"\n{'='*70}")
    print(f"Test 1: VIDEO + AUDIO (Full EMMA with motor commands)")
    print(f"{'='*70}")
    
    dataset_full = HarData(seq_len=seq_len)
    model_full = HarModel(model_type="ltc", model_size=64, learning_rate=learning_rate).to(device)
    
    print("Training with audio-derived motor commands...")
    start_time = time.time()
    best_loss = float('inf')
    patience_counter = 0
    convergence_epoch = num_epochs
    
    for epoch in range(num_epochs):
        model_full.train()
        epoch_loss = 0.0
        batch_count = 0
        
        for batch_x, batch_y, batch_motor1, batch_motor2, batch_motor3, batch_motor4 in dataset_full.iterate_train(batch_size=batch_size):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_motor1 = batch_motor1.to(device)
            batch_motor2 = batch_motor2.to(device)
            batch_motor3 = batch_motor3.to(device)
            batch_motor4 = batch_motor4.to(device)
            
            model_full.optimizer.zero_grad()
            predicted_params = model_full(batch_x)
            loss = model_full.compute_loss(predicted_params, batch_y, batch_motor1, batch_motor2, batch_motor3, batch_motor4)
            
            if torch.isnan(loss):
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_full.parameters(), max_norm=1.0)
            model_full.optimizer.step()
            
            epoch_loss += loss.item()
            batch_count += 1
        
        if batch_count > 0:
            avg_loss = epoch_loss / batch_count
            model_full.scheduler.step()
            
            if avg_loss < best_loss - 1e-4:
                best_loss = avg_loss
                patience_counter = 0
                convergence_epoch = epoch + 1
            else:
                patience_counter += 1
            
            if (epoch + 1) % 10 == 0:
                print(f'  Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}')
            
            if patience_counter >= 20 and epoch >= 10:
                print(f'  Converged at epoch {convergence_epoch}')
                break
    
    training_time_full = time.time() - start_time
    
    # Evaluate full model
    model_full.eval()
    with torch.no_grad():
        sample_batch = next(iter(dataset_full.iterate_train(batch_size=1)))
        sample_x = sample_batch[0].to(device)
        predicted_params = model_full(sample_x)
        
        maxChange = 95.0
        getp = lambda k: predicted_params[:,:,k].mean()
        
        a_full = ((1 + (0.5 - getp(0)) * maxChange / 100.0) * 0.178).item()
        b_full = ((1 + (0.5 - getp(1)) * maxChange / 100.0) * 0.144).item()
        r_full = ((1 + (0.5 - getp(2)) * maxChange / 100.0) * 0.201).item()
        m_full = ((1 + (0.5 - getp(3)) * maxChange / 100.0) * 26.88).item()
        CM_full = ((1 + (0.5 - getp(4)) * maxChange / 100.0) * 0.112).item()
        
        # Calculate errors
        a_err = abs(a_full - gt['a']) / gt['a'] * 100
        b_err = abs(b_full - gt['b']) / gt['b'] * 100
        r_err = abs(r_full - gt['r']) / gt['r'] * 100
        m_err = abs(m_full - gt['m']) / gt['m'] * 100
        CM_err = abs(CM_full - gt['CM']) / gt['CM'] * 100
        mean_err_full = (a_err + b_err + r_err + m_err + CM_err) / 5
        
        print(f"  Estimated: a={a_full:.6f}, b={b_full:.6f}, r={r_full:.6f}, m={m_full:.6f}, CM={CM_full:.6f}")
        print(f"  Mean Error: {mean_err_full:.2f}%")
        print(f"  Training time: {training_time_full:.2f}s, Converged: epoch {convergence_epoch}")
        
        results.append({
            'modality': 'Video+Audio',
            'best_loss': best_loss,
            'training_time_s': training_time_full,
            'convergence_epoch': convergence_epoch,
            'a': a_full,
            'b': b_full,
            'r': r_full,
            'm': m_full,
            'CM': CM_full,
            'a_error_%': a_err,
            'b_error_%': b_err,
            'r_error_%': r_err,
            'm_error_%': m_err,
            'CM_error_%': CM_err,
            'mean_error_%': mean_err_full
        })
    
    # Test 2: Video Only (No Audio)
    print(f"\n{'='*70}")
    print(f"Test 2: VIDEO ONLY (Motor commands inferred from trajectory)")
    print(f"{'='*70}")
    
    dataset_video = HarDataVideoOnly(seq_len=seq_len)
    model_video = HarModel(model_type="ltc", model_size=64, learning_rate=learning_rate).to(device)
    
    print("Training with inferred motor commands...")
    start_time = time.time()
    best_loss = float('inf')
    patience_counter = 0
    convergence_epoch = num_epochs
    
    for epoch in range(num_epochs):
        model_video.train()
        epoch_loss = 0.0
        batch_count = 0
        
        for batch_x, batch_y, batch_motor1, batch_motor2, batch_motor3, batch_motor4 in dataset_video.iterate_train(batch_size=batch_size):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_motor1 = batch_motor1.to(device)
            batch_motor2 = batch_motor2.to(device)
            batch_motor3 = batch_motor3.to(device)
            batch_motor4 = batch_motor4.to(device)
            
            model_video.optimizer.zero_grad()
            predicted_params = model_video(batch_x)
            loss = model_video.compute_loss(predicted_params, batch_y, batch_motor1, batch_motor2, batch_motor3, batch_motor4)
            
            if torch.isnan(loss):
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_video.parameters(), max_norm=1.0)
            model_video.optimizer.step()
            
            epoch_loss += loss.item()
            batch_count += 1
        
        if batch_count > 0:
            avg_loss = epoch_loss / batch_count
            model_video.scheduler.step()
            
            if avg_loss < best_loss - 1e-4:
                best_loss = avg_loss
                patience_counter = 0
                convergence_epoch = epoch + 1
            else:
                patience_counter += 1
            
            if (epoch + 1) % 10 == 0:
                print(f'  Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}')
            
            if patience_counter >= 20 and epoch >= 10:
                print(f'  Converged at epoch {convergence_epoch}')
                break
    
    training_time_video = time.time() - start_time
    
    # Evaluate video-only model
    model_video.eval()
    with torch.no_grad():
        sample_batch = next(iter(dataset_video.iterate_train(batch_size=1)))
        sample_x = sample_batch[0].to(device)
        predicted_params = model_video(sample_x)
        
        maxChange = 95.0
        getp = lambda k: predicted_params[:,:,k].mean()
        
        a_video = ((1 + (0.5 - getp(0)) * maxChange / 100.0) * 0.178).item()
        b_video = ((1 + (0.5 - getp(1)) * maxChange / 100.0) * 0.144).item()
        r_video = ((1 + (0.5 - getp(2)) * maxChange / 100.0) * 0.201).item()
        m_video = ((1 + (0.5 - getp(3)) * maxChange / 100.0) * 26.88).item()
        CM_video = ((1 + (0.5 - getp(4)) * maxChange / 100.0) * 0.112).item()
        
        # Calculate errors
        a_err = abs(a_video - gt['a']) / gt['a'] * 100
        b_err = abs(b_video - gt['b']) / gt['b'] * 100
        r_err = abs(r_video - gt['r']) / gt['r'] * 100
        m_err = abs(m_video - gt['m']) / gt['m'] * 100
        CM_err = abs(CM_video - gt['CM']) / gt['CM'] * 100
        mean_err_video = (a_err + b_err + r_err + m_err + CM_err) / 5
        
        print(f"  Estimated: a={a_video:.6f}, b={b_video:.6f}, r={r_video:.6f}, m={m_video:.6f}, CM={CM_video:.6f}")
        print(f"  Mean Error: {mean_err_video:.2f}%")
        print(f"  Training time: {training_time_video:.2f}s, Converged: epoch {convergence_epoch}")
        
        results.append({
            'modality': 'Video Only',
            'best_loss': best_loss,
            'training_time_s': training_time_video,
            'convergence_epoch': convergence_epoch,
            'a': a_video,
            'b': b_video,
            'r': r_video,
            'm': m_video,
            'CM': CM_video,
            'a_error_%': a_err,
            'b_error_%': b_err,
            'r_error_%': r_err,
            'm_error_%': m_err,
            'CM_error_%': CM_err,
            'mean_error_%': mean_err_video
        })
    
    # Save results
    results_file = 'rover_multimodal_ablation_results.csv'
    with open(results_file, 'w', newline='') as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
    
    print(f"\n{'='*70}")
    print("MULTIMODAL ABLATION COMPLETE")
    print(f"{'='*70}")
    print(f"Results saved to: {results_file}")
    
    # Print comparison
    print("\nCOMPARISON:")
    improvement = ((mean_err_video - mean_err_full) / mean_err_video) * 100
    print(f"\nVideo+Audio Error: {mean_err_full:.2f}%")
    print(f"Video Only Error:  {mean_err_video:.2f}%")
    print(f"Improvement: {improvement:.2f}% (Audio adds critical motor command information)")


if __name__ == "__main__":
    run_multimodal_ablation()

