#!/usr/bin/env python3
"""
Run architecture ablation for Pendulum 90° and 150°
Appends results to architecture_ablation_results.csv
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

Nloop = 0


# Import classes from architecture_ablation.py
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from architecture_ablation import (
    cut_in_sequences, 
    PendulumData, 
    ArchitectureModel
)


class Custom_Pendulum_Loss_Generic(nn.Module):
    """Physics-informed loss for pendulum with configurable ground truth."""
    
    def __init__(self, labels, logits, omega, alpha_nominal, beta_nominal):
        super().__init__()
        self.y_true = labels
        self.y_pred = logits
        self.y_omega = omega
        self.alpha_nominal = alpha_nominal
        self.beta_nominal = beta_nominal
        
    def forward(self):
        dev = self.y_pred.device
        T, B, _ = self.y_pred.shape
        
        maxChange = 95.0
        getp = lambda k: self.y_pred[:,:,k]
        
        gamma_nominal = 100.0
        
        alpha = (1 + (0.5 - getp(0)) * maxChange / 100.0) * self.alpha_nominal
        beta = (1 + (0.5 - getp(1)) * maxChange / 100.0) * self.beta_nominal
        gamma = (1 + (0.5 - getp(2)) * maxChange / 100.0) * gamma_nominal
        
        L = alpha
        tau = beta
        g = torch.tensor(9.81, device=dev)
        
        thetaVal = self.y_true[:,:,0]
        omegaVal = self.y_omega[:,:,0]
        theta = thetaVal.clone().unsqueeze(2)
        omega = omegaVal.clone().unsqueeze(2)
        
        limitLoop = min(500, T)
        tau_dt = 0.03
        
        for i in range(1, limitLoop):
            y1 = theta[:,:,i-1] + omega[:,:,i-1]*tau_dt
            y0 = omega[:,:,i-1] + (-torch.mul(tau,omega[:,:,i-1]) - torch.mul(torch.div(g,L.clamp(min=0.0001)),torch.sin(theta[:,:,i-1])))*tau_dt
            
            theta = torch.cat([theta, y1.unsqueeze(2)],dim=2)
            omega = torch.cat([omega, y0.unsqueeze(2)],dim=2)
        
        loss_Cal_theta = gamma * 0.01
        loss_Cal_omega = gamma * 0.005
        
        mse_loss = torch.abs(torch.sum(torch.square(self.y_true[:,:,0:limitLoop]-theta)/limitLoop, dim=2)-loss_Cal_theta) + \
                   torch.abs(torch.sum(torch.square(self.y_omega[:,:,0:limitLoop]-omega)/limitLoop, dim=2)-loss_Cal_omega)
        
        param_penalty = 0.0
        param_penalty += 10.0 * torch.mean(torch.relu(-alpha))
        param_penalty += 10.0 * torch.mean(torch.relu(-beta))
        param_penalty += 2.0 * torch.mean(torch.relu(alpha - 2.0))
        param_penalty += 2.0 * torch.mean(torch.relu(beta - 1.0))
        param_penalty += 1.0 * torch.mean(torch.relu(gamma - 500.0))
        
        total_loss = mse_loss + 0.001 * param_penalty
        
        self.L = L
        self.tau = tau
        
        return mse_loss


def run_ablation_for_dataset(dataset_name, data_dirs, alpha_nominal, beta_nominal):
    """
    Run architecture ablation for a specific pendulum dataset.
    
    Args:
        dataset_name: Name of dataset (e.g., "90", "150")
        data_dirs: List of data directories for each video
        alpha_nominal: Ground truth length (m)
        beta_nominal: Ground truth damping (1/s)
    """
    import random
    
    print(f"\n{'='*70}")
    print(f"Running ablation for Pendulum {dataset_name}°")
    print(f"Ground Truth: L={alpha_nominal}m, τ={beta_nominal} 1/s")
    print(f"{'='*70}")
    
    architectures = ["LTC", "LSTM", "GRU", "TRANSFORMER"]
    results = []
    
    for video_idx, data_dir in enumerate(data_dirs, start=1):
        print(f"\n{'='*70}")
        print(f"Processing Video {video_idx}/5: {data_dir}")
        print(f"{'='*70}")
        
        for arch_name in architectures:
            print(f"\n  Testing {arch_name}...")
            
            # Set seeds
            random.seed(42)
            np.random.seed(42)
            torch.manual_seed(42)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(42)
            
            # Load data
            dataset = PendulumData(seq_len=16, data_dir=data_dir)
            
            # Create model
            model = ArchitectureModel(
                model_type=arch_name.lower(),
                model_size=64,
                learning_rate=0.0003
            ).to(device)
            
            optimizer = model.optimizer
            scheduler = model.scheduler
            
            # Training
            num_epochs = 40
            batch_size = 2
            best_loss = float('inf')
            start_time = time.time()
            
            for epoch in range(num_epochs):
                model.train()
                epoch_loss = 0.0
                batch_count = 0
                
                for batch_x, batch_y, batch_omega, _ in dataset.iterate_train(batch_size=batch_size):
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)
                    batch_omega = batch_omega.to(device)
                    
                    optimizer.zero_grad()
                    predicted_params = model(batch_x)
                    
                    # Use generic loss with configurable ground truth
                    loss_fn = Custom_Pendulum_Loss_Generic(
                        batch_y, predicted_params, batch_omega,
                        alpha_nominal, beta_nominal
                    )
                    loss = loss_fn.forward()
                    
                    # Loss might be a tensor, take mean
                    if loss.dim() > 0:
                        loss = loss.mean()
                    
                    if torch.isnan(loss):
                        continue
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    batch_count += 1
                
                if batch_count > 0:
                    avg_loss = epoch_loss / batch_count
                    scheduler.step()
                    
                    if avg_loss < best_loss:
                        best_loss = avg_loss
            
            training_time = time.time() - start_time
            
            # Evaluate
            model.eval()
            with torch.no_grad():
                sample_batch = next(iter(dataset.iterate_train(batch_size=1)))
                sample_x = sample_batch[0].to(device)
                predicted_params = model(sample_x)
                
                maxChange = 95.0
                getp = lambda k: predicted_params[:,:,k].mean()
                
                L_estimated = ((1 + (0.5 - getp(0)) * maxChange / 100.0) * alpha_nominal).item()
                tau_estimated = ((1 + (0.5 - getp(1)) * maxChange / 100.0) * beta_nominal).item()
                
                print(f"    L={L_estimated:.6f}m, τ={tau_estimated:.6f} 1/s, Loss={best_loss:.2f}, Time={training_time:.2f}s")
                
                results.append({
                    'dataset': f'{dataset_name}_v{video_idx}',
                    'architecture': arch_name,
                    'hidden_units': 64,
                    'video_number': video_idx,
                    'best_loss': best_loss,
                    'training_time_s': training_time,
                    'L_estimated': L_estimated,
                    'tau_estimated': tau_estimated
                })
    
    return results


def main():
    """Run ablations for 90° and 150° pendulums and append to CSV."""
    
    # Check if data directories exist
    base_dir = "Pendulum-EMMA"
    
    # Pendulum 90° configuration
    data_dirs_90 = [
        os.path.join(base_dir, "90_v1", "data"),
        os.path.join(base_dir, "90_v2", "data"),
        os.path.join(base_dir, "90_v3", "data"),
        os.path.join(base_dir, "90_v4", "data"),
        os.path.join(base_dir, "90_v5", "data"),
    ]
    
    # Pendulum 150° configuration
    data_dirs_150 = [
        os.path.join(base_dir, "150_v1", "data"),
        os.path.join(base_dir, "150_v2", "data"),
        os.path.join(base_dir, "150_v3", "data"),
        os.path.join(base_dir, "150_v4", "data"),
        os.path.join(base_dir, "150_v5", "data"),
    ]
    
    # Check if directories exist
    print("Checking data directories...")
    for dirs, name in [(data_dirs_90, "90°"), (data_dirs_150, "150°")]:
        for d in dirs:
            if os.path.exists(d):
                print(f"  ✓ {d}")
            else:
                print(f"  ✗ {d} NOT FOUND")
    
    all_results = []
    
    # Run ablation for 90°
    if all(os.path.exists(d) for d in data_dirs_90):
        results_90 = run_ablation_for_dataset("90", data_dirs_90, 0.90, 0.05)
        all_results.extend(results_90)
    else:
        print("\n⚠️  Skipping 90° - data directories not found")
    
    # Run ablation for 150°
    if all(os.path.exists(d) for d in data_dirs_150):
        results_150 = run_ablation_for_dataset("150", data_dirs_150, 1.50, 0.05)
        all_results.extend(results_150)
    else:
        print("\n⚠️  Skipping 150° - data directories not found")
    
    # Append to existing CSV
    if all_results:
        csv_file = 'architecture_ablation_results.csv'
        print(f"\n{'='*70}")
        print(f"Appending {len(all_results)} results to {csv_file}")
        print(f"{'='*70}")
        
        with open(csv_file, 'a', newline='') as f:
            if all_results:
                writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
                # Don't write header since we're appending
                writer.writerows(all_results)
        
        print(f"✅ Results appended successfully!")
    else:
        print("\n❌ No results to append - check data directories")


if __name__ == "__main__":
    main()
