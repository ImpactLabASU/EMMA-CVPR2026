# EMMA Architecture Ablation Study: LTC vs LSTM vs GRU vs Transformer
# Critical Ablation A.1 for Paper

import os
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from ncps.torch import LTC

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

Nloop = 0


class Custom_Pendulum_Loss(nn.Module):
    """Physics-informed loss identical to run-45 pipeline."""
    
    def __init__(self, labels, logits, omega):
        super().__init__()
        self.y_true = labels
        self.y_pred = logits
        self.y_omega = omega
    
    def forward(self):
        dev = self.y_pred.device
        T, B, _ = self.y_pred.shape
        
        maxChange = 95.0
        getp = lambda k: self.y_pred[:,:,k]
        
        alpha_nominal = 0.45
        beta_nominal = 0.05
        gamma_nominal = 100.0
        
        alpha = (1 + (0.5 - getp(0)) * maxChange / 100.0) * alpha_nominal
        beta = (1 + (0.5 - getp(1)) * maxChange / 100.0) * beta_nominal
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
        param_penalty += 10.0 * torch.mean(torch.relu(-gamma))
        param_penalty += 2.0 * torch.mean(torch.relu(alpha - 2.0))
        param_penalty += 2.0 * torch.mean(torch.relu(beta - 1.0))
        param_penalty += 1.0 * torch.mean(torch.relu(gamma - 500.0))
        
        total_loss = mse_loss + 0.001 * param_penalty
        
        self.L = L
        self.tau = tau
        return total_loss


def cut_in_sequences(x, y, seq_len, inc=1):
    sequences_x, sequences_y = [], []
    for s in range(0, x.shape[0] - seq_len, inc):
        start, end = s, s + seq_len
        sequences_x.append(x[start:end])
        sequences_y.append(y[start:end])
    return np.stack(sequences_x, axis=1), np.stack(sequences_y, axis=1)


class PendulumData:
    def __init__(self, seq_len=16, data_dir="data"):
        print(f"Loading data from {data_dir}...")
        theta_data = np.loadtxt(os.path.join(data_dir, "thetaData.txt"))
        omega_data = np.loadtxt(os.path.join(data_dir, "omegaData.txt"))
        theta_traj = theta_data.T
        omega_traj = omega_data.T
        global Nloop
        Nloop = theta_traj.shape[1]
        
        train_x, train_y = cut_in_sequences(theta_traj, theta_traj, seq_len)
        train_omega, train_omega_y = cut_in_sequences(omega_traj, omega_traj, seq_len)
        
        self.train_x = torch.tensor(train_x, dtype=torch.float32)
        self.train_y = torch.tensor(train_y, dtype=torch.float32)
        self.train_omega = torch.tensor(train_omega, dtype=torch.float32)
        self.train_omega_y = torch.tensor(train_omega_y, dtype=torch.float32)
        
        print(f"Training sequences: {self.train_x.shape[1]}")
    
    def iterate_train(self, batch_size=32):
        total_seqs = self.train_x.shape[1]
        total_batches = max(1, total_seqs // batch_size)
        for i in range(total_batches):
            start = i * batch_size
            end = start + batch_size
            yield (self.train_x[:, start:end], self.train_y[:, start:end], 
                   self.train_omega[:, start:end], self.train_omega_y[:, start:end])


class ArchitectureModel(nn.Module):
    """Unified model supporting LTC, LSTM, GRU, and Transformer."""
    
    def __init__(self, model_type="ltc", model_size=64, learning_rate=0.0003):
        super().__init__()
        self.model_type = model_type.lower()
        self.model_size = model_size
        
        input_size = Nloop if Nloop > 0 else 100
        print(f"Building {model_type.upper()} model with {model_size} units...")
        
        if self.model_type == "ltc":
            self.wm = LTC(
                input_size=input_size,
                units=model_size,
                return_sequences=True,
                batch_first=False,
                mixed_memory=False,
                ode_unfolds=6,
                epsilon=1e-8
            )
            self.rnn = self.wm
            learning_rate = 0.005
        elif self.model_type == "lstm":
            self.rnn = nn.LSTM(input_size, model_size, batch_first=False)
        elif self.model_type == "gru":
            self.rnn = nn.GRU(input_size, model_size, batch_first=False)
        elif self.model_type == "transformer":
            nhead = 4
            padded_input_size = ((input_size + nhead - 1) // nhead) * nhead
            if input_size != padded_input_size:
                self.input_projection = nn.Linear(input_size, padded_input_size)
            else:
                self.input_projection = None
            
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=padded_input_size, 
                nhead=nhead, 
                dim_feedforward=model_size,
                batch_first=False
            )
            self.rnn = nn.TransformerEncoder(encoder_layer, num_layers=2)
            self.transformer_out = nn.Linear(padded_input_size, model_size)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.dense = nn.Linear(model_size, 3)
        self.sigmoid = nn.Sigmoid()
        
        self.optimizer = optim.AdamW(self.parameters(), lr=learning_rate, 
                                    weight_decay=1e-4, betas=(0.9, 0.999), eps=1e-8)
        self.to(device)
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
    
    def forward(self, x):
        if self.model_type == "ltc":
            out, _ = self.rnn(x)
        elif self.model_type == "transformer":
            if hasattr(self, 'input_projection') and self.input_projection is not None:
                T, B, _ = x.shape
                x = self.input_projection(x.reshape(T*B, -1)).reshape(T, B, -1)
            out = self.rnn(x)
            out = self.transformer_out(out)
        else:  # LSTM, GRU
            out, _ = self.rnn(x)
        
        T, B, H = out.shape
        y = self.sigmoid(self.dense(out.reshape(T*B, H))).reshape(T, B, 3)
        return y
    
    def compute_loss(self, y_pred, target_y, omega):
        loss_fn = Custom_Pendulum_Loss(target_y, y_pred, omega)
        return loss_fn.forward()


def run_architecture_ablation():
    """Run architecture comparison: LTC vs LSTM vs GRU vs Transformer."""
    
    # Get absolute path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Configuration
    pendulum_45_datasets = [
        "Pendulum-EMMA/45_v1",
        "Pendulum-EMMA/45_v2",
        "Pendulum-EMMA/45_v3",
        "Pendulum-EMMA/45_v4",
        "Pendulum-EMMA/45_v5"
    ]
    
    # Architecture comparison (all with 64 hidden units for fair comparison)
    architectures = ["ltc", "lstm", "gru", "transformer"]
    
    # Training parameters
    seq_len = 16
    batch_size = 2
    num_epochs = 40
    learning_rate = 0.0003
    
    results = []
    
    print("=" * 70)
    print("EMMA ARCHITECTURE ABLATION STUDY")
    print("Comparison: LTC vs LSTM vs GRU vs Transformer (NODE)")
    print("=" * 70)
    
    total_experiments = len(architectures) * len(pendulum_45_datasets)
    experiment_num = 0
    
    for dataset_path in pendulum_45_datasets:
        dataset_name = os.path.basename(dataset_path)
        print(f"\n{'='*70}")
        print(f"DATASET: {dataset_name}")
        print(f"{'='*70}")
        
        # Load data
        data_dir = os.path.join(script_dir, dataset_path, "data")
        if not os.path.exists(data_dir):
            print(f"[WARNING] Data directory not found: {data_dir}")
            continue
            
        dataset = PendulumData(seq_len=seq_len, data_dir=data_dir)
        
        for arch_name in architectures:
            experiment_num += 1
            print(f"\n[{experiment_num}/{total_experiments}] Architecture: {arch_name.upper()}, Dataset: {dataset_name}")
            
            # Set seed
            seed = int(dataset_name.split('_v')[-1])
            np.random.seed(seed)
            torch.manual_seed(seed)
            
            # Create model
            model = ArchitectureModel(model_type=arch_name, model_size=64, learning_rate=learning_rate).to(device)
            optimizer = model.optimizer
            scheduler = model.scheduler
            
            # Training loop with convergence tracking
            best_loss = float('inf')
            convergence_epoch = num_epochs
            loss_history = []
            import time
            start_time = time.time()
            patience = 5
            patience_counter = 0
            convergence_threshold = 1e-4  # Loss improvement threshold
            
            for epoch in range(num_epochs):
                model.train()
                epoch_loss = 0.0
                batch_count = 0
                
                for batch_x, batch_y, batch_omega, batch_omega_y in dataset.iterate_train(batch_size=batch_size):
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)
                    batch_omega = batch_omega.to(device)
                    
                    optimizer.zero_grad()
                    predicted_params = model(batch_x)
                    loss_mat = model.compute_loss(predicted_params, batch_y, batch_omega)
                    loss = loss_mat.mean()
                    
                    if torch.isnan(loss):
                        continue
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    batch_count += 1
                
                if batch_count > 0:
                    avg_loss = epoch_loss / batch_count
                    loss_history.append(avg_loss)
                    scheduler.step()
                    
                    # Track convergence (when loss improvement is minimal)
                    if avg_loss < best_loss - convergence_threshold:
                        best_loss = avg_loss
                        patience_counter = 0
                        convergence_epoch = epoch + 1
                    else:
                        patience_counter += 1
                    
                    if (epoch + 1) % 10 == 0:
                        print(f'  Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}')
                    
                    # Early stopping if converged
                    if patience_counter >= patience and epoch >= 10:
                        print(f'  Converged at epoch {convergence_epoch}')
                        break
            
            training_time = time.time() - start_time
            final_loss = loss_history[-1] if loss_history else best_loss
            
            # Evaluate
            model.eval()
            with torch.no_grad():
                sample_batch = next(iter(dataset.iterate_train(batch_size=1)))
                sample_x, sample_y, sample_omega, sample_omega_y = sample_batch
                
                sample_x = sample_x.to(device)
                predicted_params = model(sample_x)
                
                maxChange = 95.0
                getp = lambda k: predicted_params[:,:,k].mean()
                
                L = ((1 + (0.5 - getp(0)) * maxChange / 100.0) * 0.45).item()
                tau = ((1 + (0.5 - getp(1)) * maxChange / 100.0) * 0.05).item()
                
                # Calculate parameter recovery accuracy (how close to ground truth)
                L_gt = 0.45  # Ground truth for 45° pendulum
                tau_gt = 0.05  # Ground truth damping
                L_error = abs(L - L_gt) / L_gt * 100  # Percentage error
                tau_error = abs(tau - tau_gt) / tau_gt * 100
                
                print(f"  L: {L:.6f} m (error: {L_error:.2f}%), tau: {tau:.6f} 1/s (error: {tau_error:.2f}%)")
                print(f"  Time: {training_time:.2f}s, Converged: epoch {convergence_epoch}, Final loss: {final_loss:.6f}")
                
                results.append({
                    'dataset': dataset_name,
                    'architecture': arch_name.upper(),
                    'hidden_units': 64,
                    'video_number': seed,
                    'best_loss': best_loss,
                    'final_loss': final_loss,
                    'training_time_s': training_time,
                    'convergence_epoch': convergence_epoch,
                    'L_estimated': L,
                    'tau_estimated': tau,
                    'L_error_percent': L_error,
                    'tau_error_percent': tau_error
                })
    
    # Save results
    results_file = os.path.join(script_dir, "architecture_ablation_results.csv")
    print(f"\n{'='*70}")
    print("SAVING RESULTS")
    print(f"{'='*70}")
    
    with open(results_file, 'w', newline='') as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
    
    print(f"Results saved to: {results_file}")
    
    # Print summary with key metrics
    print(f"\n{'='*70}")
    print("RESULTS SUMMARY - KEY METRICS FOR PAPER")
    print(f"{'='*70}")
    
    for arch_name in architectures:
        arch_results = [r for r in results if r['architecture'] == arch_name.upper()]
        if arch_results:
            L_values = [r['L_estimated'] for r in arch_results]
            tau_values = [r['tau_estimated'] for r in arch_results]
            L_errors = [r['L_error_percent'] for r in arch_results]
            tau_errors = [r['tau_error_percent'] for r in arch_results]
            time_values = [r['training_time_s'] for r in arch_results]
            conv_epochs = [r['convergence_epoch'] for r in arch_results]
            final_losses = [r['final_loss'] for r in arch_results]
            
            print(f"\n{arch_name.upper()}:")
            print(f"  Parameter Recovery Accuracy:")
            print(f"    L error:   {np.mean(L_errors):.2f}% ± {np.std(L_errors):.2f}%")
            print(f"    tau error: {np.mean(tau_errors):.2f}% ± {np.std(tau_errors):.2f}%")
            print(f"  Convergence Speed:")
            print(f"    Epochs to converge: {np.mean(conv_epochs):.1f} ± {np.std(conv_epochs):.1f}")
            print(f"  Stability (lower is better):")
            print(f"    L std:     {np.std(L_values):.6f} m")
            print(f"    tau std:   {np.std(tau_values):.6f} 1/s")
            print(f"    Loss std:  {np.std(final_losses):.6f}")
            print(f"  Training Time:")
            print(f"    {np.mean(time_values):.2f} ± {np.std(time_values):.2f} s")


if __name__ == "__main__":
    run_architecture_ablation()
