import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA
import time
import math
import os

# ==========================================
# 1. 向量化数据生成器 (Vectorized Generator)
# ==========================================
class VectorizedAggregatedGenerator:
    def __init__(self, num_items, num_true_components=5, device='cpu'):
        self.n = num_items
        self.n_plus_1 = num_items + 1
        self.K_true = num_true_components
        self.device = device
        
        # --- Ground Truth Initialization ---
        self.true_alpha = torch.distributions.Dirichlet(torch.ones(self.K_true)).sample().to(device)
        self.true_v = torch.ones(self.K_true, self.n_plus_1, device=device) * 0.01
        self.true_v[:, 0] = 1.0 
        
        # Disjoint Preferences (Block Structure)
        items_per_group = self.n // self.K_true
        for k in range(self.K_true):
            start = 1 + k * items_per_group
            end = min(start + items_per_group, self.n_plus_1)
            self.true_v[k, start:end] = torch.rand(end - start, device=device) * 5.0 + 5.0
            
    def generate_batch(self, num_datasets=5000, samples_per_assortment=100):
        mask = (torch.rand(num_datasets, self.n_plus_1, device=self.device) > 0.5).float()
        mask[:, 0] = 1.0 
        
        scores = self.true_v.unsqueeze(0) * mask.unsqueeze(1)
        denom = scores.sum(dim=2, keepdim=True) + 1e-9
        prob_k = scores / denom
        prob_true = torch.einsum('k, mkn -> mn', self.true_alpha, prob_k)
        
        dist = torch.distributions.Multinomial(total_count=samples_per_assortment, probs=prob_true)
        counts = dist.sample()
        y_hat = counts / samples_per_assortment
        
        return torch.cat([y_hat, mask], dim=1)

# ==========================================
# 2. VAE Model (Robust Architecture)
# ==========================================
class DGRA_VAE(nn.Module):
    def __init__(self, num_items, latent_dim, num_components, hidden_dim=64):
        super().__init__()
        self.n_plus_1 = num_items + 1
        self.K = num_components
        
        # Input: [Choice Vector; Availability Mask]
        self.enc_net = nn.Sequential(
            nn.Linear(2 * self.n_plus_1, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        self.dec_hidden = nn.Linear(latent_dim, hidden_dim)
        self.fc_alpha = nn.Linear(hidden_dim, num_components)
        self.fc_v = nn.Linear(hidden_dim, num_components * self.n_plus_1)

    def forward(self, x):
        h = self.enc_net(x)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        std = torch.exp(0.5 * logvar)
        z = mu + torch.randn_like(std) * std
        
        h_dec = F.relu(self.dec_hidden(z))
        alpha = F.softmax(self.fc_alpha(h_dec), dim=1)
        v = torch.exp(self.fc_v(h_dec)).view(-1, self.K, self.n_plus_1)
        
        return alpha, v, mu, logvar

    def compute_choice_probs(self, alpha, v, mask):
        mask_exp = mask.unsqueeze(1)
        denom = torch.sum(v * mask_exp, dim=2, keepdim=True) + 1e-9
        prob_k = (v * mask_exp) / denom
        return torch.sum(alpha.unsqueeze(2) * prob_k, dim=1)

# ==========================================
# 3. Validation Report (自动保存图片版)
# ==========================================
class ValidationReport:
    def __init__(self, vae, generator, device):
        self.vae = vae
        self.gen = generator
        self.device = device
        self.N = generator.n
        self.K = generator.K_true
        
    def run_all_checks(self, save_path="vae_validation_report.png"):
        print("\n" + "="*60)
        print("📊 GENERATING VALIDATION REPORT...")
        print("="*60)
        
        # 创建一个大画布，包含 3 个子图
        fig = plt.figure(figsize=(20, 6))
        
        # Plot 1: Latent Space
        self._plot_latent_space(fig, 131)
        
        # Plot 2: Preference Recovery
        self._plot_preference_recovery(fig, 132)
        
        # Plot 3: Predictive Correlation
        self._plot_predictive_correlation(fig, 133)
        
        plt.tight_layout()
        
        # --- 保存图片 ---
        print(f"\n💾 Saving report to: {save_path}")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig) # 关闭画布释放内存
        
        print("✅ Report saved successfully!")
        print("="*60)

    def _plot_latent_space(self, fig, subplot_code):
        print(f"[1/3] Mapping Latent Space...")
        ax = fig.add_subplot(subplot_code)
        
        z_points = []
        labels = []
        items_per_group = self.N // self.K
        
        with torch.no_grad():
            for k in range(self.K):
                # 每类生成 50 个样本
                for _ in range(50):
                    target_item = np.random.randint(1 + k*items_per_group, 1 + (k+1)*items_per_group)
                    y_test = torch.zeros(1, self.N+1, device=self.device)
                    y_test[0, target_item] = 1.0
                    mask_test = torch.ones(1, self.N+1, device=self.device)
                    u_test = torch.cat([y_test, mask_test], dim=1)
                    
                    _, _, mu, _ = self.vae(u_test)
                    z_points.append(mu.cpu().numpy().flatten())
                    labels.append(k)
        
        z_points = np.array(z_points)
        pca = PCA(n_components=2)
        z_2d = pca.fit_transform(z_points)
        
        scatter = ax.scatter(z_2d[:, 0], z_2d[:, 1], c=labels, cmap='tab10', s=40, alpha=0.8, edgecolors='w')
        ax.set_title("Latent Space (PCA)\n(Distinct clusters = Success)", fontsize=12, fontweight='bold')
        ax.set_xlabel("PC 1")
        ax.set_ylabel("PC 2")
        ax.grid(True, linestyle='--', alpha=0.3)
        # Add legend manually
        legend1 = ax.legend(*scatter.legend_elements(), title="True Types")
        ax.add_artist(legend1)

    def _plot_preference_recovery(self, fig, subplot_code):
        print(f"[2/3] Recovering Preferences...")
        ax = fig.add_subplot(subplot_code)
        
        true_v = self.gen.true_v.cpu().numpy()[:, 1:] 
        true_v_norm = true_v / (true_v.max(axis=1, keepdims=True) + 1e-9)

        recovered_v = []
        items_per_group = self.N // self.K
        
        with torch.no_grad():
            for k in range(self.K):
                target_item = 1 + k*items_per_group 
                y_test = torch.zeros(1, self.N+1, device=self.device)
                y_test[0, target_item] = 1.0
                mask_test = torch.ones(1, self.N+1, device=self.device)
                u_test = torch.cat([y_test, mask_test], dim=1)
                
                alpha, v, _, _ = self.vae(u_test)
                v_weighted = torch.sum(alpha.unsqueeze(2) * v, dim=1)
                v_np = v_weighted.cpu().numpy().flatten()[1:]
                recovered_v.append(v_np)
        
        recovered_v = np.array(recovered_v)
        recovered_v_norm = recovered_v / (recovered_v.max(axis=1, keepdims=True) + 1e-9)
        
        combined = np.vstack([true_v_norm, np.zeros((1, self.N)) - 0.1, recovered_v_norm])
        
        sns.heatmap(combined, ax=ax, cmap="Blues", cbar=True, cbar_kws={'label': 'Normalized Preference Score'})
        ax.set_title("Ground Truth (Top) vs Learned (Bottom)\n(Should look mirrored)", fontsize=12, fontweight='bold')
        
        # Custom Y-axis labels
        ax.set_yticks([2.5, self.K + 3.5])
        ax.set_yticklabels(["True Types", "Learned Types"], rotation=90, va="center")
        ax.set_xlabel("Item ID")
        ax.hlines(self.K, *ax.get_xlim(), colors='red', linestyles='--')

    def _plot_predictive_correlation(self, fig, subplot_code):
        print(f"[3/3] Verifying Predictions...")
        ax = fig.add_subplot(subplot_code)
        
        u_m = self.gen.generate_batch(num_datasets=200, samples_per_assortment=100)
        y_true = u_m[:, :self.N+1]
        mask = u_m[:, self.N+1:]
        
        with torch.no_grad():
            alpha, v, _, _ = self.vae(u_m)
            y_pred = self.vae.compute_choice_probs(alpha, v, mask)
            
        y_true_np = y_true.cpu().numpy().flatten()
        y_pred_np = y_pred.cpu().numpy().flatten()
        
        indices = np.random.choice(len(y_true_np), 1000, replace=False)
        
        ax.scatter(y_true_np[indices], y_pred_np[indices], alpha=0.4, color='purple', s=15)
        ax.plot([0, 1], [0, 1], 'r--', alpha=0.8, linewidth=2, label="Perfect Fit")
        
        ax.set_title("True Freq vs Predicted Prob\n(Points should hug the red line)", fontsize=12, fontweight='bold')
        ax.set_xlabel("True Empirical Freq")
        ax.set_ylabel("Model Predicted Prob")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.legend()

def compute_data_entropy(data_tensor, num_items):
    n_plus_1 = num_items + 1
    y_true = data_tensor[:, :n_plus_1]
    entropy = -torch.sum(y_true * torch.log(y_true + 1e-9), dim=1).mean()
    return entropy.item()

# ==========================================
# 4. Main Training Loop
# ==========================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">>> Using device: {device}")
    
    # Config
    N = 20
    D = 8
    K = 5
    BATCH = 128
    EPOCHS = 150 # Enough to converge
    LR = 1e-2
    
    # 1. Generate Data
    print(">>> Generating Data...")
    gen = VectorizedAggregatedGenerator(N, num_true_components=K, device=device)
    all_data = gen.generate_batch(num_datasets=10000, samples_per_assortment=100)
    
    # 2. Theoretical Limit
    lower_bound = compute_data_entropy(all_data, N)
    print(f"📉 THEORETICAL LOWER BOUND: {lower_bound:.4f}")
    
    loader = DataLoader(TensorDataset(all_data), batch_size=BATCH, shuffle=True)
    vae = DGRA_VAE(N, D, K).to(device)
    opt = optim.Adam(vae.parameters(), lr=LR)
    
    # Free Bits Strategy
    FREE_BITS = math.log(K) + 0.5
    print(f">>> Dynamic Free Bits set to: {FREE_BITS:.4f} nats")
    
    print(">>> Start Training (Optimized Strategy)...")
    
    for epoch in range(EPOCHS):
        total_nll = 0
        total_kl = 0
        
        # Low Beta Strategy to prevent collapse
        MAX_BETA = 0.02
        if epoch < 120: 
            beta = 0.0001
        else:
            beta = min(MAX_BETA, (epoch - 20) / 50.0 * MAX_BETA)
        
        for batch in loader:
            u_m = batch[0]
            y_hat = u_m[:, :N+1]
            mask = u_m[:, N+1:]
            
            opt.zero_grad()
            alpha, v, mu, logvar = vae(u_m)
            y_pred = vae.compute_choice_probs(alpha, v, mask)
            
            nll = -torch.sum(y_hat * torch.log(y_pred + 1e-9), dim=1).mean()
            kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
            
            # Hinge Loss for Free Bits
            if kl < FREE_BITS:
                loss = nll + beta * 10.0 * (FREE_BITS - kl)**2
            else:
                loss = nll + beta * kl
            
            loss.backward()
            opt.step()
            
            total_nll += nll.item()
            total_kl += kl.item()
            
        if (epoch+1) % 10 == 0:
            avg_nll = total_nll / len(loader)
            gap = avg_nll - lower_bound
            print(f"Epoch {epoch+1:03d} | NLL: {avg_nll:.4f} (Gap: +{gap:.4f}) | KL: {total_kl/len(loader):.4f} | Beta: {beta:.4f}")

    # --- 5. Generate & Save Report ---
    # This will save 'vae_validation_report.png' in the current directory
    validator = ValidationReport(vae, gen, device)
    validator.run_all_checks(save_path="vae_validation_report.png")