import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ==============================================================================
# 🛠️ 全局配置中心 (CONFIG)
# ==============================================================================
CONFIG = {
    # --- 1. 基础环境设置 ---
    'N': 20,                 # 商品总数 (Item 1 ~ 20)
    'K': 5,                  # 消费者类别数
    'price_min': 20,         # 商品最低价格
    'price_max': 100,        # 商品最高价格
    'seed': 42,              # 随机种子
    
    # --- 2. Phase I: VAE 模型训练 ---
    'D': 8,                  # 隐空间维度
    'hidden_dim': 64,        # 神经网络宽
    'batch_size': 128,       # 批大小
    'epochs_phase1': 50,     # Phase I 训练轮数
    'lr_phase1': 1e-2,       # 学习率
    'beta_max': 0.05,        # KL散度权重上限
    
    # --- 3. Phase II: 鲁棒优化 (对抗博弈) ---
    'rho_multiplier': 2.0,   # 攻击半径倍数 
    
    'adv_steps': 50,         # 内层攻击步数
    'adv_lr': 0.2,           # 攻击步长
    'n_starts': 10,          # 多重启动
    
    'outer_steps': 200,      # 外层优化总轮数
    'outer_lr': 0.1,         # 商家选品策略学习率
    
    # --- 4. 硬件 ---
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu")
}

# 设置随机种子
torch.manual_seed(CONFIG['seed'])
np.random.seed(CONFIG['seed'])

print(f">>> 当前配置: Rho倍数={CONFIG['rho_multiplier']}, 攻击步数={CONFIG['adv_steps']}, 总轮数={CONFIG['outer_steps']}")

# ==============================================================================
# PART 1: PHASE I - 数据生成与 VAE 模型
# ==============================================================================

class VectorizedAggregatedGenerator:
    def __init__(self, num_items, num_true_components, price_range, device='cpu'):
        self.n = num_items
        self.n_plus_1 = num_items + 1
        self.K_true = num_true_components
        self.device = device
        
        # Ground Truth
        self.true_alpha = torch.distributions.Dirichlet(torch.ones(self.K_true)).sample().to(device)
        self.true_v = torch.ones(self.K_true, self.n_plus_1, device=device) * 0.01
        self.true_v[:, 0] = 1.0 
        
        self.prices = torch.randint(price_range[0], price_range[1], (self.n_plus_1,), device=device).float()
        self.prices[0] = 0.0 
        
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

class DGRA_VAE(nn.Module):
    def __init__(self, num_items, latent_dim, num_components, hidden_dim=64):
        super().__init__()
        self.n_plus_1 = num_items + 1
        self.K = num_components
        self.latent_dim = latent_dim
        
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
        return self.decode(z, mu, logvar)

    def decode(self, z, mu=None, logvar=None):
        h_dec = F.relu(self.dec_hidden(z))
        alpha = F.softmax(self.fc_alpha(h_dec), dim=1)
        v = torch.exp(self.fc_v(h_dec)).view(-1, self.K, self.n_plus_1)
        return alpha, v, mu, logvar

    def compute_choice_probs(self, alpha, v, mask):
        mask_exp = mask.unsqueeze(1) 
        scores = v * mask_exp 
        denom = torch.sum(scores, dim=2, keepdim=True) + 1e-9
        prob_k = scores / denom
        return torch.sum(alpha.unsqueeze(2) * prob_k, dim=1)

# ==============================================================================
# PART 2: PHASE II - 鲁棒优化核心逻辑
# ==============================================================================

class StraightThroughEstimator(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits):
        probs = torch.sigmoid(logits)
        x_binary = (probs >= 0.5).float()
        ctx.save_for_backward(probs)
        return x_binary

    @staticmethod
    def backward(ctx, grad_output):
        probs, = ctx.saved_tensors
        grad_logits = grad_output * probs * (1 - probs)
        return grad_logits

class RobustAssortmentOptimizer:
    def __init__(self, vae, prices, historical_data, config):
        self.vae = vae
        self.prices = prices
        self.data = historical_data
        self.cfg = config
        self.device = config['device']
        
        for param in self.vae.parameters():
            param.requires_grad = False
        self.vae.eval()
        
        self.N = vae.n_plus_1 - 1
        self.latent_dim = vae.latent_dim
        
        self.rho = math.sqrt(self.latent_dim) * config['rho_multiplier']
        self.n_starts = config['n_starts']
        self.adv_lr = config['adv_lr']
        self.adv_steps = config['adv_steps']

    def get_revenue(self, x_binary, z_particles):
        alpha, v, _, _ = self.vae.decode(z_particles)
        mask = x_binary.unsqueeze(1) 
        scores = v * mask 
        denom = torch.sum(scores, dim=2, keepdim=True) + 1e-9
        prob_k = scores / denom
        prices_exp = self.prices.view(1, 1, -1)
        rev_k = torch.sum(prob_k * prices_exp, dim=2) 
        total_rev = torch.sum(alpha * rev_k, dim=1) 
        return total_rev

    def solve_inner_adversary(self, x_binary):
        z = torch.randn(self.n_starts, self.latent_dim, device=self.device, requires_grad=True)
        optimizer = optim.SGD([z], lr=self.adv_lr)
        
        for step in range(self.adv_steps):
            optimizer.zero_grad()
            rev = self.get_revenue(x_binary.detach(), z)
            loss = rev.sum() 
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                norms = torch.norm(z, p=2, dim=1, keepdim=True)
                factor = torch.min(torch.ones_like(norms), self.rho / (norms + 1e-9))
                z.data = z.data * factor
                
        with torch.no_grad():
            final_revs = self.get_revenue(x_binary, z)
            min_idx = torch.argmin(final_revs)
            z_worst = z[min_idx].unsqueeze(0)
            worst_rev = final_revs[min_idx].item()
            
        return z_worst, worst_rev

    def optimize_assortment(self):
        steps = self.cfg['outer_steps']
        lr = self.cfg['outer_lr']
        
        # 【关键修正】：使用 torch.full 创建叶子节点张量
        pi = torch.full((1, self.N+1), 2.0, device=self.device, requires_grad=True)
        
        outer_opt = optim.Adam([pi], lr=lr)
        STE = StraightThroughEstimator.apply
        
        print(f"\n>>> [Phase II] 启动鲁棒选品进化 (Steps={steps}, Rho={self.rho:.2f})...")
        print(f"{'Step':<5} | {'Robust Rev':<12} | {'Nominal Rev':<12} | {'Action Summary'}")
        print("-" * 65)
        
        best_rev = -1.0
        best_x = None
        
        history_x = [] 
        z_mean = torch.zeros(1, self.latent_dim, device=self.device)
        prev_items = set(range(1, self.N+1))
        
        for step in range(steps):
            outer_opt.zero_grad()
            
            x_binary = STE(pi)
            x_fixed = x_binary.clone()
            x_fixed[:, 0] = 1.0 
            
            current_x_np = x_fixed.detach().cpu().numpy().flatten()
            history_x.append(current_x_np)
            
            z_adv, robust_rev = self.solve_inner_adversary(x_fixed)
            
            revenue = self.get_revenue(x_fixed, z_adv.detach())
            loss = -revenue
            loss.backward()
            outer_opt.step()
            
            if robust_rev > best_rev:
                best_rev = robust_rev
                best_x = current_x_np
            
            if (step+1) % 10 == 0:
                with torch.no_grad():
                    nominal_rev = self.get_revenue(x_fixed, z_mean).item()
                
                current_items = set(np.where(current_x_np > 0.5)[0])
                current_items.discard(0)
                
                dropped = prev_items - current_items
                added = current_items - prev_items
                
                change_str = ""
                if dropped: change_str += f"Drop {len(dropped)} "
                if added:   change_str += f"Add {len(added)}"
                if not change_str: change_str = "Stable"
                
                items_count = len(current_items)
                print(f"{step+1:<5} | ${robust_rev:<11.2f} | ${nominal_rev:<11.2f} | Items: {items_count} ({change_str})")
                prev_items = current_items

        return best_x, best_rev, np.array(history_x)

# ==============================================================================
# 📊 绘图工具
# ==============================================================================
def plot_evolution_heatmap(history_x, prices, filename="assortment_evolution.png"):
    plt.figure(figsize=(14, 8))
    
    item_prices = prices[1:].cpu().numpy()
    sorted_indices = np.argsort(item_prices)[::-1]
    
    matrix = history_x[:, 1:]
    matrix_sorted = matrix[:, sorted_indices].T 
    
    ax = sns.heatmap(matrix_sorted, cmap="Greys", cbar=False, linewidths=0.05, linecolor='gray')
    
    plt.title(f"Robust Assortment Evolution", fontsize=16, fontweight='bold')
    plt.xlabel("Optimization Steps (Time)", fontsize=14)
    plt.ylabel("Items", fontsize=14)
    
    y_labels = [f"Item {idx+1} (${int(item_prices[i])})" for i, idx in enumerate(sorted_indices)]
    plt.yticks(np.arange(len(y_labels)) + 0.5, y_labels, rotation=0, fontsize=10)
    
    plt.axvline(x=0, color='red', linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    if os.path.exists(filename): os.remove(filename)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"\n📸 [绘图完成] 图片已保存至: {os.path.abspath(filename)}")

# ==============================================================================
# 🚀 主程序入口
# ==============================================================================
if __name__ == "__main__":
    device = CONFIG['device']
    print(f">>> 使用设备: {device}")
    
    print(">>> [Phase I] 生成模拟数据...")
    gen = VectorizedAggregatedGenerator(
        num_items=CONFIG['N'], 
        num_true_components=CONFIG['K'], 
        price_range=(CONFIG['price_min'], CONFIG['price_max']),
        device=device
    )
    all_data = gen.generate_batch(num_datasets=5000, samples_per_assortment=100)
    
    print(">>> [Phase I] 训练 VAE 模型...")
    vae = DGRA_VAE(CONFIG['N'], CONFIG['D'], CONFIG['K'], CONFIG['hidden_dim']).to(device)
    opt = optim.Adam(vae.parameters(), lr=CONFIG['lr_phase1'])
    loader = DataLoader(TensorDataset(all_data), batch_size=CONFIG['batch_size'], shuffle=True)
    
    FREE_BITS = math.log(CONFIG['K']) + 0.5
    
    for epoch in range(CONFIG['epochs_phase1']):
        beta = min(CONFIG['beta_max'], epoch / 50.0 * CONFIG['beta_max'])
        total_loss = 0
        for batch in loader:
            u_m = batch[0]
            opt.zero_grad()
            alpha, v, mu, logvar = vae(u_m)
            y_pred = vae.compute_choice_probs(alpha, v, u_m[:, CONFIG['N']+1:])
            nll = -torch.sum(u_m[:, :CONFIG['N']+1] * torch.log(y_pred + 1e-9), dim=1).mean()
            kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
            loss = nll + beta * max(kl, torch.tensor(FREE_BITS).to(device))
            loss.backward()
            opt.step()
            total_loss += loss.item()
        if (epoch+1)%10==0: print(f"    Epoch {epoch+1} | Loss: {total_loss/len(loader):.4f}")

    print("\n>>> [Phase I] 完成. 模型已冻结.")

    # ----------------------------------------------------
    # 3. Phase II: 鲁棒优化
    # ----------------------------------------------------
    optimizer = RobustAssortmentOptimizer(vae, gen.prices, all_data, CONFIG)
    best_assortment, best_revenue, history = optimizer.optimize_assortment()
    
    # ----------------------------------------------------
    # 4. 结果展示
    # ----------------------------------------------------
    print("\n" + "="*60)
    print("🏆 最终结果 (FINAL RESULTS)")
    print("="*60)
    
    selected_indices = np.where(best_assortment > 0.5)[0]
    selected_indices = selected_indices[selected_indices != 0] 
    
    print(f"Optimal Items: {selected_indices}")
    print(f"Robust Revenue: ${best_revenue:.2f}")
    
    all_ones = torch.ones(1, CONFIG['N']+1, device=device)
    _, rev_full = optimizer.solve_inner_adversary(all_ones)
    print(f"Baseline Rev:   ${rev_full:.2f} (Full Assortment)")
    if rev_full > 0:
        print(f"Improvement:    +{(best_revenue - rev_full)/rev_full*100:.1f}%")
    else:
        print(f"Improvement:    Infinite (Baseline collapsed)")
    
    print("\n>>> 正在生成进化热力图...")
    plot_evolution_heatmap(history, gen.prices)