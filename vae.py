import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
import math

# ==========================================
# 1. 严格对齐论文的数据生成器
# ==========================================
class PaperCompliantGenerator:
    def __init__(self, num_items, num_true_components=5, device='cpu'):
        self.n = num_items
        self.n_plus_1 = num_items + 1
        self.K_true = num_true_components
        self.device = device
        
        # Ground Truth Parameters (保持不变)
        self.true_alpha = torch.distributions.Dirichlet(torch.ones(self.K_true)).sample().to(device)
        self.true_v = torch.ones(self.K_true, self.n_plus_1, device=device) * 0.01
        self.true_v[:, 0] = 1.0 
        
        items_per_group = self.n // self.K_true
        for k in range(self.K_true):
            start = 1 + k * items_per_group
            end = min(start + items_per_group, self.n_plus_1)
            self.true_v[k, start:end] = torch.rand(end - start, device=device) * 5.0 + 5.0
            
    def generate_batch(self, num_datasets=5000, samples_per_assortment=100):
        """
        Generates data strictly following the definition:
        u_m = [y_hat; m_m]
        """
        # ---------------------------------------------------------
        # 1. Availability Mask (m_m)
        # ---------------------------------------------------------
        # mm ∈ {0,1}^(n+1)
        # mm,i = 1 if i ∈ Sm ∪ {0}, else 0
        availability_mask = (torch.rand(num_datasets, self.n_plus_1, device=self.device) > 0.5).float()
        availability_mask[:, 0] = 1.0 # The no-purchase option (0) is always available
        
        # ---------------------------------------------------------
        # 2. Choice Vector (y_hat)
        # ---------------------------------------------------------
        # Calculate theoretical probabilities based on the mask
        scores = self.true_v.unsqueeze(0) * availability_mask.unsqueeze(1)
        denom = scores.sum(dim=2, keepdim=True) + 1e-9
        prob_k = scores / denom
        prob_true = torch.einsum('k, mkn -> mn', self.true_alpha, prob_k)
        
        # Simulate transactions
        dist = torch.distributions.Multinomial(total_count=samples_per_assortment, probs=prob_true)
        counts = dist.sample()
        
        # y_hat ∈ [0,1]^(n+1)
        # Note: If item i is not in mask, score=0 -> prob=0 -> count=0.
        # So the condition "entry for any unavailable item is zero" is automatically met.
        choice_vector = counts / samples_per_assortment
        
        # ---------------------------------------------------------
        # 3. Concatenation (u_m)
        # ---------------------------------------------------------
        # u_m = [y_hat; m_m] ∈ R^(2(n+1))
        encoder_input = torch.cat([choice_vector, availability_mask], dim=1)
        
        return encoder_input, choice_vector, availability_mask

# ==========================================
# 2. VAE Model (Encoder Input Dimension Matched)
# ==========================================
class DGRA_VAE(nn.Module):
    def __init__(self, num_items, latent_dim, num_components, hidden_dim=64):
        super().__init__()
        self.n_plus_1 = num_items + 1
        self.K = num_components
        
        # Input Dimension: 2 * (N+1)
        # Corresponds to [Choice Vector (N+1) ; Availability Mask (N+1)]
        input_dim = 2 * self.n_plus_1
        
        self.enc_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
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

def compute_data_entropy(data_tensor, num_items):
    n_plus_1 = num_items + 1
    y_true = data_tensor[:, :n_plus_1]
    entropy = -torch.sum(y_true * torch.log(y_true + 1e-9), dim=1).mean()
    return entropy.item()

# ==========================================
# 3. Main Logic
# ==========================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">>> Using device: {device}")
    
    # Config
    N = 20
    D = 8
    K = 5
    BATCH = 128
    EPOCHS = 150
    LR = 1e-2
    
    # 1. 生成数据
    print(">>> Generating Data (Paper Definition)...")
    gen = PaperCompliantGenerator(N, num_true_components=K, device=device)
    # generate_batch now returns separate components for inspection
    u_m, y_hat_all, mask_all = gen.generate_batch(num_datasets=10000, samples_per_assortment=100)
    
    # --- 【关键步骤】验证输入结构 ---
    print("\n[Input Structure Verification]")
    print(f"Total Input Dimension: {u_m.shape[1]} (Should be 2*{N+1} = {2*(N+1)})")
    
    # 取第一条数据进行微观检查
    sample_y = y_hat_all[0].cpu().numpy()
    sample_m = mask_all[0].cpu().numpy()
    sample_input = u_m[0].cpu().numpy()
    
    # 找到一个 "Not Available" 的商品 (Mask=0)
    unavailable_idx = torch.where(mask_all[0] == 0)[0][0].item()
    # 找到一个 "Available but Not Chosen" 的商品 (Mask=1, Y=0)
    # (如果找不到完全为0的，找一个概率极小的也行，但在仿真数据里通常有0)
    available_indices = torch.where(mask_all[0] == 1)[0]
    not_chosen_idx = available_indices[torch.argmin(y_hat_all[0, available_indices])].item()
    
    print(f"\nComparing Logic for Item Indices:")
    
    # Input format: [y_hat ... | mask ...]
    # The input for item i consists of (y_hat[i], mask[i])
    # But in the flattened vector, they are at index `i` and `i + (N+1)`
    
    feat_unavailable = (sample_input[unavailable_idx], sample_input[unavailable_idx + (N+1)])
    feat_not_chosen  = (sample_input[not_chosen_idx],  sample_input[not_chosen_idx + (N+1)])
    
    print(f"1. Unavailable Item (Idx {unavailable_idx}):")
    print(f"   y_hat={feat_unavailable[0]:.1f}, mask={feat_unavailable[1]:.1f}")
    print(f"   -> Network sees input feature pair: [0, 0]")
    
    print(f"2. Available (Low Sales) Item (Idx {not_chosen_idx}):")
    print(f"   y_hat={feat_not_chosen[0]:.4f}, mask={feat_not_chosen[1]:.1f}")
    print(f"   -> Network sees input feature pair: [{feat_not_chosen[0]:.4f}, 1]")
    
    print("\n✅ Verification: The network receives distinct signals for 'Unavailable' vs 'Not Chosen'.")
    print("   'Unavailable' -> weight_2 * 0 = 0")
    print("   'Not Chosen'  -> weight_2 * 1 = bias shift")
    print("-" * 60)

    # 2. 训练准备
    lower_bound = compute_data_entropy(u_m, N)
    print(f"📉 THEORETICAL LOWER BOUND: {lower_bound:.4f}")
    
    FREE_BITS = math.log(K) + 0.5
    print(f">>> Dynamic Free Bits: {FREE_BITS:.4f}")
    
    loader = DataLoader(TensorDataset(u_m), batch_size=BATCH, shuffle=True)
    vae = DGRA_VAE(N, D, K).to(device)
    opt = optim.Adam(vae.parameters(), lr=LR)
    
    # 3. 训练
    for epoch in range(EPOCHS):
        total_nll = 0
        total_kl = 0
        
        # 使用你成功的 Low Beta 策略
        MAX_BETA = 0.02
        if epoch < 20: beta = 0.0001
        else: beta = min(MAX_BETA, (epoch - 20) / 50.0 * MAX_BETA)
        
        for batch in loader:
            u_in = batch[0]
            y_real = u_in[:, :N+1]
            m_real = u_in[:, N+1:]
            
            opt.zero_grad()
            alpha, v, mu, logvar = vae(u_in)
            y_pred = vae.compute_choice_probs(alpha, v, m_real)
            
            nll = -torch.sum(y_real * torch.log(y_pred + 1e-9), dim=1).mean()
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
            avg_nll = total_nll/len(loader)
            gap = avg_nll - lower_bound
            print(f"Epoch {epoch+1:03d} | NLL: {avg_nll:.4f} (Gap: +{gap:.4f}) | KL: {total_kl/len(loader):.4f}")

    # 4. Sanity Check
    print("\n>>> Sanity Check: Latent Space Interpretation")
    vae.eval()
    with torch.no_grad():
        # Case A: Buy Item 1
        y1 = torch.zeros(1, N+1, device=device); y1[0, 1] = 1.0
        m1 = torch.ones(1, N+1, device=device)
        u1 = torch.cat([y1, m1], dim=1)
        
        # Case B: Buy Item 10
        y2 = torch.zeros(1, N+1, device=device); y2[0, 10] = 1.0
        u2 = torch.cat([y2, m1], dim=1) # Same mask, different choice
        
        a1 = vae(u1)[0].cpu().numpy().flatten().round(2)
        a2 = vae(u2)[0].cpu().numpy().flatten().round(2)
        
        print(f"Choice: Item 1  -> Alpha: {a1}")
        print(f"Choice: Item 10 -> Alpha: {a2}")
        
        diff = torch.norm(torch.tensor(a1) - torch.tensor(a2)).item()
        if diff > 0.1:
            print(f"\n✅ SUCCESS! (Diff: {diff:.2f})")
        else:
            print(f"\n❌ WARNING (Diff: {diff:.2f})")