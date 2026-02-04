import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time

# ==========================================
# 1. 你的数据生成器 (保持原样)
# ==========================================
class AggregatedDataGenerator:
    def __init__(self, num_items, num_true_components=5):
        self.n = num_items
        self.n_plus_1 = num_items + 1
        self.K_true = num_true_components
        
        self.true_alpha = torch.distributions.Dirichlet(torch.ones(self.K_true)).sample()
        self.true_v = torch.ones(self.K_true, self.n_plus_1) * 0.01
        self.true_v[:, 0] = 1.0 
        
        items_per_group = self.n // self.K_true
        for k in range(self.K_true):
            start = 1 + k * items_per_group
            end = min(start + items_per_group, self.n_plus_1)
            self.true_v[k, start:end] = torch.rand(end - start) * 5.0 + 5.0
            
        print(f"[Generator] Ground Truth: {self.K_true} disjoint consumer types.")

    def compute_true_prob(self, mask):
        scores = self.true_v * mask.unsqueeze(0)
        denom = scores.sum(dim=1, keepdim=True) + 1e-9
        prob_k = scores / denom 
        prob_true = torch.matmul(self.true_alpha, prob_k)
        return prob_true

    def generate(self, num_datasets=2000, samples_per_assortment=100):
        data_list = []
        for _ in range(num_datasets):
            mask = (torch.rand(self.n_plus_1) > 0.5).float()
            mask[0] = 1.0
            p_true = self.compute_true_prob(mask)
            counts = torch.multinomial(p_true, samples_per_assortment, replacement=True)
            y_hat = torch.bincount(counts, minlength=self.n_plus_1).float()
            y_hat = y_hat / samples_per_assortment
            u_m = torch.cat([y_hat, mask], dim=0)
            data_list.append(u_m)
        return torch.stack(data_list)

# ==========================================
# 2. 简单的 MLP Baseline (无 VAE 结构)
# ==========================================
class SimpleMLP(nn.Module):
    def __init__(self, num_items, hidden_dim=128):
        super().__init__()
        self.n_plus_1 = num_items + 1
        input_dim = 2 * self.n_plus_1
        
        # 一个 3 层的 MLP，甚至比 VAE 还简单
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.n_plus_1) # 直接输出 Logits
        )

    def forward(self, x, mask):
        # x: [Batch, 2*(N+1)]
        logits = self.net(x)
        
        # --- 关键：手动 Apply Mask ---
        # 对于 Mask 为 0 的商品，将其 Logit 设为负无穷
        # 这样 Softmax 后概率为 0，与 VAE 的行为保持一致
        logits = logits.masked_fill(mask == 0, -1e9)
        
        y_pred = F.softmax(logits, dim=1)
        return y_pred

# ==========================================
# 3. 训练脚本
# ==========================================
if __name__ == "__main__":
    # Config
    N = 20
    BATCH = 128
    EPOCHS = 100
    LR = 1e-3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">>> Running MLP Baseline on: {device}")

    # 1. 生成数据
    print(">>> Generating Data...")
    gen = AggregatedDataGenerator(N, num_true_components=5)
    # 生成 5000 条数据
    data = gen.generate(num_datasets=5000, samples_per_assortment=100)
    
    loader = DataLoader(TensorDataset(data), batch_size=BATCH, shuffle=True)

    # 2. 初始化模型
    model = SimpleMLP(N, hidden_dim=128).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    print(">>> Start Training MLP Baseline...")
    
    history_nll = []

    for epoch in range(EPOCHS):
        total_nll = 0
        
        for batch in loader:
            u_m = batch[0].to(device)
            y_hat = u_m[:, :N+1]
            mask = u_m[:, N+1:]
            
            optimizer.zero_grad()
            
            # MLP Forward
            y_pred = model(u_m, mask)
            
            # Loss: 只有 NLL，没有 KL
            nll = -torch.sum(y_hat * torch.log(y_pred + 1e-9), dim=1).mean()
            
            nll.backward()
            optimizer.step()
            
            total_nll += nll.item()
            
        avg_nll = total_nll / len(loader)
        history_nll.append(avg_nll)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:03d} | NLL: {avg_nll:.4f}")

    print("\n>>> Baseline Result:")
    print(f"Final MLP NLL: {history_nll[-1]:.4f}")
    
    # 理论极限计算 (Entropy of Data)
    # 如果数据是完全确定的，NLL 应该接近 0。
    # 如果数据本身带有噪声（采样带来的），NLL 不可能为 0。
    # 我们算一下 batch 中 y_hat 自己的熵，作为理论极限参考
    data_gpu = data.to(device)
    y_hat_all = data_gpu[:, :N+1]
    # Self-Entropy: -sum(y * log(y))
    # 只有当模型完美预测出 y_hat 时，Loss 等于这个熵
    theoretical_limit = -torch.sum(y_hat_all * torch.log(y_hat_all + 1e-9), dim=1).mean().item()
    print(f"Theoretical Lower Bound (Data Entropy): {theoretical_limit:.4f}")
    
    if history_nll[-1] < theoretical_limit + 0.1:
        print("结论：MLP 已经完美过拟合数据。如果 VAE 达不到这个值，是 VAE 结构的问题。")
    else:
        print("结论：MLP 还没拟合好，可能是网络太小，或者训练步数不够。")