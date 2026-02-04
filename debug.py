import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys

# 设置打印格式，不显示科学计数法，方便看清 0.0000
torch.set_printoptions(precision=4, sci_mode=False)

class DebugVAE(nn.Module):
    def __init__(self, n, d, k):
        super().__init__()
        self.n_plus_1 = n + 1
        self.K = k
        
        # 强力 Encoder & Decoder
        self.enc = nn.Sequential(nn.Linear(2 * self.n_plus_1, 64), nn.ReLU(), nn.Linear(64, d * 2))
        self.dec = nn.Sequential(nn.Linear(d, 64), nn.ReLU())
        
        # MoPL Heads
        self.head_alpha = nn.Linear(64, k)
        self.head_v = nn.Linear(64, k * self.n_plus_1)

    def forward(self, x):
        # Encoder
        mu, _ = self.enc(x).chunk(2, dim=1)
        z = mu # Debug模式：直接取均值，排除随机噪声干扰
        
        # Decoder
        h = self.dec(z)
        alpha = F.softmax(self.head_alpha(h), dim=1)
        
        # --- 关键点：这里决定了梯度是否消失 ---
        v_logits = self.head_v(h).view(-1, self.K, self.n_plus_1)
        
        # 方案 A: Softplus (论文原版，容易梯度消失)
        # v = F.softplus(v_logits) 
        
        # 方案 B: Exponential (梯度更强，建议调试用)
        v = torch.exp(v_logits) 
        
        return alpha, v

    def compute_loss(self, alpha, v, y_true, mask):
        # MoPL Probability Logic
        mask_exp = mask.unsqueeze(1)
        denom = torch.sum(v * mask_exp, dim=2, keepdim=True) + 1e-9
        prob_k = (v * mask_exp) / denom
        y_pred = torch.sum(alpha.unsqueeze(2) * prob_k, dim=1)
        
        # NLL
        loss = -torch.sum(y_true * torch.log(y_pred + 1e-9), dim=1).mean()
        return loss, y_pred

# ==========================================
# 运行脚本
# ==========================================
if __name__ == "__main__":
    print(f"{'Step':<6} | {'Loss (NLL)':<12} | {'Grad Norm (V)':<15} | {'Target Prob':<12} | {'Pred Prob':<12}")
    print("-" * 70)

    # 1. 造只有 1 条数据的 Batch
    # 目标：只买 Item 1 (Index 1)
    N = 20
    y_target = torch.zeros(1, N+1); y_target[0, 1] = 1.0
    mask = torch.ones(1, N+1)
    u_in = torch.cat([y_target, mask], dim=1)

    # 2. 初始化模型
    model = DebugVAE(n=N, d=8, k=5)
    
    # 使用较大的 LR 来强行推动
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for i in range(201):
        optimizer.zero_grad()
        
        alpha, v = model(u_in)
        loss, y_pred = model.compute_loss(alpha, v, y_target, mask)
        
        loss.backward()
        
        # --- 获取关键数据 ---
        # 1. 梯度模长：检查梯度是不是 0
        grad_norm = model.head_v.weight.grad.norm().item()
        
        # 2. 预测值：看看模型觉得 Item 1 的概率是多少
        pred_prob_target = y_pred[0, 1].item()
        
        optimizer.step()

        # 每 20 步打印一次数据
        if i % 20 == 0:
            print(f"{i:<6} | {loss.item():<12.4f} | {grad_norm:<15.4f} | {1.0:<12.1f} | {pred_prob_target:<12.4f}")

    print("-" * 70)
    print("最终详细数据对比:")
    print(f"Target Vector (前5位): {y_target[0, :5].numpy()}")
    print(f"Pred Vector   (前5位): {y_pred[0, :5].detach().numpy()}")
    
    # --- 自动诊断结论 ---
    print("\n【诊断结论】")
    if loss.item() < 0.1:
        print("✅ 通过：模型有能力拟合数据。之前的问题可能是数据太复杂或 Batch Size 太大。")
    elif grad_norm == 0.0:
        print("❌ 失败：梯度为 0 (Gradient Vanishing)。")
        print("   原因：Softplus 在初始值极小或极大时导数为 0。")
        print("   建议：把 Decoder 的激活函数改成 torch.exp()。")
    else:
        print("⚠️ 停滞：梯度存在，但 Loss 不降。")
        print("   原因：学习率 (LR) 可能太小，或者模型陷入局部最优。")
        print("   建议：尝试增大 LR 到 0.05 或 0.1。")