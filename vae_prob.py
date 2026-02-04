import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
import math

# ==========================================
# 1. Vectorized Data Generator (GPU Accelerated)
# ==========================================
class VectorizedAggregatedGenerator:
    def __init__(self, num_items, num_true_components=5, device='cpu'):
        self.n = num_items
        self.n_plus_1 = num_items + 1
        self.K_true = num_true_components
        self.device = device
        
        # --- Ground Truth Initialization ---
        # 1. True Mixture Weights (Alpha)
        self.true_alpha = torch.distributions.Dirichlet(torch.ones(self.K_true)).sample().to(device)
        
        # 2. True Preference Scores (V)
        # Initialize small scores
        self.true_v = torch.ones(self.K_true, self.n_plus_1, device=device) * 0.01
        self.true_v[:, 0] = 1.0 # No-buy option
        
        # Create Disjoint Preferences (Block Diagonal Structure)
        # This ensures each consumer type has distinct favorite items
        items_per_group = self.n // self.K_true
        for k in range(self.K_true):
            start = 1 + k * items_per_group
            end = min(start + items_per_group, self.n_plus_1)
            # Assign high scores to specific items for this group
            self.true_v[k, start:end] = torch.rand(end - start, device=device) * 5.0 + 5.0
            
    def generate_batch(self, num_datasets=5000, samples_per_assortment=100):
        """
        Generates a batch of synthetic sales data.
        Returns: Tensor [M, 2*(N+1)] -> [Empirical Frequencies, Assortment Mask]
        """
        # 1. Random Assortments (Masks)
        mask = (torch.rand(num_datasets, self.n_plus_1, device=self.device) > 0.5).float()
        mask[:, 0] = 1.0 # Ensure no-buy option is always available
        
        # 2. Compute True Choice Probabilities (MoPL Model)
        # Scores: [M, K, N+1]
        scores = self.true_v.unsqueeze(0) * mask.unsqueeze(1)
        denom = scores.sum(dim=2, keepdim=True) + 1e-9
        prob_k = scores / denom # Prob of choosing item i given user type k
        
        # Mixture: [M, N+1]
        prob_true = torch.einsum('k, mkn -> mn', self.true_alpha, prob_k)
        
        # 3. Simulate Empirical Sales (Multinomial Sampling)
        # We simulate 'samples_per_assortment' customers visiting the store
        dist = torch.distributions.Multinomial(total_count=samples_per_assortment, probs=prob_true)
        counts = dist.sample()
        
        # Empirical Frequency Vector (y_hat)
        y_hat = counts / samples_per_assortment
        
        # Concatenate y_hat and mask
        return torch.cat([y_hat, mask], dim=1)

# ==========================================
# 2. VAE Model (Robust Architecture)
# ==========================================
class DGRA_VAE(nn.Module):
    def __init__(self, num_items, latent_dim, num_components, hidden_dim=64):
        super().__init__()
        self.n_plus_1 = num_items + 1
        self.K = num_components
        
        # Encoder: Maps (y_hat, mask) -> z
        self.enc_net = nn.Sequential(
            nn.Linear(2 * self.n_plus_1, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder: Maps z -> (alpha, V)
        self.dec_hidden = nn.Linear(latent_dim, hidden_dim)
        self.fc_alpha = nn.Linear(hidden_dim, num_components)
        self.fc_v = nn.Linear(hidden_dim, num_components * self.n_plus_1)

    def forward(self, x):
        # 1. Encode
        h = self.enc_net(x)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        
        # 2. Reparameterize
        std = torch.exp(0.5 * logvar)
        z = mu + torch.randn_like(std) * std
        
        # 3. Decode
        h_dec = F.relu(self.dec_hidden(z))
        
        # Output Parameters
        alpha = F.softmax(self.fc_alpha(h_dec), dim=1)
        
        # [CRITICAL] Use exp() instead of softplus() to prevent gradient vanishing
        v = torch.exp(self.fc_v(h_dec)).view(-1, self.K, self.n_plus_1)
        
        return alpha, v, mu, logvar

    def compute_choice_probs(self, alpha, v, mask):
        """
        Computes P(item | S) using the mixture of Plackett-Luce models
        """
        mask_exp = mask.unsqueeze(1)
        denom = torch.sum(v * mask_exp, dim=2, keepdim=True) + 1e-9
        prob_k = (v * mask_exp) / denom
        # Mixture Sum
        return torch.sum(alpha.unsqueeze(2) * prob_k, dim=1)

# ==========================================
# 3. Utility: Data Entropy Calculation
# ==========================================
def compute_data_entropy(data_tensor, num_items):
    """
    Calculates the Theoretical Lower Bound (Self-Entropy) of the dataset.
    NLL cannot be lower than this value.
    """
    n_plus_1 = num_items + 1
    y_true = data_tensor[:, :n_plus_1]
    
    # Entropy = - sum( p * log(p) )
    entropy = -torch.sum(y_true * torch.log(y_true + 1e-9), dim=1).mean()
    return entropy.item()

# ==========================================
# 4. Main Training Loop
# ==========================================
if __name__ == "__main__":
    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">>> Using device: {device}")
    
    # Config
    N = 20          # Number of items
    D = 8           # Latent dimension
    K = 5           # Number of consumer types
    BATCH = 128     # Batch size
    EPOCHS = 200    # Training epochs
    LR = 1e-2       # Learning rate
    
    # 1. Generate Data
    print(">>> Generating Data...")
    gen = VectorizedAggregatedGenerator(N, num_true_components=K, device=device)
    all_data = gen.generate_batch(num_datasets=10000, samples_per_assortment=100)
    
    # 2. Calculate & Print Theoretical Limit
    lower_bound = compute_data_entropy(all_data, N)
    print("="*60)
    print(f"📉 THEORETICAL LOWER BOUND (Data Entropy): {lower_bound:.4f}")
    print(f"   (Target NLL is {lower_bound:.4f}. Gap < 0.05 is excellent.)")
    print("="*60)

    # 3. Prepare Model
    loader = DataLoader(TensorDataset(all_data), batch_size=BATCH, shuffle=True)
    vae = DGRA_VAE(N, D, K).to(device)
    opt = optim.Adam(vae.parameters(), lr=LR)
    
    # 4. Calculate Dynamic Free Bits
    # We want KL to be at least ln(K) to distinguish K types.
    # Adding 0.5 buffer ensures clusters are well-separated.
    FREE_BITS = math.log(K) + 0.5
    print(f">>> Dynamic Free Bits set to: {FREE_BITS:.4f} nats")
    
    print(">>> Start Training...")
    
    for epoch in range(EPOCHS):
        total_nll = 0
        total_kl = 0
        total_loss = 0
        
        # Beta Strategy: Warm-up from 0.001 to 1.0
        # We start small to let NLL converge, then increase to enforce structure.
        if epoch <  100: 
            beta = 0.001 
        else:
            beta = min(0.05, (epoch - 20) / 50.0)
        
        for batch in loader:
            u_m = batch[0] # Data is already on GPU (if generated on GPU)
            y_hat = u_m[:, :N+1]
            mask = u_m[:, N+1:]
            
            opt.zero_grad()
            
            # Forward
            alpha, v, mu, logvar = vae(u_m)
            y_pred = vae.compute_choice_probs(alpha, v, mask)
            
            # --- Loss Calculation ---
            
            # 1. NLL (Reconstruction Loss)
            nll = -torch.sum(y_hat * torch.log(y_pred + 1e-9), dim=1).mean()
            
            # 2. KL Divergence
            kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
            
            # 3. Free Bits (Hinge Loss)
            # This is the "Tax Free" zone. 
            # If KL < FREE_BITS, kl_loss_term = 0. 
            # If KL > FREE_BITS, we penalize the excess.
            # Ideally, KL will hover slightly above FREE_BITS.
            kl_loss_term = torch.max(kl, torch.tensor(FREE_BITS, device=device))
            
            # Total Loss
            loss = nll + beta * kl_loss_term
            
            loss.backward()
            opt.step()
            
            total_nll += nll.item()
            total_kl += kl.item()
            total_loss += loss.item()
            
        # Logging
        if (epoch+1) % 10 == 0:
            avg_nll = total_nll / len(loader)
            avg_kl = total_kl / len(loader)
            gap = avg_nll - lower_bound
            print(f"Epoch {epoch+1:03d} | NLL: {avg_nll:.4f} (Gap: +{gap:.4f}) | KL: {avg_kl:.4f} | Beta: {beta:.2f}")

    # --- 5. Sanity Check ---
    print("\n>>> Sanity Check: Interpreting Latent Space")
    vae.eval()
    with torch.no_grad():
        # Construct test cases on device
        # Case A: User who exclusively bought Item 1
        y_test_1 = torch.zeros(1, N+1, device=device); y_test_1[0, 1] = 1.0
        mask_test = torch.ones(1, N+1, device=device)
        u_test_1 = torch.cat([y_test_1, mask_test], dim=1)
        
        # Case B: User who exclusively bought Item 10
        y_test_2 = torch.zeros(1, N+1, device=device); y_test_2[0, 10] = 1.0
        u_test_2 = torch.cat([y_test_2, mask_test], dim=1)
        
        # Get predictions
        alpha_1 = vae(u_test_1)[0].cpu().numpy().flatten().round(2)
        alpha_2 = vae(u_test_2)[0].cpu().numpy().flatten().round(2)
        
        print(f"Input: Bought Item 1  -> Predicted Alpha: {alpha_1}")
        print(f"Input: Bought Item 10 -> Predicted Alpha: {alpha_2}")
        
        diff = torch.norm(torch.tensor(alpha_1) - torch.tensor(alpha_2)).item()
        
        if diff > 0.1:
            print(f"\n✅ SUCCESS! (Diff: {diff:.2f})")
            print("The model successfully generates different consumer types based on input behavior.")
            print("You are ready for Phase II.")
        else:
            print(f"\n❌ WARNING (Diff: {diff:.2f})")
            print("Posterior Collapse detected. Try increasing FREE_BITS or reducing model size.")