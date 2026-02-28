import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class VectorizedAggregatedGenerator:
    def __init__(self, num_items, num_true_components=5, device="cpu"):
        self.n = num_items
        self.n_plus_1 = num_items + 1
        self.K_true = num_true_components
        self.device = device

        self.true_alpha = torch.distributions.Dirichlet(torch.ones(self.K_true)).sample().to(device)
        self.true_v = torch.ones(self.K_true, self.n_plus_1, device=device) * 0.01
        self.true_v[:, 0] = 1.0

        min_favs = max(1, self.n // (2 * self.K_true))
        max_favs = max(min_favs, self.n // self.K_true)
        for k in range(self.K_true):
            fav_count = int(torch.randint(min_favs, max_favs + 1, (1,), device=device).item())
            fav_idx = torch.randperm(self.n, device=device)[:fav_count] + 1
            self.true_v[k, fav_idx] = torch.rand(fav_count, device=device) * 5.0 + 5.0

    def true_choice_probs(self, mask):
        scores = self.true_v.unsqueeze(0) * mask.unsqueeze(1)
        denom = scores.sum(dim=2, keepdim=True) + 1e-9
        prob_k = scores / denom
        return torch.einsum("k, mkn -> mn", self.true_alpha, prob_k)

    def generate_batch(self, num_datasets=5000, samples_per_assortment=100):
        mask = (torch.rand(num_datasets, self.n_plus_1, device=self.device) > 0.5).float()
        mask[:, 0] = 1.0

        scores = self.true_v.unsqueeze(0) * mask.unsqueeze(1)
        denom = scores.sum(dim=2, keepdim=True) + 1e-9
        prob_k = scores / denom
        prob_true = torch.einsum("k, mkn -> mn", self.true_alpha, prob_k)

        dist = torch.distributions.Multinomial(total_count=samples_per_assortment, probs=prob_true)
        counts = dist.sample()
        y_hat = counts / samples_per_assortment

        return torch.cat([y_hat, mask], dim=1)


class DGRA_VAE(nn.Module):
    def __init__(self, num_items, latent_dim, num_components, hidden_dim=64):
        super().__init__()
        self.n_plus_1 = num_items + 1
        self.K = num_components

        self.enc_net = nn.Sequential(
            nn.Linear(2 * self.n_plus_1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        self.dec_hidden = nn.Linear(latent_dim, hidden_dim)
        self.fc_alpha = nn.Linear(hidden_dim, num_components)
        self.fc_v = nn.Linear(hidden_dim, num_components * self.n_plus_1)

    def forward(self, x):
        h = self.enc_net(x)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        logvar = torch.clamp(logvar, min=-20.0, max=20.0)

        std = torch.exp(0.5 * logvar)
        z = mu + torch.randn_like(std) * std

        h_dec = F.relu(self.dec_hidden(z))
        alpha = F.softmax(self.fc_alpha(h_dec), dim=1)
        v_logits = torch.clamp(self.fc_v(h_dec), min=-12.0, max=12.0)
        v = torch.exp(v_logits).view(-1, self.K, self.n_plus_1)

        return alpha, v, mu, logvar

    def compute_choice_probs(self, alpha, v, mask):
        mask_exp = mask.unsqueeze(1)
        denom = torch.sum(v * mask_exp, dim=2, keepdim=True) + 1e-9
        prob_k = (v * mask_exp) / denom
        return torch.sum(alpha.unsqueeze(2) * prob_k, dim=1)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def save_matrix_csv(path, matrix, row_prefix="type_", col_prefix="type_"):
    k = matrix.shape[0]
    with open(path, "w", encoding="utf-8") as f:
        header = ["id"] + [f"{col_prefix}{j}" for j in range(k)]
        f.write(",".join(header) + "\n")
        for i in range(k):
            row = [f"{row_prefix}{i}"] + [f"{float(matrix[i, j]):.8f}" for j in range(k)]
            f.write(",".join(row) + "\n")


def export_true_type_behavior(run_dir, true_alpha, true_v, top_k=5):
    probs = true_v / (true_v.sum(axis=1, keepdims=True) + 1e-12)
    k, n_plus_1 = probs.shape
    item_probs = probs[:, 1:]
    top_k = min(top_k, item_probs.shape[1])

    top_items = np.argsort(item_probs, axis=1)[:, ::-1][:, :top_k] + 1
    top_probs = np.take_along_axis(item_probs, top_items - 1, axis=1)

    profiles = []
    for i in range(k):
        profiles.append(
            {
                "type_id": int(i),
                "mixture_weight": float(true_alpha[i]),
                "no_buy_prob": float(probs[i, 0]),
                "top_items": [int(x) for x in top_items[i].tolist()],
                "top_item_probs": [float(x) for x in top_probs[i].tolist()],
            }
        )
    save_json(os.path.join(run_dir, "true_type_profiles.json"), profiles)

    with open(os.path.join(run_dir, "true_type_profiles.csv"), "w", encoding="utf-8") as f:
        f.write("type_id,mixture_weight,no_buy_prob,top_items,top_item_probs\n")
        for p in profiles:
            items = "|".join(str(x) for x in p["top_items"])
            probs_s = "|".join(f"{x:.6f}" for x in p["top_item_probs"])
            f.write(
                f"{p['type_id']},{p['mixture_weight']:.8f},{p['no_buy_prob']:.8f},{items},{probs_s}\n"
            )

    top_sets = [set(row.tolist()) for row in top_items]
    jaccard = np.zeros((k, k), dtype=np.float64)
    cosine = np.zeros((k, k), dtype=np.float64)
    overlap = np.zeros((k, k), dtype=np.float64)
    for i in range(k):
        for j in range(k):
            inter = len(top_sets[i].intersection(top_sets[j]))
            union = len(top_sets[i].union(top_sets[j]))
            jaccard[i, j] = inter / max(union, 1)
            denom = np.linalg.norm(probs[i]) * np.linalg.norm(probs[j]) + 1e-12
            cosine[i, j] = float(np.dot(probs[i], probs[j]) / denom)
            overlap[i, j] = float(np.minimum(probs[i], probs[j]).sum())

    np.savez(
        os.path.join(run_dir, "true_type_overlap_matrices.npz"),
        jaccard_topk=jaccard,
        cosine_prob=cosine,
        overlap_prob=overlap,
    )
    save_matrix_csv(os.path.join(run_dir, "true_type_overlap_jaccard.csv"), jaccard)
    save_matrix_csv(os.path.join(run_dir, "true_type_overlap_cosine.csv"), cosine)
    save_matrix_csv(os.path.join(run_dir, "true_type_overlap_probmass.csv"), overlap)

    summary = {
        "top_k": int(top_k),
        "avg_pairwise_jaccard_topk": float(jaccard[np.triu_indices(k, 1)].mean() if k > 1 else 1.0),
        "avg_pairwise_cosine_prob": float(cosine[np.triu_indices(k, 1)].mean() if k > 1 else 1.0),
        "avg_pairwise_overlap_probmass": float(overlap[np.triu_indices(k, 1)].mean() if k > 1 else 1.0),
    }
    save_json(os.path.join(run_dir, "true_type_overlap_summary.json"), summary)
    return summary


def compute_data_entropy(data_tensor, num_items):
    n_plus_1 = num_items + 1
    y_true = data_tensor[:, :n_plus_1]
    entropy = -torch.sum(y_true * torch.log(y_true + 1e-9), dim=1).mean()
    return entropy.item()


def compute_probability_metrics(vae, gen, device, sims=5000, samples_per_assortment=100):
    mask = torch.ones(sims, gen.n + 1, device=device)

    with torch.no_grad():
        true_prob = gen.true_choice_probs(mask)[0].cpu().numpy()

    dist = torch.distributions.Multinomial(
        total_count=samples_per_assortment,
        probs=torch.tensor(true_prob, device=device),
    )
    counts = dist.sample((sims,))
    y_hat = counts / float(samples_per_assortment)

    u_m = torch.cat([y_hat, mask], dim=1)
    with torch.no_grad():
        alpha, v, _, _ = vae(u_m)
        y_pred = vae.compute_choice_probs(alpha, v, mask).cpu().numpy()

    y_pred_mean = y_pred.mean(axis=0)
    abs_err = np.abs(y_pred - true_prob.reshape(1, -1))

    mae = float(abs_err.mean())
    p95 = float(np.percentile(abs_err, 95))
    sqe = np.square(y_pred_mean - true_prob)
    centered = np.square(true_prob - true_prob.mean())
    r2 = float(1.0 - (sqe.sum() / (centered.sum() + 1e-12)))
    mean_bias = float((y_pred_mean - true_prob).mean())

    return {
        "metrics": {
            "mae": mae,
            "p95_abs_err": p95,
            "r2": r2,
            "mean_bias": mean_bias,
            "sims": int(sims),
            "samples_per_assortment": int(samples_per_assortment),
        },
        "arrays": {
            "true_prob": true_prob,
            "y_pred": y_pred,
            "y_pred_mean": y_pred_mean,
            "abs_err": abs_err,
        },
    }


def compute_sanity_diff(vae, n_items, device):
    vae.eval()
    with torch.no_grad():
        y_test_1 = torch.zeros(1, n_items + 1, device=device)
        y_test_1[0, 1] = 1.0
        mask_test = torch.ones(1, n_items + 1, device=device)
        u_test_1 = torch.cat([y_test_1, mask_test], dim=1)

        y_test_2 = torch.zeros(1, n_items + 1, device=device)
        target_item = min(10, n_items)
        y_test_2[0, target_item] = 1.0
        u_test_2 = torch.cat([y_test_2, mask_test], dim=1)

        alpha_1 = vae(u_test_1)[0].cpu().numpy().flatten()
        alpha_2 = vae(u_test_2)[0].cpu().numpy().flatten()

    return float(np.linalg.norm(alpha_1 - alpha_2))


def train_single_seed(seed, cfg, device, output_dir):
    print("\n" + "#" * 70)
    print(f"Seed {seed} start")
    print("#" * 70)
    set_seed(seed)

    run_dir = os.path.join(output_dir, f"seed_{seed}")
    os.makedirs(run_dir, exist_ok=True)

    gen = VectorizedAggregatedGenerator(cfg["N"], num_true_components=cfg["K"], device=device)
    all_data = gen.generate_batch(
        num_datasets=cfg["NUM_DATASETS"],
        samples_per_assortment=cfg["SAMPLES_PER_ASSORTMENT"],
    )

    lower_bound = compute_data_entropy(all_data, cfg["N"])
    print(f"Seed {seed} lower bound entropy: {lower_bound:.4f}")

    loader = DataLoader(TensorDataset(all_data), batch_size=cfg["BATCH"], shuffle=True)
    vae = DGRA_VAE(cfg["N"], cfg["D"], cfg["K"]).to(device)
    opt = optim.Adam(vae.parameters(), lr=cfg["LR"])

    free_bits = math.log(cfg["K"]) + 0.5

    history = {"epoch": [], "nll": [], "kl": [], "loss": [], "beta": [], "gap": []}

    t0 = time.time()
    for epoch in range(cfg["EPOCHS"]):
        total_nll = 0.0
        total_kl = 0.0
        total_loss = 0.0
        valid_batches = 0
        skipped_batches = 0

        if epoch < 100:
            beta = 0.001
        else:
            beta = min(0.05, (epoch - 20) / 50.0)

        for (u_m,) in loader:
            y_hat = u_m[:, : cfg["N"] + 1]
            mask = u_m[:, cfg["N"] + 1 :]

            opt.zero_grad()
            alpha, v, mu, logvar = vae(u_m)
            y_pred = vae.compute_choice_probs(alpha, v, mask)
            if (not torch.isfinite(alpha).all()) or (not torch.isfinite(v).all()) or (not torch.isfinite(y_pred).all()):
                skipped_batches += 1
                continue

            nll = -torch.sum(y_hat * torch.log(y_pred + 1e-9), dim=1).mean()
            kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

            kl_loss_term = torch.max(kl, torch.tensor(free_bits, device=device))
            loss = nll + beta * kl_loss_term
            if not torch.isfinite(loss):
                skipped_batches += 1
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=5.0)
            opt.step()

            total_nll += float(nll.item())
            total_kl += float(kl.item())
            total_loss += float(loss.item())
            valid_batches += 1

        if valid_batches == 0:
            raise RuntimeError(f"Seed {seed} epoch {epoch + 1}: no valid batches (all skipped due to NaN/Inf).")

        avg_nll = total_nll / valid_batches
        avg_kl = total_kl / valid_batches
        avg_loss = total_loss / valid_batches
        gap = avg_nll - lower_bound

        history["epoch"].append(epoch + 1)
        history["nll"].append(avg_nll)
        history["kl"].append(avg_kl)
        history["loss"].append(avg_loss)
        history["beta"].append(float(beta))
        history["gap"].append(gap)

        if (epoch + 1) % 10 == 0:
            print(
                f"Seed {seed} | Epoch {epoch + 1:03d} | "
                f"NLL {avg_nll:.4f} (gap {gap:+.4f}) | KL {avg_kl:.4f} | beta {beta:.4f} | skipped {skipped_batches}"
            )

    train_seconds = time.time() - t0

    prob_eval = compute_probability_metrics(
        vae,
        gen,
        device,
        sims=cfg["PROB_SIMS"],
        samples_per_assortment=cfg["SAMPLES_PER_ASSORTMENT"],
    )
    sanity_diff = compute_sanity_diff(vae, cfg["N"], device)

    np.savez(
        os.path.join(run_dir, "training_history.npz"),
        epoch=np.array(history["epoch"], dtype=np.int32),
        nll=np.array(history["nll"], dtype=np.float32),
        kl=np.array(history["kl"], dtype=np.float32),
        loss=np.array(history["loss"], dtype=np.float32),
        beta=np.array(history["beta"], dtype=np.float32),
        gap=np.array(history["gap"], dtype=np.float32),
    )
    save_json(os.path.join(run_dir, "training_history.json"), history)

    np.savez(
        os.path.join(run_dir, "probability_eval.npz"),
        true_prob=prob_eval["arrays"]["true_prob"],
        y_pred=prob_eval["arrays"]["y_pred"],
        y_pred_mean=prob_eval["arrays"]["y_pred_mean"],
        abs_err=prob_eval["arrays"]["abs_err"],
    )

    true_alpha_np = gen.true_alpha.detach().cpu().numpy()
    true_v_np = gen.true_v.detach().cpu().numpy()
    np.savez(
        os.path.join(run_dir, "generator_truth.npz"),
        true_alpha=true_alpha_np,
        true_v=true_v_np,
    )
    true_type_overlap_summary = export_true_type_behavior(
        run_dir=run_dir,
        true_alpha=true_alpha_np,
        true_v=true_v_np,
        top_k=5,
    )

    torch.save(vae.state_dict(), os.path.join(run_dir, "vae_state_dict.pt"))

    seed_summary = {
        "seed": int(seed),
        "train_seconds": float(train_seconds),
        "lower_bound_entropy": float(lower_bound),
        "final_nll": float(history["nll"][-1]),
        "best_nll": float(min(history["nll"])),
        "final_gap": float(history["gap"][-1]),
        "best_gap": float(min(history["gap"])),
        "final_kl": float(history["kl"][-1]),
        "best_kl": float(min(history["kl"])),
        "sanity_alpha_l2_diff": float(sanity_diff),
    }
    seed_summary.update(prob_eval["metrics"])
    seed_summary.update(true_type_overlap_summary)

    save_json(os.path.join(run_dir, "seed_summary.json"), seed_summary)
    return seed_summary


def aggregate_seed_metrics(seed_summaries, output_dir):
    metric_keys = [
        "final_nll",
        "best_nll",
        "final_gap",
        "best_gap",
        "final_kl",
        "mae",
        "p95_abs_err",
        "r2",
        "mean_bias",
        "sanity_alpha_l2_diff",
        "train_seconds",
    ]

    aggregate = {"num_seeds": len(seed_summaries), "metrics": {}}
    for key in metric_keys:
        vals = np.array([s[key] for s in seed_summaries], dtype=np.float64)
        aggregate["metrics"][key] = {
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=0)),
            "min": float(vals.min()),
            "max": float(vals.max()),
        }

    save_json(os.path.join(output_dir, "all_seed_summaries.json"), seed_summaries)
    save_json(os.path.join(output_dir, "aggregate_summary.json"), aggregate)

    csv_path = os.path.join(output_dir, "seed_metrics.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        header = ["seed"] + metric_keys
        f.write(",".join(header) + "\n")
        for row in seed_summaries:
            values = [str(row["seed"])] + [f"{row[k]:.8f}" for k in metric_keys]
            f.write(",".join(values) + "\n")

    print("\n" + "=" * 70)
    print("Multi-seed aggregate")
    print("=" * 70)
    for key in metric_keys:
        item = aggregate["metrics"][key]
        print(f"{key:20s} mean={item['mean']:.6f} std={item['std']:.6f} min={item['min']:.6f} max={item['max']:.6f}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    cfg = {
        "N": 20,
        "D": 8,
        "K": 5,
        "BATCH": 128,
        "EPOCHS": 200,
        "LR": 1e-2,
        "NUM_DATASETS": 10000,
        "SAMPLES_PER_ASSORTMENT": 100,
        "PROB_SIMS": 5000,
        "SEEDS": [0, 1, 2, 3, 4],
    }

    base_output_dir = os.path.join(os.path.dirname(__file__), "vae_multiseed_results")
    os.makedirs(base_output_dir, exist_ok=True)
    save_json(os.path.join(base_output_dir, "config.json"), cfg)

    summaries = []
    for seed in cfg["SEEDS"]:
        summaries.append(train_single_seed(seed, cfg, device, base_output_dir))

    aggregate_seed_metrics(summaries, base_output_dir)
    print(f"\nAll outputs saved under: {base_output_dir}")
