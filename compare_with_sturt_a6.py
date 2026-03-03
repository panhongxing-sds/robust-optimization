import argparse
import copy
import csv
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import main_sturt_section3 as model_run


CONJOINT_DIR = Path("/share/home/luenqiao/phx/gra/sturt/conjoint_data")


@dataclass
class A6Instance:
    instance_id: int
    n: int
    M: int
    prices: torch.Tensor
    all_data: torch.Tensor
    past_assortments: List[List[int]]
    v_obs: Dict[int, Dict[int, float]]
    rankings_local: np.ndarray
    opt_assortment: List[int]
    opt_revenue: float


def set_all_seeds(seed: int) -> None:
    model_run.set_seed(seed)


def load_conjoint_data() -> Tuple[np.ndarray, np.ndarray]:
    revenues = np.loadtxt(CONJOINT_DIR / "revenues_mat.csv", delimiter=",", dtype=np.float64)
    orderings = np.loadtxt(CONJOINT_DIR / "orderings_mat.csv", delimiter=",", dtype=np.int64)
    if revenues.ndim == 1:
        revenues = revenues[None, :]
    if revenues.shape != (1, 3584):
        raise ValueError(f"unexpected revenues shape: {revenues.shape}")
    if orderings.shape != (330, 3585):
        raise ValueError(f"unexpected orderings shape: {orderings.shape}")
    return revenues[0], orderings


def revenue_of_assortment(
    assortment: Sequence[int],
    prices: np.ndarray,
    rankings_local: np.ndarray,
) -> float:
    sub = rankings_local[:, assortment]
    chosen_local_idx = np.argmin(sub, axis=1)
    chosen_item = np.array(assortment, dtype=np.int64)[chosen_local_idx]
    return float(prices[chosen_item].mean())


def brute_force_optimal_assortment(prices: np.ndarray, rankings_local: np.ndarray, n: int) -> Tuple[List[int], float]:
    best_s = [0]
    best_rev = -1e18
    for bits in range(1 << n):
        s = [0] + [i + 1 for i in range(n) if (bits >> i) & 1]
        rev = revenue_of_assortment(s, prices, rankings_local)
        if rev > best_rev:
            best_rev = rev
            best_s = s
    return best_s, float(best_rev)


def build_rankings_local(orderings: np.ndarray, local_to_global: List[int]) -> np.ndarray:
    K = orderings.shape[0]
    n_plus_1 = len(local_to_global)
    rankings_local = np.empty((K, n_plus_1), dtype=np.int64)
    max_global = int(orderings.max())
    for k in range(K):
        pos = np.empty(max_global + 1, dtype=np.int64)
        row = orderings[k]
        for rank_idx, prod_id in enumerate(row):
            pos[int(prod_id)] = rank_idx
        sub_positions = np.array([pos[gid] for gid in local_to_global], dtype=np.int64)
        rankings_local[k] = np.argsort(np.argsort(sub_positions))
    return rankings_local


def compute_sales(
    past_assortments: Sequence[Sequence[int]],
    rankings_local: np.ndarray,
) -> Dict[int, Dict[int, float]]:
    K = rankings_local.shape[0]
    v: Dict[int, Dict[int, float]] = {}
    for m_idx, assortment in enumerate(past_assortments, start=1):
        sub = rankings_local[:, assortment]
        chosen_local_idx = np.argmin(sub, axis=1)
        chosen_item = np.array(assortment, dtype=np.int64)[chosen_local_idx]
        counts = np.bincount(chosen_item, minlength=rankings_local.shape[1]).astype(np.float64)
        probs = counts / float(K)
        v[m_idx] = {i: float(probs[i]) for i in assortment}
    return v


def build_tensor_from_history(
    n: int,
    past_assortments: Sequence[Sequence[int]],
    v_obs: Dict[int, Dict[int, float]],
    device: torch.device,
) -> torch.Tensor:
    n_plus_1 = n + 1
    rows = []
    for m_idx, assortment in enumerate(past_assortments, start=1):
        y_hat = torch.zeros(n_plus_1, dtype=torch.float32, device=device)
        mask = torch.zeros(n_plus_1, dtype=torch.float32, device=device)
        mask[assortment] = 1.0
        for i, p in v_obs[m_idx].items():
            y_hat[i] = float(p)
        rows.append(torch.cat([y_hat, mask], dim=0))
    return torch.stack(rows, dim=0)


def generate_a6_instance(
    revenues_3584: np.ndarray,
    orderings: np.ndarray,
    seed: int,
    n: int,
    M: int,
    device: torch.device,
    instance_id: int,
) -> A6Instance:
    rng = np.random.default_rng(seed)
    num_products = 3584
    outside_global = 3585

    while True:
        subset = rng.choice(np.arange(1, num_products + 1, dtype=np.int64), size=n, replace=False)
        subset = np.sort(subset)
        local_to_global = subset.tolist() + [outside_global]
        rev = np.array([revenues_3584[gid - 1] for gid in subset] + [0.0], dtype=np.float64)
        if len(np.unique(rev)) == n + 1:
            break

    sort_idx = np.argsort(rev)
    rev_sorted = rev[sort_idx]
    local_to_global = [local_to_global[int(i)] for i in sort_idx]

    rankings_local = build_rankings_local(orderings, local_to_global)

    random_integers = rng.integers(1, 2 ** (M + 1), size=n - 1, endpoint=False, dtype=np.int64)
    past_assortments = [[0] for _ in range(M)]
    for i in range(1, n):
        val = int(random_integers[i - 1])
        for m in range(M):
            if val % 2 == 1:
                past_assortments[m].append(i)
            val >>= 1
    for m in range(M):
        past_assortments[m].append(n)
        past_assortments[m] = sorted(set(past_assortments[m]))

    v_obs = compute_sales(past_assortments, rankings_local)
    all_data = build_tensor_from_history(n, past_assortments, v_obs, device)
    prices_tensor = torch.tensor(rev_sorted, dtype=torch.float32, device=device)

    opt_s, opt_rev = brute_force_optimal_assortment(rev_sorted, rankings_local, n)
    return A6Instance(
        instance_id=instance_id,
        n=n,
        M=M,
        prices=prices_tensor,
        all_data=all_data,
        past_assortments=[list(s) for s in past_assortments],
        v_obs=v_obs,
        rankings_local=rankings_local,
        opt_assortment=opt_s,
        opt_revenue=opt_rev,
    )


def build_benchmark_suite(
    revenues_3584: np.ndarray,
    orderings: np.ndarray,
    n_instances: int,
    n: int,
    M_values: Sequence[int],
    base_seed: int,
    device: torch.device,
) -> List[A6Instance]:
    suite = []
    for idx in range(n_instances):
        M = M_values[idx % len(M_values)]
        seed = base_seed + idx * 9973
        suite.append(
            generate_a6_instance(
                revenues_3584=revenues_3584,
                orderings=orderings,
                seed=seed,
                n=n,
                M=M,
                device=device,
                instance_id=idx + 1,
            )
        )
    return suite


def train_vae_for_instance(instance: A6Instance, cfg: Dict) -> model_run.DGRA_VAE:
    device = cfg["device"]
    vae = model_run.DGRA_VAE(instance.n, cfg["D"], cfg["K"], cfg["hidden_dim"]).to(device)
    opt = optim.Adam(vae.parameters(), lr=cfg["lr_phase1"])
    loader = DataLoader(
        TensorDataset(instance.all_data),
        batch_size=min(cfg["batch_size"], instance.all_data.shape[0]),
        shuffle=True,
    )
    free_bits = math.log(cfg["K"]) + 0.5
    for epoch in range(cfg["epochs_phase1"]):
        beta = min(cfg["beta_max"], epoch / 50.0 * cfg["beta_max"])
        for batch in loader:
            u_m = batch[0]
            opt.zero_grad()
            alpha, v_out, mu, logvar = vae(u_m)
            y_pred = vae.compute_choice_probs(alpha, v_out, u_m[:, instance.n + 1 :])
            nll = -torch.sum(u_m[:, : instance.n + 1] * torch.log(y_pred + 1e-9), dim=1).mean()
            kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
            loss = nll + beta * max(kl, torch.tensor(free_bits, device=device))
            loss.backward()
            opt.step()
    return vae


def solve_with_model(instance: A6Instance, cfg: Dict) -> Tuple[List[int], float]:
    set_all_seeds(cfg["seed"])
    vae = train_vae_for_instance(instance, cfg)

    best_x = None
    best_score = -1e18
    for ridx in range(cfg["num_restarts"]):
        set_all_seeds(cfg["seed"] + ridx * 17)
        optimizer = model_run.RobustAssortmentOptimizer(vae, instance.prices, instance.all_data, cfg)
        out = optimizer.optimize_assortment()
        x_try = out[0] if isinstance(out, tuple) else out
        x_tensor = torch.from_numpy(x_try).to(cfg["device"]).unsqueeze(0)
        _, score = optimizer.solve_inner_adversary(x_tensor)
        if score > best_score:
            best_score = score
            best_x = x_try.copy()
    if best_x is None:
        raise RuntimeError("no solution from optimizer")

    chosen = np.where(best_x > 0.5)[0].tolist()
    chosen = sorted(set(int(i) for i in chosen))
    if 0 not in chosen:
        chosen = [0] + chosen
    return chosen, float(best_score)


def evaluate_trial_on_suite(cfg: Dict, suite: Sequence[A6Instance]) -> Dict:
    gaps = []
    per_instance = []
    for inst in suite:
        x_you, robust_score = solve_with_model(inst, cfg)
        your_rev = revenue_of_assortment(x_you, inst.prices.detach().cpu().numpy(), inst.rankings_local)
        gap = (inst.opt_revenue - your_rev) / (inst.opt_revenue + 1e-12)
        gaps.append(gap)
        per_instance.append(
            {
                "instance_id": inst.instance_id,
                "M": inst.M,
                "x_you_csv": ",".join(str(i) for i in x_you),
                "your_rev": your_rev,
                "opt_rev": inst.opt_revenue,
                "gap": gap,
                "robust_score_proxy": robust_score,
                "opt_assortment_csv": ",".join(str(i) for i in inst.opt_assortment),
            }
        )
    avg_gap = float(np.mean(gaps))
    worst_gap = float(np.max(gaps))
    return {"avg_gap": avg_gap, "worst_gap": worst_gap, "per_instance": per_instance}


def candidate_configs(base_cfg: Dict) -> List[Dict]:
    presets = [
        {
            "name": "base_a6",
            "K": 4,
            "D": 8,
            "hidden_dim": 64,
            "epochs_phase1": 180,
            "beta_max": 0.01,
            "adv_steps": 20,
            "adv_lr": 0.1,
            "n_starts": 8,
            "adx_rounds": 15,
            "adx_time_limit": 30.0,
            "adx_b": 3,
            "num_restarts": 6,
        },
        {
            "name": "deeper_latent",
            "K": 6,
            "D": 12,
            "hidden_dim": 96,
            "epochs_phase1": 260,
            "beta_max": 0.02,
            "adv_steps": 30,
            "adv_lr": 0.12,
            "n_starts": 10,
            "adx_rounds": 20,
            "adx_time_limit": 40.0,
            "adx_b": 4,
            "num_restarts": 8,
        },
        {
            "name": "fast_probe",
            "K": 3,
            "D": 4,
            "hidden_dim": 48,
            "epochs_phase1": 120,
            "beta_max": 0.005,
            "adv_steps": 15,
            "adv_lr": 0.08,
            "n_starts": 6,
            "adx_rounds": 10,
            "adx_time_limit": 20.0,
            "adx_b": 3,
            "num_restarts": 5,
        },
    ]
    out = []
    for p in presets:
        c = copy.deepcopy(base_cfg)
        c.update(p)
        out.append(c)
    return out


def random_config(base_cfg: Dict, rng: np.random.Generator, trial_id: int) -> Dict:
    cfg = copy.deepcopy(base_cfg)
    cfg["name"] = f"rand_{trial_id:04d}"
    cfg["K"] = int(rng.choice([3, 4, 5, 6, 8]))
    cfg["D"] = int(rng.choice([3, 4, 6, 8, 10, 12]))
    cfg["hidden_dim"] = int(rng.choice([32, 48, 64, 96, 128]))
    cfg["epochs_phase1"] = int(rng.choice([100, 140, 180, 220, 300]))
    cfg["beta_max"] = float(rng.choice([0.003, 0.005, 0.01, 0.02]))
    cfg["adv_steps"] = int(rng.choice([10, 15, 20, 30, 40]))
    cfg["adv_lr"] = float(rng.choice([0.05, 0.08, 0.1, 0.12, 0.15]))
    cfg["n_starts"] = int(rng.choice([4, 6, 8, 10, 12]))
    cfg["adx_rounds"] = int(rng.choice([8, 10, 15, 20, 25]))
    cfg["adx_time_limit"] = float(rng.choice([15.0, 20.0, 30.0, 40.0, 60.0]))
    cfg["adx_b"] = int(rng.choice([2, 3, 4, 5]))
    cfg["num_restarts"] = int(rng.choice([3, 5, 6, 8, 10]))
    cfg["seed"] = int(base_cfg["seed"] + trial_id * 31)
    return cfg


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_trial_csv(path: Path, records: List[Dict]) -> None:
    if not records:
        return
    keys = list(records[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in records:
            writer.writerow(r)


def main() -> None:
    parser = argparse.ArgumentParser(description="A6 benchmark tuning: minimize optimality gap.")
    parser.add_argument("--trials", type=int, default=50, help="Total hyperparameter trials.")
    parser.add_argument("--instances", type=int, default=6, help="Number of A6-like instances.")
    parser.add_argument("--n", type=int, default=15, help="Products per subproblem.")
    parser.add_argument("--m-values", type=str, default="3,4,5", help="Comma-separated historical M values.")
    parser.add_argument("--seed", type=int, default=77, help="Global seed.")
    parser.add_argument(
        "--result-dir",
        type=str,
        default="/share/home/luenqiao/phx/gra/result/a6_benchmark",
        help="Directory for logs/results.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Override with quick settings for smoke runs.",
    )
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    ensure_dir(result_dir)

    set_all_seeds(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    m_values = [int(x.strip()) for x in args.m_values.split(",") if x.strip()]
    if not m_values:
        raise ValueError("m-values is empty")

    print("=" * 72)
    print("A6 Benchmark Tuning (VAE + adversarial + ADXOpt)")
    print("=" * 72)
    print(f"device={device} seed={args.seed} trials={args.trials} instances={args.instances} m_values={m_values}")

    revenues_3584, orderings = load_conjoint_data()
    suite = build_benchmark_suite(
        revenues_3584=revenues_3584,
        orderings=orderings,
        n_instances=args.instances,
        n=args.n,
        M_values=m_values,
        base_seed=args.seed,
        device=device,
    )

    base_cfg = copy.deepcopy(model_run.CONFIG)
    base_cfg.update(
        {
            "N": args.n,
            "device": device,
            "batch_size": 8,
            "lr_phase1": 1e-2,
            "rho_multiplier": 2.0,
            "phase2_z_init_scale": 0.0,
            "adx_init_fill": 1.0,
            "seed": args.seed,
        }
    )
    if args.quick:
        base_cfg["epochs_phase1"] = 40
        args.trials = min(args.trials, 3)
        print("quick mode enabled: reduced epochs/trials")

    trial_queue = candidate_configs(base_cfg)
    rng = np.random.default_rng(args.seed + 999)

    all_trial_rows = []
    best = None
    trial_id = 0
    while trial_id < args.trials:
        if trial_queue:
            cfg = trial_queue.pop(0)
        else:
            cfg = random_config(base_cfg, rng, trial_id + 1)
        if args.quick:
            cfg["epochs_phase1"] = min(int(cfg["epochs_phase1"]), 40)
            cfg["num_restarts"] = min(int(cfg["num_restarts"]), 2)
            cfg["adx_rounds"] = min(int(cfg["adx_rounds"]), 5)
            cfg["adx_time_limit"] = min(float(cfg["adx_time_limit"]), 5.0)
        trial_id += 1

        print("-" * 72)
        print(
            f"trial#{trial_id} {cfg['name']} | "
            f"K={cfg['K']} D={cfg['D']} hidden={cfg['hidden_dim']} epochs={cfg['epochs_phase1']} "
            f"adv_steps={cfg['adv_steps']} restarts={cfg['num_restarts']}"
        )

        metrics = evaluate_trial_on_suite(cfg, suite)
        row = {
            "trial_id": trial_id,
            "name": cfg["name"],
            "avg_gap": metrics["avg_gap"],
            "worst_gap": metrics["worst_gap"],
            "K": cfg["K"],
            "D": cfg["D"],
            "hidden_dim": cfg["hidden_dim"],
            "epochs_phase1": cfg["epochs_phase1"],
            "adv_steps": cfg["adv_steps"],
            "n_starts": cfg["n_starts"],
            "adx_rounds": cfg["adx_rounds"],
            "adx_time_limit": cfg["adx_time_limit"],
            "adx_b": cfg["adx_b"],
            "num_restarts": cfg["num_restarts"],
            "seed": cfg["seed"],
        }
        all_trial_rows.append(row)
        write_trial_csv(result_dir / "trial_summary.csv", all_trial_rows)

        if best is None or metrics["avg_gap"] < best["avg_gap"]:
            best = {
                "avg_gap": metrics["avg_gap"],
                "worst_gap": metrics["worst_gap"],
                "cfg": copy.deepcopy(cfg),
                "per_instance": metrics["per_instance"],
                "trial_id": trial_id,
                "name": cfg["name"],
            }
            print(
                f"NEW_BEST | avg_gap={100.0 * best['avg_gap']:.2f}% "
                f"| worst_gap={100.0 * best['worst_gap']:.2f}%"
            )
            write_trial_csv(result_dir / "best_per_instance.csv", best["per_instance"])
        else:
            print(
                f"trial_result | avg_gap={100.0 * metrics['avg_gap']:.2f}% "
                f"| worst_gap={100.0 * metrics['worst_gap']:.2f}% "
                f"| best_avg_gap={100.0 * best['avg_gap']:.2f}%"
            )

        if best["avg_gap"] <= 1e-12:
            print(">>> Found avg_gap == 0. Stop early.")
            break

    if best is None:
        raise RuntimeError("no successful trials")

    print("=" * 72)
    print("A6 Benchmark Search Summary")
    print("=" * 72)
    print(f"best_trial: #{best['trial_id']} {best['name']}")
    print(f"best_avg_gap: {best['avg_gap']:.6f} ({100.0 * best['avg_gap']:.2f}%)")
    print(f"best_worst_gap: {best['worst_gap']:.6f} ({100.0 * best['worst_gap']:.2f}%)")
    print("best_config:")
    print(
        f"  K={best['cfg']['K']} D={best['cfg']['D']} hidden={best['cfg']['hidden_dim']} "
        f"epochs={best['cfg']['epochs_phase1']} adv_steps={best['cfg']['adv_steps']} "
        f"n_starts={best['cfg']['n_starts']} adx_rounds={best['cfg']['adx_rounds']} "
        f"adx_time_limit={best['cfg']['adx_time_limit']} adx_b={best['cfg']['adx_b']} "
        f"restarts={best['cfg']['num_restarts']} seed={best['cfg']['seed']}"
    )
    print(f"saved: {result_dir / 'trial_summary.csv'}")
    print(f"saved: {result_dir / 'best_per_instance.csv'}")


if __name__ == "__main__":
    main()
