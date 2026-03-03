import argparse
import itertools
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.optimize import linprog


def parse_assortment(arg: str) -> List[int]:
    items = []
    for token in arg.split(","):
        token = token.strip()
        if token:
            items.append(int(token))
    return sorted(set(items))


def all_rankings(n: int) -> List[Dict[int, int]]:
    # Match section_3.jl construction:
    # sigma[i] = perm[i], i in 0..n (lower is more preferred)
    out = []
    for perm in itertools.permutations(range(n + 1)):
        out.append({i: perm[i] for i in range(n + 1)})
    return out


def construct_A(
    past_assortments: Sequence[Sequence[int]], rankings: Sequence[Dict[int, int]]
) -> np.ndarray:
    # A[k, m, i] = 1 if ranking k chooses item i in assortment m, else 0
    k_count = len(rankings)
    m_count = len(past_assortments)
    n_plus_1 = max(max(s) for s in past_assortments) + 1
    A = np.zeros((k_count, m_count, n_plus_1), dtype=float)
    for k, sigma in enumerate(rankings):
        for m, assortment in enumerate(past_assortments):
            best_i = min(assortment, key=lambda i: sigma[i])
            A[k, m, best_i] = 1.0
    return A


def evaluate_assortment(
    s_new: Sequence[int],
    r: Dict[int, float],
    past_assortments: Sequence[Sequence[int]],
    v: Dict[int, Dict[int, float]],
    n: int,
    rankings: Sequence[Dict[int, int]],
    best_case: bool,
) -> float:
    k_count = len(rankings)
    m_count = len(past_assortments)

    A_hist = construct_A(past_assortments, rankings)
    A_new = construct_A([list(s_new)], rankings)

    rev_new = np.zeros(k_count, dtype=float)
    for k in range(k_count):
        rev_new[k] = sum(A_new[k, 0, i] * r[i] for i in s_new)

    # Equality constraints:
    # 1) sum lambda = 1
    # 2) consistency for each historical (m, i in past_assortments[m])
    rows = [np.ones(k_count, dtype=float)]
    rhs = [1.0]
    for m in range(m_count):
        m_key = m + 1  # sturt's v uses 1-based m keys
        for i in past_assortments[m]:
            rows.append(A_hist[:, m, i].copy())
            rhs.append(float(v[m_key][i]))

    A_eq = np.vstack(rows)
    b_eq = np.array(rhs, dtype=float)

    c = rev_new.copy()
    if best_case:
        # maximize c^T λ -> minimize -c^T λ
        c = -c

    res = linprog(
        c=c,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=[(0.0, None)] * k_count,
        method="highs",
    )
    if not res.success:
        raise RuntimeError(f"LP failed: {res.message}")

    val = float(res.fun)
    return -val if best_case else val


def all_candidate_assortments(n: int) -> List[List[int]]:
    # Match section_3 filtering: assortment must contain outside option 0 and product n
    out = []
    universe = list(range(n + 1))
    for k in range(n + 2):
        for s in itertools.combinations(universe, k):
            if 0 in s and n in s:
                out.append(list(s))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate x_you under sturt section_3 robust metric (pure Python)."
    )
    parser.add_argument(
        "--assortment",
        type=str,
        required=False,
        help='Comma-separated assortment ids, e.g. "0,2,3,4"',
    )
    parser.add_argument(
        "--solve-optimal",
        action="store_true",
        help="Solve and print RO-optimal assortment under section_3 metric.",
    )
    args = parser.parse_args()

    # section_3 fixed instance
    n = 4
    past_assortments = [[0, 2, 3, 4], [0, 1, 2, 4]]
    revenues = [10.0, 20.0, 30.0, 100.0]
    r = {i + 1: revenues[i] for i in range(n)}
    r[0] = 0.0
    v = {
        1: {0: 0.3, 2: 0.3, 3: 0.3, 4: 0.1},
        2: {0: 0.3, 1: 0.3, 2: 0.1, 4: 0.3},
    }

    rankings = all_rankings(n)

    best_ro = -np.inf
    best_s = None
    for s in all_candidate_assortments(n):
        ro = evaluate_assortment(
            s_new=s,
            r=r,
            past_assortments=past_assortments,
            v=v,
            n=n,
            rankings=rankings,
            best_case=False,
        )
        if ro > best_ro:
            best_ro = ro
            best_s = s
    print("=" * 60)
    print("Section-3 Robust Benchmark Evaluation (sturt metric, Python)")
    print("=" * 60)
    print(f"RO-optimal assortment under same metric: {best_s}")
    print(f"RO*: {best_ro:.6f}")

    if args.solve_optimal and not args.assortment:
        return

    if not args.assortment:
        raise ValueError('Provide --assortment "0,2,3,4" or use --solve-optimal.')

    x_you = parse_assortment(args.assortment)
    if 0 not in x_you:
        raise ValueError(f"outside option 0 must be in assortment, got {x_you}")
    if any(i < 0 or i > n for i in x_you):
        raise ValueError(f"assortment ids must be in [0, {n}], got {x_you}")

    ro_you = evaluate_assortment(
        s_new=x_you,
        r=r,
        past_assortments=past_assortments,
        v=v,
        n=n,
        rankings=rankings,
        best_case=False,
    )
    gap = (best_ro - ro_you) / best_ro
    print(f"x_you assortment: {x_you}")
    print(f"RO(x_you): {ro_you:.6f}")
    print(f"optimality_gap: {gap:.6f} ({100.0 * gap:.2f}%)")


if __name__ == "__main__":
    main()
