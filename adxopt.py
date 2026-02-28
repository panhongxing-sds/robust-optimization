import time

import numpy as np


def ADXOpt(N, revenue_fn, b, A=None, B=None, S0=None, time_limit=100):
    """
    Local search for assortment optimization with optional linear constraints.

    Args:
        N (int): Number of products.
        revenue_fn (callable): Revenue function, takes a batch of 0-1 arrays and returns revenue.
        b (int): Maximum number of removals allowed per product.
        A (np.ndarray, optional): Constraint matrix.
        B (np.ndarray, optional): Constraint bounds.
        S0 (np.ndarray, optional): Initial solution.
        time_limit (float): Time limit in seconds.

    Returns:
        tuple: (best_revenue, best_assortment)
            best_revenue (float): Best revenue found.
            best_assortment (np.ndarray): Best assortment found.
    """

    def is_feasible(S, A, B):
        mask = np.ones(S.shape[0], dtype=bool)
        if A is not None and B is not None:
            constraint = A @ S.T
            mask &= np.all(constraint.T <= B, axis=1)
        return mask

    start_time = time.time()
    if S0 is None:
        S = np.zeros(N, dtype=np.float32)
    else:
        S = np.array(S0, dtype=np.float32)

    removals = np.zeros(N, dtype=int)

    while True:
        St = S.copy()
        current_revenue = revenue_fn(St[None])[0]

        if time.time() - start_time >= time_limit:
            print("Time limit reached.")
            return current_revenue, St

        available_mask = (removals < b) & (St == 0)
        add_idx = np.where(available_mask)[0]

        SA = St.copy()
        SA_revenue = current_revenue
        if len(add_idx) > 0:
            candidates = np.tile(St, (len(add_idx), 1))
            candidates[np.arange(len(add_idx)), add_idx] = 1
            feasibles = is_feasible(candidates, A, B)
            if feasibles.any():
                feasible_candidates = candidates[feasibles]
                revenues = revenue_fn(feasible_candidates)
                max_idx = np.argmax(revenues)
                SA = feasible_candidates[max_idx]
                SA_revenue = revenues[max_idx]

        if SA_revenue <= current_revenue:
            del_idx = np.where(St == 1)[0]
            SD = St.copy()
            SD_revenue = current_revenue
            if len(del_idx) > 0:
                candidates = np.tile(St, (len(del_idx), 1))
                candidates[np.arange(len(del_idx)), del_idx] = 0
                feasibles = is_feasible(candidates, A, B)
                if feasibles.any():
                    feasible_candidates = candidates[feasibles]
                    revenues = revenue_fn(feasible_candidates)
                    max_idx = np.argmax(revenues)
                    SD = feasible_candidates[max_idx]
                    SD_revenue = revenues[max_idx]

            SX = St.copy()
            SX_revenue = current_revenue
            exchange_pairs = [(i, j) for i in del_idx for j in add_idx]
            if exchange_pairs:
                candidates = np.tile(St, (len(exchange_pairs), 1))
                for idx, (i, j) in enumerate(exchange_pairs):
                    candidates[idx, i] = 0
                    candidates[idx, j] = 1
                feasibles = is_feasible(candidates, A, B)
                if feasibles.any():
                    feasible_candidates = candidates[feasibles]
                    revenues = revenue_fn(feasible_candidates)
                    max_idx = np.argmax(revenues)
                    SX = feasible_candidates[max_idx]
                    SX_revenue = revenues[max_idx]

            if SD_revenue >= SX_revenue:
                S = SD
                new_revenue = SD_revenue
            else:
                S = SX
                new_revenue = SX_revenue
        else:
            S = SA
            new_revenue = SA_revenue

        removed_items = np.where((St == 1) & (S == 0))[0]
        removals[removed_items] += 1

        if new_revenue <= current_revenue or np.all(removals >= b):
            if new_revenue <= current_revenue:
                return current_revenue, St
            return new_revenue, S
