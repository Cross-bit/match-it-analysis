from __future__ import annotations

import argparse
import math
from statistics import mean, pstdev
from typing import List, Sequence

import numpy as np

from evaluation_frameworks.consensus_evaluation.evaluation.evaluations.evaluation_context_factory import (
    build_context_holdout,
    load_evaluation_agent_sigmoid_normed,
)


def _stderr(values: Sequence[float]) -> float:
    n = len(values)
    if n <= 1:
        return float("nan")
    sd = float(np.std(values, ddof=1))
    return sd / math.sqrt(n)


def _fmt(x: float, d: int = 4) -> str:
    if x is None or not math.isfinite(x):
        return "nan"
    return f"{x:.{d}f}"


def _pick_users(
    context: dict,
    scope: str,
    max_users: int | None,
    seed: int,
) -> List[int]:
    if scope == "all":
        users = list(context["user_id_map"].keys())
    else:
        users = sorted({u for grp in context["eval_set_groups_data"] for u in grp})

    if max_users is not None and max_users > 0 and len(users) > max_users:
        rng = np.random.default_rng(seed)
        users = rng.choice(np.asarray(users, dtype=int), size=max_users, replace=False).tolist()

    return users


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=(
            "Estimate mapping population_mood_bias -> effective stars shift "
            "using sigma_u statistics from UserVoteSimulatorSigmoidNormed."
        )
    )
    p.add_argument("--eval-type", default="test", choices=["train", "validation", "test"])
    p.add_argument(
        "--group-type",
        default="similar",
        choices=["similar", "outlier", "random", "divergent", "variance"],
        help="Only affects which eval groups are loaded for scope=eval-groups.",
    )
    p.add_argument(
        "--user-scope",
        default="eval-groups",
        choices=["eval-groups", "all"],
        help=(
            "eval-groups: users appearing in eval groups only (fast, default). "
            "all: all users in filtered dataset (slow; recommend sampling via --max-users)."
        ),
    )
    p.add_argument(
        "--max-users",
        type=int,
        default=5000,
        help="Max users for sigma estimation (sampling without replacement). Use 0/negative for no cap.",
    )
    p.add_argument(
        "--biases",
        nargs="+",
        type=float,
        default=[0.0, 1.0, 2.0],
        help="Bias values to convert into stars shift.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--normalization-sample-k",
        type=int,
        default=500,
        help="Per-user item sample size for sigma_u estimation (same as simulator parameter).",
    )
    args = p.parse_args()

    max_users = args.max_users if args.max_users and args.max_users > 0 else None

    context = build_context_holdout(
        EVALUATION_SET_TYPE=args.eval_type,
        GROUP_TYPE=args.group_type,
    )
    agent = load_evaluation_agent_sigmoid_normed(
        context["filtered_evaluation_set_csr"],
        context["user_id_map"],
        context["item_id_map"],
        args.eval_type,
        global_user_bias=0.0,
        normalization_sample_k=args.normalization_sample_k,
        seed=args.seed,
    )

    users = _pick_users(
        context=context,
        scope=args.user_scope,
        max_users=max_users,
        seed=args.seed,
    )
    if not users:
        raise SystemExit("No users selected; adjust --user-scope / --max-users.")

    sigmas: List[float] = []
    for i, u in enumerate(users, start=1):
        _, sigma_u = agent._get_user_scale(int(u))
        sigmas.append(float(sigma_u))
        if i % 1000 == 0:
            print(f"[progress] users processed: {i}/{len(users)}")

    mu_sigma = float(mean(sigmas))
    sd_sigma = float(pstdev(sigmas)) if len(sigmas) > 1 else float("nan")
    se_sigma = _stderr(sigmas)
    ci95 = 1.96 * se_sigma if math.isfinite(se_sigma) else float("nan")

    print("")
    print("=== Sigma_u summary ===")
    print(f"users_n: {len(users)}")
    print(f"user_scope: {args.user_scope}")
    print(f"mean_sigma_u: {_fmt(mu_sigma)}")
    print(f"std_sigma_u: {_fmt(sd_sigma)}")
    print(f"stderr_mean_sigma_u: {_fmt(se_sigma)}")
    print(f"95% CI(mean_sigma_u): [{_fmt(mu_sigma - ci95)}, {_fmt(mu_sigma + ci95)}]")

    print("")
    print("=== Bias -> stars shift (approx) ===")
    print("Formula: Δstars ≈ bias * mean_sigma_u")
    for b in args.biases:
        shift = float(b) * mu_sigma
        lo = float(b) * (mu_sigma - ci95)
        hi = float(b) * (mu_sigma + ci95)
        print(f"bias={b:.3g}: Δstars≈{_fmt(shift)}   95% CI≈[{_fmt(lo)}, {_fmt(hi)}]")

