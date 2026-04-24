"""
Human-readable dump of a **single** evaluation result dict (stdout).

Used for quick inspection after a run — not imported by the LaTeX table pipeline. For persisted
metrics, use the structured pickles and ``print/`` reporting modules instead.
"""


def print_evaluation_result(res: dict):
    """ Prints output data to a console.

    Args:
        res (dict): _description_
    """
    meta = res["evaluation_metadata"]

    print("=== Evaluation metadata ===")
    print(f"Evaluation set: {meta['evaluation_set']}")
    print(f"Group type:     {meta['group_type']}")
    print(f"Agent meta:  {meta['agent_meta']}\n")
    print(f"Mediator meta:  {meta['mediator_meta']}\n")

    print(f"Total groups matched: {len(res['matched_groups'])}/{res['total_rounds']} ({((len(res['matched_groups'])/res['total_rounds'])*100):.02f}%)")
    print(f"📊 Average round: {res['average']:.2f}")
    print(f"📉 Variance: {res['variance']:.4f}")
    print(f"📐 Standard deviation: {res['std_dev']:.4f}")

    print("\n📈 Count per round:")
    for k in sorted(res["counts"]):
        print(f"  Round {k}: {res['counts'][k]}x ({res['ratios'][k]*100:.1f} %)")

    # --- NEW: NDCG summary (optional) ---
    ndcg = res.get("ndcg")
    if ndcg:
        k = ndcg.get("k", "?")
        print("\n🎯 NDCG summary")
        # Individuální (per-user GT)
        print(f"  • Individual NDCG@{k}: "
                f"mean={ndcg.get('per_user_ndcg_mean_overall', 0.0):.4f} | "
                f"min={ndcg.get('per_user_ndcg_min_overall', 0.0):.4f} | "
                f"max={ndcg.get('per_user_ndcg_max_overall', 0.0):.4f}")
        # Společný GT (intersection)
        if 'ndcg_com_mean_overall' in ndcg or 'ndcg_com_min_overall' in ndcg:
            print(f"  • Common-GT NDCG@{k}: "
                    f"mean={ndcg.get('ndcg_com_mean_overall', 0.0):.4f} | "
                    f"min={ndcg.get('ndcg_com_min_overall', 0.0):.4f}")