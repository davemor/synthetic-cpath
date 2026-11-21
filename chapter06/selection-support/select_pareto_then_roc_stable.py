import json
from pathlib import Path
import numpy as np
import pandas as pd

base = Path("/raid/experiments/lowgen_v2/output")
scored_path = base/"pareto_then_roc_no_label_selection/scored_candidates.csv"
outdir = base/"pareto_then_roc_no_label_selection_stability_filtered"
outdir.mkdir(parents=True, exist_ok=True)
scored = pd.read_csv(scored_path)
objectives = [
    "roc_auc_proxy",
    "fid_proxy",
    "kid_proxy",
    "precision_proxy",
    "recall_proxy",
    "stability_score",
    "support_score",
]
# Stability as a hard requirement: keep top quartile mean_pairwise_ari within dataset/encoder.
parts = []
thresholds = []
for keys, group in scored.groupby(["dataset", "encoder"]):
    threshold = group["mean_pairwise_ari"].quantile(0.75)
    kept = group[group["mean_pairwise_ari"].ge(threshold)].copy()
    parts.append(kept)
    thresholds.append({"dataset": keys[0], "encoder": keys[1], "mean_pairwise_ari_min": float(threshold), "input_rows": int(len(group)), "kept_rows": int(len(kept))})
stable = pd.concat(parts, ignore_index=True)

def mark_pareto(group):
    vals = group[objectives].to_numpy(float)
    n = len(group)
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        ge = vals >= vals[i] - 1e-12
        gt = vals > vals[i] + 1e-12
        dom = ge.all(axis=1) & gt.any(axis=1)
        dom[i] = False
        dominated[i] = bool(dom.any())
    out = group.copy()
    out["pareto_front"] = ~dominated
    return out
stable = stable.groupby(["dataset", "encoder"], group_keys=False).apply(mark_pareto)
pareto = stable[stable["pareto_front"]].copy()
best = (
    pareto.sort_values(["dataset","encoder","roc_auc_proxy","stability_score","support_score"], ascending=[True,True,False,False,False])
    .groupby(["dataset","encoder"], as_index=False)
    .head(1)
    .copy()
)
top = (
    pareto.sort_values(["dataset","encoder","roc_auc_proxy"], ascending=[True,True,False])
    .groupby(["dataset","encoder"], as_index=False)
    .head(10)
    .copy()
)
stable.to_csv(outdir/"stable_scored_candidates.csv", index=False)
pareto.to_csv(outdir/"pareto_front.csv", index=False)
best.to_csv(outdir/"best_after_stability_pareto_by_dataset_encoder.csv", index=False)
top.to_csv(outdir/"top_stable_pareto_front_by_roc_proxy.csv", index=False)
pd.DataFrame(thresholds).to_csv(outdir/"stability_thresholds.csv", index=False)
meta={"selection":"top quartile mean_pairwise_ari hard filter, then Pareto front, then max roc_auc_proxy","objectives":objectives,"thresholds":thresholds,"stable_rows":int(len(stable)),"pareto_rows":int(len(pareto))}
(outdir/"metadata.json").write_text(json.dumps(meta, indent=2, sort_keys=True)+"\n")
cols=["dataset","encoder","projector","dimension","target_name","clusterer","cluster_params_json","mean_silhouette","mean_noise_fraction","mean_n_clusters","mean_pairwise_ari","effective_support","roc_auc_proxy","fid_proxy","kid_proxy","precision_proxy","recall_proxy","stability_score","support_score"]
print("metadata", meta)
print("\nBEST")
print(best[cols].round(4).to_string(index=False))
print("\nTOP STABLE FRONT")
print(top[cols].round(4).to_string(index=False))
