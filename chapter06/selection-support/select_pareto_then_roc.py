import json
from pathlib import Path
import numpy as np
import pandas as pd

base = Path("/raid/experiments/lowgen_v2/output")
ranked_path = base/"calibrated_clustering_selection_gmm4/clustering_experiment_ranked_configs.csv"
corr_cluster_label = base/"lowgen-v2-cluster-guided-full-embedding/cluster_guided_full_embedding_gmm4_label_full_run/cluster_metric_quality_correlations.csv"
corr_cluster_only = base/"lowgen-v2-cluster-guided-full-embedding/cluster_guided_full_embedding_gmm4_full_run/cluster_metric_quality_correlations_with_sd_stability.csv"
outdir = base/"pareto_then_roc_no_label_selection"
outdir.mkdir(parents=True, exist_ok=True)

metric_map = {
    "silhouette": "mean_silhouette",
    "noise_fraction": "mean_noise_fraction",
    "n_clusters": "mean_n_clusters",
    "mean_pairwise_ari": "mean_pairwise_ari",
    "cluster_persistence": "mean_cluster_persistence",
    "sd_index": "sd_index",
    "sd_scat": "sd_scat",
    "sd_dis": "sd_dis",
}
no_label_metrics = list(metric_map)
quality_dirs = {
    "fid": -1.0,
    "kid": -1.0,
    "precision": 1.0,
    "recall": 1.0,
    "tstr_macro_roc_auc_ovr": 1.0,
    "aug_macro_roc_auc_ovr": 1.0,
}

candidates = pd.read_csv(ranked_path)
candidates = candidates[
    candidates["scenario"].eq("no_label")
    & candidates["eligible"].eq(True)
    & candidates["encoder"].eq("uni2")
    & candidates["dataset"].isin(["icaird100k", "kather100k"])
].copy()

# Use labelled utility run to calibrate ROC proxy; use cluster-only GMM4 for fidelity/diversity proxies
# because it includes the augmented sd/stability columns.
roc_corr = pd.read_csv(corr_cluster_label)
quality_corr = pd.read_csv(corr_cluster_only)
quality_corr = quality_corr[quality_corr["generator"].eq("cluster_guided_full_diag_gmm4_probe")].copy()
calibration = pd.concat(
    [
        roc_corr[roc_corr["quality_metric"].isin(["tstr_macro_roc_auc_ovr", "aug_macro_roc_auc_ovr"])],
        quality_corr[quality_corr["quality_metric"].isin(["fid", "kid", "precision", "recall"])],
    ],
    ignore_index=True,
)
calibration = calibration[
    calibration["cluster_metric"].isin(no_label_metrics)
    & calibration["quality_metric"].isin(quality_dirs)
    & calibration["spearman"].notna()
    & calibration["n"].ge(30)
].copy()
calibration["clustering_column"] = calibration["cluster_metric"].map(metric_map)
calibration = calibration[calibration["clustering_column"].isin(candidates.columns)].copy()
calibration["desired_association"] = calibration.apply(
    lambda r: quality_dirs[r["quality_metric"]] * float(r["spearman"]), axis=1
)
calibration["weight"] = calibration["desired_association"].abs()

# Build percentile proxy scores within dataset/encoder so values are comparable inside each decision group.
def add_proxy_scores(group: pd.DataFrame) -> pd.DataFrame:
    group = group.copy()
    for q in ["fid", "kid", "precision", "recall", "tstr_macro_roc_auc_ovr", "aug_macro_roc_auc_ovr"]:
        rows = calibration[calibration["quality_metric"].eq(q)]
        score = np.zeros(len(group), dtype=float)
        weight_sum = 0.0
        details = []
        for _, r in rows.iterrows():
            col = r["clustering_column"]
            values = pd.to_numeric(group[col], errors="coerce")
            if values.notna().sum() < 2:
                continue
            oriented = values if r["desired_association"] >= 0 else -values
            pct = oriented.rank(pct=True, method="average").fillna(0.5).to_numpy()
            w = float(r["weight"])
            score += w * pct
            weight_sum += w
            details.append(str(r.cluster_metric) + chr(58) + format(float(r.desired_association), chr(46)+chr(51)+chr(102)))
        group[f"{q}_proxy"] = score / weight_sum if weight_sum else np.nan
        group[f"{q}_proxy_details"] = ";".join(details)
    group["roc_auc_proxy"] = group[["tstr_macro_roc_auc_ovr_proxy", "aug_macro_roc_auc_ovr_proxy"]].mean(axis=1)
    group["stability_score"] = pd.to_numeric(group["mean_pairwise_ari"], errors="coerce").rank(pct=True, method="average")
    group["support_score"] = pd.to_numeric(group["effective_support"], errors="coerce").rank(pct=True, method="average")
    return group

scored = candidates.groupby(["dataset", "encoder"], group_keys=False).apply(add_proxy_scores)
objectives = [
    "roc_auc_proxy",
    "fid_proxy",
    "kid_proxy",
    "precision_proxy",
    "recall_proxy",
    "stability_score",
    "support_score",
]
scored = scored.dropna(subset=objectives).copy()

# Non-dominated set within each dataset/encoder. Higher is better for every objective.
def mark_pareto(group: pd.DataFrame) -> pd.DataFrame:
    vals = group[objectives].to_numpy(float)
    n = len(group)
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        if dominated[i]:
            continue
        ge = vals >= vals[i] - 1e-12
        gt = vals > vals[i] + 1e-12
        dom = ge.all(axis=1) & gt.any(axis=1)
        dom[i] = False
        if dom.any():
            dominated[i] = True
    out = group.copy()
    out["pareto_front"] = ~dominated
    return out

scored = scored.groupby(["dataset", "encoder"], group_keys=False).apply(mark_pareto)
pareto = scored[scored["pareto_front"]].copy()
best = (
    pareto.sort_values(
        ["dataset", "encoder", "roc_auc_proxy", "stability_score", "support_score"],
        ascending=[True, True, False, False, False],
    )
    .groupby(["dataset", "encoder"], as_index=False)
    .head(1)
    .copy()
)

# Include a few alternatives on the front for context.
top_front = (
    pareto.sort_values(["dataset", "encoder", "roc_auc_proxy"], ascending=[True, True, False])
    .groupby(["dataset", "encoder"], as_index=False)
    .head(10)
    .copy()
)
for df in [scored, pareto, best, top_front]:
    if "cluster_params_json" in df:
        pass
scored.to_csv(outdir/"scored_candidates.csv", index=False)
pareto.to_csv(outdir/"pareto_front.csv", index=False)
best.to_csv(outdir/"best_after_pareto_by_dataset_encoder.csv", index=False)
top_front.to_csv(outdir/"top_pareto_front_by_roc_proxy.csv", index=False)
calibration.to_csv(outdir/"no_label_proxy_calibration.csv", index=False)
meta = {
    "candidate_rows": int(len(candidates)),
    "scored_rows": int(len(scored)),
    "pareto_rows": int(len(pareto)),
    "objectives": objectives,
    "selection": "highest roc_auc_proxy among non-dominated eligible configs",
    "filters_inherited": "scenario=no_label, eligible=True from calibrated_clustering_selection_gmm4, encoder=uni2, datasets icaird100k/kather100k",
}
(outdir/"metadata.json").write_text(json.dumps(meta, indent=2, sort_keys=True)+"\n")
cols = [
    "dataset","encoder","projector","dimension","target_name","clusterer","cluster_params_json",
    "mean_silhouette","mean_noise_fraction","mean_n_clusters","mean_pairwise_ari","effective_support",
    "roc_auc_proxy","fid_proxy","kid_proxy","precision_proxy","recall_proxy","stability_score","support_score"
]
print("metadata", meta)
print("\nBEST")
print(best[cols].round(4).to_string(index=False))
print("\nPARETO COUNTS")
print(pareto.groupby(["dataset","encoder"]).size().to_string())
print("\nTOP FRONT")
print(top_front[cols].round(4).to_string(index=False))
