import pandas as pd
from pathlib import Path
paths = {
    "cluster_only": Path("/raid/experiments/lowgen_v2/output/lowgen-v2-cluster-guided-full-embedding/cluster_guided_full_embedding_gmm4_full_run/cluster_guided_full_embedding_results_with_sd_stability.csv"),
    "cluster_label": Path("/raid/experiments/lowgen_v2/output/lowgen-v2-cluster-guided-full-embedding/cluster_guided_full_embedding_gmm4_label_full_run/cluster_guided_full_embedding_results.csv"),
    "label_only": Path("/raid/experiments/lowgen_v2/output/lowgen-v2-cluster-guided-full-embedding/label_guided_full_embedding_gmm4_baseline/cluster_guided_full_embedding_results.csv"),
}
metrics = ["fid", "kid", "precision", "recall", "tstr_macro_roc_auc_ovr", "aug_macro_roc_auc_ovr"]
for name, path in paths.items():
    df = pd.read_csv(path)
    print(f"=== {name} ===")
    print("shape", df.shape)
    print("status", df["status"].value_counts(dropna=False).to_dict())
    print("generator", df["generator"].value_counts(dropna=False).to_dict())
    print("conditioning", df["conditioning"].value_counts(dropna=False).to_dict())
    print("datasets", df["dataset"].value_counts(dropna=False).to_dict())
    available = [m for m in metrics if m in df.columns]
    print("nonnull", {m:int(df[m].notna().sum()) for m in available})
    by = df.groupby("dataset")[available].agg(["mean", "median", "std"])
    print(by.round(4).to_string())
    print()

cl = pd.read_csv(paths["cluster_label"])
lo = pd.read_csv(paths["label_only"])
keys = ["dataset", "encoder", "projector", "dimension", "target_name", "projection_seed", "cluster_seed", "clusterer", "config_id"]
common = [k for k in keys if k in cl.columns and k in lo.columns]
merged = cl.merge(lo, on=common, suffixes=("_cluster_label", "_label_only"))
print("=== cluster_label minus label_only paired deltas ===")
print("paired rows", len(merged))
for dataset, g in merged.groupby("dataset"):
    print("dataset", dataset)
    for m, direction in [("fid", "lower"), ("kid", "lower"), ("precision", "higher"), ("recall", "higher"), ("tstr_macro_roc_auc_ovr", "higher"), ("aug_macro_roc_auc_ovr", "higher")]:
        a = f"{m}_cluster_label"
        b = f"{m}_label_only"
        if a in g and b in g:
            d = g[a] - g[b]
            better = (d < 0).mean() if direction == "lower" else (d > 0).mean()
            print(f"  {m}: mean_delta={d.mean():.5g} median_delta={d.median():.5g} cluster_label_better_frac={better:.3f}")

print("=== top utility configs cluster_label ===")
for dataset, g in cl.groupby("dataset"):
    print("dataset", dataset)
    cols = [c for c in ["projector","dimension","clusterer","config_id","n_clusters","silhouette","sd_index","mean_pairwise_ari","fid","kid","precision","recall","tstr_macro_roc_auc_ovr","aug_macro_roc_auc_ovr"] if c in g]
    for metric in ["tstr_macro_roc_auc_ovr", "aug_macro_roc_auc_ovr", "precision", "recall"]:
        if metric in g:
            print(" best_by", metric)
            print(g.sort_values(metric, ascending=False).head(3)[cols].round(4).to_string(index=False))
