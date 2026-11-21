import pandas as pd
from pathlib import Path
base = Path("/raid/experiments/lowgen_v2/output/lowgen-v2-cluster-guided-full-embedding")
paths = {
    "cluster_only": base/"cluster_guided_full_embedding_gmm4_full_run/cluster_guided_full_embedding_results_with_sd_stability.csv",
    "cluster_label": base/"cluster_guided_full_embedding_gmm4_label_full_run/cluster_guided_full_embedding_results.csv",
    "label_only": base/"label_guided_full_embedding_gmm4_baseline/cluster_guided_full_embedding_results.csv",
}
metrics = ["fid", "kid", "precision", "recall", "tstr_macro_roc_auc_ovr", "aug_macro_roc_auc_ovr"]
frames=[]
for name,path in paths.items():
    df=pd.read_csv(path)
    if name=="cluster_only":
        df=df[df["generator"].eq("cluster_guided_full_diag_gmm4_probe")].copy()
    df["branch"]=name
    frames.append(df)
all_df=pd.concat(frames, ignore_index=True, sort=False)
print("=== branch dataset means ===")
print(all_df.groupby(["branch","dataset"])[metrics].agg(["mean","median","std"]).round(5).to_string())
print()
print("=== branch dataset means compact ===")
print(all_df.groupby(["branch","dataset"])[metrics].mean().round(5).to_string())
print()
cl=pd.read_csv(paths["cluster_label"])
lo=pd.read_csv(paths["label_only"])
if "row_number" in cl.columns and "row_number" in lo.columns:
    common=["row_number"]
else:
    common=[c for c in ["dataset","encoder","projector","dimension","target_name","projection_seed","cluster_seed","clusterer","config_id","labels_path"] if c in cl and c in lo]
merged=cl.merge(lo,on=common,suffixes=("_cluster_label","_label_only"))
print("=== paired deltas using",common,"===")
print("paired rows",len(merged))
for dataset_col in ["dataset_cluster_label", "dataset"]:
    if dataset_col in merged:
        dataset_key=dataset_col
        break
for dataset,g in merged.groupby(dataset_key):
    print("dataset",dataset)
    for m,direction in [("fid","lower"),("kid","lower"),("precision","higher"),("recall","higher"),("tstr_macro_roc_auc_ovr","higher"),("aug_macro_roc_auc_ovr","higher")]:
        a=f"{m}_cluster_label"; b=f"{m}_label_only"
        delta=g[a]-g[b]
        better=(delta<0).mean() if direction=="lower" else (delta>0).mean()
        print(f"  {m}: mean_delta={delta.mean():.6f} median_delta={delta.median():.6f} better_frac={better:.3f}")
print()
for name, corr_path in {
    "cluster_only": base/"cluster_guided_full_embedding_gmm4_full_run/cluster_metric_quality_correlations_with_sd_stability.csv",
    "cluster_label": base/"cluster_guided_full_embedding_gmm4_label_full_run/cluster_metric_quality_correlations.csv",
    "label_only": base/"label_guided_full_embedding_gmm4_baseline/cluster_metric_quality_correlations.csv",
}.items():
    corr=pd.read_csv(corr_path)
    if name=="cluster_only" and "generator" in corr:
        corr=corr[corr["generator"].eq("cluster_guided_full_diag_gmm4_probe")]
    corr=corr[corr["n"].ge(30)].copy()
    corr["abs_spearman"]=corr["spearman"].abs()
    print(f"=== top spearman correlations {name} ===")
    qmetrics=["fid","kid","precision","recall","tstr_macro_roc_auc_ovr","aug_macro_roc_auc_ovr"]
    sub=corr[corr["quality_metric"].isin(qmetrics)]
    print(sub.sort_values("abs_spearman",ascending=False).head(20)[["generator","cluster_metric","quality_metric","n","spearman","pearson"]].round(4).to_string(index=False))
    print()
