import pandas as pd
from pathlib import Path
base = Path("/raid/experiments/lowgen_v2/output/lowgen-v2-cluster-guided-full-embedding")
files = {
 "cluster_only": base/"cluster_guided_full_embedding_gmm4_full_run/cluster_metric_quality_correlations_with_sd_stability.csv",
 "cluster_label": base/"cluster_guided_full_embedding_gmm4_label_full_run/cluster_metric_quality_correlations.csv",
}
no_label = ["silhouette","sd_index","sd_scat","sd_dis","mean_pairwise_ari","noise_fraction","n_clusters"]
quality = ["fid","kid","precision","recall","tstr_macro_roc_auc_ovr","aug_macro_roc_auc_ovr"]
for name,path in files.items():
    corr=pd.read_csv(path)
    if name=="cluster_only":
        corr=corr[corr["generator"].eq("cluster_guided_full_diag_gmm4_probe")]
    corr=corr[corr["cluster_metric"].isin(no_label)&corr["quality_metric"].isin(quality)].copy()
    corr["abs_spearman"]=corr["spearman"].abs()
    print("===",name,"no-label metrics top ===")
    print(corr.sort_values("abs_spearman",ascending=False).head(30)[["cluster_metric","quality_metric","n","spearman","pearson"]].round(4).to_string(index=False))
    print()
    print("pivot spearman")
    piv=corr.pivot(index="cluster_metric", columns="quality_metric", values="spearman")
    print(piv.round(3).to_string())
    print()
