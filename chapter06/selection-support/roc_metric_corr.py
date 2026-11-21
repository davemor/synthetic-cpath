import pandas as pd
from pathlib import Path
base=Path("/raid/experiments/lowgen_v2/output/lowgen-v2-cluster-guided-full-embedding")
paths={
 "cluster_label": base/"cluster_guided_full_embedding_gmm4_label_full_run/cluster_metric_quality_correlations.csv",
 "label_only": base/"label_guided_full_embedding_gmm4_baseline/cluster_metric_quality_correlations.csv",
}
roc=["tstr_macro_roc_auc_ovr","aug_macro_roc_auc_ovr"]
label_metrics=["label_ari","label_ami","label_nmi","label_homogeneity","label_completeness","label_v_measure","label_purity"]
no_label=["silhouette","sd_index","sd_scat","sd_dis","mean_pairwise_ari","noise_fraction","n_clusters"]
for name,path in paths.items():
    df=pd.read_csv(path)
    df=df[df.quality_metric.isin(roc)].copy()
    print("===",name,"label-aware ROC correlations ===")
    x=df[df.cluster_metric.isin(label_metrics)].sort_values(["quality_metric","spearman"],ascending=[True,False])
    print(x[["cluster_metric","quality_metric","n","spearman","pearson"]].round(4).to_string(index=False))
    print("===",name,"no-label ROC correlations ===")
    x=df[df.cluster_metric.isin(no_label)].sort_values(["quality_metric","spearman"],ascending=[True,False])
    print(x[["cluster_metric","quality_metric","n","spearman","pearson"]].round(4).to_string(index=False))
