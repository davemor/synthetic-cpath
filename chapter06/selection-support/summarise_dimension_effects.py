import pandas as pd
from pathlib import Path
p=Path("/raid/experiments/lowgen_v2/output/pareto_then_roc_no_label_selection/scored_candidates.csv")
df=pd.read_csv(p)
df=df[df.encoder.eq("uni2") & df.dataset.isin(["icaird100k","kather100k"])].copy()
df["dim_group"]="other"
df.loc[df.projector.eq("raw"),"dim_group"]="raw/full"
df.loc[df.dimension.isin([1,2,4,8]),"dim_group"]="very_low_1_8"
df.loc[df.dimension.isin([16,32,64]),"dim_group"]="mid_16_64"
df.loc[df.dimension.ge(128),"dim_group"]="high_128_plus"
metrics=["mean_silhouette","mean_noise_fraction","mean_n_clusters","mean_pairwise_ari","effective_support","roc_auc_proxy","fid_proxy","kid_proxy","precision_proxy","recall_proxy","stability_score","support_score"]
print("=== grouped means ===")
print(df.groupby(["dataset","dim_group"])[metrics].mean().round(4).to_string())
print("\n=== by projector/dim means selected metrics ===")
cols=["mean_silhouette","mean_pairwise_ari","roc_auc_proxy","fid_proxy","precision_proxy","recall_proxy"]
for dataset,g in df.groupby("dataset"):
    print("\n",dataset)
    out=g.groupby(["projector","dimension"], dropna=False)[cols].mean().reset_index()
    print(out.sort_values(["projector","dimension"], na_position="last").round(4).to_string(index=False))
