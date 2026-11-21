from pathlib import Path
import numpy as np
import pandas as pd
base=Path("/raid/experiments/lowgen_v2/output")
front=base/"pareto_then_roc_no_label_selection_stability_filtered/pareto_front.csv"
df=pd.read_csv(front)
objectives=["roc_auc_proxy","fid_proxy","kid_proxy","precision_proxy","recall_proxy","stability_score","support_score"]
# Equal-weight closest-to-ideal: minimise Euclidean distance to (1,...,1)
# Max-min: maximise the weakest objective.
df["utopia_distance"] = np.sqrt(((1.0-df[objectives])**2).mean(axis=1))
df["balanced_score"] = 1.0 - df["utopia_distance"]
df["maximin_score"] = df[objectives].min(axis=1)
# A pragmatic weighted middle: ROC/FID/precision/recall are primary, KID/stability/support secondary.
weights={"roc_auc_proxy":0.20,"fid_proxy":0.20,"kid_proxy":0.10,"precision_proxy":0.15,"recall_proxy":0.15,"stability_score":0.10,"support_score":0.10}
df["weighted_middle_score"] = sum(df[k]*v for k,v in weights.items())
cols=["dataset","encoder","projector","dimension","target_name","clusterer","cluster_params_json","mean_silhouette","mean_noise_fraction","mean_n_clusters","mean_pairwise_ari","effective_support","roc_auc_proxy","fid_proxy","kid_proxy","precision_proxy","recall_proxy","stability_score","support_score","balanced_score","maximin_score","weighted_middle_score"]
for method, sort_col, asc in [("closest_utopia","balanced_score",False),("maximin","maximin_score",False),("weighted_middle","weighted_middle_score",False)]:
    best=(df.sort_values(["dataset","encoder",sort_col,"balanced_score"], ascending=[True,True,asc,False]).groupby(["dataset","encoder"],as_index=False).head(1))
    print("\n===",method,"===")
    print(best[cols].round(4).to_string(index=False))
print("\n=== top weighted middle ===")
top=(df.sort_values(["dataset","encoder","weighted_middle_score"], ascending=[True,True,False]).groupby(["dataset","encoder"],as_index=False).head(8))
print(top[cols].round(4).to_string(index=False))
