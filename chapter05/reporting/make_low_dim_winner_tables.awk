BEGIN {
  dsorder = "kather100k spider-colorectal spider-breast spider-skin spider-thorax icaird100k"
  split(dsorder, dslist, " ")
}

FNR == 1 { next }

($4 == 1 || $4 == 2) {
  split_de($2, parts)
  ds = parts[1]
  enc = parts[2]
  dim = $4
  proj = $3
  acc = $8 + 0
  auc = $9 + 0
  accsd = $13 + 0
  aucsd = $14 + 0

  key = dim SUBSEP "Accuracy" SUBSEP ds
  if (!(key in best) || acc > best[key]) {
    best[key] = acc
    bestsd[key] = accsd
    bestenc[key] = enc
    bestproj[key] = proj
  }

  key = dim SUBSEP "ROC-AUC" SUBSEP ds
  if (!(key in best) || auc > best[key]) {
    best[key] = auc
    bestsd[key] = aucsd
    bestenc[key] = enc
    bestproj[key] = proj
  }
}

END {
  print "\\begin{table}[!ht]"
  print "\\centering"
  print "\\small"
  print "\\setlength{\\tabcolsep}{4pt}"
  print "\\begin{tabular}{lllclc}"
  print "\\toprule"
  print "Dim. & Dataset & Best by Accuracy & Accuracy & Best by ROC-AUC & ROC-AUC \\\\"
  print "\\midrule"

  for (dim = 1; dim <= 2; dim++) {
    for (i = 1; i <= 6; i++) {
      ds = dslist[i]
      acc_key = dim SUBSEP "Accuracy" SUBSEP ds
      auc_key = dim SUBSEP "ROC-AUC" SUBSEP ds
      dim_cell = i == 1 ? "\\multirow{6}{*}{" dim "}" : ""
      printf "%s & %s & %s + %s & %.4f $\\pm$ %.4f & %s + %s & %.4f $\\pm$ %.4f \\\\\n", \
        dim_cell, label_ds(ds), \
        label_enc(bestenc[acc_key]), label_proj(bestproj[acc_key]), best[acc_key], bestsd[acc_key], \
        label_enc(bestenc[auc_key]), label_proj(bestproj[auc_key]), best[auc_key], bestsd[auc_key]
    }
    if (dim == 1) {
      print "\\midrule"
    }
  }

  print "\\bottomrule"
  print "\\end{tabular}"
  print "\\vspace{2pt}"
  print "\\caption{Best encoder--projector combinations at one and two dimensions in the curve-stage logistic-regression experiment. Values are mean $\\pm$ SD over curve seeds. Best combinations are selected separately for Accuracy and ROC-AUC.}"
  print "\\label{tab:low-dim-winners}"
  print "\\end{table}"
}

function split_de(de, arr) {
  if (de ~ /_dinov2$/) {
    arr[1] = substr(de, 1, length(de) - 7)
    arr[2] = "dinov2"
  } else if (de ~ /_hibou$/) {
    arr[1] = substr(de, 1, length(de) - 6)
    arr[2] = "hibou"
  } else if (de ~ /_uni2$/) {
    arr[1] = substr(de, 1, length(de) - 5)
    arr[2] = "uni2"
  } else if (de ~ /_uni$/) {
    arr[1] = substr(de, 1, length(de) - 4)
    arr[2] = "uni"
  }
}

function label_ds(ds) {
  if (ds == "kather100k") return "Kather"
  if (ds == "spider-colorectal") return "CRC"
  if (ds == "spider-breast") return "Breast"
  if (ds == "spider-skin") return "Skin"
  if (ds == "spider-thorax") return "Thorax"
  if (ds == "icaird100k") return "iCAIRD"
  return ds
}

function label_enc(enc) {
  if (enc == "uni") return "UNI"
  if (enc == "uni2") return "UNI2"
  if (enc == "hibou") return "Hibou"
  if (enc == "dinov2") return "DINOv2"
  return enc
}

function label_proj(proj) {
  if (proj == "pca") return "PCA"
  if (proj == "umap") return "UMAP"
  if (proj == "grp") return "GRP"
  return proj
}
