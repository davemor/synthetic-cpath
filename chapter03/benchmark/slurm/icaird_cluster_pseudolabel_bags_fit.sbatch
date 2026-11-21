#!/bin/bash
#SBATCH --job-name=icaird_cluster_bag_fit
#SBATCH --partition=large-short
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/sharedscratch/dm236/synthetic_bags/icaird_cervical_conditional_edm_mlp_kmeans24_pca42_clusters_a100_sharded/slurm/bag_fit_%j.out
#SBATCH --error=/sharedscratch/dm236/synthetic_bags/icaird_cervical_conditional_edm_mlp_kmeans24_pca42_clusters_a100_sharded/slurm/bag_fit_%j.err

set -euo pipefail

cd /home/dm236/repos/icaird-cervical-benchmark

uv run icaird-cervical-benchmark synthetic-bags-fit \
  --config configs/icaird_cervical_cluster_pseudolabel_bags_conditional_edm_a100_sharded.yml
