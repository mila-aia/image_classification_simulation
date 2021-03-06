#!/bin/bash
## set --account=... or --partition=... as needed.
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=5G
#SBATCH --time=0:05:00
#SBATCH --job-name=image_classification_simulation
#SBATCH --output=logs/%x__%j.out
#SBATCH --error=logs/%x__%j.err
# to attach a tag to your run (e.g., used to track the GPU time)
# uncomment the following line and add replace `my_tag` with the proper tag:
##SBATCH --wckey=my_tag
# remove one # if you prefer receiving emails
##SBATCH --mail-type=all
##SBATCH --mail-user=aldo.zaimi@mila.quebec

export MLFLOW_TRACKING_URI='mlruns'

main --data ../data --output output --config config.yaml --tmp-folder ${SLURM_TMPDIR} --disable-progressbar
