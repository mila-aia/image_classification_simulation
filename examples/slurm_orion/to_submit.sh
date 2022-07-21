#!/bin/bash
#SBATCH --job-name=image_classification_simulation
## set --account=... or --partition=... as needed.
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=5G
#SBATCH --time=0:05:00
#SBATCH --output=logs/%x__%j.out
#SBATCH --error=logs/%x__%j.err
# remove one # if you prefer receiving emails
##SBATCH --mail-type=all
##SBATCH --mail-user=aldo.zaimi@mila.quebec

export MLFLOW_TRACKING_URI='mlruns'
export ORION_DB_ADDRESS='orion_db.pkl'
export ORION_DB_TYPE='pickleddb'

orion -v hunt --config /home/mila/a/aldo.zaimi/demos_code/image_classification_simulation/examples/slurm_orion/orion_config.yaml \
    main --data /home/mila/a/aldo.zaimi/data/domain_adaptation_images/amazon/images --config /home/mila/a/aldo.zaimi/demos_code/image_classification_simulation/examples/conv_ae/config.yaml --disable-progressbar \
    --output '{exp.working_dir}/{exp.name}_{trial.id}/' \
    --log '{exp.working_dir}/{exp.name}_{trial.id}/exp.log' \
    --tmp-folder ${SLURM_TMPDIR}
