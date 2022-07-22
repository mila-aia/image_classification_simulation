#!/bin/bash
#SBATCH --job-name=image_classification_simulation
## set --account=... or --partition=... as needed.
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=5G
#SBATCH --time=4:00:00
#SBATCH --output=logs/%x__%j.out
#SBATCH --error=logs/%x__%j.err
# remove one # if you prefer receiving emails
##SBATCH --mail-type=all
##SBATCH --mail-user=aldo.zaimi@mila.quebec

export MLFLOW_TRACKING_URI='mlruns'
export ORION_DB_ADDRESS='orion_db.pkl'
export ORION_DB_TYPE='pickleddb'

orion -v hunt --config orion_config.yaml \
    main --data $HOME/data/domain_adaptation_images/amazon/images --config config.yaml --start-from-scratch --gpus 0 \
    --output '{exp.working_dir}/{exp.name}_{trial.id}/' --log '{exp.working_dir}/{exp.name}_{trial.id}/exp.log' --tmp-folder ${SLURM_TMPDIR}
