#!/bin/bash
#SBATCH --job-name=pii_ner_train
#SBATCH --partition=rtx3090
#SBATCH --gres=gpu:rtx_3090:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --account=robertmo
#SBATCH --output=outputs/train_%j.log
#SBATCH --error=outputs/train_%j.err

cd ~/pii_detection
echo "Python: $(which python)"
echo "Pip: $(which pip)"
python -c "import sys; print('sys.path:', sys.path)"
pip install --user packaging protobuf transformers datasets accelerate sentencepiece seqeval
python train_ner.py
