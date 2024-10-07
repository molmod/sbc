# Rare Event Sampling using Smooth Basin Classification ([arXiv:2404.03777](https://arxiv.org/abs/2404.03777))

This repository extends MACE with a classification layer in order to learn *rigorously invariant* collective variables for reactive and nonreactive rare events.

![model](https://github.com/user-attachments/assets/403a75fb-849f-42b6-9f95-44c161278766)

# Usage

SBC works by augmenting existing MACE models (e.g. MACE-MP) with a phase readout layer whose weights are optimized using a custom cross-entropy loss function. Its training requires an XYZ dataset of atomic geometries which are annotated with a global `phase` label; no energies or forces are required. Compared to vanilla MACE, the train script implements the following additional keyword arguments:

```sh
python scripts/run_train_classifier.py \
    --name="MACE_model" \
    --train_file="combined.xyz" \
    --valid_fraction=0.1 \
    --config_type_weights='{"Default":1.0}' \
    --E0s='average' \
    --batch_size=512 \
    --valid_batch_size=64 \
    --max_num_epochs=10000 \
    --amsgrad \
    --restart_latest \
    --device="cuda" \
    --save_cpu \
    --lr=0.01 \
    --patience=500 \
    --scheduler_patience=50 \
    --default_dtype="float32" \
    --seed=2 \
    --loss='cross_entropy' \
    --energy_weight=1.0 \
    --error_table='cross_entropy' \
    --label_smoothing=0.0 \
    --scaling="no_scaling" \
    --base_model="universal_mace.pth" \
    --classifier='EnergyBasedClassifier' \
    --classifier_readout='[16, 8]' \
    --classifier_mixing=0 \

```
