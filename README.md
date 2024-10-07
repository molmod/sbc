# Rare Event Sampling using Smooth Basin Classification ([arXiv:2404.03777](https://arxiv.org/abs/2404.03777))

This repository extends MACE with a classification layer in order to learn *rigorously invariant* collective variables for reactive and nonreactive rare events.

![model](https://github.com/user-attachments/assets/403a75fb-849f-42b6-9f95-44c161278766)

# Usage

SBC works by augmenting existing MACE models (e.g. MACE-MP) with a phase readout layer whose weights are optimized using a custom cross-entropy loss function. Its training requires an XYZ dataset of atomic geometries which are annotated with a global `phase` label; no energies or forces are required. Compared to vanilla MACE, the train script implements the following additional keyword arguments:

- `--base_model`: the existing MACE model which is to be augmented with a phase readout layer
- `--classifier`: classifier architecture; for the moment, this should be set to `EnergyBasedClassifier`
- `--classifier_readout`: layer sizes for the classifier readout. `[16, 8]` means that it will create two hidden layers with respective sizes 16 and 8 to convert the (scalar part of the) node embeddings into per-phase log probabilities.
- `--classifier_mixing`: determines whether to mix features of different interaction layers in the readout.
- `--uncertainty_weight`: weight of the regularization term in Equation 12 of the preprint. We found that a value of 1.0 works well for a variety of systems.

See below for an example:
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
    --uncertainty_weight=1.0

```

# Setup
A Python environment with `torch`, and `mace` v0.3.0
