###########################################################################################
# Training script for MACE
# Authors: Ilyes Batatia, Gregor Simm, David Kovacs
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

import ast
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch.nn.functional
from e3nn import o3
from torch.optim.swa_utils import SWALR, AveragedModel
from torch_ema import ExponentialMovingAverage

import mace
from mace import data, modules, tools
from mace.tools import torch_geometric
from mace.tools.scripts_utils import (
    LRScheduler,
    #nreate_error_table,
    #get_dataset_from_xyz,
)
#from transfermace.modules import TransferMACE, extract_kwargs
#from transfermace.tools.arg_parser import build_default_arg_parser

from phace.modules import ClassifierMACE, MultilevelMACE, extract_kwargs
from phace.modules.loss import ClassificationLoss
import phace.tools
from phace.tools.scripts_utils import get_dataset_from_xyz, create_error_table
from phace.tools.utils import PhaseTable
from phace.tools.arg_parser import build_default_arg_parser
from phace.data import FeaturizedData


def main() -> None:
    args = build_default_arg_parser().parse_args()
    tag = tools.get_tag(name=args.name, seed=args.seed)

    # Setup
    tools.set_seeds(args.seed)
    tools.setup_logger(level=args.log_level, tag=tag, directory=args.log_dir)
    try:
        logging.info(f"MACE version: {mace.__version__}")
    except AttributeError:
        logging.info("Cannot find MACE version, please install MACE via pip")
    logging.info(f"Configuration: {args}")
    device = tools.init_device(args.device)
    tools.set_default_dtype(args.default_dtype)

    config_type_weights = {"Default": 1.0}

    # load model and get extra kwargs
    base_model = torch.load(args.base_model, map_location='cpu')
    model_kwargs = extract_kwargs(base_model)

    # Data preparation
    collections, atomic_energies_dict = get_dataset_from_xyz(
        train_path=args.train_file,
        valid_path=args.valid_file,
        valid_fraction=args.valid_fraction,
        config_type_weights=config_type_weights,
        test_path=args.test_file,
        seed=args.seed,
        energy_key=args.energy_key,
        forces_key=args.forces_key,
        stress_key=args.stress_key,
        virials_key=args.virials_key,
        dipole_key=args.dipole_key,
        charges_key=args.charges_key,
    )

    logging.info(
        f"Total number of configurations: train={len(collections.train)}, valid={len(collections.valid)}, "
        f"tests=[{', '.join([name + ': ' + str(len(test_configs)) for name, test_configs in collections.tests])}]"
    )

    # Atomic number table
    # yapf: disable
    z_table = tools.AtomicNumberTable([int(n) for n in base_model.atomic_numbers])
    #z_table = tools.get_atomic_number_table_from_zs(
    #    z
    #    for configs in (collections.train, collections.valid)
    #    for config in configs
    #    for z in config.atomic_numbers
    #)
    p_table = PhaseTable.from_phases(
        config.phase
        for configs in (collections.train, collections.valid)
        for config in configs
    )
    # yapf: enable
    logging.info(z_table)
    logging.info(p_table)
    if args.phase_weights is None:
        phase_counts_dict = {p: 0 for p in p_table.phases}
        for configuration in collections.train:
            phase_counts_dict[configuration.phase] += 1
        logging.info('found the following phases and counts: {}'.format(phase_counts_dict))
        total = sum([count for count in phase_counts_dict.values()])
        phase_weights = {phase: total / count for phase, count in phase_counts_dict.items()}
    else:
        phase_weights = ast.literal_eval(args.phase_weights)

    if atomic_energies_dict is None or len(atomic_energies_dict) == 0:
        if args.E0s is not None:
            logging.info(
                "Atomic Energies not in training file, using command line argument E0s"
            )
            if args.E0s.lower() == "average":
                logging.info(
                    "Computing average Atomic Energies using least squares regression"
                )
                atomic_energies_dict = data.compute_average_E0s(
                    collections.train, z_table
                )
            else:
                try:
                    atomic_energies_dict = ast.literal_eval(args.E0s)
                    assert isinstance(atomic_energies_dict, dict)
                except Exception as e:
                    raise RuntimeError(
                        f"E0s specified invalidly, error {e} occured"
                    ) from e
        else:
            raise RuntimeError(
                "E0s not found in training file and not specified in command line"
            )
    atomic_energies: np.ndarray = np.array(
        [atomic_energies_dict[z] for z in z_table.zs]
    )
    logging.info(f"Atomic energies: {atomic_energies.tolist()}")


    # Build model
    logging.info("Building model")
    classifier_readout = ast.literal_eval(args.classifier_readout)
    classifier_mixing = args.classifier_mixing
    if classifier_mixing == 0:
        classifier_mixing = None
    model = ClassifierMACE.from_model(
            base_model,
            phases=p_table.phases,
            classifier=args.classifier, 
            classifier_readout=classifier_readout,
            classifier_mixing=classifier_mixing,
            )
    model.to(device)

    loss_fn = ClassificationLoss(
            weight=[phase_weights[p] for p in p_table.phases],
            label_smoothing=args.label_smoothing,
            uncertainty_weight=args.uncertainty_weight,
            )
    loss_fn.to(device)
    logging.info(loss_fn)

    param_options = dict(
        params=[
            {
                "name": "classifier",
                "params": model.classifier.parameters(),
                "weight_decay": args.weight_decay,
            },
        ],
        lr=args.lr,
        amsgrad=args.amsgrad,
    )

    optimizer: torch.optim.Optimizer
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(**param_options)
    else:
        optimizer = torch.optim.Adam(**param_options)

    lr_scheduler = LRScheduler(optimizer, args)

    start_epoch = 0

    logging.info(model)
    logging.info(f"Number of parameters: {tools.count_parameters(model)}")
    logging.info(f"Optimizer: {optimizer}")

    # evaluate node_feats over dataset and spit out FeaturizedData
    logging.info('Parsing train and valid datasets into featurized versions')
    train_loader = torch_geometric.dataloader.DataLoader(
        dataset=[
            FeaturizedData.from_config(config, z_table=z_table, p_table=p_table, cutoff=model_kwargs['r_max'], model=model)
            for config in collections.train
        ],
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )
    valid_loader = torch_geometric.dataloader.DataLoader(
        dataset=[
            FeaturizedData.from_config(config, z_table=z_table, p_table=p_table, cutoff=model_kwargs['r_max'], model=model)
            for config in collections.valid
        ],
        batch_size=args.valid_batch_size,
        shuffle=False,
        drop_last=False,
    )

    phace.tools.train(
        model=model,
        loss_fn=loss_fn,
        train_loader=train_loader,
        valid_loader=valid_loader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        eval_interval=args.eval_interval,
        max_num_epochs=args.max_num_epochs,
        patience=args.patience,
        device=device,
        max_grad_norm=args.clip_grad,
    )

    model = model.to("cpu")
    torch.save(model, 'classifier.pth')
    logging.info("Done")


if __name__ == "__main__":
    main()
