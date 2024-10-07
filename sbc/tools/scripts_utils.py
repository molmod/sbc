###########################################################################################
# Training utils
# Authors: David Kovacs, Ilyes Batatia, Sander Vandenhaute
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

import dataclasses
import logging
from typing import Dict, List, Optional, Tuple

import torch
from prettytable import PrettyTable

#from mace import data
#from mace.data import AtomicData
import mace.data
from mace.tools import AtomicNumberTable, torch_geometric

from sbc import data
from sbc.data import AtomicData
from sbc.tools.utils import PhaseTable


@dataclasses.dataclass
class SubsetCollection:
    train: data.Configurations
    valid: data.Configurations
    tests: List[Tuple[str, data.Configurations]]


def get_dataset_from_xyz(
    train_path: str,
    valid_path: str,
    valid_fraction: float,
    config_type_weights: Dict,
    test_path: str = None,
    seed: int = 1234,
    energy_key: str = "energy",
    forces_key: str = "forces",
    stress_key: str = "stress",
    virials_key: str = "virials",
    dipole_key: str = "dipoles",
    charges_key: str = "charges",
) -> Tuple[SubsetCollection, Optional[Dict[int, float]]]:
    """Load training and test dataset from xyz file"""
    atomic_energies_dict, all_train_configs = data.load_from_xyz(
        file_path=train_path,
        config_type_weights=config_type_weights,
        energy_key=energy_key,
        forces_key=forces_key,
        stress_key=stress_key,
        virials_key=virials_key,
        dipole_key=dipole_key,
        charges_key=charges_key,
        extract_atomic_energies=True,
    )
    logging.info(
        f"Loaded {len(all_train_configs)} training configurations from '{train_path}'"
    )
    if valid_path is not None:
        _, valid_configs = data.load_from_xyz(
            file_path=valid_path,
            config_type_weights=config_type_weights,
            energy_key=energy_key,
            forces_key=forces_key,
            stress_key=stress_key,
            virials_key=virials_key,
            dipole_key=dipole_key,
            charges_key=charges_key,
            extract_atomic_energies=False,
        )
        logging.info(
            f"Loaded {len(valid_configs)} validation configurations from '{valid_path}'"
        )
        train_configs = all_train_configs
    else:
        logging.info(
            "Using random %s%% of training set for validation", 100 * valid_fraction
        )
        train_configs, valid_configs = mace.data.random_train_valid_split(
            all_train_configs, valid_fraction, seed
        )

    test_configs = []
    if test_path is not None:
        _, all_test_configs = data.load_from_xyz(
            file_path=test_path,
            config_type_weights=config_type_weights,
            energy_key=energy_key,
            forces_key=forces_key,
            dipole_key=dipole_key,
            charges_key=charges_key,
            extract_atomic_energies=False,
        )
        # create list of tuples (config_type, list(Atoms))
        test_configs = mace.data.test_config_types(all_test_configs)
        logging.info(
            f"Loaded {len(all_test_configs)} test configurations from '{test_path}'"
        )
    return (
        SubsetCollection(train=train_configs, valid=valid_configs, tests=test_configs),
        atomic_energies_dict,
    )


class LRScheduler:
    def __init__(self, optimizer, args) -> None:
        self.scheduler = args.scheduler
        if args.scheduler == "ExponentialLR":
            self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer=optimizer, gamma=args.lr_scheduler_gamma
            )
        elif args.scheduler == "ReduceLROnPlateau":
            self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer,
                factor=args.lr_factor,
                patience=args.scheduler_patience,
            )
        else:
            raise RuntimeError(f"Unknown scheduler: '{args.scheduler}'")

    def step(self, metrics=None, epoch=None):  # pylint: disable=E1123
        if self.scheduler == "ExponentialLR":
            self.lr_scheduler.step(epoch=epoch)
        elif self.scheduler == "ReduceLROnPlateau":
            self.lr_scheduler.step(metrics=metrics, epoch=epoch)

    def __getattr__(self, name):
        if name == "step":
            return self.step
        return getattr(self.lr_scheduler, name)


def create_error_table(
    table_type: str,
    all_collections: list,
    z_table: AtomicNumberTable,
    p_table: PhaseTable,
    r_max: float,
    valid_batch_size: int,
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    output_args: Dict[str, bool],
    log_wandb: bool,
    device: str,
) -> PrettyTable:
    if log_wandb:
        import wandb
    table = PrettyTable()
    #table.field_names = [
    #        'true_positive',
    for name, subset in all_collections:
        total_loss = 0
        logging.info(f"Evaluating {name} ...")
        data_loader = torch_geometric.dataloader.DataLoader(
            dataset=[
                AtomicData.from_config(config, z_table=z_table, p_table=p_table, cutoff=r_max)
                for config in subset
            ],
            batch_size=valid_batch_size,
            shuffle=False,
            drop_last=False,
        )
        for batch in data_loader:
            batch = batch.to(device)
            batch_dict = batch.to_dict()
            output = model(
                batch_dict,
                training=False,
                compute_force=output_args["forces"],
                compute_virials=output_args["virials"],
                compute_stress=output_args["stress"],
            )
            for key, value in output.items():
                if value is not None:
                    output[key] = value.detach()
            #batch = batch.cpu()
            #output['logits'] = output['logits'].cpu()
            loss = loss_fn(pred=output, ref=batch)
            total_loss += loss.detach().cpu().item()
        logging.info('loss: {}'.format(total_loss / len(data_loader)))

        #_, metrics = evaluate(
        #    model,
        #    loss_fn=loss_fn,
        #    data_loader=data_loader,
        #    output_args=output_args,
        #    device=device,
        #)
        #table.add_row(name, 
    return table

