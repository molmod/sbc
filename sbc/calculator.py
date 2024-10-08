###########################################################################################
# Atomic Data Class for handling molecules as graphs
# Authors: Sander Vandenhaute
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################


from typing import Optional
import torch
import numpy as np

from ase.units import GPa
from ase.stress import full_3x3_to_voigt_6_stress
from ase.calculators.calculator import Calculator, all_changes

import mace.data
from mace.tools import torch_geometric, torch_tools, utils

from sbc import data
from sbc.tools.utils import PhaseTable


class MACECalculator(Calculator):
    """MACE ASE Calculator"""

    implemented_properties = ["energy", "free_energy", "forces", "stress"]

    def __init__(
        self,
        model_path: str,
        device: str,
        energy_units_to_eV: float = 1.0,
        length_units_to_A: float = 1.0,
        default_dtype="float64",
        **kwargs
    ):
        Calculator.__init__(self, **kwargs)
        self.results = {}

        torch_tools.set_default_dtype(default_dtype)
        self.device = torch_tools.init_device(device)
        self.energy_units_to_eV = energy_units_to_eV
        self.length_units_to_A = length_units_to_A
        self.model = torch.load(f=model_path, map_location=device)
        if default_dtype == 'float64':
            self.model = self.model.double()
        self.model.to(device)
        self.r_max = float(self.model.r_max)
        self.z_table = utils.AtomicNumberTable(
            [int(z) for z in self.model.atomic_numbers]
        )

    def as_data(self, atoms):
        config = data.config_from_atoms(atoms)
        data_loader = torch_geometric.dataloader.DataLoader(
            dataset=[
                data.AtomicData.from_config(
                    config, z_table=self.z_table, p_table=self.p_table, cutoff=self.r_max
                )
            ],
            batch_size=1,
            shuffle=False,
            drop_last=False,
        )
        batch = next(iter(data_loader))
        batch.to(self.device)
        return batch


    # pylint: disable=dangerous-default-value
    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        """
        Calculate properties.
        :param atoms: ase.Atoms object
        :param properties: [str], properties to be computed, used by ASE internally
        :param system_changes: [str], system changes since last calculation, used by ASE internally
        :return:
        """
        # call to base-class to set atoms attribute
        Calculator.calculate(self, atoms)

        # prepare data
        config = mace.data.config_from_atoms(atoms)
        data_loader = torch_geometric.dataloader.DataLoader(
            dataset=[
                mace.data.AtomicData.from_config(
                    config, z_table=self.z_table, cutoff=self.r_max
                )
            ],
            batch_size=1,
            shuffle=False,
            drop_last=False,
        )
        batch = next(iter(data_loader)).to(self.device)

        # predict + extract data
        out = self.model(batch.to_dict(), compute_stress=True)
        energy = out["energy"].detach().cpu().item()
        forces = out["forces"].detach().cpu().numpy()

        # store results
        E = energy * self.energy_units_to_eV
        self.results = {
            "energy": E,
            "free_energy": E,
            # force has units eng / len:
            "forces": forces * (self.energy_units_to_eV / self.length_units_to_A),
        }

        # even though compute_stress is True, stress can be none if pbc is False
        # not sure if correct ASE thing is to have no dict key, or dict key with value None
        if out["stress"] is not None:
            stress = out["stress"].detach().cpu().numpy()
            # stress has units eng / len^3:
            self.results["stress"] = (
                stress * (self.energy_units_to_eV / self.length_units_to_A**3)
            )[0]
            self.results["stress"] = full_3x3_to_voigt_6_stress(self.results["stress"])


def clip_forces(forces, max_force):
    for i, force in enumerate(forces):
        if np.linalg.norm(force) > max_force:
            forces[i, :] = forces[i] * max_force / np.linalg.norm(force)


def clip_stress(stress, max_stress):
    if not (np.any(np.isnan(stress)) or np.any(np.isinf(stress))):
        max_component = np.max(np.abs(stress))
        if max_component > max_stress:
            stress[:] = stress * max_stress / max_component


def logcosh(x):  # numerically stable
    s = np.sign(x) * x
    p = np.exp(-2 * s)
    return s + np.log(1 + p) - np.log(2)


def compute_bias(x: float, function: str, logcosh_scale: float = 1) -> tuple[float, float]:
    if function == 'quadratic':
        value = x ** 2
        grad = 2 * x
    elif function == 'logcosh': # scaled; logcosh > quadratic iff x in [-1, 1]
        if logcosh_scale != 1:
            raise NotImplementedError
        value = logcosh(x) / np.log(np.cosh(1))
        grad = np.tanh(x) / np.log(np.cosh(1))
    else:
        raise ValueError('unknown function {}'.format(function))
    return value, grad


class ClassifierMACECalculator(MACECalculator):
    implemented_properties = [
        'free_energy',
        'energy',
        'forces',
        'stress',
        'CV',
        'bias_energy',
        'bias_forces',
        'bias_stress',
        ]

    def __init__(
        self,
        bias_scale: float = 1,
        bias_center: Optional[float] = None,
        bias_function: str = 'quadratic',
        max_force: float = 5,
        max_stress: float = 1 * GPa,
        **kwargs,
        ):
        super().__init__(**kwargs)
        self.p_table = PhaseTable(self.model.classifier.phases)
        self.bias_scale = bias_scale
        self.bias_center = bias_center
        self.bias_function = bias_function
        self.max_force = max_force
        self.max_stress = max_stress

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        """
        When center is not None, it returns the bias energy, forces, and stress under the corresponding keys.
        This bias is added to the total energy, forces, stress in the system

        When center is None, it returns the CV with keyword 'bias_energy', and negative gradients of the CV
        with respect to atomic coordinates and box vectors with keywords 'bias_forces' and 'bias_stress'.
        In this case, the unbiased energy, forces, stress are returned
        """
        # call to base-class to set atoms attribute
        Calculator.calculate(self, atoms)

        # predict + extract data
        out = self.model(self.as_data(atoms).to_dict(), compute_stress=True)
        energy = out["energy"].detach().cpu().item()
        forces = out["forces"].detach().cpu().numpy()

        # compute bias
        eV = self.energy_units_to_eV
        A  = self.length_units_to_A
        E  = energy * eV
        self.results = {
            "free_energy": E,
            "energy": E,
            "forces": forces * eV / A,
            "CV": out["CV"].detach().cpu().item(),
        }
        if atoms.pbc.any() and out["stress"] is not None:
            stress = out["stress"].detach().cpu().numpy()[0]
            self.results["stress"] = full_3x3_to_voigt_6_stress(stress * eV / A ** 3)

        if self.bias_center is not None: # apply bias
            delta = (self.results['CV'] - self.bias_center)
            bias_energy, gradient = self.get_bias(delta)
            bias_forces = out["CV_forces"].detach().cpu().numpy() * gradient

            clip_forces(bias_forces, self.max_force)
            self.results['bias_energy'] = bias_energy * eV
            self.results['bias_forces'] = bias_forces * eV / A
            self.results['energy'] = self.results['energy'] + self.results['bias_energy']
            self.results['forces'] = self.results['forces'] + self.results['bias_forces']

            if atoms.pbc.any() and out["stress"] is not None:
                bias_stress = out["CV_stress"].detach().cpu().numpy()[0] * gradient
                bias_stress *= eV / (A ** 3)
                clip_stress(bias_stress, self.max_stress)
                self.results['bias_stress'] = full_3x3_to_voigt_6_stress(bias_stress)
                self.results['stress'] = self.results['stress'] + self.results['bias_stress']
        else:
            self.results['bias_energy'] = self.results['CV']
            self.results['bias_forces'] = out["CV_forces"].detach().cpu().numpy()
            self.results['bias_stress'] = out["CV_stress"].detach().cpu().numpy()[0]
            self.results['bias_stress'] = full_3x3_to_voigt_6_stress(self.results['bias_stress'])

    def get_bias(self, x):
        value, grad = compute_bias(x, self.bias_function)
        return self.bias_scale * value, self.bias_scale * grad
