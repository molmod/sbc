###########################################################################################
# Atomic Data Class for handling molecules as graphs
# Authors: Sander Vandenhaute
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################


from pathlib import Path
from typing import Optional
import torch
from e3nn.util import jit
import numpy as np
from scipy.special import logsumexp

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
        model = torch.load(f=model_path, map_location=device)
        if default_dtype == 'float64':
            model = model.double()
        else:
            model = model.float()
        model = model.to(self.device)
        self.model = model
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


def hills(
    logits: np.ndarray,
    centers: np.ndarray,
    height: float,
    sigma: float,
    vectors: Optional[np.ndarray] = None
) -> tuple[float, np.ndarray]:
    assert len(logits.shape) == 1
    nphases = len(logits)
    assert len(logits) == centers.shape[1]
    logits = logits.reshape((1, -1))

    delta = logits - centers

    if vectors is not None:
        projected = np.einsum('ijk,ik->ij', vectors, delta)
        distances = np.sum(projected ** 2, axis=1, keepdims=True)
        extra = (-1.0) * np.einsum('ij,ijk->ik', projected, vectors) / sigma ** 2
    else:
        distances = np.sum(delta ** 2, axis=1, keepdims=True)
        extra = (-1.0) * delta / sigma ** 2

    exponent = (-1.0) * distances / (2 * sigma ** 2)
    energy = np.sum(height * np.exp(exponent))

    # extra = (-1.0) * delta / sigma ** 2
    gradient = np.sum(height * extra * np.exp(exponent).reshape(-1, 1), axis=0)
    return energy, gradient


# def divergence(
#     logits: np.ndarray,
#     centers: np.ndarray,
#     height: float,
#     sigma: float,
# ) -> tuple[float, np.ndarray]:
#     if len(centers.shape) == 1:
#         centers = centers.reshape(1, -1)
#     assert len(logits.shape) == 1
#     assert len(logits) == centers.shape[1]
#     logits = logits.reshape((1, -1))
#
#     probabilities = np.exp(centers)
#     assert np.allclose(np.sum(probabilities, axis=1) - 1, 0.0, atol=1e-3)
#
#     div_KL = np.sum(probabilities * (centers - logits), axis=1, keepdims=True)
#     exponent = (-1.0) * div_KL / sigma
#     energy = height * np.sum(np.exp(exponent))
#
#     extra = probabilities / sigma
#     gradient = np.sum(height * np.exp(exponent) * extra, axis=0)
#     return energy, gradient


class MetadynamicsCalculator(MACECalculator):
    implemented_properties = [
        'free_energy',
        'energy',
        'forces',
        'stress',
        ]

    def __init__(
        self,
        path_hills: Path,
        height: float,
        sigma: float,
        frequency: int,
        use_svd: bool = False,
        svd_dimensions: int = 2,
        **kwargs,
        ):
        super().__init__(**kwargs)
        self.model.eval()
        self.p_table = PhaseTable(self.model.classifier.phases)
        self.height = height
        self.sigma = sigma
        self.frequency = frequency
        self.use_svd = use_svd
        self.svd_dimensions = 2
        self.path_hills = path_hills

        self.counter = 0

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        # call to base-class to set atoms attribute
        Calculator.calculate(self, atoms)

        # predict + extract data
        out = self.model(self.as_data(atoms).to_dict(), compute_stress=True)
        energy = out["energy"].detach().cpu().item()
        forces = out["forces"].detach().cpu().numpy()
        stress = np.zeros(6)
        if atoms.pbc.any() and out["stress"] is not None:
            stress = out["stress"].detach().cpu().numpy()[0]
            stress = full_3x3_to_voigt_6_stress(stress)

        logits = out['logits'].detach().cpu().numpy()[0]
        logits_forces = out['logits_forces'].detach().cpu().numpy()
        logits_stress = np.zeros((len(logits), 3, 3))

        bias_energy = 0.0
        bias_forces = np.zeros_like(forces)
        bias_stress = np.zeros(6)
        nphases = len(logits)
        if self.path_hills.exists():
            hills_data = np.loadtxt(self.path_hills)
            if len(hills_data.shape) == 1:
                nhills = 1
            else:
                nhills = hills_data.shape[0]
            hills_data = hills_data.reshape((nhills, -1))
            if self.use_svd:
                centers = hills_data[:, :nphases]
                vectors = hills_data[:, nphases:].reshape(nhills, -1, nphases)
            else:
                centers = hills_data
                vectors = None

            bias_energy, gradients = hills(
                logits,
                centers,
                self.height,
                self.sigma,
                vectors=vectors,
            )
            bias_forces = np.einsum('i,ijk->jk', gradients, logits_forces)
            bias_stress = np.einsum('i,ijk->jk', gradients, logits_stress)
            bias_stress = full_3x3_to_voigt_6_stress(bias_stress)

        if self.frequency > 0:
            energy += bias_energy
            forces += bias_forces
            stress += bias_stress

            if self.counter % self.frequency == 0:  # add hill
                if self.use_svd:
                    U, sigma, Vh = np.linalg.svd(logits_forces.reshape(nphases, -1))
                    vectors = U.T[:self.svd_dimensions].flatten()  # transpose to get vectors as rows
                    centers = logits
                    line = ' '.join([str(c) for c in np.concatenate((centers, vectors))])
                else:
                    line = ' '.join([str(c) for c in logits])
                with open(self.path_hills, "a") as f:
                    f.write(line)
                    f.write('\n')

        self.counter += 1

        self.results = {
            "free_energy": energy,
            "energy": energy,
            "forces": forces,
            "stress": stress,
            "logits": logits,
            "bias_energy": bias_energy,
            "bias_forces": bias_forces,
            "bias_stress": bias_stress,
        }
