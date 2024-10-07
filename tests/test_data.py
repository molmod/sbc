import numpy as np
import torch

from ase import Atoms

from mace.tools.utils import AtomicNumberTable
from mace.data.atomic_data import get_data_loader

from sbc.tools.utils import PhaseTable
from sbc.data.utils import config_from_atoms
from sbc.data.atomic_data import AtomicData, get_data_loader


def test_phase_table():
    atoms = Atoms(numbers=np.ones(3), positions=np.eye(3), pbc=False)
    atoms.info['phase'] = 'A'
    atoms_list = [atoms]

    atoms_ = atoms_list[-1].copy()
    atoms_.info['phase'] = 'C'
    atoms_list.append(atoms_)

    atoms_ = atoms_list[-1].copy()
    atoms_.info['phase'] = 'A'
    atoms_list.append(atoms_)

    atoms_ = atoms_list[-1].copy()
    atoms_.info['phase'] = None
    atoms_list.append(atoms_)

    p_table = PhaseTable.from_phases([a.info['phase'] for a in atoms_list])
    assert len(p_table) == 2
    z_table = AtomicNumberTable([1])

    configs = [config_from_atoms(a) for a in atoms_list]
    dataset = [AtomicData.from_config(c, z_table, p_table, cutoff=3.0) for c in configs]
    assert np.allclose(
            dataset[0].phase.cpu().numpy(),
            np.array([0]),
            )
    assert np.allclose(
            dataset[1].phase.cpu().numpy(),
            np.array([1]),
            )
    assert torch.all(torch.isnan(dataset[-1].phase))


def test_missing_data():
    atoms = Atoms(numbers=np.ones(3), positions=np.eye(3), pbc=False)
    atoms.info['phase'] = 'A'
    atoms_list = [atoms]

    atoms_ = atoms_list[-1].copy()
    atoms_.info['phase'] = 'C'
    atoms_list.append(atoms_)

    atoms_ = atoms_list[-1].copy()
    atoms_.info['phase'] = 'A'
    atoms_.arrays['forces'] = (-1.0) * np.ones((len(atoms_), 3))
    atoms_list.append(atoms_)

    atoms_ = atoms_list[-1].copy()
    atoms_.info['phase'] = None
    atoms_.arrays['forces'] = np.zeros((len(atoms_), 3))
    atoms_list.append(atoms_)

    p_table = PhaseTable.from_phases([a.info['phase'] for a in atoms_list])
    z_table = AtomicNumberTable([1])
    configs = [config_from_atoms(a) for a in atoms_list]
    dataset = [AtomicData.from_config(c, z_table, p_table, cutoff=3.0) for c in configs]

    batch_size = 4
    data_loader = get_data_loader(dataset, batch_size=batch_size)
    for batch in iter(data_loader):
        assert type(batch['forces']) is torch.Tensor # no more None's
        assert type(batch['phase']) is torch.Tensor

    pred = {
            'logits': torch.ones((batch_size, 2)),
            'CV_forces': torch.ones((batch_size * len(atoms_), 3)),
        }
