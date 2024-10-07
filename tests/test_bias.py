import numpy as np
import copy
import torch
import torch.nn.functional
from e3nn import o3
from ase import Atoms

from mace import modules, tools

import sbc.data
import sbc.tools
from sbc.modules import ClassifierMACE
from sbc.calculator import logcosh, compute_bias


torch.set_default_dtype(torch.float64)

table = tools.AtomicNumberTable([1, 8])
atomic_energies = np.array([1.0, 3.0], dtype=float)
config_template = dict(
    r_max=5,
    num_bessel=8,
    num_polynomial_cutoff=6,
    max_ell=2,
    interaction_cls=modules.interaction_classes[
        "RealAgnosticResidualInteractionBlock"
    ],
    interaction_cls_first=modules.interaction_classes[
        "RealAgnosticResidualInteractionBlock"
    ],
    num_interactions=5,
    num_elements=2,
    hidden_irreps=o3.Irreps("32x0e + 32x1o"),
    MLP_irreps=o3.Irreps("16x0e"),
    gate=torch.nn.functional.silu,
    atomic_energies=atomic_energies,
    avg_num_neighbors=8,
    atomic_numbers=table.zs,
    correlation=3,
    atomic_inter_scale=1.0,
    atomic_inter_shift=0.0,
)

def test_classifier_mace():
    phases = ['A', 'B']
    config = copy.deepcopy(config_template)
    config['phases'] = phases
    config['classifier'] = 'EnergyBasedClassifier'
    config['classifier_readout'] = [16]
    config['classifier_mixing'] = None
    model = ClassifierMACE(**config)

    atoms = Atoms(
            numbers=[8, 1, 1],
            positions=np.eye(3),
            pbc=False,
            )
    atoms.info['phase'] = 'A'
    as_data = sbc.data.AtomicData.from_config(
            sbc.data.config_from_atoms(atoms),
            z_table=table,
            p_table=sbc.tools.PhaseTable(phases),
            cutoff=5.0,
            )
    batch = next(iter(sbc.data.get_data_loader([as_data], batch_size=1)))

    model_output = model(batch)
    class_output = model.classifier(
            node_feats=model_output['node_feats'],
            batch=batch['batch'],
            ptr=batch['ptr'],
            )
    logits  = class_output['logits'].detach().cpu().numpy()
    logits_ = model_output['logits'].detach().cpu().numpy()
    assert np.allclose(logits, logits_)
    assert not np.allclose(logits, 0)

    #config = copy.deepcopy(config_template)
    #config['phases'] = phases
    #config['classifier_readout'] = [2]
    #config['classifier_mixing'] = 4
    #model = ClassifierMACE(**config)

    #output = model(batch)
    #logits  = model.classifier(
    #        node_feats=output['node_feats'],
    #        batch=batch['batch'],
    #        ptr=batch['ptr'],
    #        ).detach().cpu().numpy()
    #logits_ = output['logits'].detach().cpu().numpy()
    #assert np.allclose(logits, logits_)


def test_bias_function():
    for i in range(4):
        assert np.allclose(np.log(np.cosh(10 ** (-i))), logcosh(10 ** (-i))) 
    assert np.allclose(10 ** 2, logcosh(10 ** 2) + np.log(2))

    value, grad = compute_bias(2, function='quadratic')
    assert value == 4
    assert grad == 4

    value_, grad_ = compute_bias(2, function='logcosh')
    assert np.allclose(value_, np.log(np.cosh(2)) / np.log(np.cosh(1)))
    assert grad_ == np.tanh(2) / np.log(np.cosh(1))
    assert value > value_

    value_, grad_ = compute_bias(0.9, function='logcosh')
    assert np.allclose(value_, np.log(np.cosh(0.9)) / np.log(np.cosh(1)))
    assert grad_ == np.tanh(0.9) / np.log(np.cosh(1))
    assert value_ > compute_bias(0.9, function='quadratic')[0]
