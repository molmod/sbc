import functools
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
from sbc.calculator import hills, MetadynamicsCalculator


torch.set_default_dtype(torch.float64)

table = tools.AtomicNumberTable([1, 8])
atomic_energies = np.array([1.0, 3.0], dtype=float)
config_template = dict(
    r_max=5,
    num_bessel=8,
    num_polynomial_cutoff=6,
    max_ell=2,
    interaction_cls=modules.interaction_classes[
        "RealAgnosticDensityResidualInteractionBlock"
    ],
    interaction_cls_first=modules.interaction_classes[
        "RealAgnosticDensityInteractionBlock"
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
    pair_repulsion=True,
    # distance_transform='Agnesi',  # disable, or initial logits ~ 1e20 somehow
    radial_type='bessel',
)

def test_classifier_mace(tmp_path):
    phases = ['A', 'B', 'C', 'D', 'E']
    config = copy.deepcopy(config_template)
    config['phases'] = phases
    config['classifier_layer_sizes'] = [16, 8]
    model = ClassifierMACE(**config)

    atoms = Atoms(
            numbers=[8, 1, 1],
            positions=np.eye(3),
            pbc=False,
            )
    atoms.info['phase'] = 'A'
    data0 = sbc.data.AtomicData.from_config(
            sbc.data.config_from_atoms(atoms),
            z_table=table,
            p_table=sbc.tools.PhaseTable(phases),
            cutoff=5.0,
            )
    atoms = atoms.copy()
    atoms.set_positions(2 * np.eye(3))
    data1 = sbc.data.AtomicData.from_config(
            sbc.data.config_from_atoms(atoms),
            z_table=table,
            p_table=sbc.tools.PhaseTable(phases),
            cutoff=5.0,
            )
    batch = next(iter(sbc.data.get_data_loader([data0, data1], batch_size=2)))

    model.eval()  # compute logit forces
    model_output = model(batch)
    batch['node_feats'] = model_output['node_feats']
    cls_output = model.classifier(batch)
    logits  = cls_output['logits'].detach().cpu().numpy()
    logits_ = model_output['logits'].detach().cpu().numpy()
    assert np.allclose(logits, logits_)
    assert not np.allclose(logits, 0)

    # forces are computed on normalized logits --> sum to zero
    logits_forces = model_output['logits_forces'].detach().cpu().numpy()
    assert np.allclose(np.sum(logits_forces, axis=1), 0.0)

    path_hills = tmp_path / 'hills'
    path_model = tmp_path / 'model.pth'
    torch.save(model, path_model)
    calculator = MetadynamicsCalculator(
        model_path=path_model,
        default_dtype='float64',
        device='cpu',
        path_hills=path_hills,
        height=0.1,
        sigma=1,
        frequency=2,
    )
    atoms.calc = calculator

    for i in range(5):
        p = np.random.uniform(-0.1, 0.1, size=(len(atoms), 3))
        atoms.set_positions(atoms.get_positions() + p)
        atoms.get_potential_energy()

    assert atoms.calc.counter == 5

    with open(path_hills, 'r') as f:
        assert len(f.readlines()) == 3  # hills at step 0, 2, 4


def check_grad(func, x, epsilon=1e-7, rtol=1e-5, atol=1e-8):
    """
    Check gradient implementation by comparing analytical gradient with numerical gradient
    computed using finite differences.

    Parameters:
    -----------
    func : callable
        Function that takes x and returns (value, grad) tuple
    x : ndarray
        Point at which to check gradient
    epsilon : float
        Step size for finite difference calculation
    rtol : float
        Relative tolerance for comparison
    atol : float
        Absolute tolerance for comparison

    Returns:
    --------
    dict containing:
        - is_correct: bool indicating if gradients match within tolerance
        - max_abs_err: maximum absolute error
        - max_rel_err: maximum relative error
        - analytical_grad: gradient returned by func
        - numerical_grad: gradient computed via finite differences
    """
    x = np.asarray(x)
    value, analytical_grad = func(x)
    numerical_grad = np.zeros_like(x)

    # Compute numerical gradient using central differences
    for i in range(x.size):
        x_plus = x.copy()
        x_plus[i] += epsilon
        value_plus = func(x_plus)[0]

        x_minus = x.copy()
        x_minus[i] -= epsilon
        value_minus = func(x_minus)[0]

        numerical_grad[i] = (value_plus - value_minus) / (2 * epsilon)

    # Compare gradients
    abs_diff = np.abs(analytical_grad - numerical_grad)
    max_abs_err = np.max(abs_diff)

    # Compute relative error, avoiding division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        rel_diff = abs_diff / np.maximum(
            np.abs(analytical_grad),
            np.abs(numerical_grad)
        )
    rel_diff[~np.isfinite(rel_diff)] = 0
    max_rel_err = np.max(rel_diff)

    # Check if gradients match within tolerance
    is_correct = np.allclose(
        analytical_grad,
        numerical_grad,
        rtol=rtol,
        atol=atol
    )

    return {
        'is_correct': is_correct,
        'max_abs_err': max_abs_err,
        'max_rel_err': max_rel_err,
        'analytical_grad': analytical_grad,
        'numerical_grad': numerical_grad
    }


def test_hills_function():
    centers = np.random.uniform(size=(100, 3))
    height = 10
    sigma = 1

    func = functools.partial(hills, centers=centers, height=height, sigma=sigma)
    assert func(np.random.uniform(size=(3,)))[0] > 0.0
    output = check_grad(func, np.random.uniform(size=(3,)))
    assert output['is_correct']

    centers = np.random.uniform(size=(100, 3))
    vectors = np.random.uniform(size=(100, 1, 3))
    vectors /= np.linalg.norm(vectors, axis=2, keepdims=True)
    height = 10
    sigma = 1
    func = functools.partial(
        hills,
        centers=centers,
        height=height,
        sigma=sigma,
        vectors=vectors,
    )
    assert func(np.random.uniform(size=(3,)))[0] > 0.0
    output = check_grad(func, np.random.uniform(size=(3,)))
    assert output['is_correct']



def test_load_model():
    model = modules.ScaleShiftMACE(**config_template)
    kwargs = {
        'phases': ['A', 'B'],
        'classifier_layer_sizes': [16],
    }
    cls_model = ClassifierMACE.from_model(
        model,
        **kwargs,
    )

    # create some data and evaluate energy
    atoms = Atoms(
        numbers=[8, 1, 1],
        positions=np.eye(3),
        pbc=False,
    )
    data = sbc.data.AtomicData.from_config(
        sbc.data.config_from_atoms(atoms),
        z_table=table,
        p_table=sbc.tools.PhaseTable(['A', 'B']),
        cutoff=5.0,
    )
    batch = next(iter(sbc.data.get_data_loader([data], batch_size=1)))

    model.eval()  # compute logit forces
    out = model(batch)
    e0 = out['interaction_energy'].detach().cpu().numpy()

    out = cls_model(batch)
    e1 = out['interaction_energy'].detach().cpu().numpy()
    assert np.allclose(e0, e1)
