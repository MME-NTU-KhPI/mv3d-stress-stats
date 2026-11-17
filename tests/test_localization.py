from pathlib import Path

import h5py
import numpy as np
import pytest

from mv3d_stress_stats.invariants import von_mises
from mv3d_stress_stats.localization import LocalizationBasis, weighted_quantiles
from mv3d_stress_stats.node_cli import main as cli_main

ROOT = Path(__file__).resolve().parent.parent
MULTI = ROOT / "result-10-5.hdf5"
SINGLE = ROOT / "ansys_angle000.00_r0.hdf5"


def _synthetic_basis(m=3, n=50, seed=0):
    rng = np.random.default_rng(seed)
    return LocalizationBasis(rng.normal(size=(m, n, 6, 6)) * 1e9)


def test_superposition_is_linear():
    b = _synthetic_basis()
    e1, e2 = np.eye(6)[0] * 1e-4, np.eye(6)[3] * 2e-4
    np.testing.assert_allclose(b.local_stress(e1 + e2), b.local_stress(e1) + b.local_stress(e2))


def test_stress_input_roundtrip():
    b = _synthetic_basis()
    b.A += np.eye(6)[None, None] * 1e11  # well-conditioned C_eff
    eps = np.array([1e-4, -2e-5, 3e-5, 1e-5, 0, 4e-5])
    sig = b.effective_stiffness() @ eps
    np.testing.assert_allclose(b.macro_strain_from_stress(sig), eps, rtol=1e-10, atol=1e-18)


def test_weighted_quantiles_match_numpy_for_equal_weights():
    v = np.random.default_rng(1).normal(size=1001)
    p = [0.05, 0.5, 0.95]
    np.testing.assert_allclose(
        weighted_quantiles(v, np.ones_like(v), p), np.quantile(v, p, method="hazen")
    )


def test_nodal_statistics_matches_node_distribution():
    b = _synthetic_basis()
    b.weights = np.array([0.5, 0.3, 0.2])
    eps = np.random.default_rng(2).normal(size=(4, 6)) * 1e-4
    st = b.nodal_statistics("von_mises", macro_strains=eps, probs=[0.1, 0.9], thresholds=[1e5])
    for j in range(4):
        nd = b.node_distribution("von_mises", macro_strain=eps[j])
        np.testing.assert_allclose(st["mean"][j], nd.stats["mean"], rtol=1e-12)
        np.testing.assert_allclose(st["quantiles"][j], nd.quantiles([0.1, 0.9]), rtol=1e-12)
        np.testing.assert_allclose(st["exceedance"][j, 0], nd.exceedance(1e5), rtol=1e-12)
        np.testing.assert_allclose(np.sum(nd.pdf * np.diff(nd.bin_edges)), 1.0, rtol=1e-12)


@pytest.mark.skipif(not MULTI.is_file(), reason="result-10-5.hdf5 not present")
def test_reproduces_all_fe_load_steps():
    b = LocalizationBasis.from_file(MULTI, set_indices=range(10))
    assert b.metadata["shear_in_file"] == "tensor"
    for c in b.verify_against_file():
        assert c.max_rel_error < 1e-5, (c.load_step, c.max_rel_error)


@pytest.mark.skipif(not MULTI.is_file(), reason="result-10-5.hdf5 not present")
def test_combined_step_distribution_equals_fe():
    sets = list(range(10))
    b = LocalizationBasis.from_file(MULTI, set_indices=sets)
    with h5py.File(MULTI, "r") as f:
        eps = f["1/ls_9/eps_as_loading"][:].astype(float)
        eps[3:] *= 2.0  # tensor -> engineering shear
        ids = b.metadata["set_ids"]
        fe = np.concatenate([von_mises(f[f"{s}/ls_9/results"][:, 7:13].astype(float)) for s in ids])
    nd = b.node_distribution("von_mises", macro_strain=eps)
    p = [0.01, 0.05, 0.5, 0.95, 0.99]
    np.testing.assert_allclose(nd.quantiles(p), np.quantile(fe, p, method="hazen"), rtol=1e-5)


@pytest.mark.skipif(not SINGLE.is_file(), reason="ansys_angle000.00_r0.hdf5 not present")
def test_single_file_basis_and_shear_detection():
    b = LocalizationBasis.from_file(SINGLE)
    assert b.num_realizations == 1
    assert b.metadata["shear_in_file"] == "tensor"
    assert max(c.max_rel_error for c in b.verify_against_file()) < 1e-5


@pytest.mark.skipif(not MULTI.is_file(), reason="result-10-5.hdf5 not present")
def test_cli_roundtrip(tmp_path):
    inp = tmp_path / "nodes.csv"
    inp.write_text("node_id,X,Y,Z,XY,YZ,XZ\n1,1e-4,0,0,0,0,0\n2,1e-4,1e-4,0,0,0,0\n")
    out = tmp_path / "stats.csv"
    rc = cli_main(["--basis", str(MULTI), "--sets", "0,1,2", "--input", str(inp),
                   "--probs", "0.5,0.95", "--thresholds", "2e7", "--out", str(out)])
    assert rc == 0
    lines = out.read_text().strip().splitlines()
    assert lines[0].startswith("node_id,q_macro,mean,std,min,max,q0.5,q0.95,P_gt_2e+07")
    assert len(lines) == 3


def test_direction_sweep_matches_node_distribution():
    from mv3d_stress_stats.localization import direction_sweep

    b = _synthetic_basis()
    b.A += np.eye(6)[None, None] * 1e11
    b.weights = np.array([0.5, 0.3, 0.2])
    th = [0.0, 30.0, 90.0]
    sw = direction_sweep(b, "von_mises", 0, 3, 50e6, th, probs=[0.1, 0.5, 0.9])
    for k, t in enumerate(th):
        s = np.zeros(6)
        s[0], s[3] = 50e6 * np.cos(np.deg2rad(t)), 50e6 * np.sin(np.deg2rad(t))
        nd = b.node_distribution("von_mises", macro_stress=s)
        np.testing.assert_allclose(sw.quantiles[k], nd.quantiles([0.1, 0.5, 0.9]), rtol=1e-9)
        np.testing.assert_allclose(sw.mean[k], nd.stats["mean"], rtol=1e-9)
        np.testing.assert_allclose(sw.q_macro[k], von_mises(s), rtol=1e-9)
        np.testing.assert_allclose(np.sum(sw.pdf[k] * np.diff(sw.bin_edges)), 1.0, rtol=1e-12)
    eta = direction_sweep(b, "von_mises", 0, 3, 50e6, th, probs=[0.1, 0.5, 0.9], normalize=True)
    np.testing.assert_allclose(eta.quantiles * sw.q_macro[:, None], sw.quantiles, rtol=1e-9)
    with pytest.raises(ValueError):
        direction_sweep(b, "hydrostatic_stress", 0, 3, 50e6, [90.0], normalize=True)
