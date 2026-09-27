from pathlib import Path
import numpy as np
import pytest
from mv3d_stress_stats import (
    RVEReader,
    RVESimulation,
    RVEEnsemble,
    von_mises,
    j2,
    j3,
    hydrostatic_stress,
    principal_stresses,
)

HDF5_PATH = Path("ansys_angle000.00_r0.hdf5")


@pytest.mark.skipif(not HDF5_PATH.is_file(), reason="ansys_angle000.00_r0.hdf5 not found")
def test_ansys_prototype_quantities():
    """Validate Phase 1-2 & Phase 3 on ansys_angle000.00_r0.hdf5 as required by AGENTS.md Section 30."""
    reader = RVEReader(HDF5_PATH)
    assert len(reader.set_ids) >= 1
    
    # Check load steps
    ls_list = reader.load_steps(0)
    assert 1 in ls_list
    
    # Read load step 1
    data = reader.read_stress(set_index=0, load_step=1, as_grid=True)
    assert data.stress.shape == (25, 25, 25, 6)
    
    # 1. SX validation
    sx = data.stress[..., 0]
    assert np.all(np.isfinite(sx))
    assert sx.mean() == pytest.approx(1.684e7, rel=1e-3)
    
    # 2. SEQV / independently calculated von Mises vs ANSYS SEQV
    calc_vm = von_mises(data.stress)
    assert calc_vm.shape == (25, 25, 25)
    if data.seqv_ansys is not None:
        # Check agreement between independently calculated von Mises and ANSYS SEQV column
        np.testing.assert_allclose(calc_vm, data.seqv_ansys, rtol=1e-3)
        
    # 3. Hydrostatic stress
    sh = hydrostatic_stress(data.stress)
    # sigma_x = 1.684e7, sigma_y = 1.214e7, sigma_z = 1.214e7
    # sh = (1.684 + 1.214 + 1.214)/3 * 1e7 = 4.112/3 * 1e7 = 1.370667e7
    assert sh.mean() == pytest.approx((1.684e7 + 2 * 1.214e7) / 3.0, rel=1e-3)
    
    # 4. J2
    j2_val = j2(data.stress)
    # von Mises = sqrt(3 * J2) -> J2 = von_mises^2 / 3
    np.testing.assert_allclose(j2_val, (calc_vm ** 2) / 3.0, rtol=1e-4)
    
    # 5. J3
    j3_val = j3(data.stress)
    assert np.all(np.isfinite(j3_val))
    
    # 6. Principal stresses
    p = principal_stresses(data.stress)
    assert p.shape == (25, 25, 25, 3)
    # Check ordering sigma_1 >= sigma_2 >= sigma_3
    assert np.all(p[..., 0] >= p[..., 1] - 1e-4)
    assert np.all(p[..., 1] >= p[..., 2] - 1e-4)
    # Trace of principal stresses equals I1 = SX + SY + SZ
    np.testing.assert_allclose(np.sum(p, axis=-1), np.sum(data.stress[..., 0:3], axis=-1), rtol=1e-4)


@pytest.mark.skipif(not HDF5_PATH.is_file(), reason="ansys_angle000.00_r0.hdf5 not found")
def test_ansys_ensemble_with_synthetic_counterparts():
    """Verify Section 30: after r0 validation, add r1, r2, ... and verify p(Q) = 1/M sum p(Q|R=r)."""
    r0 = RVESimulation(HDF5_PATH, set_index=0, load_step=1, realization_id="r0")
    
    # Create two synthetic realizations r1 and r2 by applying perturbations to r0
    # to simulate statistically equivalent microstructures
    rng = np.random.default_rng(999)
    stress_r1 = r0.stress + rng.normal(0, 0.05 * 1.684e7, size=r0.stress.shape)
    stress_r2 = r0.stress + rng.normal(0, 0.10 * 1.684e7, size=r0.stress.shape)
    
    r1 = RVESimulation(stress_r1, realization_id="r1")
    r2 = RVESimulation(stress_r2, realization_id="r2")
    
    ensemble = RVEEnsemble([r0, r1, r2], rve_ids=["r0", "r1", "r2"])
    res = ensemble.analyze(quantity="von_mises", bins=150, kde=True)
    
    # Check variance decomposition: V_total = V_within + V_between
    vd = res.variance_decomposition
    assert vd.v_total == pytest.approx(vd.v_within + vd.v_between)
    assert vd.eta_within + vd.eta_between == pytest.approx(1.0)
    
    # Check ensemble PDF is exact 1/M weighted sum of realization PDFs
    manual_sum_pdf = (res.histograms[0] + res.histograms[1] + res.histograms[2]) / 3.0
    np.testing.assert_allclose(res.ensemble_pdf, manual_sum_pdf, rtol=1e-12)


OLDER_HDF5_PATH = Path("result-10-5.hdf5")


@pytest.mark.skipif(not OLDER_HDF5_PATH.is_file(), reason="result-10-5.hdf5 not found")
def test_older_matviz3d_multi_set_file():
    """Validate backward compatibility on older MatViz3D files (19 columns, multiple sets)."""
    reader = RVEReader(OLDER_HDF5_PATH)
    assert len(reader.set_ids) == 100
    
    # Check 19-column results table
    raw = reader.results(0, 1)
    assert raw.shape == (1331, 19)
    
    # Read stress
    data = reader.read_stress(set_index=0, load_step=1, as_grid=True)
    assert data.stress.shape == (11, 11, 11, 6)
    assert data.seqv_ansys is None  # Older format lacks SEQV column
    
    # von Mises computed via tensor invariants
    vm = von_mises(data.stress)
    assert vm.shape == (11, 11, 11)
    assert np.all(vm >= 0.0)
    
    # Multi-realization ensemble via RVEEnsemble.from_file
    ensemble = RVEEnsemble.from_file(OLDER_HDF5_PATH, set_indices=[0, 1, 2, 3, 4], load_step=1)
    assert len(ensemble.sources) == 5
    assert ensemble.rve_ids[0] == "result-10-5_s1"
    assert ensemble.rve_ids[4] == "result-10-5_s5"
    
    res = ensemble.analyze(quantity="von_mises", bins=80)
    vd = res.variance_decomposition
    assert vd.v_total == pytest.approx(vd.v_within + vd.v_between)
    assert vd.eta_within + vd.eta_between == pytest.approx(1.0)
    assert vd.eta_within > 0.8  # Most variability is within-RVE spatial heterogeneity

