import numpy as np
import pytest
from mv3d_stress_stats.ensemble import RVEEnsemble, VarianceDecomposition
from mv3d_stress_stats.single_rve import RVESimulation


def test_variance_decomposition_analytical():
    # 3 synthetic realizations with known analytical within and between variance
    weights = [0.5, 0.3, 0.2]
    
    # Realization 1: mean=10, var=4
    # Realization 2: mean=20, var=9
    # Realization 3: mean=30, var=16
    n = 100000
    rng = np.random.default_rng(123)
    s1 = rng.normal(10.0, 2.0, n)
    s2 = rng.normal(20.0, 3.0, n)
    s3 = rng.normal(30.0, 4.0, n)
    
    # Pack into (n, 6) stress arrays (SX = s_i)
    arr1 = np.zeros((n, 6))
    arr1[:, 0] = s1
    arr2 = np.zeros((n, 6))
    arr2[:, 0] = s2
    arr3 = np.zeros((n, 6))
    arr3[:, 0] = s3
    
    ens = RVEEnsemble(
        sources=[arr1, arr2, arr3],
        weights=weights,
        rve_ids=["R1", "R2", "R3"],
    )
    
    res = ens.analyze(quantity="SX", bins=100)
    vd = res.variance_decomposition
    
    # Expected analytical:
    # V_within = 0.5*4 + 0.3*9 + 0.2*16 = 7.9
    # mean_of_means = 0.5*10 + 0.3*20 + 0.2*30 = 17.0
    # V_between = 0.5*(10-17)^2 + 0.3*(20-17)^2 + 0.2*(30-17)^2 = 24.5 + 2.7 + 33.8 = 61.0
    # V_total = 7.9 + 61.0 = 68.9
    assert vd.v_within == pytest.approx(7.9, rel=0.03)
    assert vd.mean_of_means == pytest.approx(17.0, rel=0.01)
    assert vd.v_between == pytest.approx(61.0, rel=0.03)
    assert vd.v_total == pytest.approx(68.9, rel=0.03)
    assert vd.eta_within + vd.eta_between == pytest.approx(1.0)
    assert vd.v_total == pytest.approx(vd.v_within + vd.v_between)


def test_ensemble_pdf_mixture():
    # Verify ensemble PDF is a valid density integrating to 1.0
    arr1 = np.zeros((5000, 6))
    arr1[:, 0] = np.random.normal(50.0, 5.0, 5000)
    arr2 = np.zeros((5000, 6))
    arr2[:, 0] = np.random.normal(70.0, 8.0, 5000)
    
    ens = RVEEnsemble([arr1, arr2])
    res = ens.analyze(quantity="SX", bins=100, kde=True)
    
    dx = np.diff(res.bin_edges)
    int_pdf = np.sum(res.ensemble_pdf * dx)
    assert int_pdf == pytest.approx(1.0, rel=0.01)
    
    # Check DataFrame export
    df = res.to_dataframe()
    assert len(df) == 2
    assert "mean" in df.columns
    assert "rve_id" in df.columns


def test_homogenized_response():
    # Question B: Macroscopic response across RVEs
    arr1 = np.zeros((100, 6))
    arr1[:, 0] = 100.0  # mean SX = 100
    arr2 = np.zeros((100, 6))
    arr2[:, 0] = 120.0  # mean SX = 120
    
    ens = RVEEnsemble([arr1, arr2])
    macro_res = ens.homogenized_response(quantity="SX")
    assert macro_res["mean_macro_stress"][0] == pytest.approx(110.0)
    assert macro_res["macro_scalar_mean"] == pytest.approx(110.0)
    # std of macro SX = sqrt((100-110)^2 * 0.5 + (120-110)^2 * 0.5) = 10.0
    assert macro_res["macro_scalar_std"] == pytest.approx(10.0)
