from pathlib import Path
import numpy as np
import pytest
from mv3d_stress_stats.multiload import (
    identify_loading_case,
    analyze_multiload,
    MultiloadAnalysisResult,
    LoadingCaseInfo,
)
from mv3d_stress_stats.ensemble import RVEEnsemble


def test_identify_loading_case():
    # 1. Uniaxial normal
    c_x = identify_loading_case(np.array([1e-4, 0, 0, 0, 0, 0]), 1)
    assert c_x.category == "normal"
    assert c_x.text_label == "eps_x"
    assert r"\varepsilon_x" in c_x.label

    # 2. Pure shear
    c_xy = identify_loading_case(np.array([0, 0, 0, 1e-4, 0, 0]), 4)
    assert c_xy.category == "shear"
    assert c_xy.text_label == "eps_xy"
    assert r"\varepsilon_{xy}" in c_xy.label

    # 3. Biaxial
    c_biax = identify_loading_case(np.array([1e-4, 1e-4, 0, 0, 0, 0]), 7)
    assert c_biax.category == "biaxial"
    assert "eps_x" in c_biax.text_label and "eps_y" in c_biax.text_label

    # 4. None / unspecified
    c_none = identify_loading_case(None, 2)
    assert c_none.category == "unspecified"
    assert c_none.label == "LS 2"


HDF5_FILE = Path("result-10-5.hdf5")


@pytest.mark.skipif(not HDF5_FILE.is_file(), reason="result-10-5.hdf5 not found")
def test_analyze_multiload_on_hdf5():
    """Test analyzing multiple load cases (normal vs shear) on real HDF5 file."""
    ensemble = RVEEnsemble.from_file(HDF5_FILE, set_indices=[0, 1])
    res = analyze_multiload(ensemble, quantity="von_mises", load_steps=[1, 2, 4], bins=50)

    assert isinstance(res, MultiloadAnalysisResult)
    assert 1 in res.cases
    assert 2 in res.cases
    assert 4 in res.cases

    assert res.cases[1].category == "normal"
    assert res.cases[2].category == "normal"
    assert res.cases[4].category == "shear"

    # Export to DataFrame
    df = res.to_dataframe()
    assert len(df) == 3
    assert "mean" in df.columns
    assert "category" in df.columns
    assert "v_within" in df.columns
    assert "v_between" in df.columns

    # Verify shear step has higher mean von Mises stress than normal step
    mean_normal_x = df.loc[df["load_step"] == 1, "mean"].values[0]
    mean_shear_xy = df.loc[df["load_step"] == 4, "mean"].values[0]
    assert mean_shear_xy > mean_normal_x

    # Anisotropy index
    aniso_normal = res.anisotropy_index("normal")
    assert aniso_normal >= 1.0
