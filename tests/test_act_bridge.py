"""Tests of the ACT bridge: pure helpers of the IronPython extension, the
.npz basis cache, stress units in node_cli, and the full file round trip that
Mechanical performs (nodes CSV -> python -m node_cli -> stats CSV)."""

import ast
import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from mv3d_stress_stats.localization import LocalizationBasis
from mv3d_stress_stats.node_cli import main as cli_main

ROOT = Path(__file__).resolve().parent.parent
ACT = ROOT / "act" / "RVEStressStats"
MULTI = ROOT / "result-10-5.hdf5"


def _load_bridge():
    spec = importlib.util.spec_from_file_location("rve_bridge", ACT / "rve_bridge.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bridge = _load_bridge()


def _basis(seed=0):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(3, 40, 6, 6)) * 1e9 + np.eye(6)[None, None] * 1e11
    return LocalizationBasis(A)


# ------------------------------------------------------------ pure helpers
def test_column_names_match_cli_formatting():
    for p in (0.05, 0.5, 0.95, 0.99, 0.999):
        assert bridge.quantile_column(p) == f"q{p:g}"
    for t in (0.0, 80.0, 2.5e7, 1e-3):
        assert bridge.exceedance_column(t) == f"P_gt_{t:g}"


@pytest.mark.parametrize("text,unit", [("MPa", "MPa"), ("Pa", "Pa"), ("N/mm^2", "MPa"),
                                       ("N/m^2", "Pa"), ("psi", None), (None, None)])
def test_normalize_stress_unit(text, unit):
    assert bridge.normalize_stress_unit(text) == unit


@pytest.mark.parametrize("arg", ["plain", "with space", r"C:\a b\c\\", 'q"uote', "", "trail\\"])
def test_quote_arg_roundtrips_through_windows_parser(arg):
    # let the real MS C runtime argv parser (python.exe) split our command line
    code = "import sys;print(repr(sys.argv[1:]))"
    line = bridge.quote_arg(sys.executable) + " " + bridge.join_args(["-c", code, arg, "end"])
    out = subprocess.run(line, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert ast.literal_eval(out.stdout.strip()) == [arg, "end"]


def test_extension_scripts_are_python2_compatible_syntax():
    for name in ("rve_bridge.py", "main.py"):
        src = (ACT / name).read_text(encoding="utf-8")
        tree = ast.parse(src)
        assert not any(isinstance(n, ast.JoinedStr) for n in ast.walk(tree)), f"f-string in {name}"


def test_extension_xml_callbacks_exist():
    import xml.etree.ElementTree as ET

    root = ET.parse(ACT.parent / "RVEStressStats.xml").getroot()
    names = {n.text for n in root.iter() if n.tag in ("evaluate", "onclick")}
    tree = ast.parse((ACT / "main.py").read_text(encoding="utf-8"))
    defined = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    assert names and names <= defined
    for res in root.iter("result"):
        props = [p.get("name") for p in res.iter("property")]
        assert "Geometry" in props and "Basis" in props


# ------------------------------------------------------------ cache
def test_save_load_roundtrip(tmp_path):
    b = _basis()
    b.metadata = {"k": [1, 2]}
    b.save(tmp_path / "b.npz")
    c = LocalizationBasis.load(tmp_path / "b.npz")
    np.testing.assert_array_equal(c.A, b.A)
    np.testing.assert_allclose(c.weights, b.weights)
    assert c.rve_ids == b.rve_ids and c.metadata == {"k": [1, 2]}


@pytest.mark.skipif(not MULTI.is_file(), reason="result-10-5.hdf5 not present")
def test_cached_basis_equals_fresh(tmp_path):
    b1 = LocalizationBasis.from_file_cached(MULTI, set_indices=[0, 1], cache_dir=tmp_path)
    assert len(list(tmp_path.glob("*.npz"))) == 1
    b2 = LocalizationBasis.from_file_cached(MULTI, set_indices=[0, 1], cache_dir=tmp_path)
    fresh = LocalizationBasis.from_file(MULTI, set_indices=[0, 1])
    np.testing.assert_array_equal(b2.A, fresh.A)
    np.testing.assert_array_equal(b1.A, fresh.A)
    LocalizationBasis.from_file_cached(MULTI, set_indices=[2], cache_dir=tmp_path)
    assert len(list(tmp_path.glob("*.npz"))) == 2  # different sets -> new key


# ------------------------------------------------------------ CLI round trip
def _write(path, ids, rows):
    bridge.write_nodes_csv(str(path), ids, rows)


@pytest.mark.skipif(not MULTI.is_file(), reason="result-10-5.hdf5 not present")
def test_stress_unit_scaling(tmp_path):
    sig_mpa = np.array([[60.0, 20, -10, 15, 0, 5], [10.0, 0, 0, 0, 0, 0]])
    _write(tmp_path / "mpa.csv", [1, 2], sig_mpa)
    _write(tmp_path / "pa.csv", [1, 2], sig_mpa * 1e6)
    common = ["--basis", str(MULTI), "--sets", "0,1,2", "--kind", "stress",
              "--probs", "0.5,0.95", "--cache-dir", str(tmp_path)]
    for q in ("von_mises", "j2", "triaxiality"):
        cli_main(common + ["--quantity", q, "--input", str(tmp_path / "pa.csv"),
                           "--out", str(tmp_path / "pa_out.csv"), "--thresholds", "5e7"])
        cli_main(common + ["--quantity", q, "--input", str(tmp_path / "mpa.csv"),
                           "--out", str(tmp_path / "mpa_out.csv"), "--stress-unit", "MPa",
                           "--thresholds", "50"])
        k = {"von_mises": 1, "j2": 2, "triaxiality": 0}[q]
        for col in ("q_macro", "mean", "q0.95"):
            pa = bridge.read_stats_csv(str(tmp_path / "pa_out.csv"), col)
            mpa = bridge.read_stats_csv(str(tmp_path / "mpa_out.csv"), col)
            for nid in (1, 2):
                assert mpa[nid] == pytest.approx(pa[nid] / 1e6 ** k, rel=1e-7, abs=1e-12)
        if k == 1:  # threshold given in the matching unit -> identical probability
            pa = bridge.read_stats_csv(str(tmp_path / "pa_out.csv"), "P_gt_5e+07")
            mpa = bridge.read_stats_csv(str(tmp_path / "mpa_out.csv"), "P_gt_50")
            assert pa == mpa


@pytest.mark.skipif(not MULTI.is_file(), reason="result-10-5.hdf5 not present")
def test_mechanical_style_subprocess_roundtrip(tmp_path):
    """Exactly what main.py does: bridge CSV -> separate python process with a
    PYTHONPATH -> stats CSV parsed by the bridge."""
    basis = LocalizationBasis.from_file(MULTI, set_indices=[0, 1, 2])
    rng = np.random.default_rng(3)
    eps = rng.normal(size=(5, 6)) * 2e-4
    ids = [101, 7, 33, 5000, 2]
    nodes = tmp_path / "dir with space" / "nodes.csv"
    nodes.parent.mkdir()
    _write(nodes, ids, eps)
    out = tmp_path / "dir with space" / "stats.csv"
    args = bridge.build_cli_args(str(MULTI), str(nodes), str(out), "strain", "von_mises", "MPa",
                                 probs=[0.95], thresholds=[60.0], sets="0, 1;2",
                                 plot_nodes="7", plot_dir=str(tmp_path / "plots"))
    import os

    env = dict(os.environ, PYTHONPATH=str(ROOT))  # ProcessStartInfo inherits the env too
    proc = subprocess.run(bridge.quote_arg(sys.executable) + " " + bridge.join_args(args),
                          cwd=str(tmp_path), env=env, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    q95 = bridge.read_stats_csv(str(out), "q0.95")
    pex = bridge.read_stats_csv(str(out), "P_gt_60")
    ref = basis.nodal_statistics("von_mises", macro_strains=eps, probs=[0.95], thresholds=[60e6])
    for j, nid in enumerate(ids):
        assert q95[nid] == pytest.approx(ref["quantiles"][j, 0] / 1e6, rel=1e-7)
        assert pex[nid] == pytest.approx(ref["exceedance"][j, 0], abs=1e-6)
    assert (tmp_path / "plots" / "node_7_von_mises.png").is_file()
