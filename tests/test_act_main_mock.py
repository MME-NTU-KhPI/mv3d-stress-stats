"""Run the Mechanical callbacks of act/RVEStressStats/main.py in CPython with
mocked clr / System / ExtAPI objects (only members confirmed in the ANSYS
2025 R2 ACT API docs are provided), so the callback logic and the real CLI
process are exercised end to end without Mechanical."""

import os
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from mv3d_stress_stats.localization import LocalizationBasis

ROOT = Path(__file__).resolve().parent.parent
ACT = ROOT / "act" / "RVEStressStats"
MULTI = ROOT / "result-10-5.hdf5"


# ------------------------------------------------------------ .NET mocks
class _Task:
    def __init__(self, value):
        self.Result = value


class _Stream:
    def __init__(self, text):
        self._t = text
        self._lines = text.splitlines()

    def ReadToEnd(self):
        return self._t

    def ReadLine(self):  # .NET StreamReader: None at end of stream
        return self._lines.pop(0) if self._lines else None

    def ReadToEndAsync(self):
        return _Task(self._t)


class ProcessStartInfo:
    def __init__(self, exe, args):
        self.FileName, self.Arguments = exe, args
        self.EnvironmentVariables = {}
        self.UseShellExecute = True
        self.WorkingDirectory = None


class Process:
    @staticmethod
    def Start(psi):
        assert psi.UseShellExecute is False and psi.RedirectStandardError
        env = dict(os.environ, **psi.EnvironmentVariables)
        line = '"%s" %s' % (psi.FileName, psi.Arguments)
        cp = subprocess.run(line, cwd=psi.WorkingDirectory, env=env,
                            capture_output=True, text=True)
        p = Process()
        p.StandardOutput, p.StandardError = _Stream(cp.stdout), _Stream(cp.stderr)
        p.ExitCode = cp.returncode
        return p

    def WaitForExit(self):
        pass


class _ArrayFactory:
    def __getitem__(self, _type):
        return lambda seq: list(seq)


def _install_dotnet_mocks(monkeypatch):
    clr = types.ModuleType("clr")
    clr.AddReference = lambda name: None
    system = types.ModuleType("System")
    system.Array, system.Double = _ArrayFactory(), float
    diag = types.ModuleType("System.Diagnostics")
    diag.Process, diag.ProcessStartInfo = Process, ProcessStartInfo
    monkeypatch.setitem(sys.modules, "clr", clr)
    monkeypatch.setitem(sys.modules, "System", system)
    monkeypatch.setitem(sys.modules, "System.Diagnostics", diag)


# ------------------------------------------------------------ Mechanical mocks
class _Prop:
    def __init__(self, name, value):
        self.Name, self.Value = name, value


class _Result:  # IResult of an element-nodal quantity (S, EPEL) as in Mechanical
    def __init__(self, table, mesh):
        self._t, self._mesh = table, mesh

    def SelectComponents(self, comps):
        assert list(comps) == ["X", "Y", "Z", "XY", "YZ", "XZ"]

    def GetNodeValues(self, nid):
        raise Exception("This result is not a nodal result.")

    def GetElementValues(self, eid):
        # corner-node values only, 6 components each, in CornerNodeIds order
        el = self._mesh.ElementById(eid)
        return [v for n in el.CornerNodeIds for v in self._t[n]]


class _Node:
    def __init__(self, nid, xyz, elems):
        self.Id, (self.X, self.Y, self.Z), self.ConnectedElementIds = nid, xyz, elems


class _Element:
    def __init__(self, eid, corners, mids):
        self.Id, self.CornerNodeIds, self.NodeIds = eid, corners, corners + mids


class _Mesh:  # IMeshData: a chain of quadratic 1D "elements" through the nodes
    def __init__(self, ids):
        self._el, self._nd = {}, {}
        mid0 = 100000
        for k, (a, b) in enumerate(zip(ids[:-1], ids[1:])):
            m = mid0 + k
            self._el[k + 1] = _Element(k + 1, [a, b], [m])
            self._nd.setdefault(a, (float(k), 0.0, 0.0))
            self._nd[b] = (float(k + 1), 0.0, 0.0)
            self._nd[m] = (k + 0.5, 0.0, 0.0)
        if len(ids) == 1:  # single node: one 1-node element
            self._el[1] = _Element(1, [ids[0]], [])
            self._nd[ids[0]] = (0.0, 0.0, 0.0)

    def ElementIdsFromNodeIds(self, nids):
        nids = set(nids)
        return [e.Id for e in self._el.values() if nids & set(e.NodeIds)]

    def ElementById(self, eid):
        return self._el[eid]

    def NodeById(self, nid):
        return _Node(nid, self._nd[nid], [])


class _Reader:  # IResultReader
    def __init__(self, data):
        self._d, self.CurrentResultSet, self.disposed = data, None, False
        self._mesh = _Mesh(sorted(next(iter(data.values())).keys()) if data else [])

    def GetResult(self, name):
        assert self.CurrentResultSet == 1
        return _Result(self._d[name], self._mesh)

    def CreateMeshData(self):
        return self._mesh

    def Dispose(self):
        self.disposed = True


class _Analysis:  # IAnalysis
    def __init__(self, workdir, data, unit):
        self.WorkingDir, self._data, self._unit = str(workdir), data, unit

    def GetResultsData(self):
        return _Reader(self._data)

    def CurrentConsistentUnitFromQuantityName(self, q):
        assert q == "Stress"
        return self._unit


class _UserResult:  # IUserObject / IMechanicalUserObject
    def __init__(self, analysis, props):
        self.Id, self.Analysis = 42, analysis
        self.Properties = {k: _Prop(k.split("/")[-1], v) for k, v in props.items()}


class _Collector:  # IResultCollector
    def __init__(self, ids):
        self.Ids, self.values = list(ids), {}

    def SetValues(self, nid, vals):
        self.values[nid] = list(vals)


class _Step:  # IStepInfo
    Set = 1


LOG_LINES = []


def _load_main(monkeypatch, install_dir=None):
    _install_dotnet_mocks(monkeypatch)
    LOG_LINES.clear()
    log = types.SimpleNamespace(WriteMessage=lambda m: LOG_LINES.append(("I", m)),
                                WriteWarning=lambda m: LOG_LINES.append(("W", m)),
                                WriteError=lambda m: LOG_LINES.append(("E", m)))
    ext = types.SimpleNamespace(InstallDir=str(install_dir or ACT))
    scope = {"__name__": "act_scope",
             "ExtAPI": types.SimpleNamespace(
                 Log=log, DataModel=None,
                 ExtensionManager=types.SimpleNamespace(CurrentExtension=ext))}
    for name in ("rve_bridge.py", "main.py"):  # same shared scope as the XML <script> tags
        exec(compile((ACT / name).read_text(encoding="utf-8"), name, "exec"), scope)
    return scope


@pytest.mark.skipif(not MULTI.is_file(), reason="result-10-5.hdf5 not present")
@pytest.mark.parametrize("kind,unit", [("strain", "Auto"), ("stress", "MPa")])
def test_callbacks_end_to_end(monkeypatch, tmp_path, kind, unit):
    act = _load_main(monkeypatch)
    basis = LocalizationBasis.from_file_cached(MULTI, set_indices=[0, 1, 2])
    rng = np.random.default_rng(5)
    ids = [11, 3, 250]
    eps = rng.normal(size=(3, 6)) * 2e-4
    sig_mpa = eps @ basis.effective_stiffness().T / 1e6
    data = {"EPEL": dict(zip(ids, eps)), "S": dict(zip(ids, sig_mpa))}
    analysis = _Analysis(tmp_path / "solver files", data, "MPa")
    os.makedirs(analysis.WorkingDir)
    common = {
        "Ensemble/Basis": str(MULTI), "Ensemble/Sets": "0,1,2",
        "Ensemble/Python": sys.executable, "Ensemble/Package": str(ROOT),
        "Input": act["INPUT_STRAIN"] if kind == "strain" else act["INPUT_STRESS"],
        "Quantity": "von_mises", "StressUnit": unit, "PlotNodes": "",
    }
    ref = basis.nodal_statistics("von_mises", macro_strains=eps, probs=[0.95],
                                 thresholds=[60e6])

    res = _UserResult(analysis, dict(common, Statistic="Quantile", Probability=0.95))
    col = _Collector(ids)
    act["EvaluateRVEStatistic"](res, _Step(), col)
    for j, nid in enumerate(ids):
        assert col.values[nid][0] == pytest.approx(ref["quantiles"][j, 0] / 1e6, rel=1e-6)

    res = _UserResult(analysis, dict(common, Statistic="Q(macro)", Probability=0.95))
    col = _Collector(ids)
    act["EvaluateRVEStatistic"](res, _Step(), col)
    for j, nid in enumerate(ids):
        assert col.values[nid][0] == pytest.approx(ref["q_macro"][j] / 1e6, rel=1e-6)

    res = _UserResult(analysis, dict(common, Threshold=60.0))
    col = _Collector(ids)
    act["EvaluateRVEExceedance"](res, _Step(), col)
    for j, nid in enumerate(ids):
        assert col.values[nid][0] == pytest.approx(ref["exceedance"][j, 0], abs=1e-6)

    assert (Path(analysis.WorkingDir) / "RVEStressStats" / "nodes_42_set1.csv").is_file()


def test_missing_basis_gives_clear_error(monkeypatch, tmp_path):
    act = _load_main(monkeypatch)
    analysis = _Analysis(tmp_path, {}, "Pa")
    res = _UserResult(analysis, {"Ensemble/Basis": str(tmp_path / "nope.hdf5"),
                                 "Input": act["INPUT_STRAIN"], "StressUnit": "Pa",
                                 "Statistic": "Mean", "Probability": 0.5})
    with pytest.raises(Exception, match="basis HDF5 file not found"):
        act["EvaluateRVEStatistic"](res, _Step(), _Collector([1]))


def test_cli_error_is_reported(monkeypatch, tmp_path):
    act = _load_main(monkeypatch)
    bad = tmp_path / "bad.hdf5"
    bad.write_bytes(b"not hdf5")
    analysis = _Analysis(tmp_path, {"EPEL": {1: [1e-4, 0, 0, 0, 0, 0]}}, "Pa")
    res = _UserResult(analysis, {"Ensemble/Basis": str(bad), "Ensemble/Sets": "",
                                 "Ensemble/Python": sys.executable,
                                 "Ensemble/Package": str(ROOT),
                                 "Input": act["INPUT_STRAIN"], "Quantity": "von_mises",
                                 "StressUnit": "Pa", "PlotNodes": "",
                                 "Statistic": "Mean", "Probability": 0.5})
    with pytest.raises(Exception, match="node_cli failed"):
        act["EvaluateRVEStatistic"](res, _Step(), _Collector([1]))


def test_isvalid_callbacks(monkeypatch, tmp_path):
    act = _load_main(monkeypatch)
    P = lambda v: _Prop("x", v)
    assert act["RVEOptionalValid"](None, P(""))
    assert act["RVEOptionalValid"](None, P(None))
    assert not act["RVENonEmptyValid"](None, P("  "))
    assert act["RVENonEmptyValid"](None, P("python"))
    f = tmp_path / "b.hdf5"
    f.write_bytes(b"x")
    assert act["RVEBasisValid"](None, P(str(f)))
    assert not act["RVEBasisValid"](None, P(str(tmp_path / "missing.hdf5")))
    assert not act["RVEBasisValid"](None, P(""))
    assert act["RVEProbabilityValid"](None, P(0.95))
    assert not act["RVEProbabilityValid"](None, P(1.0))
    assert not act["RVEProbabilityValid"](None, P("abc"))


@pytest.mark.skipif(not MULTI.is_file(), reason="result-10-5.hdf5 not present")
def test_blank_package_is_auto_detected_and_progress_logged(monkeypatch, tmp_path):
    act = _load_main(monkeypatch)  # extension "installed" in the repository's act folder
    analysis = _Analysis(tmp_path, {"EPEL": {1: [1e-4, 0, 0, 0, 0, 0], 2: [0, 1e-4, 0, 0, 0, 0]}}, "MPa")
    res = _UserResult(analysis, {"Ensemble/Basis": str(MULTI), "Ensemble/Sets": "0,1",
                                 "Ensemble/Python": sys.executable, "Ensemble/Package": "",
                                 "Input": act["INPUT_STRAIN"], "Quantity": "von_mises",
                                 "StressUnit": "Auto", "PlotNodes": "",
                                 "Statistic": "Mean", "Probability": 0.5})
    monkeypatch.delenv("PYTHONPATH", raising=False)
    col = _Collector([1, 2])
    act["EvaluateRVEStatistic"](res, _Step(), col)
    assert set(col.values) == {1, 2}
    text = "\n".join(m for _, m in LOG_LINES)
    assert "(auto-detected)" in text and "py> [node_cli" in text and "done: mean" in text
    logfile = tmp_path / "RVEStressStats" / "RVEStressStats.log"
    assert "von_mises: 2/2 nodes (100%)" in logfile.read_text()


def test_wrong_package_folder_gives_clear_error(monkeypatch, tmp_path):
    act = _load_main(monkeypatch, install_dir=tmp_path)
    f = tmp_path / "b.hdf5"
    f.write_bytes(b"x")
    analysis = _Analysis(tmp_path, {}, "Pa")
    res = _UserResult(analysis, {"Ensemble/Basis": str(f), "Ensemble/Package": str(tmp_path),
                                 "Input": act["INPUT_STRAIN"], "Quantity": "von_mises",
                                 "StressUnit": "Pa", "Statistic": "Mean", "Probability": 0.5})
    with pytest.raises(Exception, match="CONTAINS mv3d_stress_stats"):
        act["EvaluateRVEStatistic"](res, _Step(), _Collector([1]))
    assert any(level == "E" and "FAILED" in m for level, m in LOG_LINES)
