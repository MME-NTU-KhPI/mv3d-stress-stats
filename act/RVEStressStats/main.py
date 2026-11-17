# -*- coding: utf-8 -*-
"""
RVEStressStats - Mechanical callbacks (IronPython 2.7).

On evaluation of a result object:
  1. read the macro state (EPEL or S, 6 components) at the scoped nodes of the
     current result set,
  2. write it to <analysis working dir>/RVEStressStats/nodes_<id>.csv,
  3. run CPython:  python -m mv3d_stress_stats.node_cli ...  (numpy/h5py live
     there; IronPython cannot load them),
  4. read the requested statistic back and hand it to the collector.

Helpers (write_nodes_csv, read_stats_csv, build_cli_args, ...) come from
rve_bridge.py, which the extension XML loads into the same scope.
"""

import os
import clr

clr.AddReference("System")
from System import Array, Double
from System.Diagnostics import Process, ProcessStartInfo


# ------------------------------------------------------------------ toolbar
def _create(analysis, name):
    try:
        analysis.CreateResultObject(name, ExtAPI.ExtensionManager.CurrentExtension)
    except Exception:
        analysis.CreateResultObject(name)


def AddRVEStatistic(analysis):
    _create(analysis, "RVEStatistic")


def AddRVEExceedance(analysis):
    _create(analysis, "RVEExceedance")


# ------------------------------------------------------------------ helpers
def _prop(result, name):
    for key in ("Ensemble/" + name, name):
        try:
            p = result.Properties[key]
        except Exception:
            p = None
        if p is not None:
            return p.Value
    try:  # fallback: search the flattened property list by name
        for p in result.AllProperties:
            if p.Name == name or str(p.Name).endswith("/" + name):
                return p.Value
    except Exception:
        pass
    raise Exception("RVEStressStats: property '%s' not found" % name)


def _log(msg):
    try:
        ExtAPI.Log.WriteMessage("[RVEStressStats] " + msg)
    except Exception:
        pass


def _solver_stress_unit(result):
    chosen = str(_prop(result, "StressUnit"))
    if chosen != "Auto":
        return chosen
    text = None
    for owner in (result.Analysis, ExtAPI.DataModel):
        try:
            text = owner.CurrentConsistentUnitFromQuantityName("Stress")
            break
        except Exception:
            pass
    unit = normalize_stress_unit(text)
    if unit is None:
        raise Exception("RVEStressStats: cannot map solver stress unit '%s'; "
                        "set 'Solver stress unit' explicitly." % text)
    return unit


def _read_macro_state(result, step_set, node_ids, kind):
    reader = result.Analysis.GetResultsData()
    try:
        reader.CurrentResultSet = int(step_set)
        res = reader.GetResult("EPEL" if kind == "strain" else "S")
        res.SelectComponents(COMPONENTS)
        rows = []
        for nid in node_ids:
            vals = list(res.GetNodeValues(nid))
            rows.append(vals[:6])
        return rows
    finally:
        try:
            reader.Dispose()
        except Exception:
            pass


def _run_cli(python, package, args, workdir):
    psi = ProcessStartInfo(python, join_args(args))
    psi.UseShellExecute = False
    psi.RedirectStandardOutput = True
    psi.RedirectStandardError = True
    psi.CreateNoWindow = True
    psi.WorkingDirectory = workdir
    if package:
        old = os.environ.get("PYTHONPATH", "")
        psi.EnvironmentVariables["PYTHONPATH"] = package + (";" + old if old else "")
    proc = Process.Start(psi)
    err_task = proc.StandardError.ReadToEndAsync()
    out = proc.StandardOutput.ReadToEnd()
    proc.WaitForExit()
    err = err_task.Result
    if out.strip():
        _log(out.strip())
    if proc.ExitCode != 0:
        raise Exception("RVEStressStats: node_cli failed (exit %d)\n%s" % (proc.ExitCode, err.strip()))
    if err.strip():
        _log(err.strip())


def _evaluate(result, stepInfo, collector, column, probs, thresholds):
    node_ids = [int(i) for i in collector.Ids]
    if not node_ids:
        return
    kind = "strain" if str(_prop(result, "Input")) == INPUT_STRAIN else "stress"
    unit = _solver_stress_unit(result)
    basis = str(_prop(result, "Basis"))
    if not basis or not os.path.isfile(basis):
        raise Exception("RVEStressStats: basis HDF5 file not found: '%s'" % basis)

    workdir = os.path.join(result.Analysis.WorkingDir, "RVEStressStats")
    if not os.path.isdir(workdir):
        os.makedirs(workdir)
    tag = "%s_set%s" % (result.Id, stepInfo.Set)
    nodes_csv = os.path.join(workdir, "nodes_%s.csv" % tag)
    out_csv = os.path.join(workdir, "stats_%s.csv" % tag)

    rows = _read_macro_state(result, stepInfo.Set, node_ids, kind)
    write_nodes_csv(nodes_csv, node_ids, rows)

    args = build_cli_args(
        basis, nodes_csv, out_csv, kind, str(_prop(result, "Quantity")), unit,
        probs=probs, thresholds=thresholds, sets=str(_prop(result, "Sets") or ""),
        plot_nodes=str(_prop(result, "PlotNodes") or ""),
        plot_dir=os.path.join(workdir, "plots"),
    )
    _run_cli(str(_prop(result, "Python") or "python"), str(_prop(result, "Package") or ""),
             args, workdir)

    values = read_stats_csv(out_csv, column)
    for nid in node_ids:
        collector.SetValues(nid, Array[Double]([values[nid]]))


# ------------------------------------------------------------------ results
def EvaluateRVEStatistic(result, stepInfo, collector):
    stat = str(_prop(result, "Statistic"))
    p = float(_prop(result, "Probability"))
    if stat == "Quantile":
        if not (0.0 < p < 1.0):
            raise Exception("RVEStressStats: quantile level must be in (0, 1)")
        column = quantile_column(p)
    else:
        column = STAT_COLUMNS[stat]
    _evaluate(result, stepInfo, collector, column, probs=[p], thresholds=[])


def EvaluateRVEExceedance(result, stepInfo, collector):
    t = float(_prop(result, "Threshold"))
    _evaluate(result, stepInfo, collector, exceedance_column(t), probs=[0.5], thresholds=[t])
