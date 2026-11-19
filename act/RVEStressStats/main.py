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

Progress is logged with elapsed times to the ACT log (Extensions -> View Log
File) and to <analysis working dir>/RVEStressStats/RVEStressStats.log; the
Python output is streamed line by line while it runs.

Helpers (write_nodes_csv, read_stats_csv, build_cli_args, ...) come from
rve_bridge.py, which the extension XML loads into the same scope.
"""

import os
import time
import clr

clr.AddReference("System")
from System import Array, Double
from System.Diagnostics import Process, ProcessStartInfo

LOG_NAME = "RVEStressStats.log"


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


# ------------------------------------------------------------- validation
# <isvalid> callbacks: (entity, property) -> bool. Without one, an empty text
# property is flagged as missing input (yellow) in the Details view.
def _text(prop):
    v = prop.Value
    return "" if v is None else str(v).strip()


def RVEOptionalValid(entity, prop):
    return True


def RVENonEmptyValid(entity, prop):
    return _text(prop) != ""


def RVEBasisValid(entity, prop):
    v = _text(prop)
    return v != "" and os.path.isfile(v)


def RVEProbabilityValid(entity, prop):
    try:
        return 0.0 < float(prop.Value) < 1.0
    except Exception:
        return False


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


class _Logger(object):
    """Timestamped progress log to the ACT log and to a file in the solver
    folder. Never raises: logging must not break an evaluation."""

    def __init__(self, label=""):
        self.t0 = time.time()
        self.label = label
        self.path = None

    def attach(self, folder):
        self.path = os.path.join(folder, LOG_NAME)

    def _write(self, level, msg):
        line = "[RVEStressStats]%s +%6.2fs  %s" % (
            (" " + self.label) if self.label else "", time.time() - self.t0, msg)
        try:
            log = ExtAPI.Log
            {"error": log.WriteError, "warning": log.WriteWarning}.get(level, log.WriteMessage)(line)
        except Exception:
            pass
        if self.path:
            try:
                fh = open(self.path, "a")
                try:
                    fh.write(time.strftime("%Y-%m-%d %H:%M:%S ") + line + "\n")
                finally:
                    fh.close()
            except Exception:
                pass

    def info(self, msg):
        self._write("info", msg)

    def warning(self, msg):
        self._write("warning", msg)

    def error(self, msg):
        self._write("error", msg)


_LOG = _Logger()


def _log(msg):
    _LOG.info(msg)


def _extension_dirs():
    dirs = []
    try:
        dirs.append(str(ExtAPI.ExtensionManager.CurrentExtension.InstallDir))
    except Exception:
        pass
    try:
        dirs.append(os.path.dirname(os.path.abspath(__file__)))
    except Exception:
        pass
    return dirs


def _package_root(result, log):
    explicit = str(_prop(result, "Package") or "")
    try:
        root = find_package_root(explicit, _extension_dirs())
    except ValueError as exc:
        raise Exception("RVEStressStats: %s" % exc)
    if root:
        log.info("package folder: %s%s" % (root, "" if explicit else " (auto-detected)"))
    else:
        log.warning("package folder not set and not found next to the extension; "
                    "relying on 'pip install' of mv3d_stress_stats in the selected Python")
    return root


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


def _read_macro_state(result, step_set, node_ids, kind, log):
    name = "EPEL" if kind == "strain" else "S"
    reader = result.Analysis.GetResultsData()
    try:
        reader.CurrentResultSet = int(step_set)
        res = reader.GetResult(name)
        res.SelectComponents(COMPONENTS)
        try:  # genuinely nodal results
            rows = [list(res.GetNodeValues(nid))[:6] for nid in node_ids]
            log.info("read %s at %d nodes (nodal result)" % (name, len(rows)))
            return rows
        except Exception:
            log.info("%s is element-nodal: averaging element values at nodes" % name)
        # S / EPEL are element-nodal: average over the elements sharing a node
        # (what Mechanical shows as "Averaged" nodal values)
        mesh = reader.CreateMeshData()
        eids = list(mesh.ElementIdsFromNodeIds(node_ids))
        log.info("%d elements touch the %d scoped nodes" % (len(eids), len(node_ids)))
        elements = []
        step = max(1, len(eids) // 10)
        for k, eid in enumerate(eids):
            el = mesh.ElementById(eid)
            elements.append((list(el.NodeIds), list(el.CornerNodeIds),
                             list(res.GetElementValues(eid))))
            if (k + 1) % step == 0 or k + 1 == len(eids):
                log.info("  read elements %d/%d" % (k + 1, len(eids)))
        def coords(nid):  # only asked for nodes without data and their corners
            nd = mesh.NodeById(nid)
            return (nd.X, nd.Y, nd.Z)

        info = {}
        avg = average_element_nodal(elements, node_ids, coords, stats=info)
        log.info("averaged %s at %d nodes (ignored %d 'no data' node blocks, "
                 "filled %d nodes from their edge corners)"
                 % (name, len(avg), info.get("skipped_blocks", 0), info.get("filled_midside", 0)))
        return [avg[nid] for nid in node_ids]
    finally:
        try:
            reader.Dispose()
        except Exception:
            pass


def _run_cli(python, package, args, workdir, log):
    psi = ProcessStartInfo(python, join_args(args))
    psi.UseShellExecute = False
    psi.RedirectStandardOutput = True
    psi.RedirectStandardError = True
    psi.CreateNoWindow = True
    psi.WorkingDirectory = workdir
    if package:
        old = os.environ.get("PYTHONPATH", "")
        psi.EnvironmentVariables["PYTHONPATH"] = package + (";" + old if old else "")
    log.info("run: %s %s" % (python, psi.Arguments))
    try:
        proc = Process.Start(psi)
    except Exception as exc:
        raise Exception("RVEStressStats: cannot start Python '%s' (%s); "
                        "check 'Python executable'" % (python, exc))
    err_task = proc.StandardError.ReadToEndAsync()  # drained in parallel: no deadlock
    while True:  # stream stdout to the log while Python runs
        line = proc.StandardOutput.ReadLine()
        if line is None:
            break
        if line.strip():
            log.info("  py> " + line.rstrip())
    proc.WaitForExit()
    err = (err_task.Result or "").strip()
    if proc.ExitCode != 0:
        hint = ""
        if "No module named" in err and PACKAGE_NAME in err:
            hint = ("\nHint: set 'Package folder' to the folder that CONTAINS %s "
                    "(the repository root), or pip-install the package into %s."
                    % (PACKAGE_NAME, python))
        elif "No module named" in err:
            hint = "\nHint: the selected Python lacks a dependency (numpy, h5py, matplotlib)."
        log.error("node_cli failed (exit %d): %s" % (proc.ExitCode, err))
        raise Exception("RVEStressStats: node_cli failed (exit %d)\n%s%s"
                        % (proc.ExitCode, err, hint))
    if err:
        log.warning("python stderr: " + err)
    log.info("python finished")


def _evaluate(result, stepInfo, collector, column, probs, thresholds):
    global _LOG
    log = _LOG = _Logger("obj %s set %s" % (result.Id, stepInfo.Set))
    node_ids = [int(i) for i in collector.Ids]
    log.info("evaluate %s: %d nodes, column %s" % (_caption(result), len(node_ids), column))
    if not node_ids:
        log.warning("no nodes in scope; nothing to do")
        return
    try:
        kind = "strain" if str(_prop(result, "Input")) == INPUT_STRAIN else "stress"
        unit = _solver_stress_unit(result)
        basis = str(_prop(result, "Basis") or "")
        if not basis or not os.path.isfile(basis):
            raise Exception("RVEStressStats: basis HDF5 file not found: '%s'" % basis)
        quantity = str(_prop(result, "Quantity"))
        log.info("input %s, quantity %s, stress unit %s, basis %s" % (kind, quantity, unit, basis))

        workdir = os.path.join(result.Analysis.WorkingDir, "RVEStressStats")
        if not os.path.isdir(workdir):
            os.makedirs(workdir)
        log.attach(workdir)
        log.info("work folder %s" % workdir)
        package = _package_root(result, log)

        tag = "%s_set%s" % (result.Id, stepInfo.Set)
        nodes_csv = os.path.join(workdir, "nodes_%s.csv" % tag)
        out_csv = os.path.join(workdir, "stats_%s.csv" % tag)

        rows = _read_macro_state(result, stepInfo.Set, node_ids, kind, log)
        write_nodes_csv(nodes_csv, node_ids, rows)
        log.info("wrote %s" % nodes_csv)

        args = build_cli_args(
            basis, nodes_csv, out_csv, kind, quantity, unit,
            probs=probs, thresholds=thresholds, sets=str(_prop(result, "Sets") or ""),
            plot_nodes=str(_prop(result, "PlotNodes") or ""),
            plot_dir=os.path.join(workdir, "plots"),
        )
        _run_cli(str(_prop(result, "Python") or "python"), package, args, workdir, log)

        values = read_stats_csv(out_csv, column)
        vals, skipped = [], 0
        for nid in node_ids:
            v = values[nid]
            if v != v or abs(v) == float("inf"):  # NaN / inf: leave the node empty
                skipped += 1
                continue
            collector.SetValues(nid, Array[Double]([v]))
            vals.append(v)
        if skipped:
            log.warning("%d nodes without a valid value were left empty" % skipped)
        if vals:
            log.info("done: %s min %.6g max %.6g over %d nodes"
                     % (column, min(vals), max(vals), len(vals)))
    except Exception as exc:
        log.error("FAILED: %s" % exc)
        raise


def _caption(result):
    try:
        return "'%s'" % result.Caption
    except Exception:
        return "result"


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
    # quantiles are the expensive part: only request them when shown
    _evaluate(result, stepInfo, collector, column,
              probs=[p] if stat == "Quantile" else [], thresholds=[])


def EvaluateRVEExceedance(result, stepInfo, collector):
    t = float(_prop(result, "Threshold"))
    _evaluate(result, stepInfo, collector, exceedance_column(t), probs=[], thresholds=[t])
