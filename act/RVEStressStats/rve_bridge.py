# -*- coding: utf-8 -*-
"""
Pure-Python helpers of the RVEStressStats ACT extension.

Runs unchanged in IronPython 2.7 (inside Mechanical) and CPython 3 (tests):
no numpy, no f-strings, no csv module dependence on C extensions.
"""

COMPONENTS = ["X", "Y", "Z", "XY", "YZ", "XZ"]
STRESS_UNITS = ["Pa", "kPa", "MPa", "GPa"]
INPUT_STRAIN = "Elastic strain (EPEL)"
INPUT_STRESS = "Stress (S)"

# Statistic caption -> column of the node_cli output CSV (Quantile is built from p)
STAT_COLUMNS = {
    "Mean": "mean",
    "Std": "std",
    "Min": "min",
    "Max": "max",
    "Q(macro)": "q_macro",
}


def fmt_number(x):
    """Same text as Python's '{:g}' in node_cli (column names q<p>, P_gt_<t>)."""
    return "%g" % float(x)


def quantile_column(prob):
    return "q" + fmt_number(prob)


def exceedance_column(threshold):
    return "P_gt_" + fmt_number(threshold)


def normalize_stress_unit(unit_text):
    """Map a Mechanical unit string ('MPa', 'Pa', 'N/mm^2', ...) to a CLI unit."""
    if unit_text is None:
        return None
    u = str(unit_text).strip().replace(" ", "")
    aliases = {
        "Pa": "Pa", "N/m^2": "Pa", "N/m²": "Pa",
        "kPa": "kPa",
        "MPa": "MPa", "N/mm^2": "MPa", "N/mm²": "MPa",
        "GPa": "GPa",
    }
    return aliases.get(u)


def write_nodes_csv(path, node_ids, rows):
    """node_id,X,Y,Z,XY,YZ,XZ (engineering shear for strains)."""
    lines = ["node_id," + ",".join(COMPONENTS)]
    for nid, vals in zip(node_ids, rows):
        if len(vals) != 6:
            raise ValueError("node %s: expected 6 components, got %d" % (nid, len(vals)))
        lines.append(str(nid) + "," + ",".join(["%.12g" % float(v) for v in vals]))
    fh = open(path, "w")
    try:
        fh.write("\n".join(lines) + "\n")
    finally:
        fh.close()


def read_stats_csv(path, column):
    """Return {node_id(int): value} for one column of the node_cli output."""
    fh = open(path, "r")
    try:
        text = fh.read()
    finally:
        fh.close()
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        raise ValueError("empty statistics file: " + path)
    header = [h.strip() for h in lines[0].lstrip(u"﻿").split(",")]
    if column not in header:
        raise ValueError("column '%s' not in %s (have: %s)" % (column, path, ", ".join(header)))
    j = header.index(column)
    out = {}
    for ln in lines[1:]:
        cells = ln.split(",")
        out[int(cells[0])] = float(cells[j])
    return out


def split_ids(text):
    return [t.strip() for t in str(text or "").replace(";", ",").split(",") if t.strip()]


def build_cli_args(basis, nodes_csv, out_csv, kind, quantity, stress_unit,
                   probs=None, thresholds=None, sets="", plot_nodes="", plot_dir=""):
    """Argument list for `python -m mv3d_stress_stats.node_cli ...`."""
    if kind not in ("strain", "stress"):
        raise ValueError("kind must be 'strain' or 'stress'")
    if stress_unit not in STRESS_UNITS:
        raise ValueError("unsupported stress unit: %s" % stress_unit)
    args = ["-m", "mv3d_stress_stats.node_cli",
            "--basis", basis,
            "--input", nodes_csv,
            "--out", out_csv,
            "--kind", kind,
            "--quantity", quantity,
            "--stress-unit", stress_unit,
            "--probs", ",".join([fmt_number(p) for p in (probs or [])]),
            "--thresholds", ",".join([fmt_number(t) for t in (thresholds or [])])]
    if split_ids(sets):
        args += ["--sets", ",".join(split_ids(sets))]
    if split_ids(plot_nodes):
        args += ["--plot-nodes", ",".join(split_ids(plot_nodes)), "--plot-dir", plot_dir]
    return args


def quote_arg(a):
    """Windows command-line quoting for ProcessStartInfo.Arguments."""
    a = str(a)
    if a and not any(c in a for c in ' \t"'):
        return a
    out, bs = ['"'], 0
    for c in a:
        if c == "\\":
            bs += 1
            continue
        if c == '"':
            out.append("\\" * (2 * bs + 1) + '"')
        else:
            out.append("\\" * bs + c)
        bs = 0
    out.append("\\" * (2 * bs) + '"')
    return "".join(out)


def join_args(args):
    return " ".join([quote_arg(a) for a in args])
