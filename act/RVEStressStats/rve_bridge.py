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


NO_DATA_LIMIT = 1e30  # ANSYS reader returns ~DBL_MAX (1.8e308) where a node has no value


def _valid_block(values, start, ncomp):
    for c in range(ncomp):
        v = float(values[start + c])
        if not (abs(v) <= NO_DATA_LIMIT):  # also rejects inf and nan
            return False
    return True


def average_element_nodal(elements, wanted_ids, coords, ncomp=6, stats=None):
    """Nodal average of an element-nodal result (ANSYS S / EPEL).

    elements: iterable of (node_ids, corner_node_ids, values) per element;
        ``values`` is flat, ``ncomp`` numbers per node, either for all
        ``node_ids`` or only for ``corner_node_ids`` (in that order).
        Node blocks holding the reader's "no data" marker (|v| > 1e30, inf,
        nan) are ignored.
    wanted_ids: nodes to return.
    coords: {node_id: (x, y, z)} or a callable node_id -> (x, y, z). Only
        called for wanted nodes without data and the corner nodes of their
        elements: such a (midside) node gets the mean of the two corner nodes
        of its edge (corner pair whose midpoint is closest to the node).
    stats: optional dict filled with counters (skipped_blocks, filled_midside).

    Returns {node_id: [ncomp floats]}; raises ValueError if a wanted node gets
    no value.
    """
    get_xyz = coords if callable(coords) else coords.get
    sums, counts, elems = {}, {}, []
    skipped = 0
    for node_ids, corner_ids, values in elements:
        k = len(values) // ncomp
        if k * ncomp != len(values) or k == 0:
            raise ValueError("element values length %d is not a multiple of %d"
                             % (len(values), ncomp))
        if k == len(node_ids):
            owners = node_ids
        elif k == len(corner_ids):
            owners = corner_ids
        else:
            owners = list(node_ids)[:k]
        for a, nid in enumerate(owners):
            if not _valid_block(values, a * ncomp, ncomp):
                skipped += 1
                continue
            s = sums.get(nid)
            if s is None:
                s = sums[nid] = [0.0] * ncomp
                counts[nid] = 0
            for c in range(ncomp):
                s[c] += float(values[a * ncomp + c])
            counts[nid] += 1
        elems.append((node_ids, corner_ids))

    avg = {}
    for nid, s in sums.items():
        avg[nid] = [v / counts[nid] for v in s]

    out = {}
    missing = []
    for nid in wanted_ids:
        if nid in avg:
            out[nid] = avg[nid]
        else:
            missing.append(nid)

    if missing:
        miss_set = set(missing)
        owners_of = {}
        for node_ids, corner_ids in elems:
            for nid in node_ids:
                if nid in miss_set:
                    owners_of.setdefault(nid, []).append(corner_ids)
        xyz_cache = {}

        def xyz(n):
            if n not in xyz_cache:
                xyz_cache[n] = get_xyz(n)
            return xyz_cache[n]

        for nid in missing:
            x = xyz(nid)
            best = None
            for corner_ids in owners_of.get(nid, []):
                corners = [c for c in corner_ids if c in avg]
                for i in range(len(corners)):
                    a = xyz(corners[i])
                    for j in range(i + 1, len(corners)):
                        b = xyz(corners[j])
                        d = sum([((a[t] + b[t]) * 0.5 - x[t]) ** 2 for t in range(3)])
                        if best is None or d < best[0]:
                            best = (d, corners[i], corners[j])
            if best is None or x is None:
                raise ValueError("no element-nodal value for node %s" % nid)
            va, vb = avg[best[1]], avg[best[2]]
            out[nid] = [(va[c] + vb[c]) * 0.5 for c in range(ncomp)]

    if stats is not None:
        stats["skipped_blocks"] = skipped
        stats["filled_midside"] = len(missing)
    return out


PACKAGE_NAME = "mv3d_stress_stats"


def _is_package_root(folder):
    import os
    return bool(folder) and os.path.isfile(os.path.join(folder, PACKAGE_NAME, "__init__.py"))


def find_package_root(explicit="", start_dirs=(), max_up=6):
    """Folder to put on PYTHONPATH so that `import mv3d_stress_stats` works.

    explicit: the user's 'Package folder'. Accepted as the parent of the
        package or as the package folder itself. If set but wrong, raises.
    start_dirs: folders to search upward from when ``explicit`` is blank
        (the extension's install folder: works when the extension is loaded
        straight from the repository).
    Returns the folder, or "" when nothing is found (rely on pip install).
    """
    import os
    explicit = (explicit or "").strip().strip('"')
    if explicit:
        cand = os.path.normpath(explicit)
        if _is_package_root(cand):
            return cand
        parent = os.path.dirname(cand)
        if os.path.basename(cand) == PACKAGE_NAME and _is_package_root(parent):
            return parent
        raise ValueError("Package folder '%s' does not contain %s\\__init__.py; "
                         "select the folder that CONTAINS %s (the repository root)"
                         % (explicit, PACKAGE_NAME, PACKAGE_NAME))
    for start in start_dirs:
        cur = os.path.normpath(start) if start else ""
        for _ in range(max_up + 1):
            if not cur:
                break
            if _is_package_root(cur):
                return cur
            nxt = os.path.dirname(cur)
            if nxt == cur:
                break
            cur = nxt
    return ""


def split_ids(text):
    return [t.strip() for t in str(text or "").replace(";", ",").split(",") if t.strip()]


def build_cli_args(basis, nodes_csv, out_csv, kind, quantity, stress_unit,
                   probs=None, thresholds=None, sets="", plot_nodes="", plot_dir=""):
    """Argument list for `python -m mv3d_stress_stats.node_cli ...`."""
    if kind not in ("strain", "stress"):
        raise ValueError("kind must be 'strain' or 'stress'")
    if stress_unit not in STRESS_UNITS:
        raise ValueError("unsupported stress unit: %s" % stress_unit)
    args = ["-u",  # unbuffered: progress lines reach the ACT log while running
            "-m", "mv3d_stress_stats.node_cli",
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
