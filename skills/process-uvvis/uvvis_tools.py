#!/usr/bin/env python
"""
uvvis_tools.py -- helpers and command-line interface for the ``process-uvvis``
Claude Code skill.

Reads raw UV-vis exports (Agilent Cary multi-column CSV, or simple x/y
CSV/TXT files), subtracts solvent blanks, converts absorbance to molar
absorptivity, and writes processed .txt files, figures, an interactive HTML
viewer and a Jupyter notebook that documents every step.

Command line
------------
    python uvvis_tools.py check-env
    python uvvis_tools.py find    <folder>
    python uvvis_tools.py inspect <data file>
    python uvvis_tools.py build   <processing_config.json> [--no-execute]

Only numpy is needed to read and process data; matplotlib, plotly and
nbformat/nbconvert are imported when figures, HTML or notebooks are made.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

__version__ = "1.0.0"

HC_EV_NM = 1239.84198  # h*c in eV nm: E(eV) = 1239.84198 / lambda(nm)
COLORS = ['#000000', '#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7']
OUTPUT_MARKER = "Processed UV-vis spectrum (process-uvvis)"

X_UNITS = ("nm", "eV", "cm-1")
X_LABELS = {  # matplotlib (mathtext) labels
    "nm": "Wavelength (nm)",
    "eV": "Energy (eV)",
    "cm-1": "Wavenumber (cm$^{-1}$)",
}
X_LABELS_HTML = {
    "nm": "Wavelength (nm)",
    "eV": "Energy (eV)",
    "cm-1": "Wavenumber (cm⁻¹)",
}

# Instrument metadata worth repeating in output headers / notebook summaries.
SUMMARY_KEYS = (
    "Collection Time", "Instrument", "Start (nm)", "Stop (nm)", "X Mode", "Y Mode",
    "UV-Vis Scan Rate (nm/min)", "UV-Vis Data Interval (nm)", "UV-Vis Ave. Time (sec)",
    "Beam Mode", "Baseline Correction",
)

# Solvent densities in g/mL at 20-25 C (CRC Handbook / supplier data sheets).
# canonical name: (density, aliases matched case-insensitively in sample names)
SOLVENTS = {
    "DMF": (0.944, ["dmf", "dimethylformamide"]),
    "DMSO": (1.100, ["dmso"]),
    "MeCN": (0.786, ["mecn", "acn", "ch3cn", "acetonitrile"]),
    "MeOH": (0.792, ["meoh", "methanol"]),
    "EtOH": (0.789, ["etoh", "ethanol"]),
    "iPrOH": (0.786, ["iproh", "ipa", "isopropanol", "2-propanol"]),
    "H2O": (0.997, ["h2o", "water", "d2o_free"]),
    "D2O": (1.107, ["d2o"]),
    "THF": (0.889, ["thf", "tetrahydrofuran"]),
    "2-MeTHF": (0.854, ["2-methf", "methf", "2methf"]),
    "DCM": (1.326, ["dcm", "ch2cl2", "dichloromethane"]),
    "CHCl3": (1.489, ["chcl3", "chloroform"]),
    "CCl4": (1.594, ["ccl4"]),
    "DCE": (1.253, ["dce", "1,2-dichloroethane"]),
    "toluene": (0.867, ["toluene", "tol", "phme"]),
    "benzene": (0.876, ["benzene", "c6h6"]),
    "PhCl": (1.106, ["phcl", "chlorobenzene"]),
    "PhF": (1.024, ["phf", "fluorobenzene"]),
    "PhCN": (1.000, ["phcn", "benzonitrile"]),
    "hexane": (0.655, ["hexane", "hexanes", "hex"]),
    "pentane": (0.626, ["pentane", "pentanes"]),
    "heptane": (0.684, ["heptane"]),
    "cyclohexane": (0.779, ["cyclohexane", "cyhex"]),
    "acetone": (0.784, ["acetone"]),
    "EtOAc": (0.902, ["etoac", "ethylacetate"]),
    "Et2O": (0.713, ["et2o", "ether", "diethylether"]),
    "DME": (0.868, ["dme", "dimethoxyethane"]),
    "dioxane": (1.033, ["dioxane"]),
    "DMA": (0.937, ["dma", "dmac", "dimethylacetamide"]),
    "NMP": (1.028, ["nmp"]),
    "pyridine": (0.982, ["pyridine", "py"]),
    "MeNO2": (1.137, ["meno2", "nitromethane"]),
    "PC": (1.205, ["pc", "propylenecarbonate"]),
}
_SOLVENT_ALIASES = {}
for _name, (_rho, _aliases) in SOLVENTS.items():
    for _a in [_name] + _aliases:
        _SOLVENT_ALIASES[_a.lower()] = _name
_SOLVENT_ALIASES.pop("d2o_free", None)

BLANK_WORDS = {"blank", "solvent", "baseline", "ref", "reference", "background", "bkg", "bg"}
NEUTRAL_WORDS = {"neat", "pure", "rt", "dry", "anhydrous"}


def lookup_solvent(token):
    """Return the canonical solvent name for ``token`` (case-insensitive) or None."""
    return _SOLVENT_ALIASES.get(str(token).strip().lower())


def solvent_density(name):
    """Density (g/mL) of a solvent from the built-in table, or None if unknown."""
    canonical = lookup_solvent(name)
    return SOLVENTS[canonical][0] if canonical else None


# --------------------------------------------------------------------------
# Data containers
# --------------------------------------------------------------------------
@dataclass
class Spectrum:
    """One raw spectrum. ``x_nm`` is ascending wavelength in nm."""
    name: str
    x_nm: np.ndarray
    absorbance: np.ndarray
    metadata: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)


@dataclass
class Processed:
    """A solvent-subtracted spectrum (arrays already trimmed to the usable range)."""
    name: str
    x_nm: np.ndarray
    abs_raw: np.ndarray
    abs_blank: np.ndarray
    abs_corr: np.ndarray
    flagged: np.ndarray
    blank_name: str | None = None
    pathlength_cm: float | None = None
    conc_M: float | None = None
    conc_steps: list = field(default_factory=list)
    epsilon: np.ndarray | None = None

    @property
    def abs_norm(self):
        """Corrected absorbance normalised to its largest reliable value."""
        good = self.abs_corr[~self.flagged] if (~self.flagged).any() else self.abs_corr
        peak = np.nanmax(good) if np.isfinite(good).any() else np.nan
        return self.abs_corr / peak if peak and np.isfinite(peak) else self.abs_corr * np.nan


# --------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------
def _decode(raw: bytes) -> str:
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16")
    for enc in ("utf-8-sig", "cp1252"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            pass
    return raw.decode("latin-1")


def _is_num(cell: str) -> bool:
    try:
        float(cell)
        return True
    except ValueError:
        return False


def _is_data_row(row) -> bool:
    cells = [c for c in row if c != ""]
    return len(cells) >= 2 and all(_is_num(c) for c in cells)


def _split_lines(lines):
    sample = "\n".join(l for l in lines[:200] if not l.lstrip().startswith("#"))
    for delim in (",", "\t", ";"):
        if delim in sample:
            return [[c.strip() for c in r] for r in csv.reader(lines, delimiter=delim)]
    return [l.split() for l in lines]


_X_LIKE = re.compile(r"wavelength|wavenumber|energy|\bnm\b|cm-1|cm\^-1|1/cm|\bev\b|^x$", re.I)
_GENERIC_Y = {"abs", "absorbance", "a", "y", "%t", "t%", "transmittance", "intensity", "od"}


def _x_unit_from_label(label: str) -> str:
    lab = label.lower()
    if re.search(r"wavenumber|cm-1|cm\^-1|1/cm|cm⁻¹", lab):
        return "cm-1"
    if re.search(r"\bev\b|energy", lab):
        return "eV"
    return "nm"


def _parse_kv(line: str):
    m = re.match(r"^Saved as\s+(.*)$", line)
    if m:
        return "Saved as", m.group(1).strip()
    if ":" in line:
        k, v = line.split(":", 1)
    elif re.search(r"\s{2,}", line):
        k, v = re.split(r"\s{2,}", line, maxsplit=1)
    elif "," in line:
        k, v = line.split(",", 1)
    else:
        k, v = line, ""
    return k.strip(), v.strip().strip(",").strip()


def read_uvvis(path) -> list[Spectrum]:
    """
    Read a UV-vis export and return a list of :class:`Spectrum`.

    Supports Agilent Cary multi-column CSV (row 1 = spectrum names, row 2 =
    ``Wavelength (nm),Abs`` pairs, per-spectrum metadata footer) as well as
    generic files with x/y column pairs or one x column and several y columns.
    The x axis is converted to nm (from eV or cm-1 if the header says so) and
    %T is converted to absorbance. Each spectrum is sorted by ascending nm.
    """
    path = Path(path)
    lines = _decode(path.read_bytes()).splitlines()
    rows = _split_lines(lines)

    start = next((i for i in range(len(rows) - 1)
                  if _is_data_row(rows[i]) and _is_data_row(rows[i + 1])), None)
    if start is None:
        raise ValueError(f"{path.name}: no block of numeric data found")
    end = start
    while end < len(rows) and _is_data_row(rows[end]):
        end += 1

    ncol = max(len(r) for r in rows[start:end])
    data = np.full((end - start, ncol), np.nan)
    for i, r in enumerate(rows[start:end]):
        for j, c in enumerate(r):
            if c != "":
                data[i, j] = float(c)
    cols = [j for j in range(ncol) if np.isfinite(data[:, j]).any()]

    header = [[c.lstrip("#").strip() for c in r] for r in rows[:start]]
    labels = header[-1] if header else []
    names_row = header[-2] if len(header) >= 2 else []

    def cell(row, j):
        return row[j] if j < len(row) else ""

    # Decide between (x, y) pairs and one x column with several y columns.
    paired = len(cols) == 2
    if len(cols) > 2 and len(cols) % 2 == 0:
        xs = [cols[k] for k in range(0, len(cols), 2)]
        if all(_X_LIKE.search(cell(labels, j)) for j in xs):
            paired = True
        elif not any(cell(labels, j) for j in cols):
            x0 = data[:, cols[0]]
            paired = all(np.allclose(data[:, j], x0, equal_nan=True, rtol=1e-3) for j in xs)
    if paired:
        pairs = [(cols[k], cols[k + 1]) for k in range(0, len(cols) - 1, 2)]
    else:
        pairs = [(cols[0], j) for j in cols[1:]]

    spectra, seen = [], {}
    for k, (jx, jy) in enumerate(pairs):
        name = ""
        for cand in (cell(names_row, jx) if paired else "", cell(names_row, jy), cell(labels, jy)):
            if cand and cand.lower() not in _GENERIC_Y and not _X_LIKE.search(cand):
                name = cand
                break
        if not name:
            name = path.stem if len(pairs) == 1 else f"{path.stem}_{k + 1}"
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 1

        x, y = data[:, jx], data[:, jy]
        ok = np.isfinite(x) & np.isfinite(y)
        x, y = x[ok], y[ok]
        notes = []
        unit = _x_unit_from_label(cell(labels, jx))
        if unit != "nm":
            with np.errstate(divide="ignore"):
                x = convert_x(x, unit)  # the conversions are their own inverse
            notes.append(f"x axis converted from {unit} to nm")
        ylab = cell(labels, jy).lower()
        if "%t" in ylab or "t%" in ylab or "transmittance" in ylab:
            with np.errstate(divide="ignore", invalid="ignore"):
                y = 2.0 - np.log10(y)
            notes.append("converted from %T to absorbance: A = 2 - log10(%T)")
        order = np.argsort(x)
        spectra.append(Spectrum(name=name, x_nm=x[order], absorbance=y[order], notes=notes))

    # Footer metadata (Cary): a block per spectrum starting with its name.
    by_name = {s.name: s for s in spectra}
    current = None
    for line in lines[end:]:
        s = line.strip().rstrip(",").strip()
        if not s:
            continue
        if s in by_name:
            current = by_name[s]
            continue
        if current is None:
            continue
        k, v = _parse_kv(s)
        if k and k not in current.metadata:
            current.metadata[k] = v
    return spectra


# --------------------------------------------------------------------------
# Sample names: concentration / pathlength / solvent guesses
# --------------------------------------------------------------------------
_MASS = re.compile(r"^(\d*\.?\d+)(mg|ug|µg|μg|ng|g)$")
_VOL = re.compile(r"^(\d*\.?\d+)(mL|ml|uL|ul|µL|μL|L)$")
_CONC = re.compile(r"^(\d*\.?\d+)(M|mM|uM|µM|μM|nM)$")
_PATH = re.compile(r"^(\d*\.?\d+)(cm|mm)$")
_MASS_TO_MG = {"g": 1e3, "mg": 1.0, "ug": 1e-3, "µg": 1e-3, "μg": 1e-3, "ng": 1e-6}
_VOL_TO_ML = {"L": 1e3, "mL": 1.0, "ml": 1.0, "uL": 1e-3, "ul": 1e-3, "µL": 1e-3, "μL": 1e-3}
_CONC_TO_M = {"M": 1.0, "mM": 1e-3, "uM": 1e-6, "µM": 1e-6, "μM": 1e-6, "nM": 1e-9}


def parse_name(name: str) -> dict:
    """
    Guess quantities encoded in a spectrum name such as
    ``Cu_imnt_1.8mg_3.42g_DMF_0.1cm``. These are only guesses; always confirm
    with the user. Returns a dict with keys solute_mass_mg, solvent_mass_g,
    solution_volume_mL, conc_M, pathlength_cm, solvent, solvent_density_g_mL,
    blank_keyword and other_tokens.
    """
    out = dict(solute_mass_mg=None, solvent_mass_g=None, solution_volume_mL=None,
               conc_M=None, pathlength_cm=None, solvent=None, solvent_density_g_mL=None,
               blank_keyword=False, other_tokens=[])
    masses = []  # (value_mg, unit)
    for tok in re.split(r"[_\s,;()\[\]]+", name):
        if not tok:
            continue
        t = re.sub(r"(\d)p(\d)", r"\1.\2", tok)  # 0p1cm -> 0.1cm
        if m := _MASS.match(t):
            masses.append((float(m.group(1)) * _MASS_TO_MG[m.group(2)], m.group(2)))
        elif m := _VOL.match(t):
            out["solution_volume_mL"] = float(m.group(1)) * _VOL_TO_ML[m.group(2)]
        elif m := _CONC.match(t):
            out["conc_M"] = float(m.group(1)) * _CONC_TO_M[m.group(2)]
        elif m := _PATH.match(t):
            out["pathlength_cm"] = float(m.group(1)) * (0.1 if m.group(2) == "mm" else 1.0)
        elif lookup_solvent(t) and out["solvent"] is None:
            out["solvent"] = lookup_solvent(t)
        elif t.lower() in BLANK_WORDS:
            out["blank_keyword"] = True
        elif re.fullmatch(r"[\d.\-/]+", t) or t.lower() in NEUTRAL_WORDS:
            pass  # dates, run numbers, neutral words
        else:
            out["other_tokens"].append(tok)

    small = [v for v, u in masses if u != "g"]
    grams = sorted(v for v, u in masses if u == "g")
    if small:
        out["solute_mass_mg"] = small[0]
        if grams:
            out["solvent_mass_g"] = grams[-1] / 1e3
    elif len(grams) >= 2:
        out["solute_mass_mg"], out["solvent_mass_g"] = grams[0], grams[-1] / 1e3
    elif len(grams) == 1:
        if out["solvent"] and grams[0] >= 100:  # >= 0.1 g with a solvent named: solvent mass
            out["solvent_mass_g"] = grams[0] / 1e3
        else:
            out["solute_mass_mg"] = grams[0]
    if out["solvent"]:
        out["solvent_density_g_mL"] = solvent_density(out["solvent"])
    return out


def is_blank(name: str) -> bool:
    """True if a spectrum name looks like a solvent/blank (e.g. ``DMF_0.1cm``)."""
    p = parse_name(name)
    if p["blank_keyword"]:
        return True
    has_solute = p["solute_mass_mg"] is not None or p["conc_M"] is not None
    return not has_solute and p["solvent"] is not None and not p["other_tokens"]


def pair_blanks(spectra) -> dict:
    """Propose a blank for every non-blank spectrum: {sample_name: blank_name or None}."""
    blanks = [s.name for s in spectra if is_blank(s.name)]
    pairs = {}
    for s in spectra:
        if s.name in blanks:
            continue
        ps = parse_name(s.name)
        best, best_score = None, -1
        for b in blanks:
            pb = parse_name(b)
            score = 0
            for key in ("solvent", "pathlength_cm"):
                if ps[key] is not None and pb[key] is not None:
                    score += 2 if ps[key] == pb[key] else -3
            if score > best_score:
                best, best_score = b, score
        pairs[s.name] = best if (best is not None and (best_score >= 0 or len(blanks) == 1)) else None
    return pairs


# --------------------------------------------------------------------------
# Processing
# --------------------------------------------------------------------------
def convert_x(x_nm, unit: str):
    """Convert wavelength in nm to ``unit`` ("nm", "eV" or "cm-1")."""
    x_nm = np.asarray(x_nm, dtype=float)
    if unit == "nm":
        return x_nm
    if unit == "eV":
        return HC_EV_NM / x_nm
    if unit == "cm-1":
        return 1e7 / x_nm
    raise ValueError(f"unknown x unit {unit!r}; use one of {X_UNITS}")


def blank_on_grid(sample: Spectrum, blank: Spectrum):
    """Blank absorbance interpolated onto the sample's wavelengths (NaN outside its range)."""
    if blank.x_nm.shape == sample.x_nm.shape and np.allclose(blank.x_nm, sample.x_nm):
        return blank.absorbance.copy()
    return np.interp(sample.x_nm, blank.x_nm, blank.absorbance, left=np.nan, right=np.nan)


def concentration_M(conc_M=None, solute_mass_mg=None, molar_mass_g_mol=None,
                    solution_volume_mL=None, solvent_mass_g=None, solvent=None,
                    solvent_density_g_mL=None, **_ignored):
    """
    Concentration in mol/L, plus a list of human-readable calculation steps.

    Give either ``conc_M`` directly, or ``solute_mass_mg`` + ``molar_mass_g_mol``
    together with ``solution_volume_mL`` or ``solvent_mass_g`` (the density is
    ``solvent_density_g_mL`` or looked up from ``solvent``). With a solvent mass
    the solution volume is approximated by the solvent volume (dilute solution).
    """
    steps = []
    if conc_M is not None:
        steps.append(f"c given directly = {conc_M:.4g} M")
        return float(conc_M), steps
    if solute_mass_mg is None or molar_mass_g_mol is None:
        raise ValueError("need conc_M, or solute_mass_mg and molar_mass_g_mol")
    n_mol = solute_mass_mg * 1e-3 / molar_mass_g_mol
    steps.append(f"n = m / MW = {solute_mass_mg:g} mg / {molar_mass_g_mol:g} g/mol = {n_mol:.4e} mol")
    if solution_volume_mL is None:
        if solvent_mass_g is None:
            raise ValueError("need solution_volume_mL or solvent_mass_g")
        rho = solvent_density_g_mL if solvent_density_g_mL is not None else solvent_density(solvent)
        if rho is None:
            raise ValueError(f"unknown density for solvent {solvent!r}; give solvent_density_g_mL")
        solution_volume_mL = solvent_mass_g / rho
        steps.append(f"V ≈ m_solvent / ρ = {solvent_mass_g:g} g / {rho:g} g/mL = {solution_volume_mL:.4f} mL"
                     " (solution volume ≈ solvent volume; fine for dilute solutions)")
    c = n_mol / (solution_volume_mL * 1e-3)
    steps.append(f"c = n / V = {n_mol:.4e} mol / {solution_volume_mL * 1e-3:.4e} L = {c:.4e} M")
    return c, steps


def process(sample: Spectrum, blank: Spectrum | None = None, pathlength_cm=None,
            concentration: dict | None = None, solvent_cutoff_abs=1.0, saturation_abs=2.5) -> Processed:
    """
    Blank subtraction, trimming, saturation flags and (if possible) epsilon in one call.
    The generated notebook performs the same steps one at a time.
    """
    A_raw = sample.absorbance
    A_blank = blank_on_grid(sample, blank) if blank is not None else np.zeros_like(A_raw)
    A_corr = A_raw - A_blank
    keep = np.isfinite(A_blank) & (A_blank <= solvent_cutoff_abs)
    flagged = A_raw > saturation_abs
    p = Processed(name=sample.name, x_nm=sample.x_nm[keep], abs_raw=A_raw[keep],
                  abs_blank=A_blank[keep], abs_corr=A_corr[keep], flagged=flagged[keep],
                  blank_name=blank.name if blank is not None else None,
                  pathlength_cm=pathlength_cm)
    if concentration:
        p.conc_M, p.conc_steps = concentration_M(**concentration)
    if p.conc_M and p.pathlength_cm:
        p.epsilon = p.abs_corr / (p.conc_M * p.pathlength_cm)
    return p


# --------------------------------------------------------------------------
# Output: text files
# --------------------------------------------------------------------------
def safe_filename(name: str) -> str:
    """Make a spectrum name safe to use as a file name on Windows/macOS/Linux."""
    s = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).strip(" .")
    return s or "spectrum"


def write_txt(p: Processed, path, source=None, metadata=None,
              solvent_cutoff_abs=None, saturation_abs=None):
    """Write a processed spectrum as a tab-separated .txt file with a '#' header."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    head = [p.name, OUTPUT_MARKER + f" v{__version__}"]
    if source:
        head.append(f"Source file: {source}")
    head.append(f"Blank subtracted: {p.blank_name or 'none'}")
    if solvent_cutoff_abs is not None:
        head.append(f"Points where blank Abs > {solvent_cutoff_abs} removed (solvent cutoff)")
    if saturation_abs is not None:
        head.append(f"reliable = 0 where raw sample Abs > {saturation_abs} (near detector saturation)")
    head.append(f"Pathlength (cm): {p.pathlength_cm if p.pathlength_cm else 'unknown'}")
    if p.conc_M:
        head.append(f"Concentration (M): {p.conc_M:.6e}")
        head += [f"  {s}" for s in p.conc_steps]
    else:
        head.append("Concentration (M): unknown (no epsilon column)")
    for k in SUMMARY_KEYS:
        if metadata and k in metadata:
            head.append(f"{k}: {metadata[k]}")

    cols = [("Wavelength_nm", p.x_nm, "%.4f"),
            ("Energy_eV", convert_x(p.x_nm, "eV"), "%.5f"),
            ("Wavenumber_cm-1", convert_x(p.x_nm, "cm-1"), "%.2f"),
            ("Abs_raw", p.abs_raw, "%.6g"),
            ("Abs_blank", p.abs_blank, "%.6g"),
            ("Abs_corrected", p.abs_corr, "%.6g")]
    if p.epsilon is not None:
        cols.append(("epsilon_M-1cm-1", p.epsilon, "%.6g"))
    cols.append(("reliable", (~p.flagged).astype(int), "%d"))
    head.append("\t".join(c[0] for c in cols))

    data = np.column_stack([c[1] for c in cols])
    np.savetxt(path, data, fmt=[c[2] for c in cols], delimiter="\t",
               header="\n".join("# " + h for h in head), comments="", encoding="utf-8")
    return path


# --------------------------------------------------------------------------
# Output: figures (matplotlib)
# --------------------------------------------------------------------------
def _style_ax(ax, xlabel, ylabel, title=None):
    ax.set_xlabel(xlabel, fontweight="bold", fontsize=12)
    ax.set_ylabel(ylabel, fontweight="bold", fontsize=12)
    if title:
        ax.set_title(title, fontweight="bold", fontsize=14)
    ax.grid(alpha=0.2)


def _nm_top_axis(ax, x_unit):
    """Secondary wavelength axis on top when plotting in eV or cm-1."""
    if x_unit == "nm":
        return None
    k = HC_EV_NM if x_unit == "eV" else 1e7

    def inv(x):
        x = np.array(x, dtype=float)
        near_zero = np.isclose(x, 0)
        x[near_zero] = np.inf
        x[~near_zero] = k / x[~near_zero]
        return x

    sec = ax.secondary_xaxis("top", functions=(inv, inv))
    # nm ticks are crowded at long wavelengths: keep ones at least 9% of the axis apart
    lo, hi = sorted(ax.get_xlim())
    cands = [150, 175, 200, 225, 250, 275, 300, 350, 400, 450, 500, 600, 700, 800, 1000,
             1200, 1500, 2000, 2500, 3000]
    ticks, last = [], None
    for nm in cands:
        pos = k / nm
        if lo <= pos <= hi and (last is None or abs(last - pos) >= 0.09 * (hi - lo)):
            ticks.append(nm)
            last = pos
    sec.set_xticks(ticks)
    sec.set_xlabel("Wavelength (nm)", fontweight="bold", fontsize=12)
    return sec


def _eps_scale(max_value):
    """Power of 1000 used to keep epsilon tick labels short (e.g. 10^3)."""
    if not np.isfinite(max_value) or max_value < 1e4:
        return 1
    return 10 ** (3 * int(np.floor(np.log10(max_value) / 3)))


def _y_data(p: Processed, kind: str):
    return {"abs": p.abs_corr, "epsilon": p.epsilon, "norm": p.abs_norm, "raw": p.abs_raw}[kind]


def _y_label(kind, scale=1):
    if kind == "epsilon":
        pre = "" if scale == 1 else f"10$^{{{int(np.log10(scale))}}}$ "
        return f"ε ({pre}M$^{{-1}}$ cm$^{{-1}}$)"
    return {"abs": "Absorbance", "norm": "Normalized absorbance", "raw": "Absorbance (raw)"}[kind]


def _plot_with_flags(ax, x, y, flagged, color, label, flag_label):
    good = np.where(flagged, np.nan, y)
    ax.plot(x, good, color=color, lw=1.5, label=label)
    if flagged.any():
        grown = flagged | np.r_[flagged[1:], False] | np.r_[False, flagged[:-1]]
        lab = None if getattr(ax, "_uv_flag_labelled", False) else flag_label
        ax.plot(x, np.where(grown, y, np.nan), color="gray", lw=1.5, ls=":", label=lab)
        ax._uv_flag_labelled = True


def plot_spectrum(p: Processed, x_unit="nm", y_kind="auto", saturation_abs=2.5, title=None):
    """Figure of one solvent-subtracted spectrum (epsilon if available, else absorbance)."""
    import matplotlib.pyplot as plt
    if y_kind == "auto":
        y_kind = "epsilon" if p.epsilon is not None else "abs"
    y = _y_data(p, y_kind)
    scale = _eps_scale(np.nanmax(np.abs(y))) if y_kind == "epsilon" else 1
    fig, ax = plt.subplots(figsize=(6, 4))
    _plot_with_flags(ax, convert_x(p.x_nm, x_unit), y / scale, p.flagged, COLORS[0], p.name,
                     f"raw A > {saturation_abs:g} (less reliable)")
    ax.axhline(0, color="gray", lw=0.8, alpha=0.7)
    _style_ax(ax, X_LABELS[x_unit], _y_label(y_kind, scale), title if title is not None else p.name)
    _nm_top_axis(ax, x_unit)
    ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=8)
    fig.tight_layout()
    return fig, ax


def plot_overlay(processed, x_unit="nm", y_kind="auto", saturation_abs=2.5, title=None):
    """Overlay of several processed spectra (epsilon if all have it, else absorbance)."""
    import matplotlib.pyplot as plt
    if y_kind == "auto":
        y_kind = "epsilon" if all(p.epsilon is not None for p in processed) else "abs"
    ys = [_y_data(p, y_kind) for p in processed]
    scale = _eps_scale(max(np.nanmax(np.abs(y)) for y in ys)) if y_kind == "epsilon" else 1
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, (p, y) in enumerate(zip(processed, ys)):
        ls_cycle = ["-", "--", ":"][min(i // 8, 2)]
        good = np.where(p.flagged, np.nan, y)
        ax.plot(convert_x(p.x_nm, x_unit), good / scale, color=COLORS[i % 8], ls=ls_cycle, lw=1.5, label=p.name)
    ax.axhline(0, color="gray", lw=0.8, alpha=0.7)
    _style_ax(ax, X_LABELS[x_unit], _y_label(y_kind, scale), title)
    _nm_top_axis(ax, x_unit)
    ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=8)
    if any(p.flagged.any() for p in processed):
        ax.text(0.99, 0.01, f"points with raw A > {saturation_abs:g} omitted", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=7, color="gray")
    fig.tight_layout()
    return fig, ax


# --------------------------------------------------------------------------
# Output: interactive HTML (plotly)
# --------------------------------------------------------------------------
def _jsonable(a):
    return [None if not np.isfinite(v) else float(f"{v:.6g}") for v in np.asarray(a, dtype=float)]


def write_interactive_html(processed, path, title="UV-vis spectra", default_x="nm", saturation_abs=None):
    """
    Standalone HTML viewer (works offline, no Python needed): dropdowns switch the
    x unit and the y quantity; clicking legend entries shows/hides spectra.
    """
    import plotly.graph_objects as go

    blanks = {}
    for p in processed:
        if p.blank_name and p.blank_name not in blanks:
            blanks[p.blank_name] = p
    have_eps = [p.epsilon is not None for p in processed]
    ymodes = [("abs", "Absorbance (solvent-subtracted)")]
    if any(have_eps):
        ymodes.append(("epsilon", "ε (M⁻¹ cm⁻¹)"))
    ymodes += [("norm", "Normalized absorbance"), ("raw", "Raw absorbance (before subtraction)")]
    default_y = "epsilon" if all(have_eps) else "abs"

    # Traces: every sample (reliable points) + its flagged points (grey, toggled with the sample
    # through a legend group), then blanks (only drawn in the raw view).
    traces = []
    for p in processed:
        traces.append((p.name, p, "sample"))
        if p.flagged.any():
            traces.append((p.name, p, "flagged"))
    traces += [(f"Blank: {b}", p, "blank") for b, p in blanks.items()]

    def y_for(p, role, mode):
        if role == "blank":
            return p.abs_blank if mode == "raw" else np.full_like(p.x_nm, np.nan)
        y = _y_data(p, mode)
        if y is None:
            return np.full_like(p.x_nm, np.nan)
        if role == "flagged":
            grown = p.flagged | np.r_[p.flagged[1:], False] | np.r_[False, p.flagged[:-1]]
            return np.where(grown, y, np.nan)
        return y if mode == "raw" else np.where(p.flagged, np.nan, y)

    xs = {u: [_jsonable(convert_x(p.x_nm, u)) for _, p, _ in traces] for u in X_UNITS}
    ys = {m: [_jsonable(y_for(p, role, m)) for _, p, role in traces] for m, _ in ymodes}

    fig = go.Figure()
    color_of, flag_legend_shown = {}, False
    for i, (name, p, role) in enumerate(traces):
        if role == "flagged":
            fig.add_trace(go.Scatter(
                x=xs[default_x][i], y=ys[default_y][i], name=f"raw A > {saturation_abs:g} (less reliable)",
                legendgroup=name, showlegend=not flag_legend_shown, mode="lines",
                line=dict(color="gray", width=2, dash="dot"),
                hovertemplate="%{x:.4g}, %{y:.4g}<extra>" + name + " (less reliable)</extra>"))
            flag_legend_shown = True
            continue
        color_of[name] = COLORS[len(color_of) % 8]
        fig.add_trace(go.Scatter(
            x=xs[default_x][i], y=ys[default_y][i], name=name, legendgroup=name, mode="lines",
            line=dict(color=color_of[name], width=2, dash="dash" if role == "blank" else "solid"),
            hovertemplate="%{x:.4g}, %{y:.4g}<extra>" + name + "</extra>"))

    ylabels = dict(ymodes)
    x_buttons = [dict(label=X_LABELS_HTML[u], method="update",
                      args=[{"x": xs[u]}, {"xaxis.title.text": X_LABELS_HTML[u], "xaxis.autorange": True}])
                 for u in X_UNITS]
    y_buttons = [dict(label=lab, method="update",
                      args=[{"y": ys[m]}, {"yaxis.title.text": lab, "yaxis.autorange": True}])
                 for m, lab in ymodes]
    notes = ["Click a legend entry to hide/show a spectrum, double-click to show only that one.",
             "Drag to zoom, double-click the plot to reset. Camera icon (top right) saves a PNG."]
    if not all(have_eps) and any(have_eps):
        notes.append("Spectra without a known concentration are empty in the ε view.")
    if saturation_abs is not None and any(p.flagged.any() for p in processed):
        notes.append(f"Grey dotted: raw absorbance above {saturation_abs:g}, near detector saturation, "
                     "so less reliable.")
    notes.append("Blanks (dashed) only appear in the raw-absorbance view.")

    menu = dict(direction="down", showactive=True, xanchor="left", yanchor="middle", y=1.1,
                bgcolor="white", bordercolor="lightgray")
    fig.update_layout(
        title=dict(text=f"<b>{title}</b>", x=0.5, xanchor="center", y=0.97, yref="container"),
        template="simple_white",
        xaxis=dict(title=X_LABELS_HTML[default_x], showgrid=True, gridcolor="rgba(0,0,0,0.08)"),
        yaxis=dict(title=ylabels[default_y], showgrid=True, gridcolor="rgba(0,0,0,0.08)",
                   zeroline=True, zerolinecolor="rgba(0,0,0,0.3)"),
        updatemenus=[
            dict(menu, buttons=x_buttons, active=X_UNITS.index(default_x), x=0.07),
            dict(menu, buttons=y_buttons, active=[m for m, _ in ymodes].index(default_y), x=0.42),
        ],
        annotations=[
            dict(text="<b>X axis:</b>", x=0.0, y=1.1, xref="paper", yref="paper", showarrow=False,
                 xanchor="left", yanchor="middle"),
            dict(text="<b>Y axis:</b>", x=0.35, y=1.1, xref="paper", yref="paper", showarrow=False,
                 xanchor="left", yanchor="middle"),
            dict(text="<br>".join(notes), x=0, y=-0.2, xref="paper", yref="paper", showarrow=False,
                 xanchor="left", yanchor="top", align="left", font=dict(size=11, color="gray")),
        ],
        legend=dict(bgcolor="rgba(255,255,255,0.8)", bordercolor="lightgray", borderwidth=1),
        margin=dict(t=140, b=170),
        height=700,
    )
    path = Path(path)
    fig.write_html(path, include_plotlyjs=True, full_html=True,
                   config={"displaylogo": False, "responsive": True,
                           "toImageButtonOptions": {"format": "png", "scale": 3}})
    return path


# --------------------------------------------------------------------------
# Notebook generation
# --------------------------------------------------------------------------
def _py_literal(obj, indent=0):
    """Readable Python literal for dicts/lists/scalars (used to write the Settings cell)."""
    pad, pad1 = " " * indent, " " * (indent + 4)
    if isinstance(obj, dict):
        if not obj:
            return "{}"
        items = [f"{pad1}{k!r}: {_py_literal(v, indent + 4)}," for k, v in obj.items()]
        return "{\n" + "\n".join(items) + f"\n{pad}}}"
    if isinstance(obj, list):
        if not obj:
            return "[]"
        items = [f"{pad1}{_py_literal(v, indent + 4)}," for v in obj]
        return "[\n" + "\n".join(items) + f"\n{pad}]"
    return repr(obj)


def load_config(path) -> dict:
    """Read a processing_config.json and fill in defaults."""
    path = Path(path)
    cfg = json.loads(path.read_text(encoding="utf-8"))
    data_file = Path(cfg["data_file"])
    if not data_file.is_absolute():
        data_file = (path.parent / data_file).resolve()
    cfg["data_file"] = str(data_file)
    stem = data_file.stem
    cfg.setdefault("output_dir", None)
    out = Path(cfg["output_dir"]) if cfg["output_dir"] else data_file.parent / f"{stem}_processed"
    if not out.is_absolute():
        out = (path.parent / out).resolve()
    cfg["output_dir"] = str(out)
    cfg["notebook_name"] = (cfg.get("notebook_name") or f"{stem}_processing").removesuffix(".ipynb")
    cfg.setdefault("x_unit", "nm")
    if cfg["x_unit"] not in X_UNITS:
        raise ValueError(f"x_unit must be one of {X_UNITS}")
    cfg.setdefault("solvent_cutoff_abs", 1.0)
    cfg.setdefault("saturation_abs", 2.5)
    for s in cfg["samples"]:
        s.setdefault("blank", None)
        s.setdefault("pathlength_cm", None)
        s.setdefault("concentration", None)
    return cfg


def make_notebook(cfg):
    """Build (unexecuted) the processing notebook for a config from :func:`load_config`."""
    import nbformat
    from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

    data_file = Path(cfg["data_file"])
    out = Path(cfg["output_dir"])
    try:
        rel = os.path.relpath(data_file, out)
    except ValueError:  # different drive on Windows
        rel = str(data_file)
    stem = data_file.stem
    multi = len(cfg["samples"]) > 1
    any_conc = any(s["concentration"] and s["pathlength_cm"] for s in cfg["samples"])

    md, code = new_markdown_cell, new_code_cell
    cells = [
        md(f"# UV-vis processing: `{data_file.name}`\n\n"
           "This notebook was generated by the **process-uvvis** skill. It goes step by step from the raw "
           "spectrometer export to solvent-subtracted spectra"
           + (", molar absorptivity (ε)" if any_conc else "") + " and the output files.\n\n"
           "**How to use it:** run every cell from top to bottom (*Run → Run All Cells* in JupyterLab or "
           "VS Code). To change something (units, concentration, pathlength, thresholds), edit the "
           "**Settings** cell and run everything again. The output files are overwritten.\n\n"
           "**Outputs** (in the folder that holds this notebook):\n"
           "- `processed_data/`: one tab-separated `.txt` per spectrum, named after the spectrum's header\n"
           "- `figures/`: one PNG per solvent-subtracted spectrum"
           + (f", plus `{stem}_overlay.png` with all of them" if multi else "") + "\n"
           f"- `{stem}_interactive.html`: double-click to open it in a web browser, then browse the "
           "spectra and switch units (no Python needed)"),
        code("from pathlib import Path\n\n"
             "import numpy as np\n"
             "import matplotlib.pyplot as plt\n\n"
             "import uvvis_tools as uv  # helper module saved next to this notebook\n\n"
             f"colors = {COLORS!r}"),
        md("## 1. Settings\n\n"
           "- `DATA_FILE`: the raw export, relative to this notebook\n"
           "- `X_UNIT`: default x axis for the figures: `\"nm\"`, `\"eV\"` or `\"cm-1\"`\n"
           "- `SOLVENT_CUTOFF_ABS`: wavelengths where the **blank** absorbs more than this are removed. "
           "Below the solvent cutoff almost no light reaches the detector, so the subtraction there is meaningless.\n"
           "- `SATURATION_ABS`: points where the **raw sample** absorbance is above this are kept but marked "
           "less reliable (stray light makes very high absorbances read low)\n"
           "- `SAMPLES`: one entry per sample spectrum. `blank` names the solvent spectrum to subtract "
           "(`None` = no subtraction). `concentration` can be `conc_M`, or a solute mass + molar mass with either "
           "a solution volume or a solvent mass (the density comes from the solvent name or "
           "`solvent_density_g_mL`). `None` means unknown, and then only absorbance is plotted."),
        code(f"DATA_FILE = Path(r\"{rel}\")\n"
             f"X_UNIT = {cfg['x_unit']!r}\n"
             f"SOLVENT_CUTOFF_ABS = {cfg['solvent_cutoff_abs']!r}\n"
             f"SATURATION_ABS = {cfg['saturation_abs']!r}\n\n"
             f"SAMPLES = {_py_literal(cfg['samples'])}"),
        md("## 2. Load the raw data\n\n"
           "`uv.read_uvvis` reads the export. For a Cary CSV, row 1 holds the spectrum names, row 2 the "
           "column labels, then the data, then a block of instrument settings for each spectrum. "
           "Each spectrum is sorted by increasing wavelength."),
        code("spectra = {s.name: s for s in uv.read_uvvis(DATA_FILE)}\n\n"
             "for s in spectra.values():\n"
             "    print(f\"{s.name:45s} {len(s.x_nm):5d} points  {s.x_nm.min():.0f}-{s.x_nm.max():.0f} nm  \"\n"
             "          f\"max A = {np.nanmax(s.absorbance):.3f}\")\n"
             "    for note in s.notes:\n"
             "        print('    note:', note)\n\n"
             "first = next(iter(spectra.values()))\n"
             "if first.metadata:\n"
             "    print(f\"\\nInstrument settings recorded for {first.name}:\")\n"
             "    for key in uv.SUMMARY_KEYS:\n"
             "        if key in first.metadata:\n"
             "            print(f\"    {key}: {first.metadata[key]}\")"),
        md("## 3. Raw spectra\n\nEverything in the file as measured, with blanks drawn dashed."),
        code("blank_names = {cfg['blank'] for cfg in SAMPLES if cfg['blank']}\n\n"
             "fig, ax = plt.subplots(figsize=(6, 4))\n"
             "for i, s in enumerate(spectra.values()):\n"
             "    ax.plot(s.x_nm, s.absorbance, color=colors[i % 8], lw=1.5,\n"
             "            ls='--' if s.name in blank_names else '-', label=s.name)\n"
             "ax.set_xlabel('Wavelength (nm)', fontweight='bold', fontsize=12)\n"
             "ax.set_ylabel('Absorbance', fontweight='bold', fontsize=12)\n"
             "ax.set_title('Raw spectra', fontweight='bold', fontsize=14)\n"
             "ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=8)\n"
             "ax.grid(alpha=0.2)\n"
             "plt.tight_layout()\n"
             "plt.show()"),
        md("## 4. Solvent (blank) subtraction\n\n"
           "$$A_\\mathrm{corr}(\\lambda) = A_\\mathrm{sample}(\\lambda) - A_\\mathrm{blank}(\\lambda)$$\n\n"
           "The blank is interpolated onto the sample's wavelengths first (they are usually identical). "
           "Points where the blank absorbance exceeds `SOLVENT_CUTOFF_ABS` are removed, and points where the "
           "raw sample absorbance exceeds `SATURATION_ABS` are flagged."),
        code("results = {}\n"
             "for cfg in SAMPLES:\n"
             "    sample = spectra[cfg['name']]\n"
             "    x_nm, A_raw = sample.x_nm, sample.absorbance\n"
             "    if cfg['blank']:\n"
             "        A_blank = uv.blank_on_grid(sample, spectra[cfg['blank']])\n"
             "    else:\n"
             "        A_blank = np.zeros_like(A_raw)  # no blank: nothing to subtract\n\n"
             "    A_corr = A_raw - A_blank                                      # the subtraction\n"
             "    keep = np.isfinite(A_blank) & (A_blank <= SOLVENT_CUTOFF_ABS)  # outside the solvent cutoff\n"
             "    flagged = A_raw > SATURATION_ABS                              # near detector saturation\n\n"
             "    results[cfg['name']] = uv.Processed(\n"
             "        name=cfg['name'], x_nm=x_nm[keep], abs_raw=A_raw[keep], abs_blank=A_blank[keep],\n"
             "        abs_corr=A_corr[keep], flagged=flagged[keep], blank_name=cfg['blank'],\n"
             "        pathlength_cm=cfg['pathlength_cm'])\n\n"
             "    print(f\"{cfg['name']}: blank = {cfg['blank']}\")\n"
             "    if (~keep).any():\n"
             "        print(f\"    removed {(~keep).sum()} points (blank A > {SOLVENT_CUTOFF_ABS}) between \"\n"
             "              f\"{x_nm[~keep].min():.0f} and {x_nm[~keep].max():.0f} nm\")\n"
             "    if flagged[keep].any():\n"
             "        xf = x_nm[keep][flagged[keep]]\n"
             "        print(f\"    {flagged[keep].sum()} points with raw A > {SATURATION_ABS} between \"\n"
             "              f\"{xf.min():.0f} and {xf.max():.0f} nm: shown grey dotted, reliable = 0 in the .txt\")"),
        code("fig, axes = plt.subplots(len(results), 1, figsize=(6, 4 * len(results)), squeeze=False,\n"
             "                         constrained_layout=True)\n"
             "for ax, p in zip(axes[:, 0], results.values()):\n"
             "    ax.plot(p.x_nm, p.abs_raw, color=colors[0], lw=1.5, label='sample (raw)')\n"
             "    ax.plot(p.x_nm, p.abs_blank, color=colors[2], lw=1.5, ls='--', label=f'blank: {p.blank_name}')\n"
             "    ax.plot(p.x_nm, p.abs_corr, color=colors[6], lw=1.5, label='sample - blank')\n"
             "    ax.axhline(0, color='gray', lw=0.8, alpha=0.7)\n"
             "    ax.set_xlabel('Wavelength (nm)', fontweight='bold', fontsize=12)\n"
             "    ax.set_ylabel('Absorbance', fontweight='bold', fontsize=12)\n"
             "    ax.set_title(p.name, fontweight='bold', fontsize=14)\n"
             "    ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=8)\n"
             "    ax.grid(alpha=0.2)\n"
             "plt.show()"),
        md("## 5. Concentration\n\n"
           "From masses: $n = m_\\mathrm{solute} / M_W$, then "
           "$V \\approx m_\\mathrm{solvent} / \\rho_\\mathrm{solvent}$ (if no volume was measured), "
           "then $c = n / V$. Using the solvent volume for the solution volume is fine for dilute solutions."),
        code("for cfg in SAMPLES:\n"
             "    p = results[cfg['name']]\n"
             "    if cfg['concentration']:\n"
             "        p.conc_M, p.conc_steps = uv.concentration_M(**cfg['concentration'])\n"
             "        print(cfg['name'])\n"
             "        for step in p.conc_steps:\n"
             "            print('   ', step)\n"
             "    else:\n"
             "        print(f\"{cfg['name']}: concentration unknown, so only absorbance is plotted\")"),
        md("## 6. Molar absorptivity (Beer–Lambert)\n\n"
           "$$A = \\varepsilon\\, c\\, l \\quad\\Rightarrow\\quad \\varepsilon = \\frac{A_\\mathrm{corr}}{c\\, l}"
           "\\qquad [\\mathrm{M^{-1}\\,cm^{-1}}]$$"),
        code("for p in results.values():\n"
             "    if p.conc_M and p.pathlength_cm:\n"
             "        p.epsilon = p.abs_corr / (p.conc_M * p.pathlength_cm)\n"
             "        reliable = np.where(p.flagged, np.nan, p.epsilon)\n"
             "        if np.isfinite(reliable).any():\n"
             "            i = np.nanargmax(reliable)\n"
             "            print(f\"{p.name}: c = {p.conc_M:.3e} M, l = {p.pathlength_cm} cm -> largest reliable ε = \"\n"
             "                  f\"{p.epsilon[i]:,.0f} M^-1 cm^-1 at {p.x_nm[i]:.0f} nm\")\n"
             "            if p.flagged.any():\n"
             "                print(f\"    (saturated points excluded, so a band maximum inside the grey region may be higher)\")\n"
             "    else:\n"
             "        print(f\"{p.name}: no concentration or pathlength, so no ε\")"),
        md("## 7. Units\n\n"
           "$$E\\,(\\mathrm{eV}) = \\frac{1239.84}{\\lambda\\,(\\mathrm{nm})} \\qquad "
           "\\tilde{\\nu}\\,(\\mathrm{cm^{-1}}) = \\frac{10^7}{\\lambda\\,(\\mathrm{nm})}$$\n\n"
           "Absorbance and ε are values at each point, not densities, so they are not rescaled when the x axis "
           "changes. Change `X_UNIT` in the Settings cell to redraw every figure in another unit."),
        code("for lam in (300.0, 500.0, 800.0):\n"
             "    print(f\"{lam:5.0f} nm = {uv.convert_x(lam, 'eV'):.3f} eV = {uv.convert_x(lam, 'cm-1'):,.0f} cm^-1\")\n"
             "print('\\nFigures below use X_UNIT =', repr(X_UNIT))"),
        md("## 8. Figures\n\nOne figure per solvent-subtracted spectrum"
           + (", then an overlay of all of them" if multi else "")
           + ". They are saved to `figures/` (300 dpi PNG)."),
        code("processed = list(results.values())\n"
             "fig_dir = Path('figures')\n"
             "fig_dir.mkdir(exist_ok=True)\n\n"
             "for p in processed:\n"
             "    fig, ax = uv.plot_spectrum(p, x_unit=X_UNIT, saturation_abs=SATURATION_ABS)\n"
             "    fig.savefig(fig_dir / f'{uv.safe_filename(p.name)}.png', dpi=300, bbox_inches='tight', facecolor='white')\n"
             "    plt.show()\n\n"
             "if len(processed) > 1:\n"
             "    fig, ax = uv.plot_overlay(processed, x_unit=X_UNIT, saturation_abs=SATURATION_ABS, title=DATA_FILE.stem)\n"
             "    fig.savefig(fig_dir / f'{DATA_FILE.stem}_overlay.png', dpi=300, bbox_inches='tight', facecolor='white')\n"
             "    plt.show()"),
        md("## 9. Save the processed data\n\n"
           "Each spectrum is saved as a tab-separated `.txt` in `processed_data/`, named after its header. "
           "The `#` lines at the top record the source, blank, concentration and instrument settings. Columns: "
           "wavelength (nm), energy (eV), wavenumber (cm⁻¹), raw absorbance, blank, corrected absorbance, "
           "ε (when the concentration is known) and `reliable` (0 = above `SATURATION_ABS`). "
           "Origin, Excel and `np.loadtxt` all read these files."),
        code("data_dir = Path('processed_data')\n"
             "for p in processed:\n"
             "    path = uv.write_txt(p, data_dir / f'{uv.safe_filename(p.name)}.txt', source=DATA_FILE.name,\n"
             "                        metadata=spectra[p.name].metadata, solvent_cutoff_abs=SOLVENT_CUTOFF_ABS,\n"
             "                        saturation_abs=SATURATION_ABS)\n"
             "    print('saved', path)"),
        md("## 10. Interactive viewer\n\n"
           f"`{stem}_interactive.html` opens in any web browser and needs no Python. Dropdowns switch the x axis "
           "(nm / eV / cm⁻¹) and the y quantity. Click legend entries to choose which spectra to overlay."),
        code(f"html = uv.write_interactive_html(processed, '{stem}_interactive.html', title=DATA_FILE.stem,\n"
             "                                 default_x=X_UNIT, saturation_abs=SATURATION_ABS)\n"
             "print('saved', html)"),
    ]
    nb = new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.metadata["language_info"] = {"name": "python"}
    nbformat.validate(nb)
    return nb


def _run_in_process(nb, cwd):
    """Fallback when no Jupyter kernel is available: run the code cells here (figures not embedded)."""
    import matplotlib
    matplotlib.use("Agg")
    old_cwd, old_path = os.getcwd(), list(sys.path)
    os.chdir(cwd)
    sys.path.insert(0, str(cwd))
    try:
        ns = {"__name__": "__main__"}
        for c in nb.cells:
            if c.cell_type == "code":
                exec(compile(c.source, "<notebook>", "exec"), ns)
    finally:
        os.chdir(old_cwd)
        sys.path[:] = old_path
        import matplotlib.pyplot as plt
        plt.close("all")


def build(config_path, execute=True) -> dict:
    """Write the notebook and its helper next to the outputs, run it, return a summary."""
    import nbformat
    cfg = load_config(config_path)
    data_file, out = Path(cfg["data_file"]), Path(cfg["output_dir"])

    names = {s.name for s in read_uvvis(data_file)}
    for s in cfg["samples"]:
        for key in ("name", "blank"):
            if s[key] and s[key] not in names:
                raise ValueError(f"spectrum {s[key]!r} not in {data_file.name}; available: {sorted(names)}")
        if s["concentration"]:
            concentration_M(**s["concentration"])  # fail early on incomplete input

    out.mkdir(parents=True, exist_ok=True)
    here = Path(__file__).resolve()
    if here != (out / "uvvis_tools.py").resolve():
        shutil.copy2(here, out / "uvvis_tools.py")
    (out / "processing_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    nb = make_notebook(cfg)
    nb_path = out / f"{cfg['notebook_name']}.ipynb"
    status = "not executed"
    if execute:
        try:
            from nbconvert.preprocessors import ExecutePreprocessor
            from jupyter_client.kernelspec import NoSuchKernel
        except ImportError:
            ExecutePreprocessor = None
        if ExecutePreprocessor is not None:
            ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
            try:
                ep.preprocess(nb, {"metadata": {"path": str(out)}})
                status = "executed (outputs embedded)"
            except NoSuchKernel:
                ExecutePreprocessor = None
            except Exception:
                nbformat.write(nb, nb_path)  # keep partial outputs so the failing cell is visible
                raise
        if ExecutePreprocessor is None:
            _run_in_process(nb, out)
            status = "processed without Jupyter (open the notebook and Run All to see outputs inline)"
    nbformat.write(nb, nb_path)

    stem = data_file.stem
    return {
        "output_dir": str(out),
        "notebook": str(nb_path),
        "notebook_status": status,
        "interactive_html": str(out / f"{stem}_interactive.html"),
        "processed_data": sorted(str(p) for p in (out / "processed_data").glob("*.txt")),
        "figures": sorted(str(p) for p in (out / "figures").glob("*.png")),
    }


# --------------------------------------------------------------------------
# Inspection helpers for the skill
# --------------------------------------------------------------------------
def inspect(path) -> dict:
    """Summary of a data file used by the skill to ask the user the right questions."""
    path = Path(path)
    spectra = read_uvvis(path)
    pairs = pair_blanks(spectra)
    info = {"file": str(path.resolve()), "n_spectra": len(spectra), "spectra": [], "proposed_blanks": pairs,
            "warnings": []}
    for s in spectra:
        i = int(np.nanargmax(s.absorbance))
        guess = parse_name(s.name)
        info["spectra"].append({
            "name": s.name,
            "role": "blank" if is_blank(s.name) else "sample",
            "n_points": int(len(s.x_nm)),
            "range_nm": [round(float(s.x_nm.min()), 2), round(float(s.x_nm.max()), 2)],
            "max_abs": round(float(s.absorbance[i]), 4),
            "max_abs_at_nm": round(float(s.x_nm[i]), 1),
            "n_points_above_2.5": int((s.absorbance > 2.5).sum()),
            "name_guesses": {k: v for k, v in guess.items() if v not in (None, [], False)},
            "metadata": {k: s.metadata[k] for k in SUMMARY_KEYS if k in s.metadata},
            "notes": s.notes,
        })
        if s.metadata.get("Baseline Correction", "").lower() == "on" and is_blank(s.name):
            info["warnings"].append(f"{s.name}: instrument baseline correction was ON for this blank")
    if not any(is_blank(s.name) for s in spectra):
        info["warnings"].append("no spectrum looks like a solvent blank; ask the user which one is (if any)")
    for sample, blank in pairs.items():
        if blank is None and any(is_blank(s.name) for s in spectra):
            info["warnings"].append(f"{sample}: no matching blank found automatically")
    return info


def find_data(folder, recursive=False) -> list[dict]:
    """List files in ``folder`` that parse as UV-vis data (skipping process-uvvis outputs)."""
    folder = Path(folder)
    pattern = "**/*" if recursive else "*"
    found = []
    for f in sorted(folder.glob(pattern)):
        if f.suffix.lower() not in (".csv", ".txt", ".dat", ".tsv") or not f.is_file():
            continue
        if any(part.endswith("_processed") for part in f.parts):
            continue
        try:
            with open(f, "rb") as fh:
                if OUTPUT_MARKER.encode() in fh.read(2000):
                    continue
            spectra = read_uvvis(f)
        except Exception:
            continue
        lo = min(s.x_nm.min() for s in spectra)
        hi = max(s.x_nm.max() for s in spectra)
        if 100 <= lo and hi <= 3500:
            found.append({"file": str(f.resolve()), "n_spectra": len(spectra),
                          "names": [s.name for s in spectra], "range_nm": [round(lo, 1), round(hi, 1)]})
    return found


def check_env() -> dict:
    """Report which optional packages are importable."""
    import importlib
    pkgs = {}
    for mod in ("numpy", "matplotlib", "plotly", "nbformat", "nbconvert", "ipykernel", "jupyter_client"):
        try:
            m = importlib.import_module(mod)
            pkgs[mod] = getattr(m, "__version__", "installed")
        except Exception:
            pkgs[mod] = None
    kernels = None
    try:
        from jupyter_client.kernelspec import KernelSpecManager
        kernels = sorted(KernelSpecManager().find_kernel_specs())
    except Exception:
        pass
    missing = [k for k in ("numpy", "matplotlib", "plotly", "nbformat") if not pkgs[k]]
    return {"python": sys.executable, "version": sys.version.split()[0], "packages": pkgs,
            "jupyter_kernels": kernels, "missing_required": missing,
            "can_execute_notebook": bool(pkgs["nbconvert"] and pkgs["ipykernel"]),
            "install_hint": ("pip install " + " ".join(missing)) if missing else None}


def main(argv=None):
    ap = argparse.ArgumentParser(description="UV-vis processing helpers for the process-uvvis skill")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check-env", help="report available packages")
    f = sub.add_parser("find", help="list UV-vis data files in a folder")
    f.add_argument("folder", nargs="?", default=".")
    f.add_argument("-r", "--recursive", action="store_true")
    i = sub.add_parser("inspect", help="summarise a data file (spectra, guesses, blanks)")
    i.add_argument("file")
    b = sub.add_parser("build", help="write + run the processing notebook from a config JSON")
    b.add_argument("config")
    b.add_argument("--no-execute", action="store_true")
    args = ap.parse_args(argv)

    if args.cmd == "check-env":
        result = check_env()
    elif args.cmd == "find":
        result = find_data(args.folder, args.recursive)
    elif args.cmd == "inspect":
        result = inspect(args.file)
    else:
        result = build(args.config, execute=not args.no_execute)
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
