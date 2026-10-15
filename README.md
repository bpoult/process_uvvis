# process-uvvis

A [Claude Code](https://code.claude.com) skill that turns raw UV-vis spectrometer exports into
processed spectra. You talk to Claude, and it asks a few questions and does the rest:

1. **Finds** UV-vis data files in your folder: Agilent Cary multi-column CSV, or simple x/y CSV/TXT.
2. **Subtracts the solvent blank** when the file contains a solvent spectrum.
3. **Asks for concentration and pathlength**, and suggests values it reads from your sample names
   (e.g. `Cu_imnt_1.8mg_3.42g_DMF_0.1cm` → 1.8 mg solute, 3.42 g DMF, 0.1 cm) for you to
   confirm. You can give a concentration directly, or masses + molar mass, and it works out the
   concentration (using a built-in solvent density table if you give a solvent mass).
4. **Asks your preferred x axis**: nm, eV or cm⁻¹.
5. **Plots molar absorptivity ε (M⁻¹ cm⁻¹)** when the concentration and pathlength are known,
   otherwise absorbance.
6. **Writes a Jupyter notebook** that walks through every step.
7. **Saves processed data** as `.txt` files.
8. **Saves figures** (one per spectrum, plus an overlay).
9. **Builds an interactive HTML viewer** that opens in any web browser. No Python or Jupyter
   is needed to use it.

## Install

You need Claude Code and Python 3.9+ with `numpy matplotlib plotly nbformat`. Add `nbconvert`
and `ipykernel` if you want the notebook to come out with its results already filled in.

```bash
pip install numpy matplotlib plotly nbformat nbconvert ipykernel
```

**Option A: plugin (recommended).** Inside Claude Code:

```
/plugin marketplace add bpoult/process_uvvis
/plugin install process-uvvis@process-uvvis
```

**Option B: copy the skill.** Clone this repo and copy `skills/process-uvvis/` to
`~/.claude/skills/process-uvvis/`. On Windows that is `C:\Users\<you>\.claude\skills\process-uvvis\`.

## Use

Open a terminal in the folder with your data, start `claude`, and type:

```
/process-uvvis
```

You can also just ask, e.g. *"process my UV-vis data"*. Claude will show you what it found and
ask a few questions. Answer them and the results appear in a new folder next to your data:

```
10.13.2026_Cuimnt_processed/
├── 10.13.2026_Cuimnt_processing.ipynb   step-by-step notebook (with results)
├── 10.13.2026_Cuimnt_interactive.html   ← double-click: browse spectra, switch units
├── processed_data/
│   └── Cu_imnt_1.8mg_3.42g_DMF_0.1cm.txt
├── figures/
│   ├── Cu_imnt_1.8mg_3.42g_DMF_0.1cm.png
│   └── 10.13.2026_Cuimnt_overlay.png    (when there are several samples)
├── processing_config.json               your answers, so you can re-run later
└── uvvis_tools.py                       helper module used by the notebook
```

### The interactive viewer

Open `*_interactive.html` in Chrome, Edge, Firefox or Safari. It works offline and you can
email it to collaborators.

- **X axis** dropdown: wavelength (nm), energy (eV) or wavenumber (cm⁻¹)
- **Y axis** dropdown: blank-subtracted absorbance, ε, normalized absorbance, or raw data with
  the blank
- **Legend:** click a spectrum to hide or show it; double-click to show only that one
- Drag to zoom, double-click to reset, camera icon to save a PNG

### The processed .txt files

Tab-separated. The `#` header lines record the source file, blank, concentration calculation,
pathlength and instrument settings. Columns:

| Wavelength_nm | Energy_eV | Wavenumber_cm-1 | Abs_raw | Abs_blank | Abs_corrected | epsilon_M-1cm-1 | reliable |
|---|---|---|---|---|---|---|---|

The `epsilon_M-1cm-1` column is only present when the concentration is known. `reliable` is 0
where the raw absorbance is above 2.5, which is close to detector saturation.

Origin, Excel (import as tab-delimited, comment character `#`) and Python
(`np.loadtxt(path)`) all read these files.

### The notebook

You don't need the notebook, since every output is already saved. If you want to change
something, open it in JupyterLab or VS Code, edit the **Settings** cell (units, concentration,
pathlength, thresholds), then choose **Run All**.

## What it does to your data

- **Blank subtraction:** A_corr = A_sample − A_blank, on the same wavelength grid
  (interpolated if needed).
- **Solvent cutoff:** wavelengths where the blank's absorbance is above 1.0 are removed. There
  the solvent absorbs nearly all the light and the subtraction means nothing.
- **Saturation flag:** points where the sample's raw absorbance is above 2.5 are kept, drawn as
  grey dotted lines, and marked `reliable = 0`. If an important band is flagged, dilute the
  sample or use a shorter cuvette.
- **Concentration:**
  - n = m / M_W
  - V ≈ m_solvent / ρ (solution volume taken as the solvent volume, fine for dilute solutions)
  - c = n / V
- **Molar absorptivity:** ε = A_corr / (c · l)
- **Units:**
  - E (eV) = 1239.84 / λ (nm)
  - ν̃ (cm⁻¹) = 10⁷ / λ (nm)

## Supported files

- **Agilent Cary (WinUV / Cary 60, 100, 300, 5000…) CSV exports.** Row 1 has the spectrum
  names, row 2 has `Wavelength (nm),Abs` pairs, and the instrument settings at the end are read
  too.
- **Generic CSV/TXT/TSV.**
  - Layout: `x, y` pairs, or one x column with several y columns.
  - x axis: nm, eV or cm⁻¹, detected from the header.
  - y values: absorbance, or %T if the header says so.

A file that won't load is probably in a format the parser doesn't handle yet. Please open an
issue with an example file.

## Developing

```bash
pip install pytest
pytest tests
```

The test data in `tests/data/` is synthetic.
