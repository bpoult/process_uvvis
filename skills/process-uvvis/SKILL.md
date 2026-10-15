---
name: process-uvvis
description: >-
  Process raw UV-vis absorption data into publication-ready results. Finds UV-vis
  exports (Agilent Cary multi-column CSV or simple x/y CSV/TXT), subtracts the
  solvent blank, works out concentration from masses or volumes, converts to molar
  absorptivity ε (M⁻¹ cm⁻¹), and writes a step-by-step Jupyter notebook, processed
  .txt files, PNG figures (per spectrum + overlay) and a standalone interactive HTML
  viewer. Use when the user asks to process, plot, blank-subtract or analyse UV-vis
  / UV-Vis-NIR / absorption / absorbance spectra, compute extinction coefficients or
  epsilon, or invokes /process-uvvis.
---

# process-uvvis

You guide the user through processing UV-vis data. All parsing, maths, plotting and
file writing is done by `uvvis_tools.py`, which sits in the same folder as this
SKILL.md (`${CLAUDE_SKILL_DIR}/uvvis_tools.py`). Below, `$TOOLS` means that path.
**Do not hand-write the processing code.** Your jobs are to ask the right questions,
write a config file, run the tool and report the results.

Many users have never used Jupyter. Keep questions plain, explain the outputs, and
always point them to the interactive HTML file, which they can open by double-clicking.

## 0. Python

Pick the Python interpreter:
- If another skill or CLAUDE.md says which Python or conda env to use, use that.
- Otherwise try `python`, then `python3`, then `py`.

Run `"$PY" "$TOOLS" check-env`. If `missing_required` is non-empty, show the user
the `install_hint` and ask before installing anything. If `can_execute_notebook`
is false, processing still works. The notebook is then written without embedded
outputs, and the user runs it with *Run All*.

## 1. Find the data

- If the user named a file or folder, use it. Otherwise run `"$PY" "$TOOLS" find .`
  (add `-r` to search subfolders). This lists files that parse as UV-vis data and
  skips earlier `*_processed/` outputs.
- If there are several files, ask which ones to process (multiSelect). Each data
  file gets its own output folder and notebook.
- For each chosen file run `"$PY" "$TOOLS" inspect <file>` and read the JSON:
  - spectrum names and roles (blank or sample)
  - `name_guesses` (masses, volume, concentration, pathlength, solvent and density)
  - `proposed_blanks`, the instrument `metadata`, and `warnings`
- Give the user a short summary: what was found, wavelength range, which spectra
  look like blanks, and any warnings.

## 2. Confirm blank subtraction

Show the proposed pairing (sample → blank) and ask the user to confirm or change it.
"No blank (don't subtract)" must always be an option.
- If no spectrum looks like a blank, ask whether one of them is a blank.
- Never subtract a blank the user hasn't confirmed.

## 3. Concentration and pathlength (per sample)

Ask in one AskUserQuestion round where possible. Batch samples that share a
solvent or pathlength.

- **Pathlength.** If the name or metadata gives one (e.g. `0.1cm`), ask the user
  to confirm it: "The name says 0.1 cm. Is that right?". Otherwise offer common
  values: 1 cm, 0.1 cm, 0.2 cm, 0.5 cm, plus Other. "Unknown" is also allowed.
- **Concentration.** Ask if they know it. Accepted forms:
  1. concentration directly (M / mM / µM), or
  2. solute mass + molar mass + solution volume, or
  3. solute mass + molar mass + solvent mass, with the density from the
     built-in table (`SOLVENTS` in `uvvis_tools.py`) or given by the user.
  - If the name gives masses or a volume (e.g. `1.8mg_3.42g_DMF`), state your
    reading and ask them to confirm it: "1.8 mg of solute in 3.42 g DMF (density
    0.944 g/mL → 3.62 mL)?".
  - The molar mass never appears in names, so always ask for it when masses are
    used. If the solvent isn't in the table, ask for its density.
- If the concentration or pathlength is unknown, process that sample anyway and
  plot absorbance only. Tell the user they can add the values later by editing
  the notebook's Settings cell.

## 4. Display options

Ask together:
- **Default x axis:** nm / eV / cm⁻¹. All three are always in the .txt files and
  the HTML viewer. This choice only affects the PNG figures and the notebook.
- **Notebook name:** default `<data file stem>_processing`, or the user's own name.

Use the defaults for the thresholds below without asking. Mention them in the
final report so the user knows they can change them:
- `solvent_cutoff_abs` (1.0): points where the blank absorbs more than this are
  removed (below the solvent cutoff).
- `saturation_abs` (2.5): points where the raw sample absorbance is above this
  are kept but flagged as less reliable.

## 5. Build

Write `processing_config.json`. A good place is a scratch path, or the output
folder if it already exists. `build` copies it into the output folder anyway.

```json
{
  "data_file": "C:/path/to/10.13.2026_Cuimnt.csv",
  "output_dir": null,
  "notebook_name": "10.13.2026_Cuimnt_processing",
  "x_unit": "nm",
  "solvent_cutoff_abs": 1.0,
  "saturation_abs": 2.5,
  "samples": [
    {
      "name": "Cu_imnt_1.8mg_3.42g_DMF_0.1cm",
      "blank": "DMF_0.1cm",
      "pathlength_cm": 0.1,
      "concentration": {"solute_mass_mg": 1.8, "molar_mass_g_mol": 469.3,
                        "solvent_mass_g": 3.42, "solvent": "DMF"}
    }
  ]
}
```

Config details:
- `output_dir: null` means `<data folder>/<data file stem>_processed/`.
- `x_unit` is one of `"nm"`, `"eV"`, `"cm-1"`.
- `samples` should list only the spectra to process. Blanks are referenced by
  name and are not listed as samples.
- The `concentration` value takes one of these forms:
  - `null`, when unknown
  - `{"conc_M": 5e-5}`
  - `{"solute_mass_mg":…, "molar_mass_g_mol":…, "solution_volume_mL":…}`
  - `{"solute_mass_mg":…, "molar_mass_g_mol":…, "solvent_mass_g":…, "solvent":"DMF"}`
    (add `"solvent_density_g_mL"` to override the table)
- Spectrum names must match the `inspect` output exactly.

Then run `"$PY" "$TOOLS" build <config.json>`. It can take 10–30 s while the
notebook executes. If it fails, read the error. It usually points to a name
mismatch or a missing concentration field. Fix the config and re-run.

## 6. Check and report

- Look at at least one generated PNG (Read the image) to check the result makes
  sense: baseline near zero, no blown-up noise, sensible ε magnitude. Mention
  anything odd, for example a negative band after subtraction or a large flagged
  region.
- Tell the user what was made, with the folder layout:
  - `<name>.ipynb`: the step-by-step record. Open it in JupyterLab or VS Code and
    use *Run All* after editing the Settings cell.
  - `<stem>_interactive.html`: double-click to open it in a browser. Works
    offline and can be emailed.
    - Dropdowns switch nm/eV/cm⁻¹ and Abs/ε/normalized/raw, and clicking legend
      entries shows or hides spectra.
    - A table lets the user edit pathlength, masses, molar mass and
      concentration, with ε updating live. This is handy if they didn't know
      the molar mass yet.
    - *Save snapshot* writes a PNG/SVG of the current view, plus an
      `_info.txt` of the values used. In Chrome/Edge it saves into the
      `figures/` folder (they pick it once); other browsers save to Downloads.
  - `processed_data/<spectrum name>.txt`: tab-separated, with a `#` header.
  - `figures/<spectrum name>.png` (+ `<stem>_overlay.png` if there are several
    samples).
  - `uvvis_tools.py` + `processing_config.json`: copied in so the folder is
    self-contained and can be re-run on another computer.
- Report the key numbers: concentration used and the largest reliable ε with its
  wavelength. Note any flagged (near-saturated) regions, and suggest diluting or
  using a shorter pathlength if a band of interest is flagged.

## Re-running

To change units or values later, either:
- edit the notebook's Settings cell and Run All, or
- edit `processing_config.json` in the output folder and run `build` on it again.

Rebuilding regenerates the notebook, so it overwrites any manual edits to it.
Warn the user before rebuilding over a notebook they may have edited.
