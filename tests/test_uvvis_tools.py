import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "skills" / "process-uvvis"))
import uvvis_tools as uv  # noqa: E402

DATA = Path(__file__).parent / "data"
CARY = DATA / "synthetic_cary.csv"
EXAMPLE = ROOT / "10.13.2026_Cuimnt.csv"  # real Cary 60 export (not in git)


def test_read_cary_names_and_order():
    spectra = uv.read_uvvis(CARY)
    assert [s.name for s in spectra] == ["MeCN_1cm", "FeL_0.52mg_2.1g_MeCN_1cm", "CoL_10uM_MeCN_1cm",
                                         "DMF_1cm", "NiL_2mg_5mL_DMF_1cm"]
    for s in spectra:
        assert np.all(np.diff(s.x_nm) > 0)
        assert len(s.x_nm) == len(s.absorbance) == 301
    assert spectra[0].metadata["Instrument"] == "Cary 60"
    assert spectra[4].metadata["Collection Time"] == "1/2/2026 1:00:00 PM"


@pytest.mark.skipif(not EXAMPLE.exists(), reason="example data not present")
def test_read_real_cary_export():
    spectra = uv.read_uvvis(EXAMPLE)
    assert [s.name for s in spectra] == ["DMF_0.1cm", "Cu_imnt_1.8mg_3.42g_DMF_0.1cm"]
    assert spectra[1].metadata["Instrument"] == "Cary 60"
    assert spectra[1].metadata["Collection Time"] == "10/13/2026 1:23:18 PM"
    assert uv.pair_blanks(spectra) == {"Cu_imnt_1.8mg_3.42g_DMF_0.1cm": "DMF_0.1cm"}


def test_read_generic_ev_file_converts_to_nm():
    (s,) = uv.read_uvvis(DATA / "generic_ev.txt")
    assert s.x_nm.min() == pytest.approx(200, abs=0.01) and s.x_nm.max() == pytest.approx(800, abs=0.01)
    assert s.x_nm[np.argmax(s.absorbance)] == pytest.approx(450, abs=1)
    assert any("eV" in n for n in s.notes)


def test_read_multi_y_transmittance():
    spectra = uv.read_uvvis(DATA / "multi_y_T.csv")
    assert [s.name for s in spectra] == ["sampleA", "sampleB"]
    a = spectra[0]
    # %T named only "sampleA", so no conversion happens; values stay as given
    assert a.x_nm[np.argmin(a.absorbance)] == pytest.approx(450, abs=1)


def test_parse_name():
    p = uv.parse_name("Cu_imnt_1.8mg_3.42g_DMF_0.1cm")
    assert p["solute_mass_mg"] == 1.8 and p["solvent_mass_g"] == 3.42
    assert p["pathlength_cm"] == 0.1 and p["solvent"] == "DMF" and p["solvent_density_g_mL"] == 0.944
    q = uv.parse_name("CoL_10uM_MeCN_1cm")
    assert q["conc_M"] == pytest.approx(1e-5) and q["solvent"] == "MeCN"
    r = uv.parse_name("NiL_2mg_5mL_DMF_10mm")
    assert r["solution_volume_mL"] == 5 and r["pathlength_cm"] == 1.0


def test_blank_detection_and_pairing():
    spectra = uv.read_uvvis(CARY)
    assert [uv.is_blank(s.name) for s in spectra] == [True, False, False, True, False]
    assert uv.pair_blanks(spectra) == {
        "FeL_0.52mg_2.1g_MeCN_1cm": "MeCN_1cm",
        "CoL_10uM_MeCN_1cm": "MeCN_1cm",
        "NiL_2mg_5mL_DMF_1cm": "DMF_1cm",
    }
    assert uv.is_blank("blank_1") and not uv.is_blank("sample_A")


def test_concentration():
    c, steps = uv.concentration_M(solute_mass_mg=1.8, molar_mass_g_mol=800, solvent_mass_g=3.42, solvent="DMF")
    assert c == pytest.approx((1.8e-3 / 800) / (3.42 / 0.944 * 1e-3))
    assert len(steps) == 3
    c2, _ = uv.concentration_M(solute_mass_mg=2, molar_mass_g_mol=500, solution_volume_mL=5)
    assert c2 == pytest.approx(8e-4)
    assert uv.concentration_M(conc_M=1e-4)[0] == 1e-4
    with pytest.raises(ValueError):
        uv.concentration_M(solute_mass_mg=1, molar_mass_g_mol=100, solvent_mass_g=1, solvent="unobtainium")


def test_unit_conversion_round_trip():
    x = np.array([250.0, 500.0, 1000.0])
    assert uv.convert_x(500, "eV") == pytest.approx(2.4797, abs=1e-4)
    assert uv.convert_x(500, "cm-1") == pytest.approx(20000)
    for u in uv.X_UNITS:
        assert np.allclose(uv.convert_x(uv.convert_x(x, u), u), x)


def test_process_subtraction_epsilon_and_masks():
    spectra = {s.name: s for s in uv.read_uvvis(CARY)}
    p = uv.process(spectra["FeL_0.52mg_2.1g_MeCN_1cm"], spectra["MeCN_1cm"], pathlength_cm=1.0,
                   concentration={"conc_M": 1e-4}, solvent_cutoff_abs=1.0, saturation_abs=2.5)
    assert p.abs_blank.max() <= 1.0  # solvent cutoff trimmed
    i = np.argmin(np.abs(p.x_nm - 450))
    assert p.abs_corr[i] == pytest.approx(0.5, abs=1e-3)
    assert p.epsilon[i] == pytest.approx(5000, rel=1e-3)
    assert not p.flagged.any()


def test_write_txt_round_trip(tmp_path):
    spectra = {s.name: s for s in uv.read_uvvis(CARY)}
    p = uv.process(spectra["CoL_10uM_MeCN_1cm"], spectra["MeCN_1cm"], pathlength_cm=1.0,
                   concentration={"conc_M": 1e-5})
    path = uv.write_txt(p, tmp_path / "a.txt", source="x.csv", metadata=spectra["CoL_10uM_MeCN_1cm"].metadata)
    text = path.read_text(encoding="utf-8")
    assert text.startswith("# CoL_10uM_MeCN_1cm\n")
    arr = np.loadtxt(path, comments="#")
    assert arr.shape == (len(p.x_nm), 8)
    assert np.allclose(arr[:, 5], p.abs_corr, rtol=1e-5)


def test_safe_filename():
    assert uv.safe_filename('a/b:c*?"d') == "a_b_c___d"


def test_build_end_to_end(tmp_path):
    cfg = {
        "data_file": str(CARY),
        "output_dir": str(tmp_path / "out"),
        "x_unit": "eV",
        "samples": [
            {"name": "FeL_0.52mg_2.1g_MeCN_1cm", "blank": "MeCN_1cm", "pathlength_cm": 1.0,
             "concentration": {"solute_mass_mg": 0.52, "molar_mass_g_mol": 600, "solvent_mass_g": 2.1,
                               "solvent": "MeCN"}},
            {"name": "CoL_10uM_MeCN_1cm", "blank": "MeCN_1cm", "pathlength_cm": 1.0, "concentration": None},
            {"name": "NiL_2mg_5mL_DMF_1cm", "blank": "DMF_1cm", "pathlength_cm": 1.0,
             "concentration": {"conc_M": 2e-3}},
        ],
    }
    cfg_path = tmp_path / "processing_config.json"
    cfg_path.write_text(json.dumps(cfg))
    result = uv.build(cfg_path, execute=True)
    out = Path(result["output_dir"])
    assert len(result["processed_data"]) == 3
    assert {Path(f).name for f in result["figures"]} == {
        "FeL_0.52mg_2.1g_MeCN_1cm.png", "CoL_10uM_MeCN_1cm.png", "NiL_2mg_5mL_DMF_1cm.png",
        "synthetic_cary_overlay.png"}
    html = Path(result["interactive_html"]).read_text(encoding="utf-8")
    assert "updatemenus" in html and "Wavenumber" in html
    assert (out / "synthetic_cary_processing.ipynb").exists()
    assert (out / "uvvis_tools.py").exists()
