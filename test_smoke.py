"""Smoke tests. Run: python -m pytest test_smoke.py -q"""
import numpy as np
import pytest

from core.database import SpeciesDB, SpeciesDBError
from core.inference import free_energy, normalised_entropy, softmax
from core.morphkey import CHARACTERS, MATRIX, run_key


@pytest.fixture(scope="module")
def db():
    return SpeciesDB.load()


def test_database_validates(db):
    assert len(db) >= 25


def test_every_species_has_a_key_profile(db):
    assert not [i for i in db.ids if i not in MATRIX], "species missing from key matrix"
    assert not [i for i in MATRIX if i not in db.ids], "key matrix has unknown species"


def test_key_states_are_declared():
    for sid, profile in MATRIX.items():
        for char, expected in profile.items():
            assert char in CHARACTERS, f"{sid}: unknown character {char}"
            if expected is None:
                continue
            states = expected if isinstance(expected, tuple) else (expected,)
            for s in states:
                assert s in CHARACTERS[char][1], f"{sid}.{char}: bad state {s}"


def test_confusion_pairs_resolve(db):
    for sp in db:
        for pair in sp.confusion_with:
            assert pair["species_id"] in db


def test_flapshell_resolves_uniquely():
    survivors = [r.species_id for r in run_key(
        {"shell_surface": "soft", "femoral_flaps": "yes"}) if not r.contradicted]
    assert survivors == ["lissemys_punctata"]


def test_star_tortoise_resolves_uniquely():
    survivors = [r.species_id for r in run_key(
        {"limbs": "columnar", "carapace_pattern": "star"}) if not r.contradicted]
    assert survivors == ["geochelone_elegans"]


def test_tecta_smithii_separate_on_plastron():
    tecta = [r.species_id for r in run_key(
        {"keels": "one", "third_vertebral": "spined",
         "plastron_colour": "coral"}) if not r.contradicted]
    smithii = [r.species_id for r in run_key(
        {"keels": "one", "third_vertebral": "low",
         "plastron_colour": "blotched"}) if not r.contradicted]
    assert tecta == ["pangshura_tecta"]
    assert "pangshura_smithii" in smithii
    assert "pangshura_tecta" not in smithii


def test_legally_divergent_pair_is_flagged(db):
    """The tecta/smithii split changes the offence category — it must differ."""
    assert db.get("pangshura_tecta").wpa != db.get("pangshura_smithii").wpa
    assert db.discriminator("pangshura_tecta", "pangshura_smithii")


def test_geographic_prior_ordering(db):
    mp = "Madhya Pradesh"
    assert db.get("lissemys_punctata").geo_prior(mp) > db.get("geochelone_elegans").geo_prior(mp)
    assert db.get("geochelone_elegans").geo_prior(mp) > db.get("nilssonia_nigricans").geo_prior(mp)
    assert db.get("nilssonia_nigricans").geo_prior(mp) > 0, "out-of-range must never be ruled out"


def test_temperature_softens_confidence():
    logits = np.array([6.0, 1.0, 0.5, 0.2])
    assert softmax(logits, 1.0).max() > softmax(logits, 3.0).max()
    assert pytest.approx(softmax(logits, 2.0).sum()) == 1.0


def test_entropy_bounds():
    assert normalised_entropy(np.array([1.0, 0.0, 0.0, 0.0])) == 0.0
    assert normalised_entropy(np.repeat(0.25, 4)) == pytest.approx(1.0)


def test_energy_separates_confident_from_flat():
    confident = np.array([8.0, 0.1, 0.0, 0.0])
    flat = np.array([0.1, 0.1, 0.1, 0.1])
    assert free_energy(confident) < free_energy(flat), "in-distribution must score lower"


def test_energy_is_not_degenerate_on_true_logits():
    """Regression: log-probabilities give a constant 0 energy that rejects nothing."""
    log_probs = np.log(np.array([0.97, 0.02, 0.005, 0.005]))
    assert abs(free_energy(log_probs, 1.0)) < 1e-9, "log-probs are degenerate, as expected"
    assert abs(free_energy(np.array([8.0, 0.1, 0.0, 0.0]), 1.0)) > 1.0, "true logits must vary"


def test_bad_temperature_rejected():
    with pytest.raises(ValueError):
        softmax(np.array([1.0, 2.0]), 0.0)


# ---------------------------------------------------------------- contributions

def test_coordinates_are_scrubbed():
    from core.contributions import scrub_free_text
    for text in [
        "Found at 23.4521, 78.9012 near the causeway",
        "22° 58' 12.4\" N 78° 39' 40.1\" E on the sandbar",
    ]:
        cleaned, removed = scrub_free_text(text)
        assert "coordinates" in removed
        assert "23.4521" not in cleaned and "78.9012" not in cleaned


def test_long_numbers_scrubbed():
    from core.contributions import scrub_free_text
    cleaned, removed = scrub_free_text("contact 9876543210")
    assert "9876543210" not in cleaned
    assert removed


def test_ordinary_notes_survive_scrubbing():
    from core.contributions import scrub_free_text
    text = "Large female basking on a midstream rock, about 40 cm"
    cleaned, removed = scrub_free_text(text)
    assert cleaned == text and not removed


def test_proposal_requires_citation():
    from core.contributions import ContributionError, submit_proposal
    with pytest.raises(ContributionError):
        submit_proposal(
            kind="correction", species_id="lissemys_punctata", field="habitat",
            current_value="x", proposed_value="y", citation="", contributor="t",
        )


# ---------------------------------------------------------------- TRAFFIC crosswalk

def test_wpa_matches_traffic_enforcement_reference(db):
    """The offence category must agree with the cards issued to enforcement."""
    import json
    from config import DATA_DIR
    ref = json.loads((DATA_DIR / "traffic_2023.json").read_text(encoding="utf-8"))["taxa"]
    mismatched = [
        f"{db.get(sid).scientific_name}: local {db.get(sid).wpa} / TRAFFIC {e['wpa']}"
        for sid, e in ref.items() if sid in db and db.get(sid).wpa != e["wpa"]
    ]
    assert not mismatched, mismatched


def test_schedule_two_taxa_are_exactly_three(db):
    """Under the 2022 amendment only these three chelonians are Schedule II."""
    sched2 = {sp.id for sp in db if sp.wpa == "Schedule II"}
    assert sched2 == {"melanochelys_trijuga", "pangshura_smithii", "cyclemys_gemeli"}


def test_iucn_client_degrades_without_token(tmp_path):
    from core.iucn import IUCNClient
    client = IUCNClient(token=None, cache_path=tmp_path / "c.json")
    assert client.configured is False
    assert client.fetch_species("Batagur kachuga") is None  # returns None, never raises
