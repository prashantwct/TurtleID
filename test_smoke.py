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


# ---------------------------------------------------------------- reference plates

def test_plate_manifest_ids_are_real_species(db):
    """A manifest entry for an unknown id would never render and never be noticed."""
    import json

    from config import REFERENCE_MANIFEST
    if not REFERENCE_MANIFEST.is_file():
        pytest.skip("no reference manifest in this checkout")
    manifest = json.loads(REFERENCE_MANIFEST.read_text(encoding="utf-8"))
    unknown = [sid for sid in manifest["images"] if sid not in db]
    assert not unknown, unknown


def test_plate_manifest_records_a_source_for_every_image():
    """Provenance is the whole reason the manifest is tracked."""
    import json

    from config import REFERENCE_MANIFEST
    if not REFERENCE_MANIFEST.is_file():
        pytest.skip("no reference manifest in this checkout")
    manifest = json.loads(REFERENCE_MANIFEST.read_text(encoding="utf-8"))
    for species_id, entries in manifest["images"].items():
        for entry in entries:
            assert entry.get("source") in manifest["sources"], f"{species_id}: {entry}"
            assert manifest["sources"][entry["source"]].get("rights"), "source states no rights"


def test_missing_plate_files_are_not_offered():
    """The tracked manifest with untracked images is the normal deployed state."""
    from core.plates import plates_for
    for plate in plates_for("lissemys_punctata"):
        assert plate.path.is_file()


def test_plate_lookup_survives_an_unknown_species():
    from core.plates import plates_for
    assert plates_for("no_such_species") == []


# ---------------------------------------------------------------- durable storage

class FakeS3:
    """Records what would be sent. Nothing here talks to a network."""

    def __init__(self, fail_on: str | None = None):
        self.objects: dict[str, bytes] = {}
        self.fail_on = fail_on

    def put_object(self, *, Bucket, Key, Body, ContentType):  # noqa: N803
        if self.fail_on and self.fail_on in Key:
            raise RuntimeError("bucket unreachable")
        self.objects[Key] = Body

    def get_object(self, *, Bucket, Key):  # noqa: N803
        import io
        return {"Body": io.BytesIO(self.objects[Key])}

    def list_objects_v2(self, **kwargs):
        prefix = kwargs.get("Prefix", "")
        return {"Contents": [{"Key": k} for k in sorted(self.objects) if k.startswith(prefix)],
                "IsTruncated": False}


@pytest.fixture
def s3_settings():
    from core import storage
    # Credentials long enough that a substring check means something — a
    # one-character secret matches inside any URL and proves nothing.
    return storage.Settings(
        bucket="chelonid", access_key="AKIAEXAMPLEACCESSKEY",
        secret_key="wJalrXUtnFEMIEXAMPLEKEYzrfiCYEXAMPLEKEY",
        endpoint="https://example.invalid", region="us-east-1", prefix="field/",
    )


def test_storage_is_off_until_a_bucket_is_named(monkeypatch):
    from core import storage
    monkeypatch.setattr(storage, "env_value", lambda key: None)
    assert storage.settings() is None
    assert storage.configured() is False
    assert storage.describe() == "local only"


def test_a_bucket_without_credentials_is_an_error_not_a_default(monkeypatch):
    """Silently falling back to local would look identical until the first reboot."""
    from core import storage
    monkeypatch.setattr(
        storage, "env_value",
        lambda key: "chelonid" if key == storage.BUCKET_ENV else None,
    )
    with pytest.raises(storage.StorageError):
        storage.settings()
    assert storage.configured() is False          # never raises at the UI boundary
    assert "misconfigured" in storage.describe()


def test_keys_carry_the_configured_prefix(s3_settings):
    from core import storage
    fake = FakeS3()
    key = storage.put_image("lissemys_punctata_abc.jpg", b"jpegbytes",
                            s3=fake, config=s3_settings)
    assert key == "field/images/lissemys_punctata_abc.jpg"
    assert fake.objects[key] == b"jpegbytes"


def test_records_round_trip_through_storage(s3_settings):
    import json

    from core import storage
    fake = FakeS3()
    record = {"id": "4f2a9c1e", "species_id": "pangshura_tecta", "image_file": "x.jpg"}
    key = storage.put_record(record, s3=fake, config=s3_settings)
    assert storage.list_records(s3=fake, config=s3_settings) == [key]
    assert json.loads(storage.fetch(key, s3=fake, config=s3_settings)) == record


def test_a_failed_upload_raises_rather_than_reporting_success(s3_settings):
    """The submission must not be acknowledged if the photograph was not stored."""
    from core import storage
    fake = FakeS3(fail_on="images/")
    with pytest.raises(storage.StorageError):
        storage.put_image("x.jpg", b"data", s3=fake, config=s3_settings)


def test_describe_never_leaks_a_credential(s3_settings, monkeypatch):
    from core import storage
    monkeypatch.setattr(storage, "settings", lambda: s3_settings)
    described = storage.describe()
    assert s3_settings.secret_key not in described
    assert s3_settings.access_key not in described
    assert s3_settings.bucket in described


# ---------------------------------------------------------------- dataset splitting

def test_capture_id_groups_a_burst_together():
    from pathlib import Path

    from training.prepare_dataset import capture_id
    assert capture_id(Path("kanha-2024-06-11--03.jpg")) == "kanha-2024-06-11"
    assert capture_id(Path("kanha-2024-06-11--04.jpg")) == "kanha-2024-06-11"
    assert capture_id(Path("IMG_0431.jpg")) == "IMG_0431"


def test_no_capture_lands_in_two_splits():
    """Split by animal, not by photograph — the whole point of the grouping."""
    import random

    from training.prepare_dataset import split_captures
    captures = [f"capture{n}" for n in range(8)]
    train, val, test = split_captures(captures, 0.2, 0.1, random.Random(17))
    assert sorted(train + val + test) == sorted(captures)
    assert not (set(train) & set(val)) and not (set(train) & set(test))
    assert not (set(val) & set(test))
    assert val, "eight captures must yield a validation split"


def test_capture_ids_cannot_contain_the_frame_separator():
    """`one--two` as a capture id would split into a different capture on read."""
    import re

    from training.ingest_field_images import CAPTURE_RE
    assert CAPTURE_RE.match("chambal-2026-08-19")
    assert CAPTURE_RE.match("IMG_0431")
    assert not CAPTURE_RE.match("with space")
    assert not CAPTURE_RE.match("-leading")
    assert re.search("--", "two--parts"), "the guard in ingest also rejects this"


def test_ingested_names_round_trip_through_the_splitter():
    """What ingest writes must be what prepare_dataset groups on."""
    from pathlib import Path

    from training.prepare_dataset import capture_id
    assert capture_id(Path("chambal-2026-08-19--01.jpg")) == "chambal-2026-08-19"
    assert capture_id(Path("chambal-2026-08-19--12.jpg")) == "chambal-2026-08-19"


def test_contributions_from_one_person_on_one_day_are_one_capture():
    """Grouping too hard costs flexibility; grouping too little leaks an animal."""
    from training.promote_contributions import capture_for
    first = {"contributor": "R. Officer, Churna", "submitted_utc": "2026-08-22T04:10:00+00:00"}
    second = {"contributor": "R. Officer, Churna", "submitted_utc": "2026-08-22T06:55:00+00:00"}
    other_day = {"contributor": "R. Officer, Churna", "submitted_utc": "2026-08-23T04:10:00+00:00"}
    assert capture_for(first, None) == capture_for(second, None)
    assert capture_for(first, None) != capture_for(other_day, None)
    assert capture_for(first, "second-animal") != capture_for(first, None)


def test_promoted_capture_ids_survive_the_splitter():
    """A capture id that the splitter re-reads differently would leak silently."""
    from pathlib import Path

    from training.prepare_dataset import capture_id
    from training.promote_contributions import capture_for
    capture = capture_for(
        {"contributor": "R. Officer, Churna", "submitted_utc": "2026-08-22T04:10:00+00:00"},
        None,
    )
    assert "--" not in capture
    assert capture_id(Path(f"{capture}--01.jpg")) == capture


def test_a_single_capture_class_gets_no_validation_split():
    """Better an admitted blind spot than a validation number that means nothing."""
    import random

    from training.prepare_dataset import split_captures
    train, val, test = split_captures(["only"], 0.2, 0.1, random.Random(17))
    assert train == ["only"]
    assert val == [] and test == []
