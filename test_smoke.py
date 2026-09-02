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


# --------------------------------------------------------------- folder import

def test_species_folders_may_be_named_three_ways():
    """A folder a person types by hand should resolve without a lookup table."""
    from training.import_folders import species_aliases, normalise

    aliases = species_aliases()
    assert aliases[normalise("lissemys_punctata")] == "lissemys_punctata"
    assert aliases[normalise("Lissemys punctata")] == "lissemys_punctata"
    assert aliases[normalise("Indian Flapshell Turtle")] == "lissemys_punctata"


def test_messy_folder_names_become_usable_capture_ids():
    from training.import_folders import capture_from_name
    from training.ingest_field_images import capture_id_error

    for messy in ("chambal aug 19", "Rescue crate #3", "  2026-08-19 (b) ", "a -- b"):
        capture = capture_from_name(messy)
        assert capture, f"{messy!r} produced no capture id"
        assert "--" not in capture, f"{messy!r} -> {capture!r} would collide with the frame separator"
        assert capture_id_error(capture) is None, f"{messy!r} -> {capture!r} rejected downstream"


def test_capture_ids_do_not_collide_after_tidying():
    from training.import_folders import unique_capture

    taken = {"pond-b"}
    assert unique_capture(taken, "pond-b") == "pond-b-2"
    assert unique_capture(taken, "pond-c") == "pond-c"


def test_each_animal_folder_is_one_capture(tmp_path):
    """The folder is what keeps one individual out of both sides of the split."""
    from training.import_folders import scan, species_aliases

    animal = tmp_path / "lissemys_punctata" / "smoketest animal a"
    animal.mkdir(parents=True)
    for n in range(3):
        (animal / f"IMG_{n}.jpg").write_bytes(b"")

    plan = scan(tmp_path, species_aliases(), reimport=False)
    assert len(plan.captures) == 1
    species_id, capture, files, _ = plan.captures[0]
    assert species_id == "lissemys_punctata"
    assert capture == "smoketest-animal-a"
    assert len(files) == 3


def test_loose_photographs_are_separate_animals_and_are_reported(tmp_path):
    from training.import_folders import scan, species_aliases

    species_dir = tmp_path / "geochelone_elegans"
    species_dir.mkdir(parents=True)
    (species_dir / "smoketest-loose-one.jpg").write_bytes(b"")
    (species_dir / "smoketest-loose-two.jpg").write_bytes(b"")

    plan = scan(tmp_path, species_aliases(), reimport=False)
    assert len(plan.captures) == 2, "loose files must not be merged into one animal"
    assert len(plan.loose) == 2, "the guess must be reported, not made silently"


def test_unknown_species_folder_is_refused_not_guessed(tmp_path):
    from training.import_folders import scan, species_aliases

    stray = tmp_path / "some turtle I think"
    stray.mkdir(parents=True)
    (stray / "a.jpg").write_bytes(b"")

    plan = scan(tmp_path, species_aliases(), reimport=False)
    assert plan.captures == []
    assert [p.name for p in plan.unknown_folders] == ["some turtle I think"]


# --------------------------------------------------------------- gallery matching

def _toy_gallery():
    """Three species, two captures each, in a space where the axes are species."""
    from core.matcher import Gallery, normalise_rows

    vectors, species, captures = [], [], []
    for axis, species_id in enumerate(("lissemys_punctata", "pangshura_tecta",
                                       "geochelone_elegans")):
        # Two animals per species, tilted differently off the species axis, so
        # frames of one animal really are closer to each other than to the other
        # animal's — which is the situation leave-one-capture-out exists for.
        for capture, tilt in (("a", 0.05), ("b", 0.40)):
            for frame in (0, 1):
                vector = np.zeros(3)
                vector[axis] = 1.0
                vector[(axis + 1) % 3] = tilt
                vector[(axis + 2) % 3] = 0.01 * frame
                vectors.append(vector)
                species.append(species_id)
                captures.append(capture)
    return Gallery(
        vectors=normalise_rows(np.array(vectors, dtype=np.float32)),
        species=np.array(species),
        captures=np.array(captures),
        classes=sorted(set(species)),
    )


def test_species_score_is_the_mean_of_the_best_matches():
    from core.matcher import ABSENT, species_scores

    similarities = np.array([[0.9, 0.8, 0.1, 0.2]])
    entry_species = np.array(["a", "a", "b", "b"])
    scores = species_scores(similarities, entry_species, ["a", "b", "c"], neighbours=2)

    assert scores[0][0] == pytest.approx(0.85)   # mean of 0.9 and 0.8
    assert scores[0][1] == pytest.approx(0.15)
    assert scores[0][2] == ABSENT, "a species with nothing in the gallery cannot win"


def test_gallery_identifies_a_held_out_capture():
    gallery = _toy_gallery()
    query = gallery.vectors[:1]
    scores, best = gallery.score(query)

    assert gallery.classes[int(scores.argmax())] == "lissemys_punctata"
    assert best[0] == pytest.approx(1.0, abs=1e-5)


def test_leave_one_capture_out_excludes_the_same_animal():
    """The whole point: a photograph must not be identified by its own animal."""
    gallery = _toy_gallery()
    query = gallery.vectors[:1]
    key = gallery.keys[0]

    similarities = query @ gallery.vectors.T
    _, best_without = gallery.score(query, exclude_keys=[key])

    same_capture = gallery.keys == key
    assert same_capture.sum() == 2, "fixture should hold two frames of this animal"
    # The excluded frames were the best matches; the best remaining one is not.
    assert best_without[0] < similarities[0][same_capture].max()
    assert best_without[0] == pytest.approx(similarities[0][~same_capture].max(), abs=1e-6)


def test_leave_one_out_never_matches_across_species_by_capture_name():
    """Capture ids repeat between species; masking must not reach into another."""
    gallery = _toy_gallery()
    keys = gallery.keys
    assert (keys == "lissemys_punctata/a").sum() == 2
    assert (keys == "pangshura_tecta/a").sum() == 2, "same capture name, different species"

    _, best = gallery.score(gallery.vectors[:1], exclude_keys=["lissemys_punctata/a"])
    assert best[0] > 0.0, "masking one species' capture must leave the others intact"


def test_temperature_fitting_improves_likelihood():
    from core.inference import softmax
    from core.matcher import fit_temperature

    scores = np.array([[0.90, 0.60, 0.55], [0.85, 0.50, 0.45], [0.40, 0.88, 0.30]])
    labels = np.array([0, 0, 1])

    fitted = fit_temperature(scores, labels)

    def nll(temperature):
        return -np.mean([np.log(softmax(r, temperature)[l]) for r, l in zip(scores, labels)])

    assert 0.0 < fitted <= 1.05
    assert nll(fitted) < nll(1.0), "fitting must beat the uncalibrated default"


def test_gallery_survives_a_save_and_load(tmp_path):
    gallery = _toy_gallery()
    gallery.temperature = 0.042
    gallery.similarity_floor = 0.61
    gallery.calibrated = True
    gallery.metrics = {"accuracy": 0.75}

    path = tmp_path / "gallery.npz"
    gallery.save(path)

    from core.matcher import Gallery
    loaded = Gallery.load(path)
    assert loaded.classes == gallery.classes
    assert loaded.temperature == pytest.approx(0.042)
    assert loaded.similarity_floor == pytest.approx(0.61)
    assert loaded.calibrated is True
    assert loaded.metrics["accuracy"] == 0.75
    assert np.allclose(loaded.vectors, gallery.vectors)
    assert list(loaded.species) == list(gallery.species)


def test_identifier_prefers_a_trained_model_over_a_gallery(tmp_path):
    """A gallery makes the tab usable early; it must not shadow a real model."""
    from core.database import SpeciesDB
    from core.inference import ChelonidIdentifier

    weights = tmp_path / "chelonid_cls.pt"
    gallery = tmp_path / "gallery.npz"
    db = SpeciesDB.load()

    def identifier():
        return ChelonidIdentifier(db, classifier_path=weights,
                                  calibration_path=tmp_path / "none.json",
                                  gallery_path=gallery,
                                  published_gallery_path=tmp_path / "none.npz")

    assert identifier().backend is None
    gallery.write_bytes(b"")
    assert identifier().backend == "gallery"
    weights.write_bytes(b"")
    assert identifier().backend == "classifier"


def test_a_stale_identifier_is_reported_not_raised():
    """Streamlit can hand back an identifier built before this module changed."""
    from core.inference import STALE_BACKEND, backend_of

    class FromAnOlderVersion:
        available = False
        calibrated = False

    assert backend_of(FromAnOlderVersion()) == STALE_BACKEND


def test_backend_of_reads_a_current_identifier(tmp_path):
    from core.database import SpeciesDB
    from core.inference import ChelonidIdentifier, backend_of

    gallery = tmp_path / "gallery.npz"
    identifier = ChelonidIdentifier(
        SpeciesDB.load(),
        classifier_path=tmp_path / "none.pt",
        calibration_path=tmp_path / "none.json",
        gallery_path=gallery,
        published_gallery_path=tmp_path / "none.npz",
    )
    assert backend_of(identifier) is None
    gallery.write_bytes(b"")
    assert backend_of(identifier) == "gallery"


# --------------------------------------------------------------- publishing

def test_publishing_a_gallery_drops_the_capture_ids():
    """Capture ids name places. Identification never reads them."""
    gallery = _toy_gallery()
    gallery.temperature = 0.03
    gallery.similarity_floor = 0.5
    gallery.calibrated = True

    published = gallery.published()

    assert set(published.captures) == {"unpublished"}
    assert "a" not in set(published.captures) and "b" not in set(published.captures)
    assert np.array_equal(published.vectors, gallery.vectors)
    assert list(published.species) == list(gallery.species)
    assert published.calibrated and published.temperature == pytest.approx(0.03)
    assert published.similarity_floor == pytest.approx(0.5)
    assert published.metrics["published"] is True
    assert set(gallery.captures) == {"a", "b"}, "the original must not be mutated"


def test_a_published_gallery_serves_a_deployment_with_no_local_one(tmp_path):
    from core.database import SpeciesDB
    from core.inference import ChelonidIdentifier

    local, published = tmp_path / "gallery.npz", tmp_path / "published.npz"

    def identifier():
        return ChelonidIdentifier(
            SpeciesDB.load(),
            classifier_path=tmp_path / "none.pt",
            calibration_path=tmp_path / "none.json",
            gallery_path=local,
            published_gallery_path=published,
        )

    assert identifier().backend is None

    published.write_bytes(b"")
    assert identifier().backend == "gallery"
    assert identifier().active_gallery_path == published

    local.write_bytes(b"")
    assert identifier().active_gallery_path == local, "a local build is the newer one"


# --------------------------------------------------------------- github storage

def test_github_storage_is_off_unless_a_repo_is_named(monkeypatch):
    from core import github_storage

    monkeypatch.delenv("CHELONID_GITHUB_REPO", raising=False)
    monkeypatch.delenv("CHELONID_GITHUB_TOKEN", raising=False)
    assert github_storage.settings() is None
    assert github_storage.configured() is False


def test_a_repo_without_a_token_is_reported_not_ignored(monkeypatch):
    """The mistake otherwise looks identical to working, until the first restart."""
    from core import github_storage

    monkeypatch.setenv("CHELONID_GITHUB_REPO", "owner/repo")
    monkeypatch.delenv("CHELONID_GITHUB_TOKEN", raising=False)

    with pytest.raises(github_storage.GitHubStorageError):
        github_storage.settings()
    assert github_storage.configured() is False
    assert github_storage.describe().startswith("misconfigured")


def test_a_malformed_repo_name_is_refused(monkeypatch):
    from core import github_storage

    monkeypatch.setenv("CHELONID_GITHUB_TOKEN", "t")
    for bad in ("not-a-repo", "too/many/parts", "owner/", "/repo"):
        monkeypatch.setenv("CHELONID_GITHUB_REPO", bad)
        with pytest.raises(github_storage.GitHubStorageError):
            github_storage.settings()


def test_github_settings_normalise_the_path_and_default_the_branch(monkeypatch):
    from core import github_storage

    monkeypatch.setenv("CHELONID_GITHUB_REPO", "owner/repo")
    monkeypatch.setenv("CHELONID_GITHUB_TOKEN", "t")
    monkeypatch.setenv("CHELONID_GITHUB_PATH", "/field/photos")
    monkeypatch.delenv("CHELONID_GITHUB_BRANCH", raising=False)

    config = github_storage.settings()
    assert config.path == "field/photos/"
    assert config.branch == "main"
    assert config.path_for("images/x.jpg") == "field/photos/images/x.jpg"


def test_the_github_description_never_carries_the_token(monkeypatch):
    from core import github_storage

    monkeypatch.setenv("CHELONID_GITHUB_REPO", "owner/repo")
    monkeypatch.setenv("CHELONID_GITHUB_TOKEN", "ghp_secret_value")
    config = github_storage.settings()

    assert "ghp_secret_value" not in config.public_description
    assert "ghp_secret_value" not in repr(config.public_description)


def test_github_errors_explain_themselves_without_echoing_the_response(monkeypatch):
    import urllib.error

    from core import github_storage

    monkeypatch.setenv("CHELONID_GITHUB_REPO", "owner/repo")
    monkeypatch.setenv("CHELONID_GITHUB_TOKEN", "ghp_secret_value")
    config = github_storage.settings()

    for code, expected in ((403, "Contents:write"), (404, "not found"),
                           (409, "does not exist"), (422, "already"), (429, "rate-limit")):
        error = urllib.error.HTTPError("u", code, "msg", {}, None)
        message = github_storage._explain(error, config, "submissions/images/x.jpg")
        assert expected in message
        assert "ghp_secret_value" not in message


def test_storage_prefers_github_and_translates_its_errors(monkeypatch):
    """contributions.py only knows StorageError; the backend must not leak through."""
    from core import github_storage, storage

    monkeypatch.setenv("CHELONID_GITHUB_REPO", "owner/repo")
    monkeypatch.setenv("CHELONID_GITHUB_TOKEN", "t")
    monkeypatch.delenv("CHELONID_S3_BUCKET", raising=False)
    assert storage.configured() is True

    def explode(*_args, **_kwargs):
        raise github_storage.GitHubStorageError("commit refused")

    monkeypatch.setattr(github_storage, "put_image", explode)
    with pytest.raises(storage.StorageError, match="commit refused"):
        storage.put_image("x.jpg", b"data")


def test_committed_submissions_are_taken_up_from_the_working_tree(tmp_path, monkeypatch):
    """With the GitHub backend there is nothing to download; git pull did it."""
    import argparse
    import json

    from core import contributions
    from training import pull_contributions

    submissions = tmp_path / "submissions"
    (submissions / "records").mkdir(parents=True)
    (submissions / "images").mkdir(parents=True)
    (submissions / "images" / "lissemys_punctata_ff.jpg").write_bytes(b"jpeg")
    (submissions / "records" / "abc123.json").write_text(json.dumps({
        "id": "abc123", "kind": "image", "species_id": "lissemys_punctata",
        "contributor": "RO Churna", "certainty": "confident",
        "image_file": "lissemys_punctata_ff.jpg", "status": "pending",
    }), encoding="utf-8")

    contrib_dir = tmp_path / "contributions"
    image_dir = contrib_dir / "images"
    monkeypatch.setattr(pull_contributions, "CONTRIB_DIR", contrib_dir)
    monkeypatch.setattr(pull_contributions, "IMAGE_DIR", image_dir)
    monkeypatch.setattr(pull_contributions, "PROPOSAL_FILE", contrib_dir / "proposals.jsonl")
    monkeypatch.setattr(pull_contributions, "local_record_ids", lambda: set())

    args = argparse.Namespace(list=False, from_repo=True, submissions=submissions)
    pull_contributions.pull(args)

    assert (image_dir / "lissemys_punctata_ff.jpg").read_bytes() == b"jpeg"
    written = (contrib_dir / "proposals.jsonl").read_text(encoding="utf-8").strip()
    assert json.loads(written)["id"] == "abc123"


def test_taking_up_submissions_twice_adds_nothing(tmp_path, monkeypatch):
    import argparse
    import json

    from training import pull_contributions

    submissions = tmp_path / "submissions"
    (submissions / "records").mkdir(parents=True)
    (submissions / "images").mkdir(parents=True)
    (submissions / "records" / "abc123.json").write_text(
        json.dumps({"id": "abc123", "species_id": "lissemys_punctata"}), encoding="utf-8")

    contrib_dir = tmp_path / "contributions"
    monkeypatch.setattr(pull_contributions, "CONTRIB_DIR", contrib_dir)
    monkeypatch.setattr(pull_contributions, "IMAGE_DIR", contrib_dir / "images")
    monkeypatch.setattr(pull_contributions, "PROPOSAL_FILE", contrib_dir / "proposals.jsonl")
    monkeypatch.setattr(pull_contributions, "local_record_ids", lambda: {"abc123"})

    args = argparse.Namespace(list=False, from_repo=True, submissions=submissions)
    pull_contributions.pull(args)

    assert not (contrib_dir / "proposals.jsonl").exists(), "a known record must not be re-appended"


def test_a_missing_submissions_directory_says_to_git_pull(tmp_path):
    import argparse

    from training import pull_contributions

    args = argparse.Namespace(list=False, from_repo=True, submissions=tmp_path / "nothing")
    with pytest.raises(SystemExit, match="git pull"):
        pull_contributions.pull(args)


def test_a_storage_reason_reaches_the_contributor(monkeypatch, tmp_path):
    """"Please try again" hides the one line that says what to fix."""
    from core import contributions, storage

    # Pillow is not installed in CI, and this test is about the error path, not
    # about encoding: stand in for the scrubber rather than skipping the test.
    monkeypatch.setattr(contributions, "strip_exif", lambda data: (b"jpeg-bytes", False))
    monkeypatch.setattr(storage, "configured", lambda: True)
    monkeypatch.setattr(storage, "put_image", lambda *a, **k: (_ for _ in ()).throw(
        storage.StorageError("GitHub refused the token (403). It needs Contents:write")))
    monkeypatch.setattr(contributions, "IMAGE_DIR", tmp_path / "images")

    with pytest.raises(contributions.ContributionError) as caught:
        contributions.submit_image(
            b"raw", species_id="lissemys_punctata", view="dorsal",
            state=None, contributor="RO", certainty="confident",
        )
    assert "Contents:write" in str(caught.value)
    assert "403" in str(caught.value)


def test_a_credential_never_reaches_the_contributor(monkeypatch):
    from core import storage

    monkeypatch.setenv("CHELONID_GITHUB_TOKEN", "ghp_supersecrettokenvalue")
    monkeypatch.setenv("CHELONID_S3_SECRET_KEY", "s3secretkeyvalue12345")

    leaky = storage.StorageError(
        "rejected: token=ghp_supersecrettokenvalue key=s3secretkeyvalue12345"
    )
    cleaned = storage.safe_reason(leaky)

    assert "ghp_supersecrettokenvalue" not in cleaned
    assert "s3secretkeyvalue12345" not in cleaned
    assert cleaned.count("<redacted>") == 2


def test_scrubbing_leaves_an_ordinary_message_intact(monkeypatch):
    from core import storage

    monkeypatch.delenv("CHELONID_GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("CHELONID_S3_SECRET_KEY", raising=False)
    monkeypatch.delenv("CHELONID_S3_ACCESS_KEY", raising=False)

    message = "GitHub reports owner/repo as not found (404)."
    assert storage.safe_reason(storage.StorageError(message)) == message


# --------------------------------------------------------------- token handling

def test_a_quoted_token_is_normalised(monkeypatch):
    """env_value strips quotes from .env but not from an environment variable."""
    from core import github_storage

    monkeypatch.setenv("CHELONID_GITHUB_REPO", "owner/repo")
    monkeypatch.setenv("CHELONID_GITHUB_TOKEN", '"github_pat_value"')
    assert github_storage.settings().token == "github_pat_value"

    monkeypatch.setenv("CHELONID_GITHUB_TOKEN", "  github_pat_value  ")
    assert github_storage.settings().token == "github_pat_value"

    monkeypatch.setenv("CHELONID_GITHUB_TOKEN", "'github_pat_value'")
    assert github_storage.settings().token == "github_pat_value"


def test_a_token_of_only_quotes_is_refused(monkeypatch):
    from core import github_storage

    monkeypatch.setenv("CHELONID_GITHUB_REPO", "owner/repo")
    monkeypatch.setenv("CHELONID_GITHUB_TOKEN", '""')
    with pytest.raises(github_storage.GitHubStorageError, match="nothing usable"):
        github_storage.settings()


def test_401_and_403_are_told_apart(monkeypatch):
    """401 is the token value; 403 is its permissions. Conflating them misdirects."""
    import urllib.error

    from core import github_storage

    monkeypatch.setenv("CHELONID_GITHUB_REPO", "owner/repo")
    monkeypatch.setenv("CHELONID_GITHUB_TOKEN", "t")
    config = github_storage.settings()

    unauthorised = github_storage._explain(
        urllib.error.HTTPError("u", 401, "m", {}, None), config, "p")
    forbidden = github_storage._explain(
        urllib.error.HTTPError("u", 403, "m", {}, None), config, "p")

    assert "Bad credentials" in unauthorised
    assert "Contents:write" not in unauthorised, "401 must not send anyone to permissions"
    assert "Contents:write" in forbidden
    assert "revoked" in unauthorised


# --------------------------------------------------------------- the checker

def test_check_stops_at_the_first_thing_that_is_wrong(monkeypatch):
    from core import github_storage

    monkeypatch.setenv("CHELONID_GITHUB_REPO", "owner/repo")
    monkeypatch.setenv("CHELONID_GITHUB_TOKEN", "github_pat_value")
    monkeypatch.setattr(github_storage, "_get", lambda path, config: None)

    rows = github_storage.check()
    labels = [label for label, _, _ in rows]

    assert labels == ["Configuration", "Token shape", "Token authenticates"]
    assert rows[-1][1] is False
    assert "permissions are not the issue" in rows[-1][2]


def test_check_reports_a_read_only_token(monkeypatch):
    from core import github_storage

    monkeypatch.setenv("CHELONID_GITHUB_REPO", "owner/repo")
    monkeypatch.setenv("CHELONID_GITHUB_TOKEN", "github_pat_value")

    def fake_get(path, config):
        if path == "/user":
            return {"login": "someone"}
        if path == "/repos/owner/repo":
            return {"private": False, "permissions": {"push": False}}
        return {"name": "main"}

    monkeypatch.setattr(github_storage, "_get", fake_get)
    rows = dict((label, (ok, detail)) for label, ok, detail in github_storage.check())

    assert rows["Token authenticates"][0] is True
    assert rows["Write permission"][0] is False
    assert "Read and write" in rows["Write permission"][1]


def test_check_passes_a_working_configuration(monkeypatch):
    from core import github_storage

    monkeypatch.setenv("CHELONID_GITHUB_REPO", "owner/repo")
    monkeypatch.setenv("CHELONID_GITHUB_TOKEN", "github_pat_value")

    def fake_get(path, config):
        if path == "/user":
            return {"login": "someone"}
        if path == "/repos/owner/repo":
            return {"private": True, "permissions": {"push": True}}
        return {"name": "main"}

    monkeypatch.setattr(github_storage, "_get", fake_get)
    assert all(ok for _, ok, _ in github_storage.check())


def test_check_never_prints_the_token(monkeypatch):
    from core import github_storage

    monkeypatch.setenv("CHELONID_GITHUB_REPO", "owner/repo")
    monkeypatch.setenv("CHELONID_GITHUB_TOKEN", "github_pat_supersecretvalue")
    monkeypatch.setattr(github_storage, "_get", lambda path, config: None)

    rendered = " ".join(f"{label} {detail}" for label, _, detail in github_storage.check())
    assert "supersecretvalue" not in rendered


# --------------------------------------------------------------- promotion bar

def test_verified_photographs_are_promotable():
    """`verified` is the top of the scale; testing == 'confident' refused the best."""
    from training.promote_contributions import certain_enough

    assert certain_enough("verified") is True
    assert certain_enough("confident") is True


def test_uncertain_photographs_are_held():
    from training.promote_contributions import certain_enough

    assert certain_enough("probable") is False
    assert certain_enough("possible") is False
    assert certain_enough("unidentified") is False
    assert certain_enough(None) is False
    assert certain_enough("") is False


def test_the_promotion_bar_sits_on_the_scale_the_app_offers():
    """If the tab's options and this order drift apart, promotion silently changes."""
    from training.promote_contributions import CERTAINTY_ORDER, MINIMUM_CERTAINTY

    assert MINIMUM_CERTAINTY in CERTAINTY_ORDER
    # The order the Contribute tab presents, weakest to strongest.
    assert CERTAINTY_ORDER == ("possible", "probable", "confident", "verified")
    above = CERTAINTY_ORDER[CERTAINTY_ORDER.index(MINIMUM_CERTAINTY):]
    assert above == ("confident", "verified"), "everything at or above the bar promotes"


# --------------------------------------------------------------- what is loaded

def test_the_summary_names_what_is_loaded(tmp_path):
    """Without counts on screen, a stale deployment looks like a fresh one."""
    from core.database import SpeciesDB
    from core.inference import ChelonidIdentifier, backend_summary
    from core.matcher import Gallery, normalise_rows

    path = tmp_path / "gallery.npz"
    Gallery(
        vectors=normalise_rows(np.eye(3, dtype=np.float32)),
        species=np.array(["lissemys_punctata", "lissemys_punctata", "pangshura_tecta"]),
        captures=np.array(["a", "a", "b"]),
        classes=["lissemys_punctata", "pangshura_tecta"],
    ).save(path)

    identifier = ChelonidIdentifier(
        SpeciesDB.load(),
        classifier_path=tmp_path / "none.pt",
        calibration_path=tmp_path / "none.json",
        gallery_path=path,
        published_gallery_path=tmp_path / "none.npz",
    )
    assert backend_summary(identifier) == "3 photographs, 2 species"


def test_the_summary_survives_an_unreadable_gallery(tmp_path):
    from core.database import SpeciesDB
    from core.inference import ChelonidIdentifier, backend_summary

    broken = tmp_path / "gallery.npz"
    broken.write_bytes(b"not an npz")

    identifier = ChelonidIdentifier(
        SpeciesDB.load(),
        classifier_path=tmp_path / "none.pt",
        calibration_path=tmp_path / "none.json",
        gallery_path=broken,
        published_gallery_path=tmp_path / "none.npz",
    )
    assert backend_summary(identifier) == "could not be read"


def test_the_summary_is_empty_when_nothing_is_installed(tmp_path):
    from core.database import SpeciesDB
    from core.inference import ChelonidIdentifier, backend_summary

    identifier = ChelonidIdentifier(
        SpeciesDB.load(),
        classifier_path=tmp_path / "none.pt",
        calibration_path=tmp_path / "none.json",
        gallery_path=tmp_path / "none.npz",
        published_gallery_path=tmp_path / "also-none.npz",
    )
    assert backend_summary(identifier) == ""


def test_species_counts_show_where_contributions_landed(tmp_path):
    from core.database import SpeciesDB
    from core.inference import ChelonidIdentifier, gallery_species_counts
    from core.matcher import Gallery, normalise_rows

    path = tmp_path / "gallery.npz"
    Gallery(
        vectors=normalise_rows(np.eye(4, dtype=np.float32)),
        species=np.array(["geoclemys_hamiltonii"] * 3 + ["pangshura_tecta"]),
        captures=np.array(["a", "a", "a", "b"]),
        classes=["geoclemys_hamiltonii", "pangshura_tecta"],
    ).save(path)

    identifier = ChelonidIdentifier(
        SpeciesDB.load(),
        classifier_path=tmp_path / "none.pt",
        calibration_path=tmp_path / "none.json",
        gallery_path=path,
        published_gallery_path=tmp_path / "none.npz",
    )
    assert gallery_species_counts(identifier) == {
        "geoclemys_hamiltonii": 3, "pangshura_tecta": 1,
    }


def test_a_stale_identifier_yields_no_summary_rather_than_raising():
    from core.inference import backend_summary, gallery_species_counts

    class FromAnOlderVersion:
        available = False

    assert backend_summary(FromAnOlderVersion()) == ""
    assert gallery_species_counts(FromAnOlderVersion()) == {}


# --------------------------------------------------------------- fitness to use

def _gallery_at(path, *, reliable, accuracy, chance):
    from core.matcher import Gallery, normalise_rows

    gallery = Gallery(
        vectors=normalise_rows(np.eye(2, dtype=np.float32)),
        species=np.array(["lissemys_punctata", "pangshura_tecta"]),
        captures=np.array(["a", "b"]),
        classes=["lissemys_punctata", "pangshura_tecta"],
        calibrated=True,
    )
    gallery.reliable = reliable
    gallery.metrics = {"accuracy": accuracy, "chance": chance, "n_evaluated": 16,
                       "per_class_recall": {"lissemys_punctata": {"n": 8, "recall": 0.0}}}
    gallery.save(path)
    return gallery


def _identifier_for(tmp_path, gallery_path):
    from core.database import SpeciesDB
    from core.inference import ChelonidIdentifier

    return ChelonidIdentifier(
        SpeciesDB.load(),
        classifier_path=tmp_path / "none.pt",
        calibration_path=tmp_path / "none.json",
        gallery_path=gallery_path,
        published_gallery_path=tmp_path / "none.npz",
    )


def test_a_gallery_at_chance_is_reported_unfit(tmp_path):
    """Calibrated and useless are compatible: a fitted temperature over scores
    that carry nothing reports 1/n for everything, with an excellent ECE."""
    from core.inference import gallery_is_unfit

    path = tmp_path / "gallery.npz"
    _gallery_at(path, reliable=False, accuracy=0.0, chance=0.036)

    unfit = gallery_is_unfit(_identifier_for(tmp_path, path))
    assert unfit is not None
    assert unfit["accuracy"] == 0.0
    assert unfit["chance"] == 0.036
    assert unfit["per_class_recall"]["lissemys_punctata"]["recall"] == 0.0


def test_a_gallery_that_beat_chance_is_not_flagged(tmp_path):
    from core.inference import gallery_is_unfit

    path = tmp_path / "gallery.npz"
    _gallery_at(path, reliable=True, accuracy=0.62, chance=0.036)
    assert gallery_is_unfit(_identifier_for(tmp_path, path)) is None


def test_reliability_survives_saving_and_publishing(tmp_path):
    from core.matcher import Gallery

    path = tmp_path / "gallery.npz"
    gallery = _gallery_at(path, reliable=False, accuracy=0.0, chance=0.036)
    assert Gallery.load(path).reliable is False

    published = tmp_path / "published.npz"
    gallery.published().save(published)
    assert Gallery.load(published).reliable is False, "publishing must not launder it"


def test_a_gallery_predating_the_check_is_unproven_not_condemned(tmp_path):
    """Absent evidence is not evidence of failure; those galleries stay usable."""
    import json

    from core.matcher import Gallery, normalise_rows

    path = tmp_path / "gallery.npz"
    np.savez_compressed(
        path,
        vectors=normalise_rows(np.eye(2, dtype=np.float32)),
        species=np.array(["lissemys_punctata", "pangshura_tecta"]),
        captures=np.array(["a", "b"]),
        meta=np.array(json.dumps({
            "classes": ["lissemys_punctata", "pangshura_tecta"],
            "backbone": "resnet50", "neighbours": 3, "temperature": 0.07,
            "similarity_floor": 0.3, "calibrated": True, "metrics": {},
        })),
    )
    assert Gallery.load(path).reliable is True


def test_the_unfit_guard_is_quiet_when_no_gallery_is_installed(tmp_path):
    from core.inference import gallery_is_unfit

    assert gallery_is_unfit(_identifier_for(tmp_path, tmp_path / "absent.npz")) is None

    class FromAnOlderVersion:
        available = False

    assert gallery_is_unfit(FromAnOlderVersion()) is None


def test_an_older_gallery_is_judged_on_the_accuracy_it_recorded(tmp_path):
    """The galleries most needing the verdict are the ones already deployed."""
    import json

    from core.matcher import Gallery, normalise_rows

    def write(metrics):
        path = tmp_path / f"g{len(metrics)}{metrics.get('accuracy')}.npz"
        np.savez_compressed(
            path,
            vectors=normalise_rows(np.eye(2, dtype=np.float32)),
            species=np.array(["lissemys_punctata", "pangshura_tecta"]),
            captures=np.array(["a", "b"]),
            meta=np.array(json.dumps({
                "classes": ["lissemys_punctata", "pangshura_tecta"],
                "backbone": "resnet50", "neighbours": 3, "temperature": 0.07,
                "similarity_floor": 0.3, "calibrated": True, "metrics": metrics,
            })),
        )
        return Gallery.load(path)

    # Two classes, so chance is 0.5.
    assert write({"accuracy": 0.0}).reliable is False
    assert write({"accuracy": 0.5}).reliable is False, "matching chance is not beating it"
    assert write({"accuracy": 0.9}).reliable is True
    assert write({}).reliable is True, "nothing recorded means unproven, not bad"


def test_the_unfit_report_always_has_numbers_to_render(tmp_path):
    """The app formats both as percentages; a None would take the page down."""
    from core.inference import gallery_is_unfit
    from core.matcher import Gallery, normalise_rows

    path = tmp_path / "gallery.npz"
    gallery = Gallery(
        vectors=normalise_rows(np.eye(2, dtype=np.float32)),
        species=np.array(["lissemys_punctata", "pangshura_tecta"]),
        captures=np.array(["a", "b"]),
        classes=["lissemys_punctata", "pangshura_tecta"],
        calibrated=True,
    )
    gallery.reliable = False
    gallery.metrics = {"accuracy": 0.0}          # no chance, no n_evaluated
    gallery.save(path)

    unfit = gallery_is_unfit(_identifier_for(tmp_path, path))
    assert unfit["chance"] == pytest.approx(0.5), "derived from the class count"
    assert f"{unfit['accuracy']:.0%}" == "0%"
    assert f"{unfit['chance']:.0%}" == "50%"
    assert unfit["n_evaluated"] == 0
