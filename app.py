"""
Chelonid-ID — field identification tool for Indian turtles and tortoises.

Run:  streamlit run app.py
"""

from __future__ import annotations

import io
import logging
import os
from pathlib import Path

import streamlit as st

from config import (
    ALLOWED_SUFFIXES,
    APP_LOG_FILE,
    APP_VERSION,
    INDIAN_STATES,
    MAX_UPLOAD_BYTES,
    STATE_CENTROIDS,
    TFTSG_CHECKLIST,
)
from core.contributions import (
    CONTRIBUTION_KINDS,
    ContributionError,
    image_coverage,
    read_proposals,
    scrub_free_text,
    submit_image,
    submit_proposal,
    summarise,
)
from core.database import SpeciesDB, SpeciesDBError
from core.iucn import IUCNClient, compare_with_local
from core import github_storage
from core.inference import STALE_BACKEND, ChelonidIdentifier, backend_of
from core.morphkey import CHARACTERS, most_discriminating, run_key
from core.plates import coverage as plate_coverage, plates_for
from core import storage
from core.records import image_fingerprint, log_determination, read_records

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.FileHandler(APP_LOG_FILE), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Chelonid-ID", page_icon="🐢", layout="wide")


def _publish_secrets() -> None:
    """Copy deployment secrets into the environment.

    core/ reads configuration with config.env_value so that the scripts and the
    tests work without Streamlit. This is the one place that knows about
    st.secrets, and it runs before anything reads a setting.

    Existing environment variables win, so a local .env or a shell export is
    not overridden by a deployment secret of the same name.
    """
    try:
        secrets = st.secrets
    except Exception:
        return  # no secrets.toml, which is the normal local case
    for key in ("IUCN_API_TOKEN", "CHELONID_S3_BUCKET", "CHELONID_S3_ACCESS_KEY",
                "CHELONID_S3_SECRET_KEY", "CHELONID_S3_ENDPOINT",
                "CHELONID_S3_REGION", "CHELONID_S3_PREFIX"):
        try:
            value = secrets[key]
        except Exception:
            continue
        if value and not os.environ.get(key):
            os.environ[key] = str(value)


_publish_secrets()

TIER_STYLE = {
    "CONFIRMED": ("✅", "success"),
    "PROBABLE": ("🟡", "warning"),
    "TENTATIVE": ("🟠", "warning"),
    "INDETERMINATE": ("🔴", "error"),
    "REJECTED": ("⛔", "error"),
}

IUCN_COLOUR = {
    "CR": "#d81e05", "EN": "#fc7f3f", "VU": "#f9e814",
    "NT": "#cce226", "LC": "#60c659", "DD": "#d1d1c6",
    "EW": "#542344", "EX": "#000000", "NE": "#ffffff",
}


# ---------------------------------------------------------------- resources
@st.cache_resource(show_spinner=False)
def get_db() -> SpeciesDB:
    return SpeciesDB.load()


@st.cache_resource(show_spinner=False)
def get_identifier(_db: SpeciesDB) -> ChelonidIdentifier:
    return ChelonidIdentifier(_db)


try:
    DB = get_db()
except SpeciesDBError as exc:
    st.error(f"The species reference database failed validation.\n\n{exc}")
    st.stop()

IDENTIFIER = get_identifier(DB)


@st.cache_resource(show_spinner=False)
def get_iucn() -> IUCNClient:
    return IUCNClient()


IUCN = get_iucn()


# ---------------------------------------------------------------- components
def status_chip(species) -> None:
    colour = IUCN_COLOUR.get(species.iucn_status, "#cccccc")
    text = "#000" if species.iucn_status in {"VU", "NT", "LC", "DD", "NE"} else "#fff"
    st.markdown(
        f"""
        <div style="display:flex;gap:8px;flex-wrap:wrap;margin:6px 0 12px 0;">
          <span style="background:{colour};color:{text};padding:3px 11px;
                border-radius:12px;font-size:0.8rem;font-weight:600;">
            IUCN {species.iucn_status} — {species.iucn_label}
          </span>
          <span style="background:#1f4e5f;color:#fff;padding:3px 11px;
                border-radius:12px;font-size:0.8rem;font-weight:600;">
            WPA 2022: {species.wpa}
          </span>
          <span style="background:#4a4a4a;color:#fff;padding:3px 11px;
                border-radius:12px;font-size:0.8rem;font-weight:600;">
            CITES {species.cites}
          </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


CHART_SEQ = 0


def chart_key(species) -> str:
    """A distinct key per rendered chart.

    Streamlit derives an element ID from the element type and its arguments, so
    two maps drawn from the same species collide — and a species can appear on
    more than one tab in a single run. The counter is a module global, which
    Streamlit re-initialises on every rerun, so the keys stay stable within a
    run without accumulating across them.
    """
    global CHART_SEQ
    CHART_SEQ += 1
    return f"map-{species.id}-{CHART_SEQ}"


def distribution_map(species) -> None:
    """Offline state-level presence map. No network call, no basemap tiles."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        st.caption("Install plotly to see the distribution map.")
        st.write("**Recorded from:** " + ", ".join(species.states))
        return

    present = [s for s in species.states if s in STATE_CENTROIDS]
    others = [s for s in STATE_CENTROIDS if s not in present]

    fig = go.Figure()
    fig.add_trace(go.Scattergeo(
        lon=[STATE_CENTROIDS[s][1] for s in others],
        lat=[STATE_CENTROIDS[s][0] for s in others],
        text=others, mode="markers", name="Not recorded",
        marker=dict(size=6, color="#d9d9d9", line=dict(width=0.5, color="#999")),
        hovertemplate="%{text}<extra>Not recorded</extra>",
    ))
    fig.add_trace(go.Scattergeo(
        lon=[STATE_CENTROIDS[s][1] for s in present],
        lat=[STATE_CENTROIDS[s][0] for s in present],
        text=present, mode="markers", name="Recorded",
        marker=dict(size=13, color="#1f77b4", opacity=0.82,
                    line=dict(width=1, color="#0b3d5c")),
        hovertemplate="%{text}<extra>Recorded</extra>",
    ))
    fig.update_geos(
        scope="asia", center=dict(lat=22.5, lon=80),
        lataxis_range=[5, 37], lonaxis_range=[67, 98],
        showcountries=True, countrycolor="#8a8a8a",
        showland=True, landcolor="#f7f7f5",
        showocean=True, oceancolor="#e8f1f5", resolution=50,
    )
    fig.update_layout(
        height=340, margin=dict(l=0, r=0, t=4, b=0),
        legend=dict(orientation="h", y=-0.05),
    )
    st.plotly_chart(fig, use_container_width=True, key=chart_key(species))
    st.caption(
        "State-level presence from the published sources cited below. Indicative "
        "only — not a modelled range polygon, and not survey effort corrected. "
        "Use the GBIF link for point occurrence data."
    )


def reference_plates(species) -> None:
    """Published identification photographs, when they are present on disk.

    A checkout that has not run training/extract_id_cards.py has none, and the
    deployed app is one of those — the manifest is tracked but the image files
    are not, because their copyright is not ours to redistribute. Nothing is
    drawn in that case, not even a placeholder: an empty frame on a species
    card reads as "no photograph exists", which is a different claim.
    """
    found = plates_for(species.id)
    if not found:
        return

    for plate in found:
        st.image(str(plate.path), use_container_width=True)
        st.caption(plate.attribution)


def card_section(label: str, *, nested: bool, expanded: bool = False):
    """A collapsible section, or a plain one when the card is already inside an
    expander. Streamlit refuses to nest expanders, and both the key results and
    the species reference render a card inside one."""
    if nested:
        st.markdown(f"**{label}**")
        return st.container()
    return st.expander(label, expanded=expanded)


def species_card(species, *, expanded_default: bool = True, nested: bool = False) -> None:
    st.markdown(f"### *{species.scientific_name}* {species.authority}")
    st.markdown(
        f"**{species.common_en}**"
        + (f" · {species.common_hi}" if species.common_hi else "")
        + f" · {species.family}"
    )
    status_chip(species)

    reference_plates(species)

    st.info(f"**Key character** — {species.key_character}")

    left, right = st.columns([1, 1])

    with left:
        st.markdown("**Field characters**")
        for d in species.diagnostics:
            st.markdown(f"- {d}")

        if species.habitat:
            st.markdown("**Habitat**")
            st.markdown(species.habitat)

        if species.max_scl_mm:
            st.markdown(f"**Maximum carapace length** — {species.max_scl_mm} mm")

        if species.mp_notes:
            st.markdown("**Madhya Pradesh**")
            st.markdown(species.mp_notes)

    with right:
        distribution_map(species)

    if species.confusion_with:
        with card_section(
            "Species this is confused with",
            nested=nested,
            expanded=expanded_default,
        ):
            for pair in species.confusion_with:
                try:
                    other = DB.get(pair["species_id"])
                    st.markdown(f"**vs. *{other.scientific_name}*** ({other.common_en})")
                except SpeciesDBError:
                    st.markdown(f"**vs. {pair['species_id']}**")
                st.markdown(f"> {pair['discriminator']}")

    cached = IUCN.cached(species.scientific_name)
    if cached:
        verdict = compare_with_local(species.iucn_status, cached)
        with card_section(
            f"IUCN Red List assessment "
            f"({'agrees' if verdict['status'] == 'match' else 'DIVERGES'})",
            nested=nested,
            expanded=(verdict["status"] == "divergent"),
        ):
            if verdict["status"] == "divergent":
                st.warning(verdict["message"])
            cols = st.columns(2)
            with cols[0]:
                if cached.get("population_trend"):
                    st.markdown(f"**Population trend** — {cached['population_trend']}")
                if cached.get("criteria"):
                    st.markdown(f"**Criteria** — {cached['criteria']}")
                if cached.get("threats"):
                    st.markdown("**Threats**")
                    for t in cached["threats"][:6]:
                        st.markdown(f"- {t}")
            with cols[1]:
                if cached.get("use_and_trade"):
                    st.markdown("**Use and trade**")
                    for u in cached["use_and_trade"][:5]:
                        st.markdown(f"- {u}")
                if cached.get("history") and len(cached["history"]) > 1:
                    trail = " -> ".join(
                        f"{h['category']} ({h['year']})"
                        for h in cached["history"] if h.get("category")
                    )
                    st.markdown(f"**Assessment history** — {trail}")
            if cached.get("citation"):
                st.caption(cached["citation"])

    with card_section("Sources", nested=nested):
        for ref in species.references:
            if ref.get("url"):
                st.markdown(f"- {ref['citation']} — [link]({ref['url']})")
            else:
                st.markdown(f"- {ref['citation']}")
        st.markdown("**Look up online**")
        for label, url in species.links.items():
            st.markdown(f"- [{label}]({url})")
        if species.iucn_note:
            st.caption(
                f"IUCN assessment note ({species.iucn_year}): {species.iucn_note}"
            )


def legal_footer() -> None:
    basis = DB.legal_basis
    st.warning(
        f"**{basis.get('caution', '')}**\n\n"
        f"Schedule source: {basis.get('wpa', '')}"
    )


# ---------------------------------------------------------------- tabs
def tab_identify() -> None:
    st.subheader("Photograph identification")

    backend = backend_of(IDENTIFIER)
    if backend == STALE_BACKEND:
        st.error(
            "**This app is running code from before the last update.** The "
            "identification pipeline was reloaded but the process was not "
            "restarted, so what is in memory and what is on disk disagree.\n\n"
            "Reboot the app — on Streamlit Cloud, **Manage app → Reboot** in the "
            "lower right; locally, stop and restart `streamlit run app.py`. The "
            "**Morphological key** tab works regardless."
        )
        return

    if not IDENTIFIER.available:
        st.error(
            "**Nothing is installed to identify photographs with.** Use the "
            "**Morphological key** tab — it is fully functional and depends on "
            "neither a model nor a gallery."
        )
        with st.expander("Two ways to change that"):
            st.markdown(
                "**Matching gallery — minutes, no training, works from a few "
                "photographs per species.** Every photograph is embedded once "
                "and a new one is identified by its nearest neighbours.\n\n"
                "```\n"
                "python -m training.import_folders --setup\n"
                "# drop photographs in: one folder per animal\n"
                "python -m training.import_folders\n"
                "python -m training.build_gallery --seed-with-reference-plates\n"
                "```\n\n"
                "It is weaker on the confusable pairs — *Pangshura tecta* against "
                "*P. smithii* turns on plastron colour, which generic features "
                "barely see — so it stays advisory and defers to the key.\n\n"
                "**Trained classifier — hours, and needs dozens of photographs of "
                "different animals per species.** Better discrimination once the "
                "photographs exist.\n\n"
                "```\n"
                "python -m training.prepare_dataset --pool ./pool --out ./dataset\n"
                "python -m training.train_classifier --data ./dataset\n"
                "python -m training.calibrate --data ./dataset --negatives ./negatives\n"
                "```\n\n"
                "Calibration is not optional on either path. An uncalibrated "
                "model reports confident numbers that are not confidence."
            )
        return

    if backend == "gallery":
        st.info(
            "Running on the **matching gallery**, not a trained model. "
            "Determinations are advisory: confirm anything that matters against "
            "the morphological key, and treat the confusable pairs as unresolved "
            "until you have checked the discriminating character yourself."
        )

    if not IDENTIFIER.calibrated:
        st.warning(
            "This model has not been calibrated. Percentages shown are upper "
            "bounds and will be optimistic. Verify every determination against "
            "the morphological key before recording it."
        )

    col_a, col_b = st.columns([2, 1])

    with col_b:
        state = st.selectbox(
            "State where the animal was found",
            ["— not recorded —"] + INDIAN_STATES,
            index=INDIAN_STATES.index("Madhya Pradesh") + 1,
            help=(
                "Used as a weak geographic prior. An out-of-range species is "
                "down-weighted but never ruled out — that pattern is what a "
                "trade seizure looks like."
            ),
        )
        state = None if state.startswith("—") else state
        observer = st.text_input("Observer / beat", placeholder="e.g. RO, Churna range")
        location_note = st.text_input(
            "Location note", placeholder="e.g. Denwa river, near Madhai"
        )

    with col_a:
        uploaded = st.file_uploader(
            "Photograph",
            type=[s.lstrip(".") for s in sorted(ALLOWED_SUFFIXES)],
            help=(
                "Best results: dorsal (carapace from directly above), ventral "
                "(plastron), lateral profile, head close-up, with a scale."
            ),
        )

    if uploaded is None:
        with st.expander("Photograph capture protocol", expanded=True):
            st.markdown(
                "Five frames, in this order. Most failed identifications are "
                "failed photographs.\n\n"
                "1. **Dorsal** — carapace square-on from directly above, whole "
                "shell in frame, no foreshortening.\n"
                "2. **Ventral** — plastron flat, whole. This single frame "
                "separates *Pangshura tecta* from *P. smithii*, and *P. smithii* "
                "is Schedule II while *P. tecta* is Schedule I.\n"
                "3. **Lateral profile** — eye-level from the side. This is the "
                "only view that shows whether the third vertebral scute is "
                "spined.\n"
                "4. **Head close-up** — filling the frame, in shade, no flash. "
                "Head markings carry more diagnostic weight than shell colour.\n"
                "5. **Scale** — a ruler, or a 10-rupee coin, in the plane of the "
                "carapace.\n\n"
                "Shoot in even shade. Direct sun blows out the pale radiating "
                "rays on a star tortoise and turns a coral plastron white."
            )
        return

    raw = uploaded.getvalue()
    if len(raw) > MAX_UPLOAD_BYTES:
        st.error(
            f"Image is {len(raw) / 1e6:.1f} MB; the limit is "
            f"{MAX_UPLOAD_BYTES / 1e6:.0f} MB. Resize and try again."
        )
        return
    if Path(uploaded.name).suffix.lower() not in ALLOWED_SUFFIXES:
        st.error("Unsupported file type.")
        return

    try:
        from PIL import Image, ImageOps
        image = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")
    except Exception as exc:
        logger.error("Could not decode upload: %s", exc)
        st.error("That file could not be read as an image.")
        return

    st.image(image, caption=uploaded.name, width=380)

    with st.spinner("Identifying..."):
        try:
            result = IDENTIFIER.identify(image, state=state)
        except (FileNotFoundError, ImportError, ValueError) as exc:
            st.error(f"Identification could not run: {exc}")
            return
        except Exception as exc:
            logger.exception("Unexpected failure during identification")
            st.error(
                "Identification failed unexpectedly. The error has been logged. "
                "Use the morphological key for this animal."
            )
            return

    log_determination(
        result.to_record(),
        image_hash=image_fingerprint(raw),
        observer=observer,
        location_note=location_note,
        method="model",
    )

    icon, level = TIER_STYLE.get(result.tier, ("", "info"))
    getattr(st, level)(f"{icon} **{result.tier}** — {result.action}")

    for w in result.warnings:
        st.warning(w)

    if not result.candidates:
        return

    top = result.candidates[0]
    m1, m2, m3 = st.columns(3)
    m1.metric("Confidence", f"{result.confidence_pct}%")
    m2.metric("Ambiguity (entropy)", f"{result.normalised_entropy:.2f}",
              help="0 = decisive, 1 = no information. Above 0.55 forces a downgrade.")
    m3.metric("Occurrence in state", top.occurrence.title())

    st.markdown("#### Ranked candidates")
    rows = []
    for c in result.candidates:
        rows.append({
            "Species": c.scientific_name,
            "Common name": c.common_en,
            "Model": f"{100 * c.model_probability:.1f}%",
            "Geography": f"×{c.geo_multiplier:.2f}",
            "Final": f"{100 * c.posterior:.1f}%",
            "Status in state": c.occurrence,
        })
    st.dataframe(rows, use_container_width=True, hide_index=True)

    st.divider()
    species_card(DB.get(top.species_id))

    with st.expander("Audit trail for this determination"):
        st.json(result.to_record())

    legal_footer()


def tab_key() -> None:
    st.subheader("Morphological key")
    st.markdown(
        "Record whatever characters you can actually see, in any order. Leave "
        "the rest blank. Taxa that contradict an observation are pushed down "
        "but not deleted, because field characters get misread and a silent "
        "elimination of the right answer is the worse failure."
    )

    observations: dict[str, str] = {}
    cols = st.columns(2)
    for i, (char, (question, states)) in enumerate(CHARACTERS.items()):
        with cols[i % 2]:
            labels = ["— not observed —"] + list(states.values())
            picked = st.selectbox(question, labels, key=f"key_{char}")
            if not picked.startswith("—"):
                observations[char] = next(
                    code for code, label in states.items() if label == picked
                )

    if not observations:
        st.info("Select at least one character to begin.")
        return

    results = run_key(observations)
    surviving = [r for r in results if not r.contradicted and r.total_scored > 0]

    st.markdown(f"#### {len(surviving)} taxa consistent with {len(observations)} character(s)")

    if not surviving:
        st.error(
            "No taxon matches every observation. One of the characters is "
            "probably being read differently from the way the key defines it — "
            "most often the keel count or the plastron colour under artificial "
            "light. Clear the least certain character and try again. The closest "
            "partial matches are listed below."
        )
        for r in results[:5]:
            sp = DB.get(r.species_id)
            st.markdown(
                f"- *{sp.scientific_name}* ({sp.common_en}) — "
                f"{r.matched}/{r.total_scored} characters matched"
            )
        return

    if len(surviving) > 1:
        nxt = most_discriminating(observations, [r.species_id for r in surviving])
        if nxt:
            st.info(
                f"**Most useful character to check next:** "
                f"{CHARACTERS[nxt][0]} — it splits the remaining candidates most evenly."
            )

    for r in surviving[:6]:
        sp = DB.get(r.species_id)
        badge = f"{r.matched}/{r.total_scored}"
        with st.expander(
            f"*{sp.scientific_name}* — {sp.common_en}  ·  {badge} characters  ·  "
            f"{sp.iucn_status} · {sp.wpa}",
            expanded=(len(surviving) == 1),
        ):
            species_card(sp, nested=True)

    if len(surviving) == 1:
        sp = DB.get(surviving[0].species_id)
        if st.button("Record this determination", type="primary"):
            ok = log_determination(
                {
                    "tier": "KEY_DETERMINATION",
                    "action": "Determined by morphological key",
                    "candidates": [{
                        "species_id": sp.id,
                        "scientific_name": sp.scientific_name,
                        "posterior": 1.0,
                    }],
                    "observations": observations,
                },
                method="morphological_key",
            )
            st.success("Recorded." if ok else "Determination shown but not logged.")

    legal_footer()


def tab_reference() -> None:
    st.subheader("Species reference")

    c1, c2, c3 = st.columns(3)
    families = ["All"] + sorted({s.family for s in DB})
    family = c1.selectbox("Family", families)
    statuses = ["All"] + ["CR", "EN", "VU", "NT", "LC"]
    status = c2.selectbox("IUCN status", statuses)
    state = c3.selectbox("Occurs in state", ["All"] + INDIAN_STATES)

    query = st.text_input("Search name", placeholder="scientific, English or Hindi name")

    selection = list(DB)
    if family != "All":
        selection = [s for s in selection if s.family == family]
    if status != "All":
        selection = [s for s in selection if s.iucn_status == status]
    if state != "All":
        selection = [s for s in selection if s.occurs_in(state) != "absent"]
    if query:
        q = query.lower()
        selection = [
            s for s in selection
            if q in s.scientific_name.lower()
            or q in s.common_en.lower()
            or q in s.common_hi
        ]

    st.caption(f"{len(selection)} of {len(DB)} taxa")
    for sp in sorted(selection, key=lambda s: (s.family, s.scientific_name)):
        with st.expander(
            f"*{sp.scientific_name}* — {sp.common_en}  ·  {sp.iucn_status} · {sp.wpa}"
        ):
            species_card(sp, nested=True)

    legal_footer()


def tab_records() -> None:
    st.subheader("Determination log")
    entries = read_records()
    if not entries:
        st.info("No determinations recorded yet.")
        return

    tiers: dict[str, int] = {}
    for e in entries:
        t = e.get("determination", {}).get("tier", "unknown")
        tiers[t] = tiers.get(t, 0) + 1

    cols = st.columns(len(tiers) or 1)
    for col, (t, n) in zip(cols, sorted(tiers.items())):
        col.metric(t, n)

    st.markdown("#### Recent")
    for e in reversed(entries[-25:]):
        det = e.get("determination", {})
        cands = det.get("candidates") or [{}]
        name = cands[0].get("scientific_name", "—")
        st.markdown(
            f"`{e.get('timestamp_utc','')}` · **{det.get('tier','')}** · "
            f"*{name}* · {e.get('observer','')} · {e.get('location_note','')}"
        )

    unresolved = [
        e for e in entries
        if e.get("determination", {}).get("tier") in {"INDETERMINATE", "REJECTED", "TENTATIVE"}
    ]
    if unresolved:
        st.info(
            f"**{len(unresolved)} unresolved determinations.** These are the "
            "images worth labelling next — every one is a case the model could "
            "not handle, which makes them worth more per image than another "
            "hundred flapshells."
        )

    st.download_button(
        "Download log (JSONL)",
        data="\n".join(str(e) for e in entries),
        file_name="chelonid_determinations.jsonl",
        mime="application/json",
    )


def publication_notice() -> str | None:
    """The warning owed to a contributor, or None when nothing is published.

    Only the GitHub backend publishes. A private repository needs no notice; a
    public one, or one whose visibility could not be established, does — an
    unanswered visibility check is treated as public, because the cost of being
    wrong runs one way only.
    """
    if not github_storage.configured():
        return None
    visible = github_storage.is_public()
    if visible is False:
        return None
    hedge = "" if visible else (
        "\n\nThe repository's visibility could not be checked just now, so this "
        "is written assuming the worst case."
    )
    return (
        "**Photographs submitted here are published publicly and permanently.** "
        "They are committed to a public GitHub repository, where anyone can see "
        "and download them, along with the name you enter and the state you "
        "select. A commit cannot be withdrawn: deleting a file later leaves it "
        "in the repository's history and does nothing about copies already "
        "taken.\n\n"
        "EXIF coordinates are still stripped and typed coordinates still "
        "redacted — but that cannot help with locality **visible in the frame**: "
        "a recognisable bank, a signboard, a number plate. For a Schedule I "
        "species that is a poaching risk.\n\n"
        "Do not submit a photograph you would not put on a public website."
        + hedge
    )


def tab_contribute() -> None:
    st.subheader("Contribute")
    st.markdown(
        "This tool is only as good as its reference data and its training "
        "images. Both come from people in the field."
    )
    st.info(
        "**Locality data is stripped on submission.** EXIF GPS is removed from "
        "every photograph before it is written to disk, and coordinates typed "
        "into a notes field are redacted. Nothing finer than a state is stored. "
        "A coordinate for a *Batagur kachuga* nesting bank is a poaching risk, "
        "and the model does not need one to learn."
    )
    _notice = publication_notice()
    if _notice:
        st.error(_notice)

    what = st.radio(
        "What are you contributing?",
        list(CONTRIBUTION_KINDS),
        format_func=lambda k: CONTRIBUTION_KINDS[k],
        horizontal=False,
    )
    contributor = st.text_input(
        "Your name and posting", placeholder="e.g. RO Churna range, STR"
    )

    if what == "image":
        _contribute_image(contributor, _notice)
    else:
        _contribute_proposal(what, contributor)

    st.divider()
    counts = summarise()
    if counts:
        st.markdown("#### Contributions received")
        cols = st.columns(len(counts))
        for col, (kind, n) in zip(cols, sorted(counts.items())):
            col.metric(CONTRIBUTION_KINDS.get(kind, kind), n)


def _contribute_image(contributor: str, notice: str | None) -> None:
    coverage = image_coverage(DB.ids)
    thin = sorted(
        ((n, sid) for sid, n in coverage.items() if n < 30),
        key=lambda x: x[0],
    )[:8]
    if thin:
        st.markdown("**Most needed right now** (under 30 images each)")
        st.markdown(
            ", ".join(f"*{DB.get(sid).scientific_name}* ({n})" for n, sid in thin)
        )

    col_a, col_b = st.columns([2, 1])
    with col_b:
        known = st.radio("Do you know the species?", ["Yes", "No / unsure"])
        species_id = None
        certainty = "unidentified"
        if known == "Yes":
            options = sorted(DB.ids, key=lambda i: DB.get(i).scientific_name)
            species_id = st.selectbox(
                "Species", options,
                format_func=lambda i: f"{DB.get(i).scientific_name} — {DB.get(i).common_en}",
            )
            certainty = st.select_slider(
                "How certain?", ["possible", "probable", "confident", "verified"],
                value="confident",
                help="'verified' means confirmed by a chelonian specialist or "
                     "from a vouchered institutional record.",
            )
        view = st.selectbox(
            "View",
            ["dorsal", "ventral (plastron)", "lateral profile", "head close-up", "other"],
        )
        state = st.selectbox("State", ["\u2014 not recorded \u2014"] + INDIAN_STATES)
        state = None if state.startswith("\u2014") else state

    with col_a:
        uploaded = st.file_uploader(
            "Photograph",
            type=[s.lstrip(".") for s in sorted(ALLOWED_SUFFIXES)],
            key="contrib_image",
        )
        notes = st.text_area(
            "Notes (optional)",
            placeholder="Circumstances, approximate size, condition. "
                        "Do not include coordinates \u2014 they will be removed.",
        )

    if uploaded is None:
        st.caption(
            "Ventral (plastron) views are the scarcest and the most valuable. "
            "The *Pangshura tecta* / *P. smithii* split cannot be learned from "
            "carapace photographs, and those two sit on different WPA schedules."
        )
        return

    raw = uploaded.getvalue()
    if len(raw) > MAX_UPLOAD_BYTES:
        st.error(f"Image exceeds {MAX_UPLOAD_BYTES / 1e6:.0f} MB.")
        return

    st.image(raw, width=320)

    # Where submissions are published, consent is a precondition of the button
    # rather than a line of small print above it.
    acknowledged = True
    if notice:
        acknowledged = st.checkbox(
            "I understand this photograph, my name and the state will be "
            "published publicly and permanently, and I have checked that the "
            "image itself does not reveal the locality.",
            key="contrib_publication_consent",
        )

    if st.button("Submit photograph", type="primary", disabled=not acknowledged):
        if not acknowledged:
            return
        try:
            record = submit_image(
                raw, species_id=species_id, view=view, state=state,
                contributor=contributor, notes=notes, certainty=certainty,
            )
        except ContributionError as exc:
            st.error(str(exc))
            return
        st.success(f"Received. Reference {record['id']}.")
        if record.get("stored_key") and notice:
            st.info(
                f"Published at `{record['stored_key']}`. If this was a mistake, "
                "tell the maintainer now — removing it from a repository's "
                "history is only realistic before it spreads."
            )
        if record["exif_gps_present_and_removed"]:
            st.warning(
                "This photograph carried GPS coordinates in its metadata. They "
                "have been removed. Worth knowing that your camera embeds them "
                "\u2014 it matters when sharing images of threatened species by "
                "any route, not just this one."
            )
        if record["redactions"]:
            st.warning(
                "Removed from your notes: " + ", ".join(record["redactions"])
            )


def _contribute_proposal(kind: str, contributor: str) -> None:
    species_id = None
    current = ""
    field = "new_record"

    if kind != "new_species":
        options = sorted(DB.ids, key=lambda i: DB.get(i).scientific_name)
        species_id = st.selectbox(
            "Species", options,
            format_func=lambda i: f"{DB.get(i).scientific_name} — {DB.get(i).common_en}",
            key="proposal_species",
        )
        sp = DB.get(species_id)
        editable = {
            "key_character": sp.key_character,
            "diagnostics": "\n".join(f"- {d}" for d in sp.diagnostics),
            "habitat": sp.habitat,
            "distribution_states": ", ".join(sp.states),
            "iucn.status": sp.iucn_status,
            "cites": sp.cites,
            "wpa_2022": sp.wpa,
            "mp_notes": sp.mp_notes,
        }
        field = st.selectbox("Which field?", list(editable))
        current = editable[field]
        st.text_area("Current value", current, disabled=True, height=120)

    proposed = st.text_area(
        "Proposed value", height=140,
        placeholder="Write the replacement text exactly as it should appear.",
    )
    citation = st.text_area(
        "Citation (required)", height=80,
        placeholder="Paper, handbook, gazette notification, or an institutional "
                    "record with an accession or specimen number.",
    )
    rationale = st.text_area("Why (optional)", height=80)

    st.caption(
        "A citation is required because everything in this database is "
        "traceable to a published source, and field staff act on these "
        "characters. Personal observation is genuinely valuable \u2014 please "
        "submit it as a photograph instead, where it carries real weight."
    )

    if st.button("Submit proposal", type="primary"):
        try:
            record = submit_proposal(
                kind=kind, species_id=species_id, field=field,
                current_value=current, proposed_value=proposed,
                citation=citation, contributor=contributor, rationale=rationale,
            )
        except ContributionError as exc:
            st.error(str(exc))
            return
        st.success(
            f"Received. Reference {record['id']}. A maintainer reviews every "
            "proposal by hand — nothing edits the database automatically."
        )
        if record["redactions"]:
            st.warning("Removed from your text: " + ", ".join(record["redactions"]))


# ---------------------------------------------------------------- shell
st.title("🐢 Chelonid-ID")
st.caption(
    f"Field identification of Indian turtles and tortoises · {len(DB)} taxa · "
    f"v{APP_VERSION}"
)

with st.sidebar:
    st.markdown("### Status")
    st.markdown(f"- Reference database: **{len(DB)} taxa**")
    _backend = backend_of(IDENTIFIER)
    _backend_label = {
        "classifier": "**trained classifier**",
        "gallery": "**matching gallery**",
        STALE_BACKEND: "**unknown — restart required**",
        None: "**not installed**",
    }.get(_backend, "**not installed**")
    st.markdown(f"- Identification: {_backend_label}")
    if _backend == "gallery":
        st.caption("No trained model; matching against embedded photographs.")
    elif _backend == STALE_BACKEND:
        st.caption(
            "This app is still running code from before the last update. "
            "Reboot it — on Streamlit Cloud, **Manage app → Reboot**."
        )
    st.markdown(f"- Calibration: {'**applied**' if IDENTIFIER.calibrated else '**none**'}")
    if IDENTIFIER.calibrated:
        st.caption(f"T = {IDENTIFIER.temperature:.3f}")
    st.markdown(
        f"- IUCN cache: **{len(IUCN.cached_species)} taxa**"
        + (f" (Red List {IUCN.red_list_version})" if IUCN.red_list_version else "")
    )
    st.markdown(f"- Contribution storage: **{storage.describe()}**")
    if not storage.configured():
        st.caption(
            "Submitted photographs are written to this machine only. On a "
            "hosted deployment that disk is wiped on every restart."
        )
    _plates_present, _plates_listed = plate_coverage()
    if _plates_listed:
        st.markdown(f"- Reference plates: **{_plates_present} of {_plates_listed} taxa**")
        if not _plates_present:
            st.caption(
                "Manifest present, image files absent. Run "
                "`python -m training.extract_id_cards --pdf <source>` to restore them."
            )
    st.divider()
    st.markdown("### Scope")
    st.caption(
        "Non-marine chelonians of India: Geoemydidae, Trionychidae, "
        "Testudinidae, plus the exotic Red-eared Slider. Marine turtles are "
        "not covered."
    )
    st.markdown(f"[TFTSG global checklist]({TFTSG_CHECKLIST})")
    st.divider()
    st.caption(
        "This tool assists identification. It does not make determinations of "
        "law. Any species entered on an official form should be confirmed by a "
        "competent authority."
    )

t1, t2, t3, t4, t5 = st.tabs(
    ["Identify", "Morphological key", "Species reference", "Contribute", "Records"]
)
with t1:
    tab_identify()
with t2:
    tab_key()
with t3:
    tab_reference()
with t4:
    tab_contribute()
with t5:
    tab_records()
