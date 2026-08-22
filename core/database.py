"""
Loading and validation of the species reference database.

The database is the part of this tool that carries authority. The model only
ranks candidates; every legal status, diagnostic character and citation shown
to a user comes from here. So it is validated on load rather than trusted.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from config import (
    GBIF_SEARCH,
    GEO_PRIOR,
    INDIA_BIODIVERSITY,
    IUCN_SEARCH,
    SPECIES_DB_PATH,
)

logger = logging.getLogger(__name__)

REQUIRED_FIELDS = (
    "id", "scientific_name", "family", "common_en", "iucn", "cites",
    "wpa_2022", "distribution_states", "mp_status", "key_character",
    "diagnostics", "references",
)

VALID_IUCN = {"EX", "EW", "CR", "EN", "VU", "NT", "LC", "DD", "NE"}

IUCN_LABEL = {
    "CR": "Critically Endangered", "EN": "Endangered", "VU": "Vulnerable",
    "NT": "Near Threatened", "LC": "Least Concern", "DD": "Data Deficient",
    "EW": "Extinct in the Wild", "EX": "Extinct", "NE": "Not Evaluated",
}


class SpeciesDBError(Exception):
    """Raised when the reference database fails validation."""


@dataclass
class Species:
    raw: dict[str, Any] = field(repr=False)

    # -- identity ------------------------------------------------------
    @property
    def id(self) -> str:
        return self.raw["id"]

    @property
    def scientific_name(self) -> str:
        return self.raw["scientific_name"]

    @property
    def authority(self) -> str:
        return self.raw.get("authority", "")

    @property
    def family(self) -> str:
        return self.raw["family"]

    @property
    def common_en(self) -> str:
        return self.raw["common_en"]

    @property
    def common_hi(self) -> str:
        return self.raw.get("common_hi", "")

    # -- status --------------------------------------------------------
    @property
    def iucn_status(self) -> str:
        return self.raw["iucn"]["status"]

    @property
    def iucn_label(self) -> str:
        return IUCN_LABEL.get(self.iucn_status, self.iucn_status)

    @property
    def iucn_year(self) -> int | None:
        return self.raw["iucn"].get("year")

    @property
    def iucn_note(self) -> str:
        return self.raw["iucn"].get("note", "")

    @property
    def cites(self) -> str:
        return self.raw["cites"]

    @property
    def wpa(self) -> str:
        return self.raw["wpa_2022"]

    @property
    def is_threatened(self) -> bool:
        return self.iucn_status in {"CR", "EN", "VU"}

    # -- distribution --------------------------------------------------
    @property
    def states(self) -> list[str]:
        return self.raw["distribution_states"]

    @property
    def mp_status(self) -> str:
        return self.raw["mp_status"]

    @property
    def mp_notes(self) -> str:
        return self.raw.get("mp_notes", "")

    @property
    def habitat(self) -> str:
        return self.raw.get("habitat", "")

    @property
    def max_scl_mm(self) -> int | None:
        return self.raw.get("max_scl_mm")

    # -- identification ------------------------------------------------
    @property
    def key_character(self) -> str:
        return self.raw["key_character"]

    @property
    def diagnostics(self) -> list[str]:
        return self.raw["diagnostics"]

    @property
    def confusion_with(self) -> list[dict[str, str]]:
        return self.raw.get("confusion_with", [])

    @property
    def references(self) -> list[dict[str, str]]:
        return self.raw["references"]

    # -- derived -------------------------------------------------------
    def occurs_in(self, state: str | None) -> str:
        """Occurrence class for a given state: resident / marginal / absent."""
        if not state:
            return "unknown"
        if state == "Madhya Pradesh":
            return self.mp_status
        if state in self.states:
            return "introduced" if self.mp_status == "introduced" else "resident"
        return "absent"

    def geo_prior(self, state: str | None) -> float:
        return GEO_PRIOR.get(self.occurs_in(state), GEO_PRIOR["unknown"])

    @property
    def links(self) -> dict[str, str]:
        q = self.scientific_name.replace(" ", "%20")
        return {
            "IUCN Red List assessment": IUCN_SEARCH.format(name=q),
            "GBIF occurrence map": GBIF_SEARCH.format(name=q),
            "India Biodiversity Portal": INDIA_BIODIVERSITY.format(name=q),
        }


class SpeciesDB:
    """Validated, indexed access to the species reference database."""

    def __init__(self, payload: dict[str, Any]):
        self._meta = {k: v for k, v in payload.items() if k != "species"}
        self._by_id: dict[str, Species] = {}
        for record in payload["species"]:
            sp = Species(record)
            self._by_id[sp.id] = sp

    # -- construction --------------------------------------------------
    @classmethod
    def load(cls, path=SPECIES_DB_PATH) -> "SpeciesDB":
        try:
            with open(path, encoding="utf-8") as fh:
                payload = json.load(fh)
        except FileNotFoundError as exc:
            raise SpeciesDBError(f"Species database not found at {path}") from exc
        except json.JSONDecodeError as exc:
            raise SpeciesDBError(f"Species database is not valid JSON: {exc}") from exc

        cls._validate(payload)
        db = cls(payload)
        logger.info("Loaded species database: %d taxa", len(db))
        return db

    @staticmethod
    def _validate(payload: dict[str, Any]) -> None:
        if "species" not in payload or not isinstance(payload["species"], list):
            raise SpeciesDBError("Database must contain a 'species' list.")

        seen: set[str] = set()
        problems: list[str] = []

        for i, rec in enumerate(payload["species"]):
            label = rec.get("id", f"<record {i}>")
            for fname in REQUIRED_FIELDS:
                if fname not in rec:
                    problems.append(f"{label}: missing required field '{fname}'")

            if rec.get("id") in seen:
                problems.append(f"{label}: duplicate id")
            seen.add(rec.get("id"))

            status = rec.get("iucn", {}).get("status")
            if status and status not in VALID_IUCN:
                problems.append(f"{label}: unrecognised IUCN category '{status}'")

            if not rec.get("diagnostics"):
                problems.append(f"{label}: empty diagnostics list")
            if not rec.get("references"):
                problems.append(f"{label}: no references cited")

        # Cross-references must resolve, or the UI will show a broken comparison.
        for rec in payload["species"]:
            for pair in rec.get("confusion_with", []):
                if pair.get("species_id") not in seen:
                    problems.append(
                        f"{rec.get('id')}: confusion_with points at unknown id "
                        f"'{pair.get('species_id')}'"
                    )

        if problems:
            raise SpeciesDBError(
                "Species database failed validation:\n  - " + "\n  - ".join(problems)
            )

    # -- access --------------------------------------------------------
    def __len__(self) -> int:
        return len(self._by_id)

    def __contains__(self, species_id: str) -> bool:
        return species_id in self._by_id

    def __iter__(self):
        return iter(self._by_id.values())

    def get(self, species_id: str) -> Species:
        try:
            return self._by_id[species_id]
        except KeyError as exc:
            raise SpeciesDBError(f"Unknown species id: {species_id}") from exc

    @property
    def ids(self) -> list[str]:
        return list(self._by_id)

    @property
    def legal_basis(self) -> dict[str, str]:
        return self._meta.get("legal_basis", {})

    def in_state(self, state: str) -> list[Species]:
        return [s for s in self if s.occurs_in(state) in {"resident", "introduced", "marginal"}]

    def discriminator(self, a_id: str, b_id: str) -> str | None:
        """The published character separating two species, if recorded."""
        for pair in self.get(a_id).confusion_with:
            if pair["species_id"] == b_id:
                return pair["discriminator"]
        for pair in self.get(b_id).confusion_with:
            if pair["species_id"] == a_id:
                return pair["discriminator"]
        return None
