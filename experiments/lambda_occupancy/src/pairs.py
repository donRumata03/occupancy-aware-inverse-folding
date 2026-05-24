from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from .utils import read_csv, write_csv


PAIR_FIELDS = ["pair_id", "x0_pdb", "x1_pdb", "chain_id", "length_hint", "notes"]


@dataclass(frozen=True)
class ConformerPair:
    pair_id: str
    x0_pdb: Path
    x1_pdb: Path
    chain_id: str = ""
    length_hint: int | None = None
    notes: str = ""

    def to_row(self) -> dict[str, Any]:
        row = asdict(self)
        row["x0_pdb"] = str(self.x0_pdb)
        row["x1_pdb"] = str(self.x1_pdb)
        row["length_hint"] = "" if self.length_hint is None else self.length_hint
        return row


def create_template_csv(path: str | Path) -> None:
    rows = [
        {
            "pair_id": "pair_001",
            "x0_pdb": "data/pairs/pair_001/X0.pdb",
            "x1_pdb": "data/pairs/pair_001/X1.pdb",
            "chain_id": "A",
            "length_hint": "120",
            "notes": "metamorphic/small",
        },
        {
            "pair_id": "pair_002",
            "x0_pdb": "data/pairs/pair_002/X0.pdb",
            "x1_pdb": "data/pairs/pair_002/X1.pdb",
            "chain_id": "A",
            "length_hint": "150",
            "notes": "hinge",
        },
        {
            "pair_id": "pair_003",
            "x0_pdb": "data/pairs/pair_003/X0.pdb",
            "x1_pdb": "data/pairs/pair_003/X1.pdb",
            "chain_id": "A",
            "length_hint": "180",
            "notes": "small two-state pair",
        },
    ]
    write_csv(path, rows, PAIR_FIELDS)


def load_pairs(path: str | Path, limit: int | None = None, create_if_missing: bool = True) -> list[ConformerPair]:
    path = Path(path)
    if not path.exists():
        if create_if_missing:
            create_template_csv(path)
        raise FileNotFoundError(
            f"Conformer-pair CSV does not exist: {path}. A template was created there; fill it with real PDB paths."
        )
    pairs: list[ConformerPair] = []
    for row in read_csv(path):
        if not row.get("pair_id"):
            continue
        length_hint = row.get("length_hint") or None
        pairs.append(
            ConformerPair(
                pair_id=row["pair_id"],
                x0_pdb=Path(row["x0_pdb"]),
                x1_pdb=Path(row["x1_pdb"]),
                chain_id=row.get("chain_id", ""),
                length_hint=int(length_hint) if length_hint else None,
                notes=row.get("notes", ""),
            )
        )
    return pairs[:limit] if limit is not None else pairs


def validate_pairs(pairs: list[ConformerPair]) -> list[dict[str, Any]]:
    metadata: list[dict[str, Any]] = []
    for pair in pairs:
        warnings: list[str] = []
        if not pair.x0_pdb.exists():
            warnings.append(f"missing X0 PDB: {pair.x0_pdb}")
        if not pair.x1_pdb.exists():
            warnings.append(f"missing X1 PDB: {pair.x1_pdb}")
        metadata.append(
            {
                "pair_id": pair.pair_id,
                "x0_pdb": str(pair.x0_pdb),
                "x1_pdb": str(pair.x1_pdb),
                "chain_id": pair.chain_id,
                "length_hint": pair.length_hint or "",
                "notes": pair.notes,
                "preprocessing_warnings": "; ".join(warnings),
            }
        )
    return metadata

