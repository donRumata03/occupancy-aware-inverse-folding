from __future__ import annotations

from pathlib import Path


def write_single_sequence_a3m(sequence_id: str, sequence: str, output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{sequence_id}.a3m"
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f">{sequence_id}\n{sequence}\n")
    return path


def pdb_residue_count(path: str | Path, chain_id: str | None = None) -> int | None:
    path = Path(path)
    if not path.exists():
        return None
    residues: set[tuple[str, str, str]] = set()
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith("ATOM"):
                continue
            chain = line[21].strip()
            if chain_id and chain_id.strip() and chain != chain_id:
                continue
            resseq = line[22:26].strip()
            icode = line[26].strip()
            resname = line[17:20].strip()
            residues.add((chain, resseq + icode, resname))
    return len(residues)

