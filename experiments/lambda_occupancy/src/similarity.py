from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from typing import Any


class AlignmentToolMissing(RuntimeError):
    pass


def find_alignment_tool(preferred: str | None = None) -> str:
    candidates = [preferred] if preferred else []
    candidates.extend(["USalign", "US-align", "TMalign", "TM-align"])
    for candidate in candidates:
        if candidate and shutil.which(candidate):
            return candidate
    raise AlignmentToolMissing(
        "No US-align/TM-align executable found on PATH. Install US-align or TM-align, "
        "or set assignment.alignment_tool in the YAML config."
    )


def parse_tm_score(output: str) -> float:
    scores = [float(x) for x in re.findall(r"TM-score\s*=\s*([0-9]*\.?[0-9]+)", output)]
    if not scores:
        raise ValueError("Could not parse TM-score from alignment output")
    return max(scores)


def tm_score(mobile: str | Path, target: str | Path, config: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    tool = find_alignment_tool(config.get("assignment", {}).get("alignment_tool"))
    command = [tool, str(mobile), str(target)]
    proc = subprocess.run(command, check=False, capture_output=True, text=True)
    metadata = {
        "command": " ".join(command),
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }
    if proc.returncode != 0:
        raise RuntimeError(f"Alignment command failed: {' '.join(command)}\n{proc.stderr}")
    return parse_tm_score(proc.stdout + "\n" + proc.stderr), metadata

