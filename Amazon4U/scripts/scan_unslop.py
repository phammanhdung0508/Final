#!/usr/bin/env python3
"""Scan markdown files for unslop violations.

Checks against patterns defined in the unslop skill:
- Em dashes and en dashes (rule 13)
- Mid-sentence colon connectors (rule 14)
- AI vocabulary (rule 7)
- Fancy ways to say 'is' (rule 8)
- Superficial -ing phrases (rule 3)
- Filler phrases (rule 23)
- Abstract jargon words (rule 26)
- Over-compression / missing spacing around punctuation (rule 33)
- Title case headings (rule 17)
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Byte-frozen outside the protocol manifests: tests/test_checkpoint_payload.py (V02)
# hashes its first 59605 bytes.
FROZEN_PREFIX_DOCS = {"docs/checkpoint-payload.md"}


def frozen_documents():
    """Paths whose bytes are pinned by a frozen protocol baseline or a prefix hash."""
    frozen = set(FROZEN_PREFIX_DOCS)
    for manifest in sorted((ROOT / "configs" / "protocols").glob("*/baseline-manifest.json")):
        frozen.update(e["source_path"] for e in json.loads(manifest.read_text())["files"])
    return frozen

AI_VOCAB = [
    r"\badditionally\b",
    r"\bcrucial\b",
    r"\bdelve\b",
    r"\benduring\b",
    r"\benhance\b",
    r"\benhancing\b",
    r"\bfostering\b",
    r"\bgarner\b",
    r"\binterplay\b",
    r"\bintricate\b",
    r"\bpivotal\b",
    r"\bshowcase\b",
    r"\bshowcasing\b",
    r"\btapestry\b",
    r"\btestament\b",
    r"\bunderscore\b",
    r"\bvibrant\b",
]

FANCY_IS = [
    r"\bserves as\b",
    r"\bstands as\b",
    r"\bboasts\b",
    # 'features' as a verb: e.g. "it features", "which features", "model features an attention"
    r"\b(?:it|which|that|system|tool|model|release|library)\s+features\b",
]

FILLER = [
    r"\bin order to\b",
    r"\bdue to the fact that\b",
    r"\bit is important to note that\b",
]

JARGON = [
    r"\blandscape\b",
    r"\bvector\s+for\b",
    r"\bvector\s+of\b",
    r"\blocus\b",
    r"\bvantage\b",
    r"\bnexus\b",
    r"\bbedrock\b",
    r"\bmodality\b",
    r"\bparadigm\b",
    r"\bgold-plating\b",
    r"\bevacuate\b",
    r"\bendgame\b",
    r"\bnorth star\b",
    r"\bflywheel\b",
]

SUPERFICIAL_ING = [
    r"\bhighlighting\b",
    r"\bensuring\b",
    r"\breflecting\b",
    r"\bfostering\b",
]

SQUISHED_PUNCTUATION = [
    r"[a-zA-Z0-9],[a-zA-Z]",  # missing space after comma
    r"[a-zA-Z0-9];[a-zA-Z]",  # missing space after semicolon
    r"[a-zA-Z0-9]\/[a-zA-Z]",  # compressed slash without spaces like full/mock or files/26
]


def check_heading_title_case(heading_text: str) -> bool:
    words = [w for w in re.split(r"[\s\-_/]+", heading_text) if len(w) > 3]
    if len(words) <= 1:
        return False
    # If more than 50% of words >3 chars are capitalized and not all-uppercase acronyms
    title_words = [w for w in words[1:] if w[0].isupper() and not w.isupper()]
    return len(title_words) >= 2


def scan_file(path: Path):
    lines = path.read_text(encoding="utf-8").splitlines()
    violations = []

    for idx, line in enumerate(lines, start=1):
        # 1. Em dashes and en dashes
        if "—" in line or "–" in line:
            violations.append((idx, "dash", "Em dash or en dash present"))

        # 2. Headings title case
        if line.startswith("#"):
            heading = line.lstrip("#").strip()
            if check_heading_title_case(heading):
                violations.append((idx, "title-case-heading", f"Title case heading: {heading}"))

        # 3. AI vocabulary
        for pat in AI_VOCAB:
            if re.search(pat, line, re.IGNORECASE):
                violations.append((idx, "ai-vocab", f"AI vocab match {pat}: {line.strip()[:60]}"))

        # 4. Fancy is
        for pat in FANCY_IS:
            if re.search(pat, line, re.IGNORECASE):
                violations.append((idx, "fancy-is", f"Fancy is match {pat}: {line.strip()[:60]}"))

        # 5. Filler phrases
        for pat in FILLER:
            if re.search(pat, line, re.IGNORECASE):
                violations.append((idx, "filler", f"Filler match {pat}: {line.strip()[:60]}"))

        # 6. Jargon
        for pat in JARGON:
            if re.search(pat, line, re.IGNORECASE):
                violations.append((idx, "jargon", f"Jargon match {pat}: {line.strip()[:60]}"))

        # 7. Superficial -ing
        for pat in SUPERFICIAL_ING:
            if re.search(pat, line, re.IGNORECASE):
                violations.append((idx, "superficial-ing", f"-ing match {pat}: {line.strip()[:60]}"))

        # 8. Squished comma or semicolon
        if re.search(r"[a-zA-Z0-9],[a-zA-Z]", line):
            violations.append((idx, "squished-comma", f"Missing space after comma: {line.strip()[:60]}"))
        if re.search(r"[a-zA-Z0-9];[a-zA-Z]", line):
            violations.append((idx, "squished-semicolon", f"Missing space after semicolon: {line.strip()[:60]}"))

    return violations


def main():
    paths = sys.argv[1:]
    if not paths:
        print("Usage: scan_unslop.py <file1> <file2> ...")
        sys.exit(1)

    total_violations = 0
    frozen = frozen_documents()
    for p_str in paths:
        p = Path(p_str)
        if not p.exists():
            print(f"File not found: {p}")
            continue
        rel = p.resolve().relative_to(ROOT).as_posix() if p.resolve().is_relative_to(ROOT) else None
        if rel in frozen:
            print(f"{p}: FROZEN, not scanned. Its bytes are pinned by configs/protocols/*/"
                  "baseline-manifest.json or a prefix-hash test; editing it breaks the frozen "
                  "baseline. Record wording changes as a protocol amendment instead.")
            continue
        v = scan_file(p)
        total_violations += len(v)
        print(f"{p}: {len(v)} violations found")
        for line_no, kind, msg in v[:10]:
            print(f"  L{line_no} [{kind}]: {msg}")
        if len(v) > 10:
            print(f"  ... and {len(v) - 10} more")

    print(f"\nTotal violations: {total_violations}")
    sys.exit(0 if total_violations == 0 else 1)


if __name__ == "__main__":
    main()
