"""Parse dependency manifest files into package lists.

Supports:
- requirements.txt (Python/pip)
- package.json (Node.js/npm)
- go.mod (Go modules)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Literal

Ecosystem = Literal["pypi", "npm", "go"]


@dataclass(frozen=True)
class ParsedDependency:
    """A dependency extracted from a manifest file."""

    name: str
    ecosystem: Ecosystem


class ManifestParseError(Exception):
    """Raised when a manifest file cannot be parsed."""


def detect_ecosystem(filename: str) -> Ecosystem:
    """Detect package ecosystem from filename."""
    name = filename.lower().rsplit("/", maxsplit=1)[-1]
    if name == "requirements.txt":
        return "pypi"
    if name == "package.json":
        return "npm"
    if name == "go.mod":
        return "go"
    msg = f"Unsupported manifest file: {filename}"
    raise ManifestParseError(msg)


def parse_requirements_txt(content: str) -> list[ParsedDependency]:
    """Parse a pip requirements.txt file.

    Handles:
    - Package names with version specifiers (flask>=2.0)
    - Comments and blank lines
    - Extras (flask[async])
    - -r/-c includes (skipped)
    - Environment markers (; python_version >= "3.8")
    """
    deps: list[ParsedDependency] = []
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        # Strip inline comments
        line = line.split("#", maxsplit=1)[0].strip()
        # Strip environment markers
        line = line.split(";", maxsplit=1)[0].strip()
        # Extract package name (before version specifiers and extras)
        match = re.match(r"^([A-Za-z0-9]([A-Za-z0-9._-]*[A-Za-z0-9])?)", line)
        if match:
            name = re.sub(r"[-_.]+", "-", match.group(1)).lower()
            deps.append(ParsedDependency(name=name, ecosystem="pypi"))
    return deps


def parse_package_json(content: str) -> list[ParsedDependency]:
    """Parse a Node.js package.json file.

    Extracts packages from both 'dependencies' and 'devDependencies'.
    """
    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        msg = f"Invalid package.json: {e}"
        raise ManifestParseError(msg) from e

    deps: list[ParsedDependency] = []
    seen: set[str] = set()
    for key in ("dependencies", "devDependencies"):
        for name in data.get(key, {}):
            if name not in seen:
                seen.add(name)
                deps.append(ParsedDependency(name=name, ecosystem="npm"))
    return deps


def parse_go_mod(content: str) -> list[ParsedDependency]:
    """Parse a Go go.mod file.

    Extracts module paths from 'require' blocks and single-line requires.
    """
    deps: list[ParsedDependency] = []
    in_require_block = False

    for raw_line in content.splitlines():
        line = raw_line.strip()
        # Strip inline comments
        line = line.split("//", maxsplit=1)[0].strip()

        if line == "require (":
            in_require_block = True
            continue
        if line == ")" and in_require_block:
            in_require_block = False
            continue

        if in_require_block:
            # Lines like: github.com/gin-gonic/gin v1.9.1
            parts = line.split()
            if len(parts) >= 2:
                deps.append(ParsedDependency(name=parts[0], ecosystem="go"))
        elif line.startswith("require "):
            # Single-line: require github.com/gin-gonic/gin v1.9.1
            parts = line.split()
            if len(parts) >= 3:
                deps.append(ParsedDependency(name=parts[1], ecosystem="go"))

    return deps


def parse_manifest(filename: str, content: str) -> list[ParsedDependency]:
    """Parse a manifest file and return extracted dependencies.

    Args:
        filename: Original filename (used for ecosystem detection).
        content: Raw file content.

    Returns:
        List of parsed dependencies.

    Raises:
        ManifestParseError: If the file type is unsupported or content is invalid.
    """
    ecosystem = detect_ecosystem(filename)
    if ecosystem == "pypi":
        return parse_requirements_txt(content)
    if ecosystem == "npm":
        return parse_package_json(content)
    return parse_go_mod(content)
