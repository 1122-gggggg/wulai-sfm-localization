#!/usr/bin/env python3
"""Run pip-audit, enforce documented vulnerability exceptions, and write an SBOM.

The exception list lives in ``pyproject.toml`` under
``[tool.security_dependency_gate]``.  An exception is a temporary, package- and
advisory-specific risk acceptance; an expired exception is always a gate
failure.  The command audits the active interpreter so the CI job can install
the hash-locked runtime/test environment first, then writes both pip-audit JSON
and CycloneDX JSON receipts.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
try:
    import tomllib
except ModuleNotFoundError:  # CPython 3.10 uses the locked tomli backport.
    import tomli as tomllib
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AuditFinding:
    package: str
    version: str
    advisory_id: str


@dataclass(frozen=True)
class SkippedDependency:
    """A dependency pip-audit could not audit and therefore skipped."""

    package: str
    version: str
    reason: str


@dataclass(frozen=True)
class VulnerabilityException:
    package: str
    advisory_id: str
    reason: str
    expires: date


@dataclass(frozen=True)
class SkippedDependencyException:
    """An exact, temporary allowlist entry for a pip-audit skip."""

    package: str
    version: str
    skip_reason: str
    reason: str
    expires: date


_SKIPPED_VERSION_RE = re.compile(r"\((?P<version>[^()]*)\)\s*$")
_SKIPPED_PACKAGE_VERSION_RE = re.compile(
    r":\s*(?P<package>[A-Za-z0-9_.-]+)\s*\((?P<version>[^()]*)\)\s*$"
)


def _as_non_empty_text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"security exception {field} must be a non-empty string")
    return value.strip()


def _package_key(value: str) -> str:
    """Use the canonical comparison form for Python distribution names."""
    return re.sub(r"[-_.]+", "-", value.strip().lower())


def _load_security_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    try:
        with path.open("rb") as stream:
            document = tomllib.load(stream)
    except OSError as exc:
        raise ValueError(f"cannot read security gate config: {path}") from exc
    raw_tool = document.get("tool", {})
    raw_gate = raw_tool.get("security_dependency_gate", {}) if isinstance(raw_tool, dict) else {}
    if not isinstance(raw_gate, dict):
        raise ValueError("security dependency gate configuration must be a table")
    return raw_gate


def load_exceptions(config_path: str | Path) -> tuple[VulnerabilityException, ...]:
    """Load and validate machine-readable vulnerability exceptions."""
    raw_exceptions = _load_security_config(config_path).get("exceptions", [])
    if not isinstance(raw_exceptions, list):
        raise ValueError("security gate exceptions must be an array of tables")

    parsed: list[VulnerabilityException] = []
    seen: set[tuple[str, str]] = set()
    for raw in raw_exceptions:
        if not isinstance(raw, dict):
            raise ValueError("each security gate exception must be a table")
        package = _package_key(_as_non_empty_text(raw.get("package"), field="package"))
        advisory_id = _as_non_empty_text(raw.get("advisory_id"), field="advisory_id")
        reason = _as_non_empty_text(raw.get("reason"), field="reason")
        raw_expiry = _as_non_empty_text(raw.get("expires"), field="expires")
        try:
            expires = date.fromisoformat(raw_expiry)
        except ValueError as exc:
            raise ValueError(
                f"security exception expires must be ISO date: {raw_expiry!r}"
            ) from exc
        key = (package, advisory_id)
        if key in seen:
            raise ValueError(f"duplicate security exception: {package}/{advisory_id}")
        seen.add(key)
        parsed.append(VulnerabilityException(package, advisory_id, reason, expires))
    return tuple(parsed)


def load_skip_exceptions(config_path: str | Path) -> tuple[SkippedDependencyException, ...]:
    """Load exact, expiring exceptions for dependencies skipped by pip-audit."""
    raw_exceptions = _load_security_config(config_path).get("skip_exceptions", [])
    if not isinstance(raw_exceptions, list):
        raise ValueError("security gate skip_exceptions must be an array of tables")

    parsed: list[SkippedDependencyException] = []
    seen: set[tuple[str, str]] = set()
    for raw in raw_exceptions:
        if not isinstance(raw, dict):
            raise ValueError("each security gate skip exception must be a table")
        package = _package_key(_as_non_empty_text(raw.get("package"), field="package"))
        version = _as_non_empty_text(raw.get("version"), field="version")
        skip_reason = _as_non_empty_text(raw.get("skip_reason"), field="skip_reason")
        reason = _as_non_empty_text(raw.get("reason"), field="reason")
        raw_expiry = _as_non_empty_text(raw.get("expires"), field="expires")
        try:
            expires = date.fromisoformat(raw_expiry)
        except ValueError as exc:
            raise ValueError(
                f"security skip exception expires must be ISO date: {raw_expiry!r}"
            ) from exc
        key = (package, version)
        if key in seen:
            raise ValueError(f"duplicate security skip exception: {package}/{version}")
        seen.add(key)
        parsed.append(
            SkippedDependencyException(package, version, skip_reason, reason, expires)
        )
    return tuple(parsed)


def _dependency_entries(document: object) -> list[object]:
    if isinstance(document, list):
        return document
    if isinstance(document, dict) and isinstance(document.get("dependencies"), list):
        return document["dependencies"]
    raise ValueError("pip-audit JSON has no dependencies list")


def _skip_version(package: str, reason: str, raw_version: object) -> str:
    """Extract a skip version and reject an embedded package/version conflict."""
    structured_match = _SKIPPED_PACKAGE_VERSION_RE.search(reason)
    embedded_version = None
    if structured_match is not None:
        embedded_package = structured_match.group("package")
        if _package_key(embedded_package) != _package_key(package):
            raise ValueError(
                f"pip-audit skip package mismatch: {package} versus {embedded_package}"
            )
        embedded_version = structured_match.group("version").strip()
        if not embedded_version:
            raise ValueError(f"pip-audit skip for {package} has no exact version: {reason}")

    if isinstance(raw_version, str) and raw_version.strip():
        version = raw_version.strip()
        if embedded_version is not None and version != embedded_version:
            raise ValueError(
                f"pip-audit skip version mismatch for {package}: "
                f"{version} versus {embedded_version}"
            )
        return version
    if embedded_version is not None:
        return embedded_version
    match = _SKIPPED_VERSION_RE.search(reason)
    if match is None or not match.group("version").strip():
        raise ValueError(f"pip-audit skip for {package} has no exact version: {reason}")
    return match.group("version").strip()


def parse_audit_json(document: object) -> list[AuditFinding]:
    """Parse pip-audit's JSON report without trusting unvalidated fields."""
    dependencies = _dependency_entries(document)

    findings: list[AuditFinding] = []
    for dependency in dependencies:
        if not isinstance(dependency, dict):
            raise ValueError("pip-audit dependency entry is not an object")
        package = _as_non_empty_text(dependency.get("name"), field="package")
        raw_vulnerabilities = dependency.get("vulns", dependency.get("vulnerabilities", []))
        if not isinstance(raw_vulnerabilities, list):
            raise ValueError(f"pip-audit vulnerabilities for {package} are not a list")
        if "skip_reason" in dependency and not raw_vulnerabilities:
            # Skipped dependencies are validated separately by
            # parse_skipped_dependencies/evaluate_skipped_dependencies.
            continue
        if (
            "skip_reason" in dependency
            and isinstance(dependency.get("version"), str)
            and dependency["version"].strip()
        ):
            version = _skip_version(
                package,
                _as_non_empty_text(dependency.get("skip_reason"), field="skip_reason"),
                dependency["version"],
            )
        elif "skip_reason" in dependency:
            reason = _as_non_empty_text(dependency.get("skip_reason"), field="skip_reason")
            version = _skip_version(package, reason, None)
        else:
            version = _as_non_empty_text(dependency.get("version"), field="version")
        for vulnerability in raw_vulnerabilities:
            if not isinstance(vulnerability, dict):
                raise ValueError(f"pip-audit vulnerability for {package} is not an object")
            advisory_id = _as_non_empty_text(
                vulnerability.get("id", vulnerability.get("advisory")),
                field="advisory_id",
            )
            findings.append(AuditFinding(package, version, advisory_id))
    return findings


def parse_skipped_dependencies(document: object) -> list[SkippedDependency]:
    """Parse pip-audit's explicit ``skip_reason`` entries fail-closed."""
    dependencies = _dependency_entries(document)
    skipped: list[SkippedDependency] = []
    for dependency in dependencies:
        if not isinstance(dependency, dict):
            raise ValueError("pip-audit dependency entry is not an object")
        if "skip_reason" not in dependency:
            continue
        package = _as_non_empty_text(dependency.get("name"), field="package")
        reason = _as_non_empty_text(dependency.get("skip_reason"), field="skip_reason")
        version = _skip_version(package, reason, dependency.get("version"))
        skipped.append(SkippedDependency(package, version, reason))
    return skipped


def evaluate_findings(
    findings: list[AuditFinding],
    exceptions: tuple[VulnerabilityException, ...],
    *,
    today: date | None = None,
) -> list[str]:
    """Return gate failures for expired or unexcepted advisories."""
    current_date = date.today() if today is None else today
    exception_by_key = {
        (_package_key(item.package), item.advisory_id): item for item in exceptions
    }
    errors: list[str] = []
    for item in exceptions:
        if item.expires <= current_date:
            errors.append(
                f"security exception expired: {item.package}/{item.advisory_id} "
                f"on {item.expires.isoformat()}"
            )
    for finding in findings:
        key = (_package_key(finding.package), finding.advisory_id)
        exception = exception_by_key.get(key)
        if exception is None:
            errors.append(
                f"unexcepted vulnerability: {finding.package}=={finding.version} "
                f"{finding.advisory_id}"
            )
        elif exception.expires <= current_date:
            errors.append(
                f"vulnerability is covered only by expired exception: "
                f"{finding.package}=={finding.version} {finding.advisory_id}"
            )
    return errors


def evaluate_skipped_dependencies(
    skipped: list[SkippedDependency],
    exceptions: tuple[SkippedDependencyException, ...],
    *,
    today: date | None = None,
) -> list[str]:
    """Require every pip-audit skip to match package, version, and reason."""
    current_date = date.today() if today is None else today
    exception_by_key = {
        (_package_key(item.package), item.version, item.skip_reason): item
        for item in exceptions
    }
    errors: list[str] = []
    for item in exceptions:
        if item.expires <= current_date:
            errors.append(
                f"security skip exception expired: {item.package}=={item.version} "
                f"on {item.expires.isoformat()}"
            )
    for dependency in skipped:
        key = (_package_key(dependency.package), dependency.version, dependency.reason)
        exception = exception_by_key.get(key)
        if exception is None:
            errors.append(
                f"unexcepted skipped dependency: {dependency.package}=={dependency.version} "
                f"{dependency.reason}"
            )
        elif exception.expires <= current_date:
            errors.append(
                f"skipped dependency is covered only by expired exception: "
                f"{dependency.package}=={dependency.version}"
            )
    return errors


def _record_skipped_component(
    components: list[Any],
    dependencies: list[Any],
    item: SkippedDependency,
) -> str | None:
    """Add one un-audited dependency to a CycloneDX document."""
    component = next(
        (
            entry
            for entry in components
            if isinstance(entry, dict)
            and isinstance(entry.get("name"), str)
            and _package_key(entry["name"]) == _package_key(item.package)
            and entry.get("version") == item.version
        ),
        None,
    )
    if component is None:
        bom_ref = f"pkg:pypi/{_package_key(item.package)}@{item.version}"
        component = {
            "bom-ref": bom_ref,
            "name": item.package,
            "type": "library",
            "version": item.version,
        }
        components.append(component)
    else:
        bom_ref = component.get("bom-ref")
        if not isinstance(bom_ref, str) or not bom_ref:
            bom_ref = f"pkg:pypi/{_package_key(item.package)}@{item.version}"
            component["bom-ref"] = bom_ref
    properties = component.setdefault("properties", [])
    if not isinstance(properties, list):
        return f"CycloneDX component has invalid properties list: {item.package}=={item.version}"
    properties[:] = [
        property_entry
        for property_entry in properties
        if not (
            isinstance(property_entry, dict)
            and property_entry.get("name") == "security_dependency_gate:skip_reason"
        )
    ]
    properties.extend(
        [
            {
                "name": "security_dependency_gate:audited",
                "value": "false",
            },
            {
                "name": "security_dependency_gate:skip_reason",
                "value": item.reason,
            },
        ]
    )
    if not any(
        isinstance(entry, dict) and entry.get("ref") == bom_ref
        for entry in dependencies
    ):
        dependencies.append({"ref": bom_ref})
    return None


def _record_skipped_in_sbom(
    sbom_path: Path,
    skipped: list[SkippedDependency],
) -> list[str]:
    """Keep skipped, un-audited wheels visible in the CycloneDX receipt."""
    if not skipped:
        return []
    try:
        document = json.loads(sbom_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"cannot update CycloneDX SBOM with skipped dependencies: {exc}"]
    if not isinstance(document, dict) or document.get("bomFormat") != "CycloneDX":
        return ["pip-audit SBOM is not a CycloneDX JSON document"]
    components = document.get("components")
    dependencies = document.get("dependencies")
    if not isinstance(components, list) or not isinstance(dependencies, list):
        return ["pip-audit SBOM has invalid components/dependencies lists"]

    for item in skipped:
        error = _record_skipped_component(components, dependencies, item)
        if error is not None:
            return [error]
    try:
        sbom_path.write_text(
            json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        return [f"cannot write CycloneDX SBOM with skipped dependencies: {exc}"]
    return []


def _run_pip_audit(
    python_executable: str,
    *,
    output_path: Path,
    sbom_path: Path,
) -> list[str]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sbom_path.parent.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []

    audit_command = [
        python_executable,
        "-m",
        "pip_audit",
        "--local",
        "--format",
        "json",
        "--output",
        str(output_path),
        "--progress-spinner",
        "off",
    ]
    audit_result = subprocess.run(audit_command, check=False, text=True, capture_output=True)
    if audit_result.returncode not in (0, 1):
        errors.append(
            f"pip-audit JSON command failed with exit={audit_result.returncode}: "
            f"{audit_result.stderr.strip()}"
        )
    if not output_path.is_file():
        errors.append(f"pip-audit did not write JSON report: {output_path}")

    sbom_command = [
        python_executable,
        "-m",
        "pip_audit",
        "--local",
        "--format",
        "cyclonedx-json",
        "--output",
        str(sbom_path),
        "--progress-spinner",
        "off",
    ]
    sbom_result = subprocess.run(sbom_command, check=False, text=True, capture_output=True)
    if sbom_result.returncode not in (0, 1):
        errors.append(
            f"pip-audit CycloneDX command failed with exit={sbom_result.returncode}: "
            f"{sbom_result.stderr.strip()}"
        )
    if not sbom_path.is_file():
        errors.append(f"pip-audit did not write CycloneDX SBOM: {sbom_path}")
    else:
        try:
            sbom_document = json.loads(sbom_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"CycloneDX SBOM is not valid JSON: {exc}")
        else:
            if (
                not isinstance(sbom_document, dict)
                or sbom_document.get("bomFormat") != "CycloneDX"
            ):
                errors.append("pip-audit SBOM is not a CycloneDX JSON document")
    return errors


def run_gate(
    *,
    config_path: str | Path,
    audit_json_path: str | Path,
    sbom_path: str | Path,
    python_executable: str = sys.executable,
) -> list[str]:
    """Run audit and return all gate failures."""
    audit_json = Path(audit_json_path)
    sbom = Path(sbom_path)
    errors = _run_pip_audit(python_executable, output_path=audit_json, sbom_path=sbom)
    if not audit_json.is_file():
        return errors
    try:
        document: Any = json.loads(audit_json.read_text(encoding="utf-8"))
        findings = parse_audit_json(document)
        skipped = parse_skipped_dependencies(document)
        exceptions = load_exceptions(config_path)
        skip_exceptions = load_skip_exceptions(config_path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        errors.append(f"cannot parse dependency audit receipt: {exc}")
        return errors
    errors.extend(evaluate_findings(findings, exceptions))
    errors.extend(evaluate_skipped_dependencies(skipped, skip_exceptions))
    errors.extend(_record_skipped_in_sbom(sbom, skipped))
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="pyproject.toml")
    parser.add_argument("--audit-json", default="outputs/security/pip-audit.json")
    parser.add_argument("--sbom", default="outputs/security/sbom.cyclonedx.json")
    parser.add_argument("--python", dest="python_executable", default=sys.executable)
    args = parser.parse_args(argv)
    errors = run_gate(
        config_path=args.config,
        audit_json_path=args.audit_json,
        sbom_path=args.sbom,
        python_executable=args.python_executable,
    )
    if errors:
        for error in errors:
            print(f"FAIL: {error}", file=sys.stderr)
        return 1
    print("dependency security gate OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
