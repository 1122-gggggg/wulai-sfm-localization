from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest

from security_dependency_gate import (
    AuditFinding,
    SkippedDependency,
    _record_skipped_in_sbom,
    _run_pip_audit,
    evaluate_findings,
    evaluate_skipped_dependencies,
    load_exceptions,
    load_skip_exceptions,
    parse_audit_json,
    parse_skipped_dependencies,
)


def _write_config(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "pyproject.toml"
    path.write_text(body, encoding="utf-8")
    return path


def test_expired_exception_is_a_gate_failure(tmp_path: Path) -> None:
    config = _write_config(
        tmp_path,
        """
[tool.security_dependency_gate]
[[tool.security_dependency_gate.exceptions]]
package = "protobuf"
advisory_id = "PYSEC-2026-899"
reason = "The flight SDK pins the wire schema."
expires = "2026-08-08"
""",
    )

    exceptions = load_exceptions(config)

    errors = evaluate_findings(
        [AuditFinding("protobuf", "3.19.4", "PYSEC-2026-899")],
        exceptions,
        today=date(2026, 8, 9),
    )

    assert any("expired" in error for error in errors)
    assert any("PYSEC-2026-899" in error for error in errors)


def test_matching_unexpired_exception_is_allowed(tmp_path: Path) -> None:
    config = _write_config(
        tmp_path,
        """
[[tool.security_dependency_gate.exceptions]]
package = "protobuf"
advisory_id = "PYSEC-2026-899"
reason = "The flight SDK pins the wire schema."
expires = "2027-08-09"
""",
    )

    errors = evaluate_findings(
        [AuditFinding("protobuf", "3.19.4", "PYSEC-2026-899")],
        load_exceptions(config),
        today=date(2026, 8, 9),
    )

    assert errors == []


def test_unlisted_finding_fails_even_when_another_is_excepted(tmp_path: Path) -> None:
    config = _write_config(
        tmp_path,
        """
[[tool.security_dependency_gate.exceptions]]
package = "protobuf"
advisory_id = "PYSEC-2026-899"
reason = "The flight SDK pins the wire schema."
expires = "2027-08-09"
""",
    )

    errors = evaluate_findings(
        [
            AuditFinding("protobuf", "3.19.4", "PYSEC-2026-899"),
            AuditFinding("requests", "2.31.0", "CVE-2099-0001"),
        ],
        load_exceptions(config),
        today=date(2026, 8, 9),
    )

    assert len(errors) == 1
    assert "requests" in errors[0]
    assert "CVE-2099-0001" in errors[0]


def test_parser_accepts_pip_audit_json_shape() -> None:
    findings = parse_audit_json(
        [
            {
                "name": "protobuf",
                "version": "3.19.4",
                "vulns": [
                    {"id": "PYSEC-2026-899"},
                    {"id": "PYSEC-2026-1805"},
                ],
            },
            {"name": "safe", "version": "1.0.0", "vulns": []},
        ]
    )

    assert findings == [
        AuditFinding("protobuf", "3.19.4", "PYSEC-2026-899"),
        AuditFinding("protobuf", "3.19.4", "PYSEC-2026-1805"),
    ]


def test_exception_schema_requires_reason_and_expiry(tmp_path: Path) -> None:
    config = _write_config(
        tmp_path,
        """
[[tool.security_dependency_gate.exceptions]]
package = "protobuf"
advisory_id = "PYSEC-2026-899"
expires = "2027-08-09"
""",
    )

    with pytest.raises(ValueError, match="reason"):
        load_exceptions(config)


def test_parser_extracts_exact_version_from_pip_audit_skip_reason() -> None:
    reason = "Dependency not found on PyPI and could not be audited: torch (2.11.0+cu128)"

    assert parse_skipped_dependencies(
        {"dependencies": [{"name": "torch", "skip_reason": reason}]}
    ) == [SkippedDependency("torch", "2.11.0+cu128", reason)]


def test_parser_rejects_skip_without_exact_version() -> None:
    with pytest.raises(ValueError, match="no exact version"):
        parse_skipped_dependencies(
            {"dependencies": [{"name": "torch", "skip_reason": "not found"}]}
        )


def test_parser_rejects_skip_version_conflicting_with_reason() -> None:
    with pytest.raises(ValueError, match="version mismatch"):
        parse_skipped_dependencies(
            {
                "dependencies": [
                    {
                        "name": "torch",
                        "version": "2.11.0+cu129",
                        "skip_reason": (
                            "Dependency not found on PyPI and could not be audited: "
                            "torch (2.11.0+cu128)"
                        ),
                    }
                ]
            }
        )


def test_matching_skip_exception_requires_package_version_and_reason(tmp_path: Path) -> None:
    reason = "Dependency not found on PyPI and could not be audited: torch (2.11.0+cu128)"
    config = _write_config(
        tmp_path,
        f"""
[[tool.security_dependency_gate.skip_exceptions]]
package = "torch"
version = "2.11.0+cu128"
skip_reason = "{reason}"
reason = "The official CUDA wheel is hash locked and offline."
expires = "2027-02-09"
""",
    )

    exceptions = load_skip_exceptions(config)
    assert evaluate_skipped_dependencies(
        [SkippedDependency("torch", "2.11.0+cu128", reason)],
        exceptions,
        today=date(2026, 8, 9),
    ) == []
    assert evaluate_skipped_dependencies(
        [SkippedDependency("torch", "2.11.0+cu129", reason)],
        exceptions,
        today=date(2026, 8, 9),
    )
    assert evaluate_skipped_dependencies(
        [SkippedDependency("torch", "2.11.0+cu128", reason + " changed")],
        exceptions,
        today=date(2026, 8, 9),
    )


def test_expired_skip_exception_is_a_gate_failure(tmp_path: Path) -> None:
    config = _write_config(
        tmp_path,
        """
[[tool.security_dependency_gate.skip_exceptions]]
package = "torch"
version = "2.11.0+cu128"
skip_reason = "Dependency not found on PyPI and could not be audited: torch (2.11.0+cu128)"
reason = "The official CUDA wheel is hash locked and offline."
expires = "2026-08-08"
""",
    )

    errors = evaluate_skipped_dependencies(
        parse_skipped_dependencies(
            {
                "dependencies": [
                    {
                        "name": "torch",
                        "skip_reason": (
                            "Dependency not found on PyPI and could not be audited: "
                            "torch (2.11.0+cu128)"
                        ),
                    }
                ]
            }
        ),
        load_skip_exceptions(config),
        today=date(2026, 8, 9),
    )

    assert any("expired" in error for error in errors)


def test_skipped_dependency_is_kept_in_cyclonedx_sbom(tmp_path: Path) -> None:
    sbom = tmp_path / "sbom.json"
    sbom.write_text(
        json.dumps(
            {
                "bomFormat": "CycloneDX",
                "specVersion": "1.4",
                "components": [],
                "dependencies": [],
            }
        ),
        encoding="utf-8",
    )
    skipped = [
        SkippedDependency(
            "torch",
            "2.11.0+cu128",
            "Dependency not found on PyPI and could not be audited: torch (2.11.0+cu128)",
        )
    ]

    assert _record_skipped_in_sbom(sbom, skipped) == []
    document = json.loads(sbom.read_text(encoding="utf-8"))
    component = document["components"][0]
    assert component["name"] == "torch"
    assert component["version"] == "2.11.0+cu128"
    assert {entry["name"] for entry in component["properties"]} == {
        "security_dependency_gate:audited",
        "security_dependency_gate:skip_reason",
    }
    assert document["dependencies"] == [{"ref": component["bom-ref"]}]


def test_pip_audit_commands_are_non_strict_to_preserve_skip_receipts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commands: list[list[str]] = []

    def fake_run(command: list[str], **_: object) -> SimpleNamespace:
        commands.append(command)
        output = Path(command[command.index("--output") + 1])
        if command[command.index("--format") + 1] == "json":
            output.write_text('{"dependencies": []}', encoding="utf-8")
        else:
            output.write_text(
                '{"bomFormat": "CycloneDX", "components": [], "dependencies": []}',
                encoding="utf-8",
            )
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr("security_dependency_gate.subprocess.run", fake_run)
    errors = _run_pip_audit(
        "python",
        output_path=tmp_path / "audit.json",
        sbom_path=tmp_path / "sbom.json",
    )

    assert errors == []
    assert len(commands) == 2
    assert all("--strict" not in command for command in commands)
