"""CI-native rendering for Clozn batch comparisons."""
from __future__ import annotations

import json
from pathlib import Path
from xml.etree import ElementTree as ET

from .compare import BatchComparison, ExperimentComparison, MetricDelta


def _format_value(value: float | bool | None) -> str:
    if value is None:
        return "missing"
    if isinstance(value, bool):
        return str(value).lower()
    return f"{value:.12g}"


def _metric_message(metric: MetricDelta) -> str:
    message = (
        f"{metric.arm}.{metric.metric}: baseline={_format_value(metric.baseline)}, "
        f"candidate={_format_value(metric.candidate)}"
    )
    if metric.delta is not None:
        message += f", delta={_format_value(metric.delta)}"
    return message


def _experiment_message(item: ExperimentComparison) -> str:
    rows = [_metric_message(metric) for metric in item.metrics if metric.regressed]
    if item.error:
        rows.insert(0, item.error)
    return "\n".join(rows) or "experiment regressed"


def comparison_to_junit_xml(comparison: BatchComparison) -> str:
    """Render one JUnit testcase per manifest comparison."""
    failures = sum(item.status == "regressed" for item in comparison.experiments)
    suite = ET.Element(
        "testsuite",
        {
            "name": "clozn.batch_comparison",
            "tests": str(len(comparison.experiments)),
            "failures": str(failures),
            "errors": "0",
            "skipped": "0",
        },
    )
    properties = ET.SubElement(suite, "properties")
    for name, value in (
        ("baseline_index", comparison.baseline_index),
        ("candidate_index", comparison.candidate_index),
        ("max_metric_delta", str(comparison.max_metric_delta)),
        ("regressions", str(comparison.regressions)),
    ):
        ET.SubElement(properties, "property", {"name": name, "value": value})

    for item in comparison.experiments:
        case = ET.SubElement(
            suite,
            "testcase",
            {
                "classname": item.schema or "clozn.experiment",
                "name": item.name or item.manifest_sha256,
            },
        )
        output = ET.SubElement(case, "system-out")
        output.text = json.dumps(item.to_json_object(), sort_keys=True, allow_nan=False)
        if item.status == "regressed":
            failure = ET.SubElement(
                case,
                "failure",
                {"type": "clozn.regression", "message": f"{item.regressions} regression(s)"},
            )
            failure.text = _experiment_message(item)

    ET.indent(suite, space="  ")
    return ET.tostring(suite, encoding="unicode", xml_declaration=True) + "\n"


def comparison_to_markdown(comparison: BatchComparison) -> str:
    """Render a compact GitHub step summary."""
    state = "✅ Passed" if comparison.regressions == 0 else "❌ Regressed"
    lines = [
        "# Clozn experiment comparison",
        "",
        f"**{state}** — {comparison.regressions} regression(s) across "
        f"{len(comparison.experiments)} experiment(s).",
        "",
        f"Tolerance: `{comparison.max_metric_delta:.12g}`",
        "",
        "| Experiment | Status | Regressions |",
        "|---|---:|---:|",
    ]
    for item in comparison.experiments:
        name = (item.name or item.manifest_sha256).replace("|", "\\|")
        lines.append(f"| `{name}` | {item.status} | {item.regressions} |")

    regressed = [item for item in comparison.experiments if item.status == "regressed"]
    if regressed:
        lines.extend(["", "## Regression details", ""])
        for item in regressed:
            lines.append(f"### {item.name or item.manifest_sha256}")
            if item.error:
                lines.append(f"- {item.error}")
            for metric in item.metrics:
                if metric.regressed:
                    lines.append(f"- `{_metric_message(metric)}`")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _escape_command_message(value: str) -> str:
    return value.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def _escape_command_property(value: str) -> str:
    return _escape_command_message(value).replace(":", "%3A").replace(",", "%2C")


def comparison_to_github_annotations(comparison: BatchComparison) -> str:
    """Render GitHub Actions workflow-command annotations, one per regression."""
    lines: list[str] = []
    for item in comparison.experiments:
        if item.status != "regressed":
            continue
        title = _escape_command_property(f"Clozn regression: {item.name or item.manifest_sha256[:12]}")
        if item.error:
            lines.append(f"::error title={title}::{_escape_command_message(item.error)}")
        for metric in item.metrics:
            if metric.regressed:
                lines.append(f"::error title={title}::{_escape_command_message(_metric_message(metric))}")
        if not item.error and not any(metric.regressed for metric in item.metrics):
            lines.append(f"::error title={title}::experiment regressed")
    return "\n".join(lines) + ("\n" if lines else "")


def write_ci_reports(
    comparison: BatchComparison,
    *,
    junit: str | Path | None = None,
    github_summary: str | Path | None = None,
    github_annotations: str | Path | None = None,
) -> None:
    """Write any requested CI reports."""
    outputs = (
        (junit, comparison_to_junit_xml(comparison)),
        (github_summary, comparison_to_markdown(comparison)),
        (github_annotations, comparison_to_github_annotations(comparison)),
    )
    for destination, text in outputs:
        if destination is not None:
            Path(destination).write_text(text, encoding="utf-8")
