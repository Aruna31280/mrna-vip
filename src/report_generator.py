
from pathlib import Path
from datetime import datetime, timezone

from .schemas import SafetyReport


def generate_demo_report(
    risk_label,
    probability,
    guideline_chunks,
    patient_summary,
):
    """Create an educational report without using an external API."""

    report = SafetyReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        patient_summary=patient_summary,
        model_output={
            "predicted_dataset_label": str(risk_label),
            "probability_for_label_1": round(float(probability), 4),
        },
        guideline_evidence=list(guideline_chunks),
        limitations=[
            "Educational demonstration only; not clinically validated.",
            "A dataset prediction is not a medical diagnosis.",
            "The available information may be incomplete.",
        ],
        disclaimer=(
            "This report is not medical advice and must not be used "
            "to make vaccination, diagnosis, or treatment decisions. "
            "Consult a qualified healthcare professional."
        ),
    )

    return report


def generate_llm_report(
    risk_label,
    probability,
    guideline_chunks,
    patient_summary,
):
    """Use demo mode when no LLM integration is configured."""
    return generate_demo_report(
        risk_label,
        probability,
        guideline_chunks,
        patient_summary,
    )


def save_report(report, stem="latest_report"):
    """Save a report as JSON and Markdown under outputs/reports."""

    output_dir = Path("outputs") / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / f"{stem}.json"
    markdown_path = output_dir / f"{stem}.md"

    report_dict = report.model_dump()

    import json
    json_path.write_text(
        json.dumps(report_dict, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    lines = [
        "# Educational Safety Report",
        "",
        f"Generated: {report_dict.get('generated_at', '')}",
        "",
        "## Patient Summary",
        str(report_dict.get("patient_summary", "")),
        "",
        "## Model Output",
        "```json",
        json.dumps(report_dict.get("model_output", {}), indent=2),
        "```",
        "",
        "## Guideline Evidence",
    ]

    for item in report_dict.get("guideline_evidence", []):
        lines.extend(["", f"- {item}"])

    lines.extend(["", "## Limitations"])

    for item in report_dict.get("limitations", []):
        lines.extend(["", f"- {item}"])

    lines.extend([
        "",
        "## Disclaimer",
        str(report_dict.get("disclaimer", "")),
    ])

    markdown_path.write_text("\n".join(lines), encoding="utf-8")

    return json_path, markdown_path
