"""
Report Renderer — generates GitHub-flavored Markdown and styled standalone HTML.
"""
from typing import Dict, Any, List


def render_markdown(report: Dict[str, Any]) -> str:
    lines = []
    title = report.get("title", "DataPilot Analysis Report")
    lines.append(f"# {title}\n")

    # Metadata
    run_id = report.get("run_id", "N/A")
    dataset_hash = report.get("dataset_hash", "N/A")
    question = report.get("question", "")
    lines.append(f"**Run ID:** `{run_id}` | **Dataset SHA-256:** `{dataset_hash[:12]}...`\n")
    if question:
        lines.append(f"### Research Question\n> {question}\n")

    # Executive Summary
    summary = report.get("executive_summary", "")
    if summary:
        lines.append(f"## Executive Summary\n{summary}\n")

    # Hypotheses from Plan
    plan_info = report.get("plan", {})
    hypotheses = plan_info.get("hypotheses", [])
    if hypotheses:
        lines.append("## Hypotheses Investigated\n")
        for h in hypotheses:
            lines.append(f"- **{h.get('id', 'H')}**: {h.get('statement', '')}")
        lines.append("")

    # Evaluator Status Badge
    eval_info = report.get("evaluation", {})
    if eval_info:
        status = eval_info.get("overall_status", "passed").upper()
        score = eval_info.get("overall_score", 1.0)
        lines.append(f"## Rigor & Evaluation\n- **Status:** `{status}`\n- **Quality Score:** `{score:.2f} / 1.00`\n")
        issues = eval_info.get("issues", [])
        if issues:
            lines.append("### Flagged Items & Caveats")
            for issue in issues:
                lines.append(f"- ⚠️ {issue}")
            lines.append("")

    # Firewall Status
    fw = report.get("firewall", {})
    if fw:
        fw_status = "PASSED" if fw.get("passed") else "FAILED"
        lines.append(f"## Hallucination Firewall: `{fw_status}`\n")
        if not fw.get("passed"):
            for v in fw.get("violations", []):
                lines.append(f"- ❌ **[{v.get('check_name')}]** {v.get('message')}")
            lines.append("")

    # Findings & Conclusions
    conclusions = report.get("conclusions", [])
    if conclusions:
        lines.append("## Findings & Hypothesis Verdicts\n")
        for c in conclusions:
            hid = c.get("hypothesis_id", "Claim")
            verdict = c.get("verdict", "").upper()
            conf = c.get("confidence", "medium")
            summ = c.get("summary", "")
            eids = ", ".join(f"`{eid}`" for eid in c.get("evidence_ids", []))
            lines.append(f"### {hid}: **{verdict}** (Confidence: *{conf}*)")
            lines.append(f"{summ}\n")
            if eids:
                lines.append(f"*Citing Evidence:* {eids}\n")
            caveats = c.get("caveats", [])
            if caveats:
                for cav in caveats:
                    lines.append(f"- *Caveat:* {cav}")
                lines.append("")

    # Visualizations
    charts = report.get("charts", [])
    if charts:
        lines.append("## Key Visualizations\n")
        for ch in charts:
            ch_type = ch.get("type", "chart")
            ch_title = ch.get("title", f"{ch_type.capitalize()} Chart")
            lines.append(f"### {ch_title} ({ch_type})\n")
            if ch_type == "bar":
                lines.append("| Category | Value |")
                lines.append("|---|---|")
                for item in ch.get("series", []):
                    lines.append(f"| {item.get('label')} | {item.get('value')} |")
                lines.append("")
            elif ch_type == "waterfall":
                lines.append("| Component | Value | Running Total |")
                lines.append("|---|---|---|")
                for b in ch.get("bars", []):
                    lines.append(f"| {b.get('label')} | {b.get('value'):.2f} | {b.get('end'):.2f} |")
                lines.append("")

    # Provenance & Audit Trail
    evidences = report.get("evidences", [])
    if evidences:
        lines.append("## Provenance & Cryptographic Audit Trail\n")
        lines.append("| Evidence ID | Kind | Producer | Status | SHA-256 Hash |")
        lines.append("|---|---|---|---|---|")
        for e in evidences:
            eid = e.get("id")
            kind = e.get("kind")
            prod = e.get("produced_by")
            st = e.get("status")
            h = e.get("hash", "")[:12] + "..." if e.get("hash") else "N/A"
            lines.append(f"| `{eid}` | {kind} | {prod} | {st} | `{h}` |")
        lines.append("")

    return "\n".join(lines)


def render_html(report: Dict[str, Any]) -> str:
    md = render_markdown(report)
    title = report.get("title", "DataPilot Analysis Report")

    # Premium dark/clean card styling
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>{title}</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      line-height: 1.6;
      color: #1e293b;
      background: #f8fafc;
      margin: 0;
      padding: 40px 20px;
    }}
    .container {{
      max-width: 900px;
      margin: 0 auto;
      background: #ffffff;
      padding: 40px;
      border-radius: 12px;
      box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
    }}
    h1 {{ color: #0f172a; border-bottom: 2px solid #e2e8f0; padding-bottom: 12px; }}
    h2 {{ color: #1e293b; margin-top: 32px; border-bottom: 1px solid #f1f5f9; padding-bottom: 8px; }}
    h3 {{ color: #334155; }}
    blockquote {{
      background: #f1f5f9;
      border-left: 4px solid #3b82f6;
      margin: 16px 0;
      padding: 12px 16px;
      border-radius: 4px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin: 20px 0;
    }}
    th, td {{
      text-align: left;
      padding: 10px 14px;
      border: 1px solid #e2e8f0;
    }}
    th {{
      background: #f8fafc;
      font-weight: 600;
    }}
    code {{
      background: #f1f5f9;
      padding: 2px 6px;
      border-radius: 4px;
      font-size: 0.9em;
      color: #0f172a;
    }}
    .badge {{
      display: inline-block;
      padding: 4px 10px;
      border-radius: 9999px;
      font-size: 0.85em;
      font-weight: 600;
    }}
    .badge-pass {{ background: #dcfce7; color: #15803d; }}
    .badge-warn {{ background: #fef9c3; color: #a16207; }}
    .badge-fail {{ background: #fee2e2; color: #b91c1c; }}
  </style>
</head>
<body>
  <div class="container">
    <div id="content">
      <pre style="white-space: pre-wrap; font-family: inherit;">{md}</pre>
    </div>
  </div>
</body>
</html>
"""
    return html
