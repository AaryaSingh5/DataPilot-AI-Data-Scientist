"""
Report Renderer — generates ranked, decision-useful Markdown and rich standalone HTML.

Ranked report design:
- Findings sorted by statistical strength (effect size × confidence)
- Every numeric claim cites the evidence ID it came from
- Code section included for each exportable evidence record
- Firewall violations clearly delineated from warnings
"""
from typing import Dict, Any, List
import html as html_lib


# ── Confidence → weight mapping ────────────────────────────────────────────────
_CONF_WEIGHT = {"high": 1.0, "medium": 0.6, "low": 0.3, "uncertain": 0.1}
_VERDICT_ICON = {
    "supported": "✅",
    "weak": "⚠️",
    "insufficient": "🔶",
    "refuted": "❌",
}


def _rank_score(conclusion: Dict[str, Any]) -> float:
    """Compute a ranking score for a conclusion based on confidence × verdict."""
    conf = conclusion.get("confidence", "low")
    verdict = conclusion.get("verdict", "insufficient")
    conf_w = _CONF_WEIGHT.get(conf, 0.1)
    verdict_w = {"supported": 1.0, "weak": 0.5, "insufficient": 0.2, "refuted": 0.3}.get(verdict, 0.1)
    return conf_w * verdict_w


def _format_result_summary(ev: Dict[str, Any]) -> str:
    """Extract a human-readable summary of numerical results from evidence."""
    result = ev.get("result", {})
    kind = ev.get("kind", "")
    lines = []
    if kind == "stat_test":
        p = result.get("p") or result.get("p_value")
        d = result.get("effect_size")
        n = result.get("n") or result.get("n_a")
        if p is not None:
            lines.append(f"p = {p:.4f}")
        if d is not None:
            lines.append(f"effect size (Cohen's d) = {d:.3f}")
        if n is not None:
            lines.append(f"n = {n:,}")
        q = result.get("q") or result.get("q_val")
        if q is not None:
            lines.append(f"q (FDR-corrected) = {q:.4f}")
    elif kind == "model":
        metrics = result.get("metrics", {})
        for k, v in metrics.items():
            if isinstance(v, float):
                lines.append(f"{k} = {v:.4f}")
    elif kind == "sql":
        n = result.get("num_rows")
        if n is not None:
            lines.append(f"rows returned: {n:,}")
    return " | ".join(lines) if lines else ""


def render_markdown(report: Dict[str, Any]) -> str:
    lines = []
    title = report.get("title", "DataPilot Analysis Report")
    lines.append(f"# {title}\n")

    # Metadata
    run_id = report.get("run_id", "N/A")
    dataset_hash = report.get("dataset_hash", "N/A")
    question = report.get("question", "")
    lines.append(f"**Run ID:** `{run_id}` | **Dataset SHA-256:** `{dataset_hash[:16]}...`\n")
    if question:
        lines.append(f"### Research Question\n> {question}\n")

    # Executive Summary
    summary = report.get("executive_summary", "")
    if summary:
        lines.append(f"## Executive Summary\n{summary}\n")

    # Firewall Status — top-level banner
    fw = report.get("firewall", {})
    if fw:
        fw_status = "PASSED ✅" if fw.get("passed") else "FAILED ❌"
        lines.append(f"## Hallucination Firewall: `{fw_status}`\n")
        violations = fw.get("violations", [])
        warnings = fw.get("warnings", [])
        if violations:
            lines.append("### ❌ Firewall Violations (claims removed from report)")
            for v in violations:
                lines.append(f"- **[{v.get('check_name', 'check')}]** {v.get('message', '')}")
            lines.append("")
        if warnings:
            lines.append("### ⚠️ Firewall Warnings")
            for w in warnings:
                lines.append(f"- {w.get('message', '')}")
            lines.append("")

    # Rigor Evaluation
    eval_info = report.get("evaluation", {})
    if eval_info:
        status = eval_info.get("overall_status", "passed").upper()
        score = eval_info.get("overall_score", 1.0)
        lines.append(f"## Statistical Rigor Evaluation\n- **Status:** `{status}`\n- **Quality Score:** `{score:.2f} / 1.00`\n")
        issues = eval_info.get("issues", [])
        if issues:
            lines.append("### Flagged Issues")
            for issue in issues:
                lines.append(f"- ⚠️ {issue}")
            lines.append("")

    # Ranked Findings
    conclusions = report.get("conclusions", [])
    if conclusions:
        ranked = sorted(conclusions, key=_rank_score, reverse=True)
        evidences = report.get("evidences", [])
        ev_by_id = {e.get("id"): e for e in evidences}

        lines.append("## Ranked Findings (by Statistical Strength)\n")
        lines.append("| Rank | Hypothesis | Verdict | Confidence | Key Statistics |")
        lines.append("|:---:|:---|:---:|:---:|:---|")
        for i, c in enumerate(ranked, 1):
            hid = c.get("hypothesis_id", "?")
            verdict = c.get("verdict", "insufficient")
            icon = _VERDICT_ICON.get(verdict, "")
            conf = c.get("confidence", "?")
            # Gather stats from cited evidence
            eids = c.get("evidence_ids", [])
            stats_parts = []
            for eid in eids:
                ev = ev_by_id.get(eid)
                if ev:
                    s = _format_result_summary(ev)
                    if s:
                        stats_parts.append(f"`{eid}`: {s}")
            stats_str = "; ".join(stats_parts) if stats_parts else "—"
            lines.append(f"| {i} | **{hid}** | {icon} {verdict.upper()} | *{conf}* | {stats_str} |")
        lines.append("")

        lines.append("## Detailed Findings\n")
        for i, c in enumerate(ranked, 1):
            hid = c.get("hypothesis_id", "Claim")
            verdict = c.get("verdict", "")
            conf = c.get("confidence", "medium")
            summ = c.get("summary", "")
            icon = _VERDICT_ICON.get(verdict, "")
            eids = c.get("evidence_ids", [])
            eids_str = ", ".join(f"`{eid}`" for eid in eids)

            lines.append(f"### #{i} — {hid}: {icon} **{verdict.upper()}** (Confidence: *{conf}*)")
            lines.append(f"{summ}\n")

            # Numeric evidence
            for eid in eids:
                ev = ev_by_id.get(eid)
                if ev:
                    s = _format_result_summary(ev)
                    if s:
                        lines.append(f"> 📊 **Evidence `{eid}`** ({ev.get('kind', '')}): {s}")
            if eids_str:
                lines.append(f"\n*Evidence chain:* {eids_str}\n")

            caveats = c.get("caveats", [])
            if caveats:
                for cav in caveats:
                    lines.append(f"- *Caveat:* {cav}")
                lines.append("")

    # Hypotheses from Plan
    plan_info = report.get("plan", {})
    hypotheses = plan_info.get("hypotheses", [])
    if hypotheses:
        lines.append("## Hypotheses Investigated\n")
        for h in hypotheses:
            lines.append(f"- **{h.get('id', 'H')}**: {h.get('statement', '')}")
        lines.append("")

    # Exportable Code Section
    evidences = report.get("evidences", [])
    exportable = [e for e in evidences if e.get("exportable") and e.get("code") and e.get("code") not in ("synthesis", "report_synthesis", "evaluator_threshold_checks")]
    if exportable:
        lines.append("## Reproducible Code\n")
        lines.append("The following code snippets were executed to produce the findings above.\n")
        for e in exportable:
            eid = e.get("id")
            purpose = e.get("purpose") or e.get("kind", "")
            code = e.get("code", "")
            lines.append(f"### `{eid}` — {purpose}\n")
            lines.append(f"```python\n{code}\n```\n")

    # Provenance Table
    if evidences:
        lines.append("## Provenance & Cryptographic Audit Trail\n")
        lines.append("| Evidence ID | Kind | Producer | Order | Status | Purpose | SHA-256 |")
        lines.append("|---|---|---|:---:|:---:|:---|---|")
        for e in evidences:
            eid = e.get("id")
            kind = e.get("kind")
            prod = e.get("produced_by")
            order = e.get("execution_order", "?")
            st = e.get("status")
            purpose = (e.get("purpose") or "")[:50]
            h = (e.get("hash", "") or "")[:12] + "..."
            lines.append(f"| `{eid}` | {kind} | {prod} | {order} | {st} | {purpose} | `{h}` |")
        lines.append("")

    return "\n".join(lines)


# ── HTML Rendering ─────────────────────────────────────────────────────────────

def render_html(report: Dict[str, Any]) -> str:
    """Render a rich, standalone HTML report with proper markdown conversion."""
    md = render_markdown(report)
    title = report.get("title", "DataPilot Analysis Report")
    fw_passed = report.get("firewall", {}).get("passed", True)
    score = report.get("evaluation", {}).get("overall_score", 1.0)

    # Simple markdown → HTML conversion (tables, code blocks, headings, bold/italic)
    def md_to_html(text: str) -> str:
        import re
        lines = text.split("\n")
        output = []
        in_code = False
        in_table = False
        table_rows = []

        i = 0
        while i < len(lines):
            line = lines[i]

            # Code blocks
            if line.startswith("```"):
                if in_code:
                    output.append("</code></pre>")
                    in_code = False
                else:
                    lang = line[3:].strip()
                    output.append(f'<pre><code class="language-{lang}">')
                    in_code = True
                i += 1
                continue

            if in_code:
                output.append(html_lib.escape(line))
                i += 1
                continue

            # Tables
            if line.startswith("|"):
                if not in_table:
                    in_table = True
                    table_rows = []
                # Skip separator rows
                if re.match(r'^\|[\s|:\-]+\|$', line):
                    i += 1
                    continue
                cells = [c.strip() for c in line.strip("|").split("|")]
                table_rows.append(cells)
                i += 1
                continue
            else:
                if in_table and table_rows:
                    output.append('<table>')
                    for ri, row in enumerate(table_rows):
                        tag = "th" if ri == 0 else "td"
                        output.append('<tr>' + ''.join(f'<{tag}>{_inline_md(c)}</{tag}>' for c in row) + '</tr>')
                    output.append('</table>')
                    table_rows = []
                    in_table = False

            # Headings
            if line.startswith("#### "):
                output.append(f"<h4>{_inline_md(line[5:])}</h4>")
            elif line.startswith("### "):
                output.append(f"<h3>{_inline_md(line[4:])}</h3>")
            elif line.startswith("## "):
                output.append(f"<h2>{_inline_md(line[3:])}</h2>")
            elif line.startswith("# "):
                output.append(f"<h1>{_inline_md(line[2:])}</h1>")
            # Blockquote
            elif line.startswith("> "):
                output.append(f"<blockquote>{_inline_md(line[2:])}</blockquote>")
            # List items
            elif re.match(r'^[-*] ', line):
                output.append(f"<li>{_inline_md(line[2:])}</li>")
            elif line.strip() == "":
                output.append("<br>")
            else:
                output.append(f"<p>{_inline_md(line)}</p>")
            i += 1

        if in_table and table_rows:
            output.append('<table>')
            for ri, row in enumerate(table_rows):
                tag = "th" if ri == 0 else "td"
                output.append('<tr>' + ''.join(f'<{tag}>{_inline_md(c)}</{tag}>' for c in row) + '</tr>')
            output.append('</table>')

        return "\n".join(output)

    fw_badge_class = "badge-pass" if fw_passed else "badge-fail"
    fw_badge_text = "FIREWALL PASSED" if fw_passed else "FIREWALL FAILED"
    score_pct = int(score * 100)
    score_color = "#22c55e" if score >= 0.85 else ("#f59e0b" if score >= 0.5 else "#ef4444")

    body = md_to_html(md)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{html_lib.escape(title)}</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
      line-height: 1.7;
      color: #1e293b;
      background: linear-gradient(135deg, #f0f4ff 0%, #f8fafc 100%);
      min-height: 100vh;
      padding: 40px 20px;
    }}
    .container {{
      max-width: 960px;
      margin: 0 auto;
      background: #ffffff;
      border-radius: 16px;
      box-shadow: 0 20px 60px -10px rgba(0,0,0,0.15);
      overflow: hidden;
    }}
    .report-header {{
      background: linear-gradient(135deg, #1e1b4b 0%, #312e81 50%, #1e40af 100%);
      color: white;
      padding: 40px 48px;
    }}
    .report-header h1 {{ font-size: 1.6rem; font-weight: 700; margin-bottom: 12px; color: white; border: none; }}
    .header-meta {{ display: flex; gap: 16px; flex-wrap: wrap; margin-top: 16px; }}
    .badge {{
      display: inline-flex; align-items: center; gap: 6px;
      padding: 6px 14px; border-radius: 9999px;
      font-size: 0.78rem; font-weight: 700; letter-spacing: 0.05em; text-transform: uppercase;
    }}
    .badge-pass {{ background: rgba(34, 197, 94, 0.2); color: #86efac; border: 1px solid rgba(34,197,94,0.3); }}
    .badge-fail {{ background: rgba(239, 68, 68, 0.2); color: #fca5a5; border: 1px solid rgba(239,68,68,0.3); }}
    .badge-score {{ background: rgba(255,255,255,0.1); color: white; border: 1px solid rgba(255,255,255,0.2); }}
    .report-body {{ padding: 48px; }}
    h1 {{ color: #0f172a; font-size: 1.8rem; font-weight: 800; border-bottom: 3px solid #e2e8f0; padding-bottom: 16px; margin: 32px 0 16px; }}
    h2 {{ color: #1e293b; font-size: 1.3rem; font-weight: 700; margin: 36px 0 12px; border-bottom: 2px solid #f1f5f9; padding-bottom: 8px; }}
    h3 {{ color: #334155; font-size: 1.1rem; font-weight: 600; margin: 24px 0 8px; }}
    h4 {{ color: #475569; font-size: 1rem; font-weight: 600; margin: 16px 0 6px; }}
    p {{ color: #334155; margin: 8px 0; }}
    li {{ color: #334155; margin: 6px 0 6px 24px; list-style-type: disc; }}
    blockquote {{
      background: linear-gradient(135deg, #eff6ff 0%, #f0fdf4 100%);
      border-left: 4px solid #3b82f6; border-radius: 0 8px 8px 0;
      margin: 16px 0; padding: 14px 18px; color: #1e40af;
    }}
    table {{ width: 100%; border-collapse: collapse; margin: 20px 0; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
    th {{ background: #1e1b4b; color: white; font-weight: 600; padding: 12px 16px; text-align: left; font-size: 0.85rem; }}
    td {{ padding: 10px 16px; border-bottom: 1px solid #f1f5f9; font-size: 0.9rem; }}
    tr:hover td {{ background: #f8fafc; }}
    code {{ background: #f1f5f9; padding: 2px 8px; border-radius: 6px; font-family: 'JetBrains Mono', monospace; font-size: 0.85em; color: #3730a3; }}
    pre {{ background: #0f172a; border-radius: 12px; padding: 24px; margin: 16px 0; overflow-x: auto; }}
    pre code {{ background: transparent; color: #e2e8f0; padding: 0; font-size: 0.88rem; }}
    .score-bar {{ height: 6px; background: #e2e8f0; border-radius: 9999px; margin: 8px 0 16px; }}
    .score-fill {{ height: 100%; border-radius: 9999px; background: {score_color}; width: {score_pct}%; }}
    .summary-cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin: 24px 0; }}
    .card {{ background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px; padding: 20px; text-align: center; }}
    .card-value {{ font-size: 1.8rem; font-weight: 800; color: #1e1b4b; }}
    .card-label {{ font-size: 0.8rem; color: #94a3b8; font-weight: 500; text-transform: uppercase; letter-spacing: 0.05em; margin-top: 4px; }}
    br {{ display: block; margin: 4px 0; content: ""; }}
  </style>
</head>
<body>
  <div class="container">
    <div class="report-header">
      <h1>🧭 {html_lib.escape(title)}</h1>
      <div class="header-meta">
        <span class="badge {fw_badge_class}">{fw_badge_text}</span>
        <span class="badge badge-score">Rigor Score: {score:.0%}</span>
      </div>
    </div>
    <div class="report-body">
      {body}
    </div>
  </div>
</body>
</html>"""
    return html


def _inline_md(text: str) -> str:
    """Convert inline markdown (bold, italic, code, links) to HTML."""
    import re
    # Escape HTML first
    text = html_lib.escape(text)
    # Bold+italic: ***text***
    text = re.sub(r'\*\*\*(.+?)\*\*\*', r'<strong><em>\1</em></strong>', text)
    # Bold: **text**
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    # Italic: *text*
    text = re.sub(r'\*(.+?)\*', r'<em>\1</em>', text)
    # Inline code: `text`
    text = re.sub(r'`([^`]+)`', r'<code>\1</code>', text)
    return text
