"""
SatQuery AI - GUI Trace Visualization Generator
===============================================
Renders self-contained, interactive HTML/JS execution timelines, waterfall Gantt charts,
agent orchestration DAG node graphs, and audit logs.
Satisfies Step 30 of the Trace Specification.
"""

import json
from typing import Any, Dict, Union
from trace.models import TraceData


def render_trace_timeline_html(trace: Union[TraceData, Dict[str, Any]]) -> str:
    """
    Step 30: Renders an interactive, standalone HTML timeline and DAG dashboard for a trace object.
    """
    if hasattr(trace, "model_dump"):
        data = trace.model_dump()
    elif hasattr(trace, "dict"):
        data = trace.dict()
    elif isinstance(trace, dict):
        data = trace
    else:
        data = {}

    trace_id = data.get("trace_id", "N/A")
    request_id = data.get("request_id", "N/A")
    query = data.get("query", "")
    total_duration_ms = data.get("total_duration_ms", 0.0) or 0.0
    status = str(data.get("status", "UNKNOWN")).upper()

    spans = data.get("spans", [])
    events = data.get("events", [])
    selected_agents = data.get("selected_agents", [])

    # Calculate min and max times for Gantt normalization
    min_time = data.get("created_at", 0.0)
    if spans:
        span_starts = [s.get("start_time", min_time) for s in spans]
        min_time = min(span_starts)

    json_str = json.dumps(data, indent=2)

    # Theme badge color based on status
    status_bg = "#10b981" if status == "SUCCESS" else "#f59e0b" if status in ["DEGRADED", "PARTIAL_SUCCESS"] else "#ef4444"

    # Build spans HTML rows
    span_rows_html = ""
    for idx, span in enumerate(spans, 1):
        s_id = span.get("span_id", "")
        name = span.get("name", "Unknown Step")
        component = span.get("component", "agent")
        agent_name = span.get("agent_name") or name
        agent_ver = span.get("agent_version", "1.0.0")
        dur_ms = span.get("duration_ms", 0.0) or 0.0
        s_status = str(span.get("status", "SUCCESS")).upper()
        confidence = span.get("confidence")
        conf_str = f"{round(confidence * 100, 1)}%" if confidence is not None else "N/A"
        retries = span.get("retries", 0)
        fallbacks = span.get("fallbacks", 0)

        # Status badge styling
        s_badge_cls = "badge-success" if s_status == "SUCCESS" else "badge-retry" if s_status == "RETRYING" else "badge-fallback" if s_status == "FALLBACK" else "badge-failure"

        # Calculate offset percentage for Gantt bar
        start_t = span.get("start_time", min_time)
        offset_ms = max(0.0, (start_t - min_time) * 1000.0)
        max_duration = max(total_duration_ms, 1.0)
        left_pct = min(95.0, (offset_ms / max_duration) * 100.0)
        width_pct = max(2.0, min(100.0 - left_pct, (dur_ms / max_duration) * 100.0))

        bar_color = "#10b981" if s_status == "SUCCESS" else "#f59e0b" if s_status in ["RETRYING", "FALLBACK"] else "#ef4444"

        span_rows_html += f"""
        <tr class="span-row" onclick="toggleDetails('{s_id}')">
            <td><span class="step-num">{idx}</span></td>
            <td>
                <div class="agent-title">{agent_name}</div>
                <div class="sub-text">{component} &bull; v{agent_ver}</div>
            </td>
            <td><span class="badge {s_badge_cls}">{s_status}</span></td>
            <td>{dur_ms:.1f} ms</td>
            <td>{conf_str}</td>
            <td>
                <div class="gantt-bar-container">
                    <div class="gantt-bar" style="left: {left_pct:.1f}%; width: {width_pct:.1f}%; background-color: {bar_color};"></div>
                </div>
            </td>
        </tr>
        <tr id="details-{s_id}" class="details-row" style="display: none;">
            <td colspan="6">
                <div class="details-box">
                    <h4>Span Details: {s_id}</h4>
                    <p><strong>Parameters:</strong> <code>{json.dumps(span.get('parameters', {}))}</code></p>
                    <p><strong>Input References:</strong> <code>{json.dumps(span.get('input_references', {}))}</code></p>
                    <p><strong>Output References:</strong> <code>{json.dumps(span.get('output_references', {}))}</code></p>
                    <p><strong>Retries:</strong> {retries} | <strong>Fallbacks:</strong> {fallbacks}</p>
                    {f'<div class="error-box"><strong>Error:</strong> {json.dumps(span.get("error"))}</div>' if span.get("error") else ''}
                </div>
            </td>
        </tr>
        """

    # Build Events Log HTML
    events_html = ""
    for evt in events:
        evt_type = evt.get("event_type", "")
        evt_comp = evt.get("component", "")
        evt_msg = evt.get("message", "")
        evt_time = evt.get("iso_timestamp", "").split("T")[-1][:8]
        events_html += f"""
        <div class="event-item">
            <span class="event-time">{evt_time}</span>
            <span class="event-comp">[{evt_comp}]</span>
            <span class="event-type">{evt_type}:</span>
            <span class="event-msg">{evt_msg}</span>
        </div>
        """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SatQuery AI Trace Visualization - {trace_id}</title>
    <style>
        :root {{
            --bg: #0f172a;
            --card-bg: #1e293b;
            --border: #334155;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --accent: #38bdf8;
            --success: #10b981;
            --warning: #f59e0b;
            --danger: #ef4444;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }}
        body {{ background-color: var(--bg); color: var(--text-main); padding: 24px; font-size: 14px; line-height: 1.5; }}
        .container {{ max-width: 1200px; margin: 0 auto; display: flex; flex-direction: column; gap: 20px; }}
        .card {{ background-color: var(--card-bg); border: 1px solid var(--border); border-radius: 12px; padding: 20px; box-shadow: 0 4px 12px rgba(0,0,0,0.3); }}
        .header {{ display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--border); padding-bottom: 16px; margin-bottom: 16px; }}
        .header-title h1 {{ font-size: 20px; color: var(--accent); display: flex; align-items: center; gap: 8px; }}
        .header-title p {{ color: var(--text-muted); font-size: 12px; margin-top: 4px; }}
        .status-pill {{ background-color: {status_bg}; color: #fff; padding: 6px 14px; border-radius: 20px; font-weight: bold; font-size: 13px; text-transform: uppercase; letter-spacing: 0.5px; }}
        .query-box {{ background: rgba(56, 189, 248, 0.1); border-left: 4px solid var(--accent); padding: 12px 16px; border-radius: 4px; font-size: 15px; margin-bottom: 16px; }}
        .stats-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 16px; }}
        .stat-card {{ background: #0f172a; padding: 12px; border-radius: 8px; border: 1px solid var(--border); text-align: center; }}
        .stat-val {{ font-size: 18px; font-weight: bold; color: var(--accent); }}
        .stat-lbl {{ font-size: 11px; color: var(--text-muted); text-transform: uppercase; margin-top: 2px; }}
        
        /* DAG Workflow */
        .dag-container {{ display: flex; align-items: center; justify-content: space-between; overflow-x: auto; padding: 16px 0; gap: 12px; }}
        .dag-node {{ background: #0f172a; border: 1px solid var(--accent); border-radius: 8px; padding: 10px 16px; text-align: center; min-width: 140px; flex-shrink: 0; }}
        .dag-node.agent-node {{ border-color: var(--success); }}
        .dag-arrow {{ color: var(--text-muted); font-size: 18px; font-weight: bold; }}

        /* Timeline Table */
        table {{ width: 100%; border-collapse: collapse; margin-top: 12px; }}
        th {{ text-align: left; padding: 10px; color: var(--text-muted); font-weight: 600; border-bottom: 1px solid var(--border); font-size: 12px; text-transform: uppercase; }}
        td {{ padding: 12px 10px; border-bottom: 1px solid var(--border); vertical-align: middle; }}
        .span-row {{ cursor: pointer; transition: background 0.2s; }}
        .span-row:hover {{ background: rgba(255,255,255,0.05); }}
        .step-num {{ background: #334155; color: #fff; width: 22px; height: 22px; border-radius: 50%; display: inline-flex; align-items: center; justify-content: center; font-size: 11px; font-weight: bold; }}
        .agent-title {{ font-weight: 600; color: #fff; }}
        .sub-text {{ font-size: 11px; color: var(--text-muted); }}
        
        .badge {{ padding: 3px 8px; border-radius: 12px; font-size: 10px; font-weight: bold; text-transform: uppercase; }}
        .badge-success {{ background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid #10b981; }}
        .badge-retry {{ background: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid #f59e0b; }}
        .badge-fallback {{ background: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid #f59e0b; }}
        .badge-failure {{ background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid #ef4444; }}

        .gantt-bar-container {{ width: 100%; height: 16px; background: #0f172a; border-radius: 8px; position: relative; overflow: hidden; border: 1px solid var(--border); }}
        .gantt-bar {{ height: 100%; position: absolute; border-radius: 6px; min-width: 4px; transition: width 0.3s; }}
        
        .details-box {{ background: #0f172a; padding: 14px; border-radius: 8px; border: 1px solid var(--border); font-size: 12px; display: flex; flex-direction: column; gap: 8px; }}
        .details-box h4 {{ color: var(--accent); margin-bottom: 4px; }}
        .details-box code {{ background: #1e293b; padding: 2px 6px; border-radius: 4px; color: #cbd5e1; word-break: break-all; }}
        .error-box {{ background: rgba(239, 68, 68, 0.15); border: 1px solid var(--danger); color: #f87171; padding: 8px; border-radius: 6px; }}

        /* Events Log */
        .events-list {{ display: flex; flex-direction: column; gap: 6px; max-height: 200px; overflow-y: auto; background: #0f172a; padding: 10px; border-radius: 8px; border: 1px solid var(--border); font-family: monospace; font-size: 12px; }}
        .event-item {{ display: flex; gap: 10px; color: var(--text-muted); }}
        .event-time {{ color: var(--accent); }}
        .event-comp {{ color: #a855f7; font-weight: bold; }}
        .event-type {{ color: #e2e8f0; font-weight: bold; }}
        .event-msg {{ color: #cbd5e1; }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Top Summary Header -->
        <div class="card">
            <div class="header">
                <div class="header-title">
                    <h1>🛰️ SatQuery AI Execution Trace</h1>
                    <p>Request ID: <strong>{request_id}</strong> &bull; Trace ID: <strong>{trace_id}</strong></p>
                </div>
                <div class="status-pill">{status}</div>
            </div>

            <div class="query-box">
                <strong>Query:</strong> "{query}"
            </div>

            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-val">{total_duration_ms:.1f} ms</div>
                    <div class="stat-lbl">Total Execution Latency</div>
                </div>
                <div class="stat-card">
                    <div class="stat-val">{len(spans)}</div>
                    <div class="stat-lbl">Execution Spans</div>
                </div>
                <div class="stat-card">
                    <div class="stat-val">{len(selected_agents)}</div>
                    <div class="stat-lbl">Selected Specialist Models</div>
                </div>
                <div class="stat-card">
                    <div class="stat-val">{len(events)}</div>
                    <div class="stat-lbl">Trace Events Recorded</div>
                </div>
            </div>
        </div>

        <!-- Orchestration DAG -->
        <div class="card">
            <h2 style="font-size: 16px; color: var(--accent); margin-bottom: 12px;">Agentic Execution Flow DAG</h2>
            <div class="dag-container">
                <div class="dag-node">
                    <div class="agent-title">Query Received</div>
                    <div class="sub-text">User Input</div>
                </div>
                <div class="dag-arrow">&rarr;</div>
                <div class="dag-node">
                    <div class="agent-title">Query Understanding</div>
                    <div class="sub-text">NLP & Task Parser</div>
                </div>
                <div class="dag-arrow">&rarr;</div>
                <div class="dag-node">
                    <div class="agent-title">SatQuery Router</div>
                    <div class="sub-text">Capability Scoring</div>
                </div>
                <div class="dag-arrow">&rarr;</div>
                {"".join([f'<div class="dag-node agent-node"><div class="agent-title">{a.get("agent_name") or a.get("agent_id")}</div><div class="sub-text">Specialist Agent</div></div><div class="dag-arrow">&rarr;</div>' for a in selected_agents])}
                <div class="dag-node">
                    <div class="agent-title">Evidence Integration</div>
                    <div class="sub-text">Final Response</div>
                </div>
            </div>
        </div>

        <!-- Execution Gantt Waterfall -->
        <div class="card">
            <h2 style="font-size: 16px; color: var(--accent); margin-bottom: 8px;">Execution Waterfall & Agent Spans</h2>
            <table>
                <thead>
                    <tr>
                        <th style="width: 40px;">#</th>
                        <th style="width: 220px;">Agent / Component</th>
                        <th style="width: 100px;">Status</th>
                        <th style="width: 100px;">Latency</th>
                        <th style="width: 80px;">Confidence</th>
                        <th>Execution Timeline Waterfall</th>
                    </tr>
                </thead>
                <tbody>
                    {span_rows_html}
                </tbody>
            </table>
        </div>

        <!-- Trace Audit Events Log -->
        <div class="card">
            <h2 style="font-size: 16px; color: var(--accent); margin-bottom: 12px;">Trace Audit Event Stream</h2>
            <div class="events-list">
                {events_html}
            </div>
        </div>

        <!-- Raw JSON Data Toggle -->
        <div class="card">
            <details>
                <summary style="cursor: pointer; color: var(--accent); font-weight: bold;">View Full Raw Trace JSON Telemetry</summary>
                <pre style="background: #0f172a; padding: 16px; border-radius: 8px; margin-top: 12px; max-height: 400px; overflow-y: auto; color: #38bdf8; font-size: 12px;">{json_str}</pre>
            </details>
        </div>
    </div>

    <script>
        function toggleDetails(spanId) {{
            const row = document.getElementById('details-' + spanId);
            if (row) {{
                row.style.display = row.style.display === 'none' ? 'table-row' : 'none';
            }}
        }}
    </script>
</body>
</html>
"""
    return html
