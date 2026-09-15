"""
SatQuery AI - Trace Retrieval REST API
======================================
FastAPI endpoints for trace retrieval, listing, exporting, and visual rendering.
Satisfies Step 29 of the Trace Specification.
"""

from typing import Optional, List, Dict, Any
try:
    from fastapi import APIRouter, HTTPException, Query, Response
    from fastapi.responses import HTMLResponse, JSONResponse
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False
    class APIRouter:
        def __init__(self, *args, **kwargs): pass
        def get(self, *args, **kwargs): return lambda f: f
        def post(self, *args, **kwargs): return lambda f: f
    class HTTPException(Exception):
        def __init__(self, status_code: int, detail: str):
            self.status_code = status_code
            self.detail = detail
            super().__init__(detail)
    Query = lambda default=None, **kwargs: default
    class Response:
        def __init__(self, content: Any, media_type: str = "text/plain", headers: Dict[str, str] = None):
            self.content = content
            self.media_type = media_type
            self.headers = headers or {}
    class HTMLResponse(Response): pass
    class JSONResponse(Response): pass


from trace.service import TraceService
from trace.visualization import render_trace_timeline_html


router = APIRouter(prefix="/api/trace", tags=["Trace System"])
_service_instance: Optional[TraceService] = None


def get_trace_service() -> TraceService:
    global _service_instance
    if _service_instance is None:
        _service_instance = TraceService()
    return _service_instance


@router.get("s", response_model=List[Dict[str, Any]])
def list_traces(
    limit: int = Query(default=20, ge=1, le=100),
    status: Optional[str] = None
):
    """
    Step 29: List recent execution traces with optional status filtering.
    """
    service = get_trace_service()
    traces = service.list_traces()
    if status:
        traces = [t for t in traces if str(t.get("status")).lower() == status.lower()]
    return traces[:limit]


@router.get("/{trace_id}")
def get_trace_detail(trace_id: str):
    """
    Step 29: Retrieve complete JSON telemetry for a specific trace_id.
    """
    service = get_trace_service()
    trace = service.get_trace(trace_id)
    if not trace:
        raise HTTPException(status_code=404, detail=f"Trace with ID '{trace_id}' not found.")
    
    if hasattr(trace, "model_dump"):
        return trace.model_dump()
    return trace.dict()


@router.get("/{trace_id}/export")
def export_trace_json(trace_id: str):
    """
    Step 29: Export downloadable trace JSON file.
    """
    service = get_trace_service()
    trace = service.get_trace(trace_id)
    if not trace:
        raise HTTPException(status_code=404, detail=f"Trace with ID '{trace_id}' not found.")

    if hasattr(trace, "model_dump_json"):
        content = trace.model_dump_json(indent=2)
    else:
        content = trace.json(indent=2)

    return Response(
        content=content,
        media_type="application/json",
        headers={"Content-Disposition": f"attachment; filename=satquery_trace_{trace_id}.json"}
    )


@router.get("/{trace_id}/viz", response_class=HTMLResponse)
def get_trace_visualization(trace_id: str):
    """
    Step 30: Render self-contained HTML visual timeline for the requested trace.
    """
    service = get_trace_service()
    trace = service.get_trace(trace_id)
    if not trace:
        raise HTTPException(status_code=404, detail=f"Trace with ID '{trace_id}' not found.")

    html_content = render_trace_timeline_html(trace)
    return HTMLResponse(content=html_content, status_code=200)
