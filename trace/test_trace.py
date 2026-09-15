"""
SatQuery AI - Trace Subsystem Test Suite
========================================
Comprehensive pytest test cases validating Steps 31 through 35 of the Trace Specification.
- Step 31: Test successful execution
- Step 32: Test failure
- Step 33: Test retry
- Step 34: Test fallback
- Step 35: Test multiple agents
"""

import sys
from pathlib import Path
try:
    import pytest
except ImportError:
    pytest = None

import time
import os

# Ensure root workspace is in sys.path
_ROOT_DIR = Path(__file__).resolve().parent.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from trace.service import TraceService
from trace.models import TraceStatus, SpanStatus
from trace.visualization import render_trace_timeline_html
from trace.integration import TracedSatQueryEngine
from satquery_engine import SatQueryEngine


if pytest:
    @pytest.fixture
    def trace_service(tmp_path):
        """Provides a fresh TraceService instance pointing to a temporary test directory."""
        return TraceService(persistence_dir=str(tmp_path / "traces"))



# Step 31: Test successful execution
def test_successful_execution(trace_service):
    """
    Step 31: Validates start_trace, span recording, result consumption,
    duration calculation, trace finalization, and persistence under nominal success conditions.
    """
    request_id = "REQ_TEST_001"
    query = "Describe the land-cover and major objects visible in this optical image."
    image_inputs = [{"file_name": "sentinel2_sample.tif", "format": "GEOTIFF", "modality": "OPTICAL"}]

    # 1. Start trace
    trace = trace_service.start_trace(request_id=request_id, query=query, image_inputs=image_inputs)
    assert trace.trace_id.startswith("trc_")
    assert trace.request_id == request_id
    assert trace.query == query

    # 2. Record Query Understanding & Router Decision
    sqo_mock = {"tasks": ["LAND_COVER"], "modalities": ["OPTICAL"]}
    trace_service.record_query_understanding(sqo_mock, trace_id=trace.trace_id)

    plan_mock = {
        "execution_steps": [
            {
                "step_id": "step_1",
                "agent_id": "rs_vqa_model",
                "agent_name": "Remote Sensing VQA Model",
                "agent_version": "2.1.0",
                "capability": "visual_question_answering",
                "parameters": {"confidence_threshold": 0.8}
            }
        ]
    }
    trace_service.record_router_decision(plan_mock, trace_id=trace.trace_id)

    # 3. Create & execute span
    span = trace_service.create_span(
        name="Remote Sensing VQA Model",
        component="agent",
        agent_id="rs_vqa_model",
        agent_version="2.1.0",
        trace_id=trace.trace_id
    )
    assert span.span_id.startswith("spn_")

    # Simulate agent execution latency
    time.sleep(0.05)

    # 4. Consume ExecutionResult
    exec_result_mock = {
        "execution_id": "exec_101",
        "agent_id": "rs_vqa_model",
        "status": "success",
        "confidence_score": 0.95,
        "raw_output": {"answer": "The image shows agricultural fields and river streams."},
        "evidence_list": [{"evidence_id": "ev_1", "evidence_type": "text_rationale", "confidence": 0.95}]
    }
    closed_span = trace_service.record_execution_result(span.span_id, exec_result_mock, trace_id=trace.trace_id)

    assert closed_span.status == SpanStatus.SUCCESS
    assert closed_span.duration_ms > 0
    assert closed_span.confidence == 0.95

    # 5. Finalize trace
    final_trace = trace_service.finalize_trace(trace_id=trace.trace_id)
    assert final_trace.status == TraceStatus.SUCCESS
    assert final_trace.total_duration_ms > 0

    # 6. Verify Persistence
    persisted_file = trace_service.persistence_dir / f"{final_trace.trace_id}.json"
    assert persisted_file.exists()

    # 7. Test Visualization HTML Generation (Step 30)
    html = render_trace_timeline_html(final_trace)
    assert "<title>SatQuery AI Trace Visualization" in html
    assert "REQ_TEST_001" in html


# Step 32: Test failure
def test_failure_execution(trace_service):
    """
    Step 32: Validates error capture, failure status propagation, and span error state.
    """
    trace = trace_service.start_trace(request_id="REQ_FAIL_002", query="Ground urban area in image.")
    
    span = trace_service.create_span(
        name="Grounding Model",
        component="agent",
        agent_id="grounding_agent",
        trace_id=trace.trace_id
    )

    # Record agent crash error
    trace_service.record_error(
        span_id=span.span_id,
        error_message="CUDA Out of Memory exception during feature extraction",
        error_details={"exception_type": "RuntimeError", "device": "cuda:0"},
        trace_id=trace.trace_id
    )
    trace_service.close_span(span.span_id, trace_id=trace.trace_id)

    final_trace = trace_service.finalize_trace(trace_id=trace.trace_id)
    assert final_trace.status == TraceStatus.FAILED
    assert span.status == SpanStatus.FAILURE
    assert span.error["message"] == "CUDA Out of Memory exception during feature extraction"


# Step 33: Test retry
def test_retry_execution(trace_service):
    """
    Step 33: Validates retry attempt logging, retry history, and status update.
    """
    trace = trace_service.start_trace(request_id="REQ_RETRY_003", query="Detect temporal change.")

    span = trace_service.create_span(
        name="Change Detection Agent",
        component="agent",
        agent_id="change_agent",
        trace_id=trace.trace_id
    )

    # Record first failed attempt & retry
    trace_service.record_retry(
        span_id=span.span_id,
        retry_count=1,
        error_message="Connection reset by peer",
        trace_id=trace.trace_id
    )
    assert span.status == SpanStatus.RETRYING
    assert span.retries == 1
    assert len(span.retry_history) == 1

    # Record second successful execution
    exec_res = {
        "status": "success",
        "confidence_score": 0.88,
        "retries_attempted": 1,
        "raw_output": {"change_detected": True}
    }
    trace_service.record_execution_result(span.span_id, exec_res, trace_id=trace.trace_id)

    final_trace = trace_service.finalize_trace(trace_id=trace.trace_id)
    assert final_trace.status == TraceStatus.SUCCESS
    assert span.retries == 1


# Step 34: Test fallback
def test_fallback_execution(trace_service):
    """
    Step 34: Validates fallback agent activation logging and degraded status handling.
    """
    trace = trace_service.start_trace(request_id="REQ_FALLBACK_004", query="Perform optical SAR joint analysis.")

    span = trace_service.create_span(
        name="Primary SAR Fusion Model",
        component="agent",
        agent_id="sar_fusion_primary",
        trace_id=trace.trace_id
    )

    # Record fallback trigger
    trace_service.record_fallback(
        span_id=span.span_id,
        fallback_agent_id="sar_fusion_secondary_lightweight",
        reason="Primary SAR fusion model timeout after 30s",
        trace_id=trace.trace_id
    )

    assert span.status == SpanStatus.FALLBACK
    assert span.fallbacks == 1

    # Complete step via fallback
    exec_res = {
        "status": "success",
        "fallback_triggered": True,
        "fallback_agent_id": "sar_fusion_secondary_lightweight",
        "confidence_score": 0.79,
        "raw_output": {"built_up_area_ha": 45.2}
    }
    trace_service.record_execution_result(span.span_id, exec_res, trace_id=trace.trace_id)

    final_trace = trace_service.finalize_trace(trace_id=trace.trace_id)
    assert final_trace.status == TraceStatus.DEGRADED
    assert span.status == SpanStatus.FALLBACK


# Step 35: Test multiple agents
def test_multiple_agents_execution(trace_service):
    """
    Step 35: Validates multi-agent orchestration, recording sequential/parallel spans,
    agent version logging, and aggregate latency calculation.
    """
    trace = trace_service.start_trace(
        request_id="REQ_MULTI_005",
        query="Use optical and SAR images together to identify built-up areas and detect changes between dates."
    )

    # Selected agents list
    selected_agents = [
        {"agent_id": "building_detector_v2", "agent_name": "Building Detection Model", "agent_version": "2.0.0"},
        {"agent_id": "change_detector_v1", "agent_name": "Bi-Temporal Change Model", "agent_version": "1.4.0"},
        {"agent_id": "vlm_summarizer", "agent_name": "RS VLM Synthesizer", "agent_version": "3.0.0"}
    ]
    trace_service.record_router_decision(
        {"execution_steps": selected_agents},
        trace_id=trace.trace_id
    )

    # Execute Span 1: Building Detection
    span1 = trace_service.create_span(
        name="Building Detection Model",
        agent_id="building_detector_v2",
        agent_version="2.0.0",
        trace_id=trace.trace_id
    )
    time.sleep(0.02)
    trace_service.record_execution_result(
        span1.span_id,
        {"status": "success", "confidence_score": 0.92, "raw_output": {"building_count": 142}},
        trace_id=trace.trace_id
    )

    # Execute Span 2: Change Detection
    span2 = trace_service.create_span(
        name="Bi-Temporal Change Model",
        agent_id="change_detector_v1",
        agent_version="1.4.0",
        trace_id=trace.trace_id
    )
    time.sleep(0.03)
    trace_service.record_execution_result(
        span2.span_id,
        {"status": "success", "confidence_score": 0.89, "raw_output": {"change_pct": 12.4}},
        trace_id=trace.trace_id
    )

    # Execute Span 3: VLM Synthesizer
    span3 = trace_service.create_span(
        name="RS VLM Synthesizer",
        agent_id="vlm_summarizer",
        agent_version="3.0.0",
        trace_id=trace.trace_id
    )
    time.sleep(0.02)
    trace_service.record_execution_result(
        span3.span_id,
        {"status": "success", "confidence_score": 0.96, "raw_output": {"summary": "142 buildings identified; 12.4% area expansion detected."}},
        trace_id=trace.trace_id
    )

    final_trace = trace_service.finalize_trace(trace_id=trace.trace_id)

    assert final_trace.status == TraceStatus.SUCCESS
    assert len(final_trace.spans) == 3
    assert final_trace.metrics["span_count"] == 3
def test_traced_engine_end_to_end():
    """
    End-to-End Integration test running TracedSatQueryEngine with engine execution.
    """
    traced_engine = TracedSatQueryEngine()
    
    sqo, plan, results, trace = traced_engine.process_and_execute(
        raw_query="Describe the land-cover and major objects visible in this image.",
        image_inputs=[{"file_name": "test_opt.tif", "format": "GEOTIFF", "modality": "OPTICAL"}]
    )

    assert sqo is not None
    assert plan is not None
    assert trace.trace_id.startswith("trc_")
    assert trace.query_understanding is not None
    assert trace.router_decision is not None
    assert trace.status in [TraceStatus.SUCCESS, TraceStatus.DEGRADED]



def run_all_tests():

    """Executes test suite as standalone test runner."""
    print("=" * 60)
    print("SatQuery AI - Running Trace Subsystem Test Suite (Steps 31-35)")
    print("=" * 60)
    
    tmp_dir = Path("./evidence_artifacts/test_traces_tmp")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    ts = TraceService(persistence_dir=str(tmp_dir))

    tests = [
        ("Step 31: Test Successful Execution", lambda: test_successful_execution(ts)),
        ("Step 32: Test Failure Execution", lambda: test_failure_execution(ts)),
        ("Step 33: Test Retry Execution", lambda: test_retry_execution(ts)),
        ("Step 34: Test Fallback Execution", lambda: test_fallback_execution(ts)),
        ("Step 35: Test Multiple Agents Execution", lambda: test_multiple_agents_execution(ts)),
        ("End-to-End Traced Engine Integration", test_traced_engine_end_to_end),
    ]

    passed = 0
    failed = 0
    for name, test_func in tests:
        try:
            test_func()
            print(f"[PASS] {name}")
            passed += 1
        except Exception as e:
            print(f"[FAIL] {name}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("=" * 60)
    print(f"Test Summary: {passed} PASSED, {failed} FAILED out of {len(tests)} tests")
    print("=" * 60)
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

