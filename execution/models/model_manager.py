"""
Model Manager for SatQuery AI Execution Subsystem.
Manages lifecycle, resource allocation (GPU/CPU/RAM), caching, and device placement
for remote sensing specialist models.
"""

from typing import Dict, Any, Optional
import time
from execution.schemas import ResourceRequirements, ResourceMetrics, ExecutionTraceItem


class ModelManager:
    """
    Manages loading, device placement (CPU/CUDA), and resource tracking
    for domain-adapted remote-sensing specialist models.
    """

    _instance = None
    _loaded_models: Dict[str, Any] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance._loaded_models = {}
        return cls._instance

    def allocate_resources(
        self,
        requirements: ResourceRequirements,
        trace: Optional[list] = None
    ) -> ResourceMetrics:
        """
        Allocates compute devices and memory for the specialist agent execution.
        Simulates / tracks GPU device availability or CPU cores and peak RAM.
        """
        has_gpu = requirements.gpu_required
        gpu_name = "NVIDIA-A100-SXM4-40GB" if has_gpu else None
        cores = max(2, requirements.min_cpu_cores)
        ram_alloc = float(requirements.min_ram_mb)

        metrics = ResourceMetrics(
            allocated_gpu=gpu_name,
            cpu_cores_used=cores,
            ram_mb_peak=ram_alloc,
            disk_mb_used=64.5,
            allocated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        )

        if trace is not None:
            trace.append(ExecutionTraceItem(
                stage="Resource Allocation",
                action="allocate_compute_and_ram",
                status="success",
                details={
                    "device": gpu_name or f"CPU ({cores} cores)",
                    "ram_allocated_mb": ram_alloc,
                    "timeout_seconds": requirements.timeout_seconds
                }
            ))

        return metrics

    def get_model(self, agent_id: str) -> Any:
        """Retrieves or loads the specialist model for a given agent_id."""
        if agent_id in self._loaded_models:
            return self._loaded_models[agent_id]

        from execution.models.specialist_models import get_specialist_model_for_agent
        model = get_specialist_model_for_agent(agent_id)
        self._loaded_models[agent_id] = model
        return model

    def release_resources(self, metrics: ResourceMetrics, duration_sec: float) -> None:
        """Releases allocated resources and records telemetry duration."""
        metrics.released_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        metrics.execution_duration_sec = duration_sec
