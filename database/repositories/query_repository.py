"""
Repository for Query Sessions, Evidence Artifacts, & Aggregated Results in PostgreSQL.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
import uuid
from sqlalchemy.orm import Session
from database.models.query_model import QuerySessionModel
from database.models.evidence_model import EvidenceArtifactModel
from database.models.result_model import QueryResultModel


class QueryRepository:
    def __init__(self, db: Session):
        self.db = db

    def create_query_session(
        self,
        raw_query: str,
        image_inputs: List[Dict[str, Any]],
        project_context: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> QuerySessionModel:
        """Creates a new query session record."""
        sid = session_id or str(uuid.uuid4())
        session = QuerySessionModel(
            session_id=sid,
            user_id=user_id,
            raw_query=raw_query,
            image_inputs=image_inputs,
            project_context=project_context or {},
            status="PENDING"
        )
        self.db.add(session)
        self.db.commit()
        self.db.refresh(session)
        return session

    def update_query_plan(
        self,
        session_id: str,
        sqo: Dict[str, Any],
        execution_plan: Dict[str, Any],
        status: str = "PROCESSING"
    ) -> Optional[QuerySessionModel]:
        """Updates SQO, execution plan, and status for a session."""
        session = self.db.query(QuerySessionModel).filter(QuerySessionModel.session_id == session_id).first()
        if session:
            session.sqo = sqo
            session.execution_plan = execution_plan
            session.status = status
            session.updated_at = datetime.now(timezone.utc)
            self.db.commit()
            self.db.refresh(session)
        return session

    def save_evidence_artifact(
        self,
        session_id: str,
        agent_id: str,
        modality: str,
        evidence_type: str,
        file_path: Optional[str] = None,
        bbox_coords: Optional[List[float]] = None,
        confidence_score: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> EvidenceArtifactModel:
        """Saves a grounded specialist evidence artifact."""
        artifact = EvidenceArtifactModel(
            artifact_id=str(uuid.uuid4()),
            session_id=session_id,
            agent_id=agent_id,
            modality=modality,
            evidence_type=evidence_type,
            file_path=file_path,
            bbox_coords=bbox_coords or [],
            confidence_score=confidence_score,
            metadata_json=metadata or {}
        )
        self.db.add(artifact)
        self.db.commit()
        self.db.refresh(artifact)
        return artifact

    def save_query_result(
        self,
        session_id: str,
        consensus_score: float,
        synthesized_text: str,
        evidence_summary: List[Dict[str, Any]],
        conflict_resolution: Optional[Dict[str, Any]] = None
    ) -> QueryResultModel:
        """Saves the final aggregated consensus query result."""
        result = QueryResultModel(
            result_id=str(uuid.uuid4()),
            session_id=session_id,
            consensus_score=consensus_score,
            synthesized_text=synthesized_text,
            evidence_summary=evidence_summary,
            conflict_resolution=conflict_resolution or {}
        )
        self.db.add(result)
        
        # Mark session as COMPLETED
        session = self.db.query(QuerySessionModel).filter(QuerySessionModel.session_id == session_id).first()
        if session:
            session.status = "COMPLETED"
            session.updated_at = datetime.now(timezone.utc)

        self.db.commit()
        self.db.refresh(result)
        return result

    def get_query_session(self, session_id: str) -> Optional[QuerySessionModel]:
        """Fetches query session by ID."""
        return self.db.query(QuerySessionModel).filter(QuerySessionModel.session_id == session_id).first()

    def get_query_result(self, session_id: str) -> Optional[QueryResultModel]:
        """Fetches final query result by session_id."""
        return self.db.query(QueryResultModel).filter(QueryResultModel.session_id == session_id).first()

    def get_evidence_artifacts(self, session_id: str) -> List[EvidenceArtifactModel]:
        """Fetches all evidence artifacts for a query session."""
        return self.db.query(EvidenceArtifactModel).filter(EvidenceArtifactModel.session_id == session_id).all()
