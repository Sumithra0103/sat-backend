"""
Query Understanding Pipeline Orchestrator for SatQuery AI.
Combines Preprocessing, Context Resolution, Semantic Parsing, Task Classification,
Ambiguity Detection, and Execution Tracing into a unified entry point.
"""

import uuid
from typing import List, Dict, Any, Optional

from .models import (
    StructuredQueryObject,
    ConversationSession,
    ConversationTurn,
    ProjectContext,
    ImageMetadata,
    ExecutionTraceStep
)
from .preprocessor import InputPreprocessor
from .context_resolver import ContextResolver
from .semantic_parser import SemanticParser
from .classifier import QueryClassifier
from .ambiguity_detector import AmbiguityDetector


class QueryUnderstandingPipeline:
    """
    Main orchestrator for the Query Understanding phase of SatQuery AI.
    Processes user query text + image inputs + session context -> returns StructuredQueryObject (SQO).
    """

    def __init__(self):
        self.preprocessor = InputPreprocessor()
        self.context_resolver = ContextResolver()
        self.semantic_parser = SemanticParser()
        self.classifier = QueryClassifier()
        self.ambiguity_detector = AmbiguityDetector()

        # In-memory storage for active chat sessions
        self._sessions: Dict[str, ConversationSession] = {}

    def process_query(
        self,
        raw_query: str,
        image_inputs: Optional[List[Any]] = None,
        project_context_dict: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None
    ) -> StructuredQueryObject:
        """
        Processes a single user query within a project & chat context session.

        Args:
            raw_query: Plain text natural language query from user.
            image_inputs: List of dicts or ImageMetadata objects attached to query/turn.
            project_context_dict: Dict with project_id, user_id, user_role, etc.
            session_id: Unique chat conversation session ID. If None, auto-generated.

        Returns:
            StructuredQueryObject: Unified JSON-serializable query object with full audit trace.
        """
        query_id = f"qo_{uuid.uuid4().hex[:8]}"
        session_id = session_id or f"sess_{uuid.uuid4().hex[:8]}"

        # 1. Parse Project Context
        project_context = self.preprocessor.parse_project_context(project_context_dict)

        # 2. Retrieve or initialize Session
        if session_id not in self._sessions:
            self._sessions[session_id] = ConversationSession(
                session_id=session_id,
                project_context=project_context,
                active_images=[],
                turns=[]
            )
        session = self._sessions[session_id]

        execution_trace: List[ExecutionTraceStep] = []

        # 3. Preprocess Query & Images
        orig_query, cleaned_query, pre_trace = self.preprocessor.preprocess_query(raw_query)
        execution_trace.append(pre_trace)

        parsed_images, img_traces = self.preprocessor.validate_and_parse_images(image_inputs or [])
        execution_trace.extend(img_traces)

        # 4. Context Resolution & Coreference Resolution
        (
            effective_images,
            effective_query,
            coref_resolved,
            inherited_turn,
            ctx_trace
        ) = self.context_resolver.resolve_context(
            cleaned_query, parsed_images, session, project_context
        )
        execution_trace.append(ctx_trace)

        # 5. Temporal & Modality Pairing Analysis
        (
            temporal_structure,
            detected_modalities,
            formats,
            sensors
        ) = self.context_resolver.determine_temporal_and_modal_structure(effective_images)

        # 6. Semantic Parsing
        entities, sem_trace = self.semantic_parser.parse(
            effective_query, coref_resolved, inherited_turn
        )
        execution_trace.append(sem_trace)

        # 7. Task Classification
        task_classification, routing_metadata, cls_trace = self.classifier.classify(
            effective_query, entities, temporal_structure, detected_modalities
        )
        execution_trace.append(cls_trace)

        # 8. Ambiguity & Compatibility Detection
        input_compatibility, ambiguity_report, amb_trace = self.ambiguity_detector.validate(
            task_classification.primary_task,
            effective_images,
            entities,
            temporal_structure,
            detected_modalities,
            formats,
            sensors,
            project_context
        )
        execution_trace.append(amb_trace)

        # 9. Construct Structured Query Object (SQO)
        sqo = StructuredQueryObject(
            query_id=query_id,
            session_id=session_id,
            original_query=orig_query,
            effective_resolved_query=effective_query,
            project_context=project_context,
            task_classification=task_classification,
            extracted_entities=entities,
            input_compatibility=input_compatibility,
            ambiguity_report=ambiguity_report,
            routing_metadata=routing_metadata,
            image_inputs=[img.to_dict() for img in effective_images],
            execution_trace=execution_trace
        )

        # 10. Update Conversation Session History
        turn_index = len(session.turns) + 1
        turn = ConversationTurn(
            turn_index=turn_index,
            raw_query=orig_query,
            sqo=sqo
        )
        session.turns.append(turn)

        return sqo

    def get_session(self, session_id: str) -> Optional[ConversationSession]:
        """Returns session history by ID."""
        return self._sessions.get(session_id)
