"""
Local Registry Cache for SatQuery AI Router.
Implements thread-safe, TTL-based caching of active agents and their capabilities/modalities.
Minimizes database access during high-frequency routing requests while keeping metadata fresh.
"""

import time
import threading
from typing import List, Optional, Dict, Any
from schemas.agent_schema import AgentResponse, AgentCapability, InputModality, AgentStatus
from client.registry_client import AgentRegistryClient


class RegistryCacheItem:
    def __init__(self, data: List[AgentResponse], ttl_seconds: float):
        self.data = data
        self.expires_at = time.time() + ttl_seconds

    @property
    def is_expired(self) -> bool:
        return time.time() > self.expires_at


class LocalRegistryCache:
    def __init__(self, client: AgentRegistryClient, default_ttl_seconds: float = 30.0):
        self.client = client
        self.default_ttl = default_ttl_seconds
        self._cache: Dict[str, RegistryCacheItem] = {}
        self._lock = threading.Lock()

    def _cache_key(
        self,
        capability: Optional[AgentCapability],
        modality: Optional[InputModality],
        status: Optional[AgentStatus]
    ) -> str:
        c = capability.value if capability else "all"
        m = modality.value if modality else "all"
        s = status.value if status else "all"
        return f"cap:{c}|mod:{m}|stat:{s}"

    def get_candidates(
        self,
        capability: Optional[AgentCapability] = None,
        modality: Optional[InputModality] = None,
        status: Optional[AgentStatus] = AgentStatus.ACTIVE,
        force_refresh: bool = False
    ) -> List[AgentResponse]:
        """
        Retrieves matching agents from cache or queries the registry client.
        """
        key = self._cache_key(capability, modality, status)

        with self._lock:
            if not force_refresh and key in self._cache:
                item = self._cache[key]
                if not item.is_expired:
                    return item.data

        # Cache miss or expired: fetch fresh from Registry Client
        agents = self.client.discover_agents(
            capability=capability,
            modality=modality,
            status=status
        )

        with self._lock:
            self._cache[key] = RegistryCacheItem(agents, self.default_ttl)

        return agents

    def invalidate(self):
        """Clears all cached registry data."""
        with self._lock:
            self._cache.clear()
