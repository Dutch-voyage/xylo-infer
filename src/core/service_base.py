import asyncio
import functools
import time
from typing import Any, Dict, List, Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.core.artifact_base import Artifact


class BaseService(ABC):
    """Base class for unified services"""

    def __init__(self):
        self.event_loop = asyncio.get_event_loop()
    
    async def _wrap_as_async(self, func, *args, **kwargs):
        func_partial = functools.partial(func, *args, **kwargs)
        return await self.event_loop.run_in_executor(None, func_partial) 
    
    # @abstractmethod
    # def register_artifact(self, artifact: 'Artifact'):
    #     """Register an artifact with the service"""
        
    # @abstractmethod
    # def register_artifact_resource(self, resource: Any):
    #     """Register a stateful resource for the artifact"""
    
    @abstractmethod
    async def _main_event_loop(self, artifact_name: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a registered artifact by name with given input data"""        
        
