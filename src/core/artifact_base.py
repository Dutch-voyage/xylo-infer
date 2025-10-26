"""
Base interfaces for artifacts

Provides the fundamental building blocks for systematic implementations
that can be registered, called, and composed within the unified service architecture.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
import asyncio
import time
from pathlib import Path
import importlib.util
import inspect
import uuid

@dataclass
class ExecutionContext:
    """Context passed to artifacts during execution"""
    artifact_name: str
    metadata: Dict[str, Any]
    request_id: Optional[str] = None


class Artifact(ABC):
    """Base interface for all artifacts - single unified API"""
    
    # """TODO let's see if there are better abstraction for the 'runnning' of artifacts"""
    # @abstractmethod
    # def execute(self, input_data: Dict[str, Any], context: ExecutionContext) -> Dict[str, Any]:
    #     """
    #     Single unified API for artifact execution
        
    #     Args:
    #         input_data: Dictionary containing all inputs
    #         context: Execution context with mode and metadata
            
    #     Returns:
    #         Dictionary with execution results
    #     """
    
    def __init__(self):
        super().__init__()
        self.registered_methods_to = {}
        self.registered_objs_to = {}
    
    @property
    def path(self):
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        pass
    
    def _register_method(self, obj_name: str, service):
        if not hasattr(self, obj_name):
            raise ValueError("No such method in the artifact to register.")
        
        if obj_name not in self.registered_methods_to.keys():
            self.registered_methods_to[obj_name] = []
        self.registered_methods_to[obj_name].append(service.name)
        
        if self.name not in service.registered_methods_by.keys():
            service.registered_methods_by[self.name] = []
        
        service.registered_methods_by[self.name].append(obj_name)
        
        original_method = getattr(self.__class__, obj_name)
        
        self_artifact = self
        self_service = service
        
        # Create a wrapper that replaces self references with artifact references
        def create_method_wrapper(original_func):
            
            def method_wrapper(dummy_self, *args, **kwargs):
                # Create a proxy object that routes attribute access
                class ArtifactProxy:
                    def __init__(self):
                        self._artifact_name = self_artifact.name
                    
                    def __getattr__(self, name):
                        # First check if it's a registered method in the service
                        if name in self_artifact.registered_methods_to.keys() or name in self_artifact.registered_objs_to.keys():
                            # print(f"[Proxy] Accessing registered attribute '{name}': {hasattr(self_service, name)}")
                            if hasattr(self_service, name):
                                return getattr(self_service, name)
                            
                            # print(name, hasattr(self_artifact, name))
                            # Then check if it's available in the artifact instance
                            raise AttributeError(f"'{type(self_service).__name__}' object has no attribute '{name}'")
                        else:
                            if hasattr(self_service, name):
                                return getattr(self_service, name)
                            elif hasattr(self_artifact, name):
                                return getattr(self_artifact, name)
                            else:
                                raise AttributeError(f"'Neither {type(self_service).__name__}' or {type(self_artifact).__name__} object has attribute '{name}'")
                    
                    def __setattr__(self, name, value):
                        if name in self_artifact.registered_methods_to.keys() or name in self_artifact.registered_objs_to.keys():
                            if hasattr(self_service, name):
                                setattr(self_service, name, value)
                                return
                            
                            raise AttributeError(f"'{type(self_service).__name__}' object has no attribute '{name}'")
                        else:
                            setattr(self_artifact, name, value)
                
                # Create the proxy and call the original method
                proxy = ArtifactProxy()

                # Call the original method with the proxy as self
                return original_func(proxy, *args, **kwargs)
            
            return method_wrapper
        
        # Create the wrapped method
        wrapped_method = create_method_wrapper(original_method)
        
        # Bind it to the service
        import types
        setattr(service, obj_name, types.MethodType(wrapped_method, service))
    
    def _register_obj(self, obj_name: str, service):
        if not hasattr(self, obj_name):
            raise ValueError("No such object in the artifact to register.")
        if obj_name not in self.registered_objs_to.keys():
            self.registered_objs_to[obj_name] = []
        self.registered_objs_to[obj_name].append(service.name)
        if self.name not in service.registered_objs_by.keys():
            service.registered_objs_by[self.name] = []
        service.registered_objs_by[self.name].append(obj_name)
        
        setattr(service, obj_name, getattr(self, obj_name))
        
        # self_artifact = self
        # self_service = service
        
        # # # original_obj = getattr(self, obj_name)
        
        # # # Create an ObjectProxy that provides cross-artifact access similar to ArtifactProxy
        # class ObjectProxy:
        #     def __init__(self, artifact_name):
        #         # self._original_obj = original_obj
        #         self._artifact_name = artifact_name
            
        #     def __getattr__(self, name):
        #         # Then check if it's a registered method in the service
        #         if name in self_artifact.registered_objs:
        #             if hasattr(self_service, name):
        #                 return getattr(self_service, name)
                    
        #             raise AttributeError(f"'{type(self_service).__name__}' object has no attribute '{name}'")
        #         else:
        #             return getattr(self_artifact, name)
            
        #     def __setattr__(self, name, value):
        #         if name in self_artifact.registered_objs:
        #             # Set attribute on the original object
        #             if hasattr(self_service, name):
        #                 setattr(self_service, name, value)

        #             raise AttributeError(f"'{type(self_service).__name__}' object has no attribute '{name}'")            
        #         else:
        #             setattr(self_artifact, name, value)
        
        # # # Create the proxy object
        # proxy_obj = ObjectProxy(artifact_name)
        # setattr(self, obj_name, proxy_obj)
        
    # @abstractmethod
    # def get_schema(self) -> Dict[str, Any]:
    #     """Return JSON schema for input validation"""
    #     pass
    
    # @abstractmethod
    # def get_description(self) -> str:
    #     """Human-readable description"""
    #     return self.__class__.__doc__ or "No description provided"
    
    # @abstractmethod
    # def validate_input(self, input_data: Dict[str, Any]) -> bool:
    #     """Validate input against schema (optional override)"""
    #     return True  # Basic validation - can be overridden
    
class ArtifactRegistry:
    pass