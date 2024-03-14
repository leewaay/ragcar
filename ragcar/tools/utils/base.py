import re
import unicodedata
from abc import abstractmethod
from dataclasses import dataclass
from typing import Optional, Union, Mapping, List

import torch


@dataclass
class ToolConfig:
    tool: str
    src: str
    model_info: Union[str, dict]
    device: Optional[str]


class RagcarToolBase:
    r"""Tool base class that implements basic functions for prediction"""

    def __init__(self, config: ToolConfig):
        self.config = config

    @property
    def model_info(self):
        return self.config.model_info

    @property
    def src(self):
        return self.config.src

    @abstractmethod
    def predict(
        self,
        text: Union[str, List[str]],
        **kwargs,
    ):
        raise NotImplementedError(
            "`predict()` function is not implemented properly!")

    def __call__(self):
        raise NotImplementedError(
            "`call()` function is not implemented properly!")

    def __repr__(self):
        return f"[TOOL]: {self.config.tool.upper()}\n[TOOL]: {self.config.tool.upper()}\n[MODEL]: {self.config.model_info}"
    
    def _normalize(self, text: str):
        """Unicode normalization and whitespace removal (often needed for contexts)"""
        text = unicodedata.normalize("NFKC", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text


class RagcarFactoryBase(object):
    r"""This is a factory base class that construct tool-specific module"""
    
    def __init__(
        self,
        tool: str,
        src: str,
        model: Optional[Union[str, dict]] = None,
        device: Optional[str] = None,
    ):
        self._available_srcs = self.get_available_srcs()
        self._available_models = self.get_available_models()
        self._model2src = {
            m: s for s, ms in self._available_models.items() for m in ms
        }
        
        # Set default src as very first supported src
        assert (
            src in self._available_srcs
        ), f"Following src are supported for this tool: {self._available_srcs}"
        
        if src is None:
            src = self._available_srcs[0]
            
        # Set default model
        if model is None:
            model = self.get_default_model(src)

        self.config = ToolConfig(tool, src, model, device)
            
    @abstractmethod
    def get_available_srcs(self) -> List[str]:
        raise NotImplementedError(
            "`get_available_tools()` is not implemented properly!")

    @abstractmethod
    def get_available_models(self) -> Mapping[str, List[str]]:
        raise NotImplementedError(
            "`get_available_models()` is not implemented properly!")
    
    @abstractmethod
    def get_default_model(self, src: str) -> str:
        return self._available_models[src][0]
    
    @classmethod
    def load(cls) -> RagcarToolBase:
        raise NotImplementedError(
            "Model load function is not implemented properly!")


class RagcarSimpleBase(RagcarToolBase):
    r"""Simple tool base wrapper class"""

    def __call__(self, text: Union[str, List[str]], **kwargs):
        return self.predict(text, **kwargs)


class RagcarAsyncSimpleBase(RagcarToolBase):
    r"""Simple tool base wrapper class"""

    async def __call__(self, text: Union[str, List[str]], **kwargs):
        return await self.predict(text, **kwargs)


class RagcarBiencoderBase(RagcarToolBase):
    r"""Bi-Encoder base wrapper class"""

    def __call__(
        self,
        sent_a: Union[str, List[str], torch.Tensor],
        sent_b: Union[str, List[str], torch.Tensor] = None,
        **kwargs,
    ):
        if isinstance(sent_a, str):
            sent_a = self._normalize(sent_a)
        elif isinstance(sent_a, list):
            sent_a = [self._normalize(t) for t in sent_a]
            
        if isinstance(sent_b, str):
            sent_b = self._normalize(sent_b)
        elif isinstance(sent_b, list):
            sent_b = [self._normalize(t) for t in sent_b]
        
        if isinstance(sent_a, torch.Tensor) and isinstance(sent_b, torch.Tensor):
            assert sent_a.shape[-1] == sent_b.shape[-1], "The last dimension of the tensors must be the same."

        return self.predict(sent_a, sent_b, **kwargs)


class RagcarAsyncBiencoderBase(RagcarToolBase):
    r"""Async Bi-Encoder base wrapper class"""

    async def __call__(
        self,
        sent_a: Union[str, List[str], torch.Tensor],
        sent_b: Union[str, List[str], torch.Tensor] = None,
        **kwargs,
    ):
        if isinstance(sent_a, str):
            sent_a = self._normalize(sent_a)
        elif isinstance(sent_a, list):
            sent_a = [self._normalize(t) for t in sent_a]
            
        if isinstance(sent_b, str):
            sent_b = self._normalize(sent_b)
        elif isinstance(sent_b, list):
            sent_b = [self._normalize(t) for t in sent_b]
        
        if isinstance(sent_a, torch.Tensor) and isinstance(sent_b, torch.Tensor):
            assert sent_a.shape[-1] == sent_b.shape[-1], "The last dimension of the tensors must be the same."

        return await self.predict(sent_a, sent_b, **kwargs)


class RagcarGenerationBase(RagcarToolBase):
    r"""Simple tool base wrapper class"""

    def __call__(self, **kwargs):
        return self.predict(**kwargs)


class RagcarAsyncGenerationBase(RagcarToolBase):
    r"""Simple tool base wrapper class"""

    async def __call__(self, **kwargs):
        return await self.predict(**kwargs)