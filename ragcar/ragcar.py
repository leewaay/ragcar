"""
Ragcar tool-specific factory class
"""

from typing import Optional

import torch

from ragcar.tools.utils.base import RagcarToolBase
from ragcar.tools.utils import (
    TransformerConfig, 
    GoogledriveInfo, 
    OpenaiConfig,
    ClovastudioConfig,
    EncoderInfo,
    ElasticsearchConfig
)
from ragcar.tools import(
    RagcarTokenizationFactory,
    RagcarSentenceFactory,
    RagcarStsFactory,
    RagcarSemanticSearchFactory,
    RagcarTextGenerationFactory,
    RagcarSegmentationFactory
)


SUPPORTED_TOOLS = {
    "tokenization": RagcarTokenizationFactory,
    "sentence_embedding": RagcarSentenceFactory,
    "sentence_similarity": RagcarStsFactory,
    "semantic_search": RagcarSemanticSearchFactory,
    "text_generation": RagcarTextGenerationFactory,
    "text_segmentation": RagcarSegmentationFactory,
}

TOOLS_WITHOUT_DEVICE = [
    "tokenization",
    "text_generation",
    "text_segmentation"
]

SRC_ALIASES = {
    "구글드라이브": "googledrive",
    "구글": "googledrive",
    "gdrive": "googledrive",
    "google": "googledrive",
    "gpt": "openai",
    "clovastudio": "clova",
    "hyperclova": "clova",
    "hc": "clova",
    "hcx": "clova",
    "엘라스틱서치": "elasticsearch",
    "es": "elasticsearch",
    "허깅페이스": "huggingface",
    "hf": "huggingface",
    "tt": "tiktoken",
    "키위": "kiwi",
}

CUSTOMIZABLE_SRC = {
    "model_name_or_path": TransformerConfig,
    "googledrive": GoogledriveInfo,
    "openai": OpenaiConfig,
    "clova": ClovastudioConfig,
    "elasticsearch": ElasticsearchConfig,
}


class Ragcar:
    r"""
    This is a generic class that will return one of the tool-specific model classes of the library
    when created with the `__new__()` method
    """
    
    def __new__(
        cls,
        tool: str,
        src: str = "clova",
        model: Optional[str] = None,
        **kwargs,
    ) -> RagcarToolBase:
        
        if tool not in SUPPORTED_TOOLS:
            raise KeyError("Unknown tool {}, available tools are {}".format(
                tool,
                list(SUPPORTED_TOOLS.keys()),
            ))

        src = src.lower()
        src = SRC_ALIASES[src] if src in SRC_ALIASES else src

        if tool in TOOLS_WITHOUT_DEVICE:
            tool_module = SUPPORTED_TOOLS[tool](
                tool,
                src,
                model,
                **kwargs,
            ).load()
        else:
            # Get device information from torch API
            if kwargs.get("device"):
                device = kwargs.get("device")
            else:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Instantiate tool-specific pipeline module, if possible
            tool_module = SUPPORTED_TOOLS[tool](
                tool,
                src,
                model,
                **kwargs,
            ).load(device)

        return tool_module

    @staticmethod
    def available_tools() -> str:
        """
        Returns a string listing the available tools in the Ragcar project.
        
        Returns:
            str: A message stating the available tools.
        """
        return "Available tools are {}".format(list(SUPPORTED_TOOLS.keys()))

    @staticmethod
    def available_models(tool: str) -> str:
        """
        Returns a string listing the names of available models for a specified tool.
        
        Args:
            tool (str): The name of the tool for which available models are requested.
        
        Returns:
            str: A string that provides the available models for the specified tool.
        
        Raises:
            KeyError: If the specified tool is not supported.
        
        Examples:
            >>> Ragcar.available_models("text_generation")
            "Available models for text_generation are ([src]: openai, [model]: gpt-3.5-turbo, MODELS_SUPPORTED(https://platform.openai.com/docs/models)), 
            ([src]: clova, [model]: HCX-002, MODELS_SUPPORTED(https://www.ncloud.com/product/aiService/clovaStudio))"
        """
        if tool not in SUPPORTED_TOOLS:
            raise KeyError(
                "Unknown tool {} ! Please check available models via `available_tools()`"
                .format(tool))

        srcs = SUPPORTED_TOOLS[tool].get_available_models()
        output = f"Available models for {tool} are "
        for src in srcs:
            srcs[src] = list(map(lambda x: 'None' if x is None else x, srcs[src]))
            output += f"([src]: {src}, [model]: {', '.join(srcs[src])}), "
        
        return output[:-2]
    
    @staticmethod
    def available_customizable_src(tool: str) -> str:
        """
        Returns a string that lists customizable sources for a specified tool in the Ragcar project.
        
        Args:
            tool (str): The name of the tool for which customizable sources are requested.
        
        Returns:
            str: A message that lists the available customizable sources for the specified tool.
        
        Raises:
            KeyError: If the specified tool is not supported or does not exist in the `SUPPORTED_TOOLS` dictionary.
        
        Examples:
            >>> Ragcar.available_customizable_src("tokenization")
            "Available customizable src for sentence_embedding are ['googledrive', 'clovax', 'openai']"
        """
        if tool not in SUPPORTED_TOOLS:
            raise KeyError(
                "Unknown tool {} ! Please check available tools via `available_tools()`"
                .format(tool))

        if tool == "tokenization":
            customizable_srcs = ["null"]
        else:
            tool_module = SUPPORTED_TOOLS[tool]
            available_srcs = tool_module.get_available_models().keys()
            customizable_srcs = set(available_srcs).intersection(CUSTOMIZABLE_SRC.keys())

        return "Available customizable src for {} are {}".format(tool, list(customizable_srcs))
    
    @staticmethod
    def available_model_fields(src: str) -> str:
        """
        Returns a string detailing the available fields and their data types for a specified customizable source (src).
        
        Args:
            src (str): The name of the customizable source.
        
        Returns:
            str: A string formatted to list each field and its corresponding data type for the specified source,
                including nested fields for complex types.
        
        Raises:
            KeyError: If the specified source is not supported or does not exist in the `CUSTOMIZABLE_SRC` dictionary.
        
        Examples:
            >>> Ragcar.available_model_fields("clova")
            'Available fields for clova are ([field]: api_key, [type]: str), ([field]: app_key, [type]: str), ([field]: model_n, [type]: str)'
        """
        if src not in CUSTOMIZABLE_SRC:
            raise KeyError(
                "Unknown customizable src {} ! Please check available customizable src via `available_customizable_src()`"
                .format(src))

        dataclass = CUSTOMIZABLE_SRC[src]
        output = f"Available fields for {src} are "
        for field, field_type in dataclass.__annotations__.items():
            if field_type == EncoderInfo:
                encoder_fields = EncoderInfo.__annotations__
                for encoder_field, encoder_field_type in encoder_fields.items():
                    output += f"([field]: {field}.{encoder_field}, [type]: {encoder_field_type.__name__}), "
            else:
                if hasattr(field_type, '__args__'):
                    type_names = ", ".join([t.__name__ for t in field_type.__args__])
                else:
                    type_names = field_type.__name__
                output += f"([field]: {field}, [type]: {type_names}), "

        return output[:-2]