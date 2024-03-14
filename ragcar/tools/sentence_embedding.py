import logging
from typing import Optional, Union, List

import torch
import numpy as np
import asyncio

from ragcar.tools.utils.base import RagcarFactoryBase, RagcarSimpleBase, RagcarAsyncSimpleBase
from ragcar.tools.utils.model_config import download_or_load, get_openai_config, get_clova_config


logger = logging.getLogger(__name__)


class RagcarSentenceFactory(RagcarFactoryBase):
    """
    Sentence embeddings to represent text as dense vectors.
    
    Examples:
        >>> se = Ragcar(tool="sentence_embedding", src="gdrive")
        >>> se("'Ragcar'는 'Retrieval-Augmented Generative Companion for Advanced Research'의 준말입니다.")
        array([-1.25807121e-01, -3.77116412e-01,  7.40933239e-01, -1.15971670e-01, 
        -6.07750058e-01,  8.26456621e-02, -4.65455428e-02, -1.41872734e-01, 
        -2.74574943e-02, -2.61776805e-01,  4.37410325e-01, -6.18349612e-01, 
        ...
        -9.96537209e-02,  3.22387397e-01,  2.22027674e-01, -3.13533843e-01], 
        dtype=float32)  # (1, hidden dim)
        
        >>> se(
        ... "'Ragcar'는 'Retrieval-Augmented Generative Companion for Advanced Research'의 준말입니다.",
        ... convert_to_tensor=True
        ... )
        tensor([-1.2581e-01, -3.7712e-01,  7.4093e-01, -1.1597e-01, -6.0775e-01, 
        8.2646e-02, -4.6546e-02, -1.4187e-01, -2.7457e-02, -2.6178e-01, 
        4.3741e-01, -6.1835e-01,  3.4471e-01,  4.4601e-02, -6.4906e-01, 
        ...
        2.8595e-01,  2.9512e-01, -5.6431e-01,  4.9555e-01, -9.9654e-02, 
        3.2239e-01,  2.2203e-01, -3.1353e-01], device='cuda:0')
    """
    
    def __init__(
        self, 
        tool: str, 
        src: str, 
        model: Optional[Union[str, dict]],
        device: Optional[str] = None,
        use_async: bool = False
    ):
        super().__init__(tool, src, model, device)
        self.use_async = use_async
    
    @staticmethod
    def get_available_srcs():
        return [
            "model_name_or_path",
            "googledrive",
            "openai",
            "clova",
        ]
    
    @staticmethod
    def get_available_models():
        return {
            "model_name_or_path": [
                "leewaay/kpf-bert-base-klueNLI-klueSTS-MSL512",
                "leewaay/klue-roberta-base-klueNLI-klueSTS-MSL512",
                "leewaay/klue-roberta-large-klueNLI-klueSTS-MSL512",
                "MODELS_SUPPORTED(https://huggingface.co/models?pipeline_tag=sentence-similarity)"
            ],
            "googledrive": [
                "YOUR_MODEL"
            ],
            "openai": [
                "text-embedding-3-large",
                "text-embedding-3-small",
                "text-embedding-ada-002",
                "MODELS_SUPPORTED(https://platform.openai.com/docs/models)"
            ],
            "clova": [
                "YOUR_MODEL(https://www.ncloud.com/product/aiService/clovaStudio)"
            ]
        }
    
    def load(self, device: str):
        """
        Loads and initializes the model based on configuration settings.

        Args:
            device (str): The computing device ('cpu' or 'cuda') where the model will be loaded.

        Returns:
            object: The initialized model ready for use.
        """
        if self.config.src == "clova":
            from ragcar.models.clova import ClovaEmbedding
            
            api_conf = get_clova_config(self.config.model_info)
            
            model = ClovaEmbedding(
                api_conf.model_n, 
                api_conf.api_key, 
                api_conf.app_key
            )
            
            if self.use_async:
                return RagcarAsyncGenerativeAISentVec(self.config, model)
            else:
                return RagcarGenerativeAISentVec(self.config, model)
        
        if self.config.src == "openai":
            from ragcar.models.openai import OpenaiEmbedding
            
            api_conf = get_openai_config(self.config.model_info)
            
            model = OpenaiEmbedding(
                api_conf.model_n,
                api_conf.api_key
            )
            
            if self.use_async:
                return RagcarAsyncGenerativeAISentVec(self.config, model)
            else:
                return RagcarGenerativeAISentVec(self.config, model)
        
        from sentence_transformers import SentenceTransformer
        
        model_conf = download_or_load(
            self.config.src, 
            self.config.model_info
        )
        
        model = SentenceTransformer(model_conf.model_n, device)
        
        return RagcarSentenceTransformersSentVec(self.config, model)


class RagcarSentenceTransformersSentVec(RagcarSimpleBase):
    
    def __init__(self, config, model):
        super().__init__(config)
        self._model = model
    
    def predict(
        self, 
        sent: Union[str, List[str]], 
        **kwargs
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Generates sentence embeddings using the Sentence Transformer model.

        Args:
            sent (Union[str, List[str]]): The sentence or list of sentences to be embedded.
            batch_size (int): Batch size for processing multiple sentences.
            convert_to_tensor (bool): If True, converts the embeddings to PyTorch tensors.

        Returns:
            Union[np.ndarray, torch.Tensor]: The embedding(s) of the input sentence(s) as NumPy arrays or PyTorch tensors.
        """
        batch_size = kwargs.get("batch_size", 32)
        convert_to_tensor = kwargs.get("convert_to_tensor", False)
        
        outputs = self._model.encode(sent, batch_size=batch_size, convert_to_tensor=convert_to_tensor)
        
        return outputs


class RagcarGenerativeAISentVec(RagcarSimpleBase):
    
    def __init__(self, config, model):
        super().__init__(config)
        self._model = model
    
    def predict(
        self, 
        sent: Union[str, List[str]], 
        **kwargs
    ) -> Union[List[float], torch.Tensor]:
        """
        Generates embeddings using a generative AI model for sentences.

        Args:
            sent (Union[str, List[str]]): The sentence or list of sentences for embedding.
            **kwargs: Additional keyword arguments including:
                - batch_size (int): Batch size for processing multiple sentences.
                - convert_to_tensor (bool): If True, converts the embeddings to PyTorch tensors.

        Returns:
            Union[List[float], torch.Tensor]: The embedding(s) of the input sentence(s), either as lists of floats or PyTorch tensors.
        """
        batch_size = kwargs.get("batch_size", 32)
        convert_to_tensor = kwargs.get("convert_to_tensor", False)
        
        if isinstance(sent, str):
            outputs = self._model.encode(sent)
            if convert_to_tensor:
                outputs = torch.tensor(self._model.encode(sent))
        
            return outputs
        
        if isinstance(sent, list):
            outputs = []
            for i in range(0, len(sent), batch_size):
                batch = sent[i:i + batch_size]
                if convert_to_tensor:
                    results = [torch.tensor(self._model.encode(s)) for s in batch]
                else:
                    results = [self._model.encode(s) for s in batch]
                outputs.extend(results)
        
            if convert_to_tensor:
                return torch.stack(outputs)
            else:
                return outputs
        

class RagcarAsyncGenerativeAISentVec(RagcarAsyncSimpleBase):
    
    def __init__(self, config, model):
        super().__init__(config)
        self._model = model
    
    async def predict(
        self, 
        sent: Union[str, List[str]], 
        **kwargs
    ) -> Union[List[float], torch.Tensor]:
        """
        Asynchronously generates embeddings using a generative AI model for sentences.

        Args:
            sent (Union[str, List[str]]): The sentence or list of sentences for embedding.
            **kwargs: Additional keyword arguments including:
                - batch_size (int): Batch size for processing multiple sentences.
                - convert_to_tensor (bool): If True, converts the embeddings to PyTorch tensors.

        Returns:
            Union[List[float], torch.Tensor]: The embedding(s) of the input sentence(s), either as lists of floats or PyTorch tensors, generated asynchronously.
        """
        batch_size = kwargs.get("batch_size", 32)
        convert_to_tensor = kwargs.get("convert_to_tensor", False)
        
        if isinstance(sent, str):
            outputs = await self._model.aencode(sent)
            if convert_to_tensor:
                outputs = torch.tensor(self._model.aencode(sent))
                
            return outputs
        
        if isinstance(sent, list):
            outputs = []
            for i in range(0, len(sent), batch_size):
                batch = sent[i:i + batch_size]
                tasks = [self._model.aencode(s) for s in batch]
                results = await asyncio.gather(*tasks)
                if convert_to_tensor:
                    outputs.extend([torch.tensor(r) for r in results])
                else:
                    outputs.extend(results)
        
            if convert_to_tensor:
                return torch.stack(outputs)
            else:
                return outputs