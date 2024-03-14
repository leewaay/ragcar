from typing import Optional, Union, List

import torch

from ragcar.tools.utils.base import RagcarFactoryBase, RagcarSimpleBase, RagcarAsyncSimpleBase
from ragcar.tools.utils.model_config import get_clova_config


class RagcarSegmentationFactory(RagcarFactoryBase):
    """
    Splits text into meaningful segments using CLOVA Studio.
    
    Examples:
        >>> ts = Ragcar(tool="text_segmentation", src="clova", model="https://clovastudio.apigw.ntruss.com/testapp/v1/api-tools/segmentation/{appId}")
        >>> ts("'Ragcar'는 'Retrieval-Augmented Generative Companion for Advanced Research'의 준말입니다.")
        [["'Ragcar'는 'Retrieval-Augmented Generative Companion for Advanced Research'의 준말입니다."]]
    """
    
    def __init__(
        self, 
        tool: str, 
        src: str, 
        model: Optional[Union[str, dict]],
        alpha: int = -100,
        seg_cnt: int = -1,
        post_process: bool = True,
        min_size: int = 300,
        max_size: int = 1000,
        use_async: bool = False
    ):
        super().__init__(tool, src, model)
        self.alpha = alpha
        self.seg_cnt = seg_cnt
        self.post_process = post_process
        self.min_size = min_size
        self.max_size = max_size
        self.use_async = use_async
    
    @staticmethod
    def get_available_srcs():
        return [
            "clova",
        ]
    
    @staticmethod
    def get_available_models():
        return {
            "clova": [
                "YOUR_MODEL(https://www.ncloud.com/product/aiService/clovaStudio)"
            ]
        }
    
    def load(self):
        """
        Loads and initializes the model based on configuration settings.

        Returns:
            object: The initialized model ready for use.
        """
        params = {
            "alpha": self.alpha,
            "seg_cnt": self.seg_cnt,
            "post_process": self.post_process,
            "min_size": self.min_size,
            "max_size": self.max_size
        }
        
        if self.config.src == "clova":
            from ragcar.models.clova import ClovaSegmentation
            
            api_conf = get_clova_config(self.config.model_info)
            
            model = ClovaSegmentation(
                api_conf.model_n, 
                api_conf.api_key, 
                api_conf.app_key
            )
            
            if self.use_async:
                return RagcarAsyncGenerativeAISegmentation(self.config, model, params)
            else:
                return RagcarGenerativeAISegmentation(self.config, model, params)


class RagcarGenerativeAISegmentation(RagcarSimpleBase):
    
    def __init__(self, config, model, params):
        super().__init__(config)
        self._model = model
        self._params = params
    
    def predict(
        self, 
        sent: Union[str, List[str]], 
        **kwargs
    ) -> List[List[str]]:
        """
        Segments text into meaningful units.

        Args:
            sent (str or List[str]): The input text or list of texts to be segmented.
            **kwargs: Additional keyword arguments for segmentation parameters.

        Returns:
            List[List[str]]: A list of segments for each input text.
        """
        outputs = self._model.segment(sent, **self._params)
        return outputs
        

class RagcarAsyncGenerativeAISegmentation(RagcarAsyncSimpleBase):
    
    def __init__(self, config, model, params):
        super().__init__(config)
        self._model = model
        self._params = params
    
    async def predict(
        self, 
        sent: Union[str, List[str]], 
        **kwargs
    ) -> Union[List[float], torch.Tensor]:
        """
        Asynchronously segments text into meaningful units.

        Args:
            sent (str or List[str]): The input text or list of texts to be segmented.
            **kwargs: Additional keyword arguments for segmentation parameters.

        Returns:
            Union[List[float], torch.Tensor]: A list of segments for each input text, or a tensor if specified.
        """
        outputs = await self._model.asegment(sent, **self._params)
        return outputs