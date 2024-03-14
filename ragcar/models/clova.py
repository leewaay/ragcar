import logging
import json
import time
from typing import Optional, Union, Dict, List, Generator, AsyncGenerator

from ragcar.models.base import ClovaBase


logger = logging.getLogger(__name__)


class ClovaSegmentation(ClovaBase):
    """Class for dividing text into separate paragraphs with Clova's model"""
    
    def __init__(self, api_url: str, api_key: str, app_key: str):
        super().__init__(api_url, api_key, app_key)
    
    def segment(
        self, 
        text: str,
        alpha: int = -100,
        seg_cnt: int = -1,
        post_process: bool = True,
        min_size: int = 300,
        max_size: int = 1000
    ) -> List[List[str]]:
        """
        Segments the given text into meaningful units.

        Args:
            text (str): The input text to be segmented.
            alpha (int, optional): Thresholds value for paragraph segmentation. Higher values increase the number of paragraphs created. Range is -1.5 to 1.5, 
                with -100 indicating that the model will automatically determine the optimal value for paragraph segmentation. Defaults to -100.
            seg_cnt (int, optional): The desired number of segments. Defaults to -1, which means automatic determination.
            post_process (bool, optional): Whether to post-process the segments for better quality. Defaults to True.
            min_size (int, optional): Minimum size of a post-processed segment. Applicable only if post_process is True. Defaults to 300.
            max_size (int, optional): Maximum size of a post-processed segment. Applicable only if post_process is True. Defaults to 1000.

        Returns:
            List[List[str]]: A list of segmented text units, each represented as a list of strings.
        """
        res, _ = self.fetch(
            text = text, 
            alpha = alpha,
            segCnt = seg_cnt,
            postProcess = post_process, 
            postProcessMinSize = min_size, 
            postProcessMaxSize = max_size
        )
        
        return res['result']['topicSeg']
    
    async def asegment(
        self, 
        text: str,
        alpha: int = -100,
        seg_cnt: int = -1,
        post_process: bool = True,
        min_size: int = 300,
        max_size: int = 1000
    ) -> List[List[str]]:
        """
        Asynchronously segments the given text into meaningful units.

        Args:
            text (str): The input text to be segmented.
            alpha (int, optional): Thresholds value for paragraph segmentation. Higher values increase the number of paragraphs created. Range is -1.5 to 1.5, 
                with -100 indicating that the model will automatically determine the optimal value for paragraph segmentation. Defaults to -100.
            seg_cnt (int, optional): The desired number of segments. Defaults to -1, which means automatic determination.
            post_process (bool, optional): Whether to post-process the segments for better quality. Defaults to True.
            min_size (int, optional): Minimum size of a post-processed segment. Applicable only if post_process is True. Defaults to 300.
            max_size (int, optional): Maximum size of a post-processed segment. Applicable only if post_process is True. Defaults to 1000.

        Returns:
            List[List[str]]: A list of segmented text units, each represented as a list of strings.
        """
        res, _ = await self.afetch(
            text = text, 
            alpha = alpha,
            segCnt = seg_cnt,
            postProcess = post_process, 
            postProcessMinSize = min_size, 
            postProcessMaxSize = max_size
        )
        
        return res['result']['topicSeg']


class ClovaChatTokenization(ClovaBase):
    """Class for calculating the number of tokens in text using HyperCLVOA X model"""
    
    def __init__(self, api_url: str, api_key: str, app_key: str):
        super().__init__(api_url, api_key, app_key)
    
    def tokenize(self, text: str) -> int:
        """
        Calculates the number of tokens in the given text using the specified model.

        Args:
            text (str): The input text for which the number of tokens is to be calculated.

        Returns:
            int: The number of tokens in the input text.
        """
        res, _ = self.fetch(messages=[
            {
                "role": "system",
                "content": text
            }
        ])
        
        return res['result']['messages'][0]['count']
    
    async def atokenize(self, text: str) -> int:
        """
        Asynchronously calculates the number of tokens in the given text using the specified model.

        Args:
            text (str): The input text for which the number of tokens is to be calculated.

        Returns:
            int: The number of tokens in the input text.
        """
        res, _ = await self.afetch(messages=[
            {
                "role": "system",
                "content": text
            }
        ])
        
        return res['result']['messages'][0]['count']


class ClovaTokenization(ClovaBase):
    """Class for calculating the number of tokens in text using HyperCLVOA model"""
    
    def __init__(self, api_url: str, api_key: str, app_key: str):
        super().__init__(api_url, api_key, app_key)
    
    def tokenize(self, text: str) -> int:
        """
        Calculates the number of tokens in the given text using the specified model.

        Args:
            text (str): The input text for which the number of tokens is to be calculated.

        Returns:
            int: The number of tokens in the input text.
        """
        res, _ = self.fetch(text=text)
        
        return res['result']['numTokens']
    
    async def atokenize(self, text: str) -> int:
        """
        Asynchronously calculates the number of tokens in the given text using the specified model.

        Args:
            text (str): The input text for which the number of tokens is to be calculated.

        Returns:
            int: The number of tokens in the input text.
        """
        res, _ = await self.afetch(text=text)
        
        return res['result']['numTokens']


class ClovaEmbedding(ClovaBase):
    """Class for generating text embedding with Clova's model"""
    
    def __init__(self, api_url: str, api_key: str, app_key: str):
        super().__init__(api_url, api_key, app_key)
    
    def encode(self, text: str) -> List[float]:
        """
        Get embedding for a given text.

        Args:
            text (str): Input text for which embedding is to be generated.

        Returns:
            List[float]: The embedding.
        """
        res, _ = self.fetch(text=text)
        
        return res['result']['embedding']

    async def aencode(self, text: str) -> List[float]:
        """
        Asynchronously get embedding for a given text.

        Args:
            text (str): Input text for which embedding is to be generated.

        Returns:
            List[float]: The embedding.
        """
        res, _ = await self.afetch(text=text)
        
        return res['result']['embedding']


class ClovaChatCompletion(ClovaBase):
    """Class for generating text with Clova's Completion model"""

    def __init__(
        self, 
        api_url: str, 
        api_key: str, 
        app_key: str, 
        stream: bool,
        formatting: bool
    ):
        super().__init__(api_url, api_key, app_key, stream)
        
        self.stream = stream
        self.formatting = formatting
        
    def _get_params(
        self, 
        messages: list, 
        max_tokens: int, 
        temperature: float, 
        top_p: float,
        frequency_penalty: float, 
        presence_penalty: float, 
        stop_before: Optional[list] = None
    ) -> Dict[str, Union[str, int, float, list, bool]]:
        params = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": presence_penalty,
            "repeat_penalty": frequency_penalty,
            "stop_before": stop_before if stop_before else [],
        }

        return {k: v for k, v in params.items() if v is not None}
    
    def _output(
        self,
        res,
        request_id,
        start_time
    ) -> Dict[str, Union[str, int, float]]:
        if self.formatting:
            response_time = time.time() - start_time
            formatted_data = self.format_response(res, request_id, response_time)
            
            logger.info(json.dumps(res, ensure_ascii=False, indent=4))
            return formatted_data
        else:
            return res
    
    def _stream_output(
        self,
        res,
        request_id,
        start_time
    ) -> Generator[str, None, None]:
        current_event = {}
        for chunk in res.iter_lines():
            if chunk:
                chunk_str = chunk.decode('utf-8')
                
                if chunk_str.startswith("id:"):
                    current_event['id'] = request_id
                
                elif chunk_str.startswith("event:"):
                    current_event['event'] = chunk_str.split("event:")[1].strip()
                    
                elif chunk_str.startswith("data:"):
                    data_str = chunk_str.split("data:")[1].strip()
                    data_json = json.loads(data_str)
                    current_event['data'] = data_json
                    
                    if self.formatting:
                        logger.info(json.dumps(current_event, ensure_ascii=False, indent=4))
                        
                        if current_event['event'] == "token":
                            content = data_json.get('message', {}).get('content')
                        
                            if content:
                                yield {
                                    "id": request_id,
                                    "event": "chunk",
                                    "content": content
                                }
                                
                        if current_event['event'] == "result":
                            response_time = time.time() - start_time
                            
                            default_format = {"result": current_event['data']}
                            
                            yield {
                                    "id": request_id,
                                    "event": "finish",
                                    "data": self.format_response(default_format, request_id, response_time)
                                }
                    else:
                        yield current_event
                
                    current_event = {}
    
    def create(self, messages: str, **kwargs) -> Union[str, Generator[str, None, None]]:
        params = self._get_params(messages, **kwargs)
        
        start_time = time.time()
        res, request_id = self.fetch(**params)
        
        if self.stream:
            return self._stream_output(res, request_id, start_time)
        
        return self._output(res, request_id, start_time)

    async def _astream_output(
        self, 
        res, 
        request_id, 
        start_time
    ) -> AsyncGenerator[str, None]:
        current_event = {}
        async for chunk in res.iter_lines():
            if chunk:
                chunk_str = chunk.decode('utf-8')
                
                if chunk_str.startswith("id:"):
                    current_event['id'] = request_id
                
                elif chunk_str.startswith("event:"):
                    current_event['event'] = chunk_str.split("event:")[1].strip()
                    
                elif chunk_str.startswith("data:"):
                    data_str = chunk_str.split("data:")[1].strip()
                    data_json = json.loads(data_str)
                    current_event['data'] = data_json
                    
                    if self.formatting:
                        logger.info(json.dumps(current_event, ensure_ascii=False, indent=4))
                        
                        if current_event['event'] == "token":
                            content = data_json.get('message', {}).get('content')
                        
                            if content:
                                yield {
                                    "id": request_id,
                                    "event": "chunk",
                                    "content": content
                                }
                                
                        if current_event['event'] == "result":
                            response_time = time.time() - start_time
                            
                            default_format = {"result": current_event['data']}
                            
                            yield {
                                    "id": request_id,
                                    "event": "finish",
                                    "data": self.format_response(default_format, request_id, response_time)
                                }
                    else:
                        yield current_event
    
    async def acreate(self, messages: str, **kwargs) -> Union[str, AsyncGenerator[str, None]]:
        params = self._get_params(messages, **kwargs)
        
        start_time = time.time()
        res, request_id = await self.afetch(**params)
        
        if params.get('stream'):
            return self._astream_output(res, request_id, start_time)
        
        return self._output(res, request_id, start_time)


class ClovaCompletion(ClovaBase):
    """Class for generating text with Clova's Completion model"""

    def __init__(self, api_url: str, api_key: str, app_key: str):
        super().__init__(api_url, api_key, app_key)

    def _get_params(
        self, 
        prompt: str, 
        max_tokens: int, 
        temperature: float, 
        top_p: float,
        frequency_penalty: float, 
        presence_penalty: float, 
        start: Optional[str] = '',
        restart: Optional[str] = '', 
        stop_before: Optional[list] = None,
        include_tokens: bool = True, 
        include_ai_filters: bool = True,
        include_probs: bool = True
    ) -> Dict[str, Union[str, int, float, list, bool]]:
        params = {
            "text": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": presence_penalty,
            "repeat_penalty": frequency_penalty,
            "start": start,
            "restart": restart,
            "stop_before": stop_before if stop_before else [],
            "include_tokens": include_tokens,
            "include_ai_filters": include_ai_filters,
            "include_probs": include_probs
        }

        return {k: v for k, v in params.items() if v is not None}
    
    def _output(self, res) -> Dict[str, Union[str, int, float]]:
        result = res['response_data'].get('result', {})
        
        res['content'] = result['text'].strip()
            
        res.pop('response_data')
        
        return res
    
    def create(self, prompt: str, **kwargs) -> str:
        params = self._get_params(prompt, **kwargs)
        
        start_time = time.time()
        res, _ = self.fetch(**params)
        response_time = time.time() - start_time
        
        if self.formatting:
            return self._output(self.format_response(res, response_time))
        
        return res

    async def acreate(self, prompt: str, **kwargs) -> str:
        params = self._get_params(prompt, **kwargs)
        
        start_time = time.time()
        res, _ = await self.afetch(**params)
        response_time = time.time() - start_time
        
        if self.formatting:
            return self._output(self.format_response(res, response_time))
        
        return res