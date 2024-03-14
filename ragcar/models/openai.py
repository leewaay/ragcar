import logging
import json
import time
from typing import Optional, Union, Dict, List, Generator, AsyncGenerator

import openai

from ragcar.models.base import OpenaiBase


logger = logging.getLogger(__name__)


class OpenaiEmbedding(OpenaiBase):
    """Class for generating text with OpenAI's Embedding model"""
    
    def __init__(self, model_n: str, api_key: str):
        super().__init__(model_n, api_key)

    def encode(self, text: str) -> List[float]:
        """
        Get embedding for a given text.

        Args:
            text (str): Input text for which embedding is to be generated.

        Returns:
            List[float]: The embedding.
        """
        res, _ = self.fetch(openai.Embedding.create, input=text)
        
        return res['data'][0]['embedding']

    async def aencode(self, text: str) -> List[float]:
        """
        Asynchronously get embedding for a given text.

        Args:
            text (str): Input text for which embedding is to be generated.

        Returns:
            List[float]: The embedding.
        """
        res, _ = await self.afetch(openai.Embedding.acreate, input=text)
        
        return res['data'][0]['embedding']


class OpenaiChatCompletion(OpenaiBase):
    """Class for generating text with OpenAI's Chat model."""
    
    def __init__(
        self, 
        model_n: str, 
        api_key: str, 
        stream: bool, 
        formatting: bool
    ):
        super().__init__(model_n, api_key)
        
        self.stream = stream
        self.formatting = formatting
        
        if self.stream and self.formatting:
            from ragcar.tools.tokenization import RagcarTokenizationFactory
            
            self.tokenizer = RagcarTokenizationFactory(
                tool="tokenization",
                src="openai",
                model=model_n
            ).load()

    def _calculate_tokens(self, prompt: Optional[list] = None, completion: Optional[str] = None):
        if prompt:
            fromatted_prompt = ["""{{"{0}": "{1}"}}""".format(item['role'], item['content']) for item in prompt]
            prompt_str = f"[{', '.join(fromatted_prompt)}]"
            return len(self.tokenizer(str(prompt_str))) - 1
        
        if completion:
            return len(self.tokenizer(completion))
    
    def _get_params(
        self, 
        messages: list, 
        max_tokens: int, 
        temperature: float, 
        top_p: float, 
        frequency_penalty: float, 
        presence_penalty: float, 
        stop_before: Optional[list] = None, 
        functions: Optional[list] = None,
        stream: Optional[bool] = None
    ) -> Dict[str, Union[str, int, float]]:
        params = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "stop": stop_before,
            "functions": functions,
            "stream": stream
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
        start_time,
        has_functions,
        prompt
    ) -> Generator[str, None, None]:
        message = ""
        for chunk in res:
            if self.formatting:
                logger.info(json.dumps(chunk, ensure_ascii=False, indent=4))
                
                choice = chunk.get('choices', [{}])[0]
                
                delta = choice.get('delta', {})
                finish_reason = choice.get('finish_reason')
                
                if has_functions:
                    function_call = delta.get('function_call', {})
                    content = function_call.get('arguments', None)
                else:
                    content = delta.get('content', None)
                    
                if content:
                    yield {
                        "id": request_id,
                        "event": "chunk",
                        "content": content
                    }
                    
                    message += content
                    
                if finish_reason:
                    response_time = time.time() - start_time
                    
                    prompt_tokens = self._calculate_tokens(prompt=prompt)
                    completion_tokens = self._calculate_tokens(completion=message)
                    
                    default_format = {
                        "choices": [{
                            "message": {
                                "role": "assistant",
                                "content": message,
                            },
                            "finish_reason": finish_reason
                        }],
                        "usage": {
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "total_tokens": prompt_tokens + completion_tokens
                        }
                    }
                    
                    yield {
                        "id": request_id,
                        "event": "finish",
                        "data": self.format_response(default_format, request_id, response_time)
                    }
            else:
                chunk["id"] = request_id
                yield chunk
    
    def create(
        self, 
        messages, 
        **kwargs
    ) -> Union[str, Dict[str, str], Generator[str, None, None]]:
        params = self._get_params(messages, **kwargs)
        
        start_time = time.time()
        res, request_id = self.fetch(openai.ChatCompletion.create, **params)
        
        if params.get('stream'):
            return self._stream_output(res, request_id, start_time, params.get("functions"), messages)
        
        return self._output(res, request_id, start_time)

    async def _astream_output(
        self,
        res,
        request_id,
        start_time,
        has_functions,
        prompt
    ) -> AsyncGenerator[str, None]:
        message = ""
        async for chunk in res:
            if self.formatting:
                logger.info(json.dumps(chunk, ensure_ascii=False, indent=4))
                
                choice = chunk.get('choices', [{}])[0]
                
                delta = choice.get('delta', {})
                finish_reason = choice.get('finish_reason')
                
                if has_functions:
                    function_call = delta.get('function_call', {})
                    content = function_call.get('arguments', None)
                else:
                    content = delta.get('content', None)
                    
                if content is not None:
                    yield {
                        "id": request_id,
                        "event": "chunk",
                        "content": content
                    }
                    
                    message += content
                    
                if finish_reason:
                    response_time = time.time() - start_time
                    
                    prompt_tokens = self._calculate_tokens(prompt=prompt)
                    completion_tokens = self._calculate_tokens(completion=message)
                    
                    default_format = {
                        "choices": [{
                            "message": {
                                "role": "assistant",
                                "content": message,
                            },
                            "finish_reason": finish_reason
                        }],
                        "usage": {
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "total_tokens": prompt_tokens + completion_tokens
                        }
                    }
                    
                    yield {
                        "id": request_id,
                        "event": "finish",
                        "data": self.format_response(default_format, request_id, response_time)
                    }
            else:
                chunk["id"] = request_id
                yield chunk
    
    async def acreate(self, messages, **kwargs) -> Union[str, AsyncGenerator[str, None]]:
        params = self._get_params(messages, **kwargs)
        
        start_time = time.time()
        res, request_id = await self.afetch(openai.ChatCompletion.acreate, **params)
        
        if params.get('stream'):
            return self._astream_output(res, request_id, start_time, params.get("functions"), messages)
        
        return self._output(res, request_id, start_time)
    

class OpenaiCompletion(OpenaiBase):
    """Class for generating text with OpenAI's Completion model"""
    
    def __init__(
        self, 
        model_n: str, 
        api_key: str, 
        stream: bool, 
        formatting: bool
    ):
        super().__init__(model_n, api_key)
        
        self.stream = stream
        self.formatting = formatting
        
        if self.stream and self.formatting:
            from ragcar.tools.tokenization import ToolvaTokenizationFactory
            
            self.tokenizer = ToolvaTokenizationFactory(
                tool="tokenization",
                src="openai",
                model=model_n
            ).load()

    def _calculate_tokens(self, prompt: Optional[list] = None, completion: Optional[str] = None):
        if prompt:
            fromatted_prompt = ["""{{"{0}": "{1}"}}""".format(item['role'], item['content']) for item in prompt]
            prompt_str = f"[{', '.join(fromatted_prompt)}]"
            return len(self.tokenizer(str(prompt_str))) - 1
        
        if completion:
            return len(self.tokenizer(completion))
    
    def _get_params(
        self, 
        prompt: str, 
        max_tokens: int, 
        temperature: float, 
        top_p: float, 
        frequency_penalty: float, 
        presence_penalty: float,
        stop_before: Optional[list] = None,
        stream: Optional[bool] = False
    ) -> Dict[str, Union[str, int, float]]:
        params = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "stop": stop_before,
            "stream": stream
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
        start_time,
        prompt
    ) -> Generator[str, None, None]:
        message = ""
        for chunk in res:
            if self.formatting:
                logger.info(json.dumps(chunk, ensure_ascii=False, indent=4))
                
                choice = chunk.get('choices', [{}])[0]
                
                content = choice[0].get('text', {})
                finish_reason = choice.get('finish_reason')
                
                if content:
                    yield {
                        "id": request_id,
                        "event": "chunk",
                        "content": content
                    }
                    
                    message += content
                
                if finish_reason:
                    response_time = time.time() - start_time
                    
                    prompt_tokens = self._calculate_tokens(prompt=prompt)
                    completion_tokens = self._calculate_tokens(completion=message)
                    
                    default_format = {
                        "choices": [{
                            "text": message,
                            "finish_reason": finish_reason
                        }],
                        "usage": {
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "total_tokens": prompt_tokens + completion_tokens
                        }
                    }
                    
                    yield {
                        "id": request_id,
                        "event": "finish",
                        "data": self.format_response(default_format, request_id, response_time)
                    }
            else:
                chunk["id"] = request_id
                yield chunk
    
    def create(self, prompt, **kwargs) -> Union[str, Generator[str, None, None]]:
        params = self._get_params(prompt, **kwargs)
        
        start_time = time.time()
        res, request_id = self.fetch(openai.Completion.create, **params)
        
        if params.get('stream'):
            return self._stream_output(res, request_id, start_time, params.get("functions"), prompt)
        
        return self._output(res, request_id, start_time)

    async def _astream_output(self, res, request_id) -> AsyncGenerator[str, None]:
        message = ""
        async for chunk in res:
            if self.formatting:
                logger.info(json.dumps(chunk, ensure_ascii=False, indent=4))
                
                choice = chunk.get('choices', [{}])[0]
                
                content = choice[0].get('text', {})
                finish_reason = choice.get('finish_reason')
                
                if content:
                    yield {
                        "id": request_id,
                        "event": "chunk",
                        "content": content
                    }
                    
                    message += content
                
                if finish_reason:
                    response_time = time.time() - start_time
                    
                    prompt_tokens = self._calculate_tokens(prompt=prompt)
                    completion_tokens = self._calculate_tokens(completion=message)
                    
                    default_format = {
                        "choices": [{
                            "text": message,
                            "finish_reason": finish_reason
                        }],
                        "usage": {
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "total_tokens": prompt_tokens + completion_tokens
                        }
                    }
                    
                    yield {
                        "id": request_id,
                        "event": "finish",
                        "data": self.format_response(default_format, request_id, response_time)
                    }
            else:
                chunk["id"] = request_id
                yield chunk
    
    async def acreate(self, prompt, **kwargs) -> Union[str, AsyncGenerator[str, None]]:
        params = self._get_params(prompt, **kwargs)
        
        start_time = time.time()
        res, request_id = await self.afetch(openai.Completion.acreate, **params)
        response_time = time.time() - start_time
        
        if params.get('stream'):
            return self._stream_output(res, request_id, start_time, params.get("functions"), prompt)
        
        return self._output(res, request_id, start_time)