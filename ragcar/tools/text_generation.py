from typing import Optional, Union

from ragcar.tools.utils.base import RagcarFactoryBase, RagcarGenerationBase, RagcarAsyncGenerationBase
from ragcar.tools.utils.model_config import get_openai_config, get_clova_config
from ragcar.utils.prompt_template import PromptTemplate


class RagcarTextGenerationFactory(RagcarFactoryBase):
    """
    Generates text responses based on input prompts.

    Examples:
        >>> generator = Ragcar(
        ... tool="text_generation", 
        ... src="openai", 
        ... model="gpt-3.5-turbo", 
        ... prompt_template=PromptTemplate("{input} 수도는?"), 
        ... max_tokens=50
        ... )
        >>> generator(input="대한민국")
        '대한민국의 수도는 서울입니다.'
        
        >>> generator = Ragcar(
        ... tool="text_generation", 
        ... src="openai", 
        ... model="gpt-3.5-turbo", 
        ... prompt_template=PromptTemplate("{input} 수도는?"), 
        ... max_tokens=50, 
        ... stream=True
        ... )
        >>> response = generator(input="대한민국")
        >>> for i in response:
        ...     print(i)
        대
        한
        민
        국
        의
         수
        도
        는
         서
        울
        입니다
        .
    """
    
    def __init__(
        self, 
        tool: str, 
        src: str, 
        model: Optional[Union[str, dict]],
        prompt_template: PromptTemplate,
        max_tokens: int=1000,
        temperature: float=0.,
        top_p: float=1.,
        frequency_penalty: float=0., 
        presence_penalty: float=2.0,
        stop_before: Optional[list] = None,
        functions: Optional[list] = None,
        stream: Optional[bool] = False, 
        use_async: Optional[bool] = False,
        formatting: Optional[bool] = False
    ):
        super().__init__(tool, src, model)
        self.prompt_template = prompt_template
        self.max_tokens = max_tokens
        self.temperature = 0.1 if src == 'clova' else temperature
        self.top_p = top_p
        self.frequency_penalty = 0.1 if src == 'clova' else frequency_penalty
        self.presence_penalty  = int(presence_penalty) if src == 'clova' else presence_penalty
        self.stop_before = stop_before
        self.functions = functions
        self.stream = stream
        self.use_async = use_async
        self.formatting = formatting
    
    @staticmethod
    def get_available_srcs():
        return [
            "openai",
            "clova",
        ]
    
    @staticmethod
    def get_available_models():
        return {
            "openai": [
                "gpt-4-turbo-preview",
                "gpt-4",
                "gpt-3.5-turbo",
                "MODELS_SUPPORTED(https://platform.openai.com/docs/models)"
            ],
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
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "stop_before": self.stop_before
        }
        params = {k: v for k, v in params.items() if v is not None}
        
        # OpenAI
        if self.config.src == "openai":
            api_conf = get_openai_config(self.config.model_info)
            
            params["stream"] = self.stream
            
            if "gpt-" in api_conf.model_n:  # >= gpt-3.5
                from ragcar.models.openai import OpenaiChatCompletion
                
                model = OpenaiChatCompletion(
                    api_conf.model_n,
                    api_conf.api_key,
                    self.stream,
                    self.formatting
                )
                
                params["functions"] = self.functions
                
                if self.use_async:
                    return RagcarAsyncChatGeneration(
                        self.config,
                        model,
                        self.prompt_template,
                        params
                    )
                else:
                    return RagcarChatGeneration(
                        self.config,
                        model,
                        self.prompt_template,
                        params
                    )
            else:  # gpt-3
                from ragcar.models.openai import OpenaiCompletion
                
                model = OpenaiCompletion(
                    api_conf.model_n,
                    api_conf.api_key,
                    self.stream,
                    self.formatting
                )
        
                if self.use_async:
                    return RagcarAsyncTextGeneration(
                        self.config,
                        model,
                        self.prompt_template,
                        params
                    )
                else:
                    return RagcarTextGeneration(
                        self.config,
                        model,
                        self.prompt_template,
                        params
                    )
        
        # CLOVA
        if self.config.src == "clova":
            api_conf = get_clova_config(self.config.model_info)
            
            if "HCX" in api_conf.model_n:  # HyperCLOVA X
                from ragcar.models.clova import ClovaChatCompletion
                
                model = ClovaChatCompletion(
                    api_conf.model_n, 
                    api_conf.api_key, 
                    api_conf.app_key, 
                    self.stream,
                    self.formatting
                )
                
                if self.use_async:
                    return RagcarAsyncChatGeneration(
                        self.config,
                        model,
                        self.prompt_template,
                        params
                    )
                else:
                    return RagcarChatGeneration(
                        self.config,
                        model,
                        self.prompt_template,
                        params
                    )
                
            else:  # HyperCLOVA
                from ragcar.models.clova import ClovaCompletion
                
                model = ClovaCompletion(
                    api_conf.model_n, 
                    api_conf.api_key, 
                    api_conf.app_key,
                    self.formatting
                )
                
                if self.use_async:
                    return RagcarAsyncTextGeneration(
                        self.config,
                        model,
                        self.prompt_template,
                        params
                    )
                else:
                    return RagcarTextGeneration(
                        self.config,
                        model,
                        self.prompt_template,
                        params
                    )


class RagcarTextGeneration(RagcarGenerationBase):
    
    def __init__(self, config, model, prompt_template, params):
        super().__init__(config)
        self._model = model
        self._prompt_template = prompt_template
        self._params = params
    
    def predict(self, **kwargs):
        """
        Generates a text response based on a formatted prompt template and model parameters.

        Args:
            **kwargs (dict): Keyword arguments used to fill placeholders in the prompt template.

        Returns:
            dict: Contains details of the response and the generated text.
        """
        formatted_prompt = self._prompt_template.format_text(**kwargs)
        
        outputs = self._model.create(formatted_prompt, **self._params)
        
        return outputs


class RagcarAsyncTextGeneration(RagcarAsyncGenerationBase):
    
    def __init__(self, config, model, prompt_template, params):
        super().__init__(config)
        self._model = model
        self._prompt_template = prompt_template
        self._params = params
    
    async def predict(self, **kwargs):
        """
        Asynchronously generates a text response based on a formatted prompt template and model parameters.

        Args:
            **kwargs (dict): Keyword arguments used to fill placeholders in the prompt template.

        Returns:
            dict: Contains details of the response and the generated text, obtained asynchronously.
        """
        formatted_prompt = self._prompt_template.format_text(**kwargs)
        
        outputs = await self._model.acreate(formatted_prompt, **self._params)
        
        return outputs


class RagcarChatGeneration(RagcarGenerationBase):
    
    def __init__(self, config, model, prompt_template, params):
        super().__init__(config)
        self._model = model
        self._prompt_template = prompt_template
        self._params = params
    
    def predict(self, **kwargs):
        """
        Generates a chat response for given messages using a specified model and prompt template.

        Args:
            **kwargs (dict): Keyword arguments representing the chat messages used to fill placeholders in the prompt template.

        Returns:
            dict: Contains details of the chat response and the generated text.
        """
        formatted_messages = self._prompt_template.format_chat(**kwargs)
        
        outputs = self._model.create(formatted_messages, **self._params)
        
        return outputs


class RagcarAsyncChatGeneration(RagcarAsyncGenerationBase):
    
    def __init__(self, config, model, prompt_template, params):
        super().__init__(config)
        self._model = model
        self._prompt_template = prompt_template
        self._params = params
    
    async def predict(self, **kwargs):
        """
        Asynchronously generates a chat response for given messages using a specified model and prompt template.

        Args:
            **kwargs (dict): Keyword arguments representing the chat messages used to fill placeholders in the prompt template.

        Returns:
            dict: Contains details of the chat response and the generated text, obtained asynchronously.
        """
        formatted_messages = self._prompt_template.format_chat(**kwargs)
        
        outputs = await self._model.acreate(formatted_messages, **self._params)
        
        return outputs