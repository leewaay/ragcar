from typing import Optional, Union, List

from ragcar.tools.utils.base import RagcarFactoryBase, RagcarSimpleBase, RagcarAsyncSimpleBase
from ragcar.tools.utils.model_config import get_clova_config


class RagcarTokenizationFactory(RagcarFactoryBase):
    """
    Tokenizing text or counting the number of tokens.
    
    Examples:
        >>> tokenizer = Ragcar(tool="tokenization", src="huggingface")
        >>> tokenizer("나는 동물을 좋아하는 사람이야")
        ['나', '##는', '동물', '##을', '좋아하', '##는', '사람', '##이', '##야']
        
        >>> tokenizer = Ragcar(tool="tokenization", src="tiktoken")
        >>> tokenizer("나는 동물을 좋아하는 사람이야")
        [
            167,
            224,
            246,
            167,
            232,
            242,
            ...
            234,
            35975,
            112,
            168,
            243,
            120
        ]
        
        >>> tokenizer = Ragcar(tool="tokenization", src="kiwi")
        >>> tokenizer("나는 동물을 좋아하는 사람이야")
        [
            Token(form='나', tag='NP', start=0, len=1),
            Token(form='는', tag='JX', start=1, len=1),
            Token(form='동물', tag='NNG', start=3, len=2),
            Token(form='을', tag='JKO', start=5, len=1),
            Token(form='좋아하', tag='VV', start=7, len=3),
            Token(form='는', tag='ETM', start=10, len=1),
            Token(form='사람', tag='NNG', start=12, len=2),
            Token(form='이', tag='VCP', start=14, len=1),
            Token(form='야', tag='EF', start=15, len=1)
        ]
    """
    
    def __init__(
        self, 
        tool: str, 
        src: str, 
        model: Optional[Union[str, dict]],
        use_async: bool = False
    ):
        super().__init__(tool, src, model)
        self.use_async = use_async
    
    @staticmethod
    def get_available_srcs():
        return [
            "huggingface",
            "tiktoken",
            "openai",
            "clova",
            "kiwi",
        ]
    
    @staticmethod
    def get_available_models():
        return {
            "huggingface": [
                "klue/roberta-large",
                "jinmang2/kpfbert",
                "MODELS_SUPPORTED(https://huggingface.co/models?library=transformers)"
            ],
            "tiktoken": [
                "cl100k_base",
                "p50k_base",
                "r50k_base",
                "gpt2",
                "MODELS_SUPPORTED(https://github.com/openai/tiktoken/blob/main/tiktoken/model.py)"
            ],
            "openai": [
                "gpt-4-turbo",
                "gpt-4",
                "gpt-3.5-turbo",
                "MODELS_SUPPORTED(https://platform.openai.com/docs/models)"
            ],
            "clova": [
                "YOUR_MODEL(https://www.ncloud.com/product/aiService/clovaStudio)"
            ],
            "kiwi": [
                None,
                "YOUR_MODEL"
            ],
        }
    
    def load(self):
        """
        Loads and initializes the model based on configuration settings.

        Returns:
            object: The initialized model ready for use.
        """
        if self.config.src == "tiktoken":
            import tiktoken
            
            model = tiktoken.get_encoding(self.config.model_info)
            
            return RagcarTiktokenTokenizer(self.config, model)
        
        if self.config.src == "openai":
            import tiktoken
            
            model = tiktoken.encoding_for_model(self.config.model_info)
            
            return RagcarTiktokenTokenizer(self.config, model)
        
        if self.config.src == "clova":  # HyperCLOVA X
            api_conf = get_clova_config(self.config.model_info)
            
            if "chat-tokenize" in api_conf.model_n:
                from ragcar.models.clova import ClovaChatTokenization
                
                model = ClovaChatTokenization(
                    api_conf.model_n, 
                    api_conf.api_key, 
                    api_conf.app_key
                )
            else:
                from ragcar.models.clova import ClovaTokenization
                
                model = ClovaTokenization(
                    api_conf.model_n, 
                    api_conf.api_key, 
                    api_conf.app_key
                )
                
            if self.use_async:
                return RagcarAsyncTokenizer(self.config, model)
        
        elif self.config.src == "kiwi":
            from kiwipiepy import Kiwi
            
            model = Kiwi()
            
            if self.config.model_info:
                model.load_user_dictionary(self.config.model_info)
        
        else:
            from transformers import AutoTokenizer

            model = AutoTokenizer.from_pretrained(self.config.model_info)
        
        return RagcarTokenizer(self.config, model)


class RagcarTiktokenTokenizer(RagcarSimpleBase):
    
    def __init__(self, config, model):
        super().__init__(config)
        self._model = model
    
    def predict(self, sent: str, **kwargs) -> List[str]:
        """
        Tokenizes the input sentence into a sequence of tokens.

        Args:
            sent (str): The sentence to be tokenized.

        Returns:
            List[str]: A list of tokens derived from the input sentence.
        """
        token_integers = self._model.encode(sent)
        outputs = self._model.decode_tokens_bytes(token_integers)
        return outputs


class RagcarTokenizer(RagcarSimpleBase):
    
    def __init__(self, config, model):
        super().__init__(config)
        self._model = model
    
    def predict(self, sent: str, **kwargs) -> List[str]:
        """
        Tokenizes a sentence into a sequence of strings.

        Args:
            sent (str): The sentence to be tokenized.

        Returns:
            List[str]: A list of tokens as strings.
        """
        outputs = self._model.tokenize(sent)
        return outputs

class RagcarAsyncTokenizer(RagcarAsyncSimpleBase):
    
    def __init__(self, config, model):
        super().__init__(config)
        self._model = model
    
    async def predict(self, sent: str, **kwargs) -> List[str]:
        """
        Asynchronously tokenizes a sentence into a sequence of strings.

        Args:
            sent (str): The sentence to be tokenized.

        Returns:
            List[str]: A list of tokens as strings, obtained asynchronously.
        """
        outputs = await self._model.atokenize(sent)
        return outputs