from typing import Optional

from ragcar.tools.utils.base import RagcarToolBase
from ragcar.tools.text_generation import RagcarTextGenerationFactory
from ragcar.utils.prompt_template import PromptTemplate


class ChatSummarizer:
    """Class for summarizing conversations between a user and a chatbot."""
    
    def __init__(self, summarizer: Optional[RagcarToolBase] = None, max_tokens: int=256, use_async: Optional[bool] = False):
        """
        Initializes the ChatSummarizer with an optional custom summarizer or the default configuration.

        Args:
            summarizer (Optional[RagcarToolBase]): An optional custom summarizer instance of RagcarToolBase or derived class.
            max_tokens (int): The maximum number of tokens allowed for the summary. Defaults to 256.
            use_async (Optional[bool]): Enables asynchronous operation mode if True. Defaults to False.
        """
        if summarizer:
            self.summarizer = summarizer
        else:
            summ_template = PromptTemplate()
            summ_template.user(
                (
                    "Given the following conversation between a user and a chatbot, "
                    "please provide only a concise summary in Korean (2-3 sentences) "
                    "using the format 'User는 ..., AI는 ...':\n\nUser: {input}\nAI: {output}"
                )
            )
            
            self.summarizer = RagcarTextGenerationFactory(
                tool="text_generation", 
                src="openai", 
                model="gpt-3.5-turbo", 
                prompt_template=summ_template, 
                max_tokens=max_tokens, 
                temperature=0., 
                top_p=1., 
                frequency_penalty=0., 
                presence_penalty=2.0, 
                async_mode=use_async,
                formatting=True
            ).load()
    
    def summarize(self, input: str, output: str) -> str:
        """
        Generates a summary for a given user input and assistant output.

        Args:
            input (str): The user's input text.
            output (str): The assistant's output text.

        Returns:
            str: A concise summary of the conversation in Korean.
        """
        return self.summarizer(input=input, output=output)
    
    async def asummarize(self, input: str, output: str) -> str:
        """
        Asynchronously generates a summary for a given user input and assistant output.

        Args:
            input (str): The user's input text.
            output (str): The assistant's output text.

        Returns:
            str: A concise summary of the conversation in Korean, generated asynchronously.
        """
        return await self.summarizer(input=input, output=output)