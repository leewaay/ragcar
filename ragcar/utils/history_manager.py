from typing import Optional, List

from ragcar.utils.token_limiter import TokenLimiter
from ragcar.utils.chat_summarizer import ChatSummarizer


class HistoryManager:
    """Class for managing and summarizing conversation memory to avoid excessive token usage."""
    
    def __init__(
        self, 
        chat_summarizer: ChatSummarizer,
        token_limiter: Optional[TokenLimiter] = None
    ):
        """
        Initializes the history manager with options for conversation summarization and token limitation.

        Args:
            chat_summarizer (ChatSummarizer): A ChatSummarizer instance used for summarizing each turn of the conversation.
            token_limiter (Optional[TokenLimiter]): An optional TokenLimiter instance to enforce token count limits.
        """
        self.history = []
        self.chat_summarizer = chat_summarizer
        
        self.token_limiter = token_limiter
    
    def save_context(self, input: str, output: str) -> List[str]:
        """
        Stores and summarizes the user input and assistant output into the conversation history.

        Args:
            input (str): User's input.
            output (str): Assistant's output.

        Returns:
            List[str]: The updated conversation history with the latest interaction summarized and added.
        """
        summarized = self.chat_summarizer.summarize(input=input, output=output)
        
        self.history.append(summarized)
        
        # apply token limitation if token_limiter is provided
        if self.token_limiter:
            self.history = self.token_limiter.cutoff(self.history)
        
        return self.history
    
    async def asave_context(self, input: str, output: str) -> List[str]:
        """
        Asynchronously stores and summarizes the user input and assistant output into the conversation history.

        Args:
            input (str): User's input.
            output (str): Assistant's output.

        Returns:
            List[str]: The updated conversation history with the latest interaction summarized and added.
        """
        summarized = await self.chat_summarizer.asummarize(input=input, output=output)
        
        self.history.append(summarized)
        
        # apply token limitation if token_limiter is provided
        if self.token_limiter:
            self.history = self.token_limiter.cutoff(self.history)
        
        return self.history