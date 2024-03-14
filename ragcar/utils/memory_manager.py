from typing import Optional

from ragcar.utils.token_limiter import TokenLimiter
from ragcar.utils.prompt_template import PromptTemplate


class MemoryManager:
    """Class for managing a conversation memory."""
    
    def __init__(
        self, 
        turn_size: int = 1,
        token_limiter: Optional[TokenLimiter] = None
    ):
        """
        Initializes the memory manager with a specified turn size and an optional token limiter.

        Args:
            turn_size (int, optional): The size of the interaction between the human and AI, representing the number of exchanges (turns) to remember. Defaults to 1.
            token_limiter (Optional[TokenLimiter]): An instance of the TokenLimiter class to apply token limits on the conversation memory, or None if no limit is applied.
        """
        self.convo = PromptTemplate()
        self.turn_size = turn_size
        
        self.token_limiter = token_limiter
    
    def save_context(self, input: str, output: str) -> PromptTemplate:
        """
        Saves a new interaction (user input and assistant output) to the conversation history, applying turn size and token limit constraints.

        Args:
            input (str): The latest user input to save.
            output (str): The corresponding assistant output to save.

        Returns:
            PromptTemplate: The updated conversation history, trimmed according to the set turn size and token limit.
        """
        self.convo.user(input)
        self.convo.assistant(output)
        
        # retain only the latest 'turn_size' pairs of conversations.
        while len(self.convo.get_prompt()) > self.turn_size * 2:  # *2 because we have pairs (input, output)
            self.convo.remove_message(0)
        
        # apply token limitation if token_limiter is provided
        if self.token_limiter:
            self.convo = self.token_limiter.cutoff(self.convo)
        
        return self.convo