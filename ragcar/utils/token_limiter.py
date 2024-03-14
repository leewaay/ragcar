from typing import Union

from ragcar.tools.utils.base import RagcarToolBase
from ragcar.utils.prompt_template import PromptTemplate


class TokenLimiter:
    """Class for limiting the number of tokens in a conversation list to a predefined maximum."""
    
    def __init__(self, tokenizer: RagcarToolBase, max_tokens: int = 1000) -> None:
        """
        Initializes a token limiter instance with a tokenizer derived from RagcarToolBase and a maximum token limit.

        Args:
            tokenizer (RagcarToolBase): An instance of a class derived from RagcarToolBase for tokenization.
            max_tokens (int, optional): The maximum number of tokens allowed. Defaults to 1000.
        """
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens

    def token_counter(self, corpus: str) -> int:
        """
        Count the number of tokens in the input string.

        Args:
            corpus (str): Input string to be tokenized.

        Returns:
            int: Number of tokens in the input string.
        """
        return len(self.tokenizer(corpus))
    
    def cutoff(self, input_data: Union[list, PromptTemplate], direction: str = "front") -> Union[list, PromptTemplate]:
        """
        Trims the input data to ensure the total token count is below the specified maximum limit. This is achieved by
        removing tokens from the start or end of the input data based on the provided direction.

        Args:
            input_data (Union[list, PromptTemplate]): The input data to be trimmed. Can be either a list of tokens or a PromptTemplate object.
            direction (str, optional): The direction from which tokens should be removed. Accepts 'front' for removal from the beginning 
                or 'back' for removal from the end. Defaults to 'front'.

        Returns:
            Union[list, PromptTemplate]: The trimmed input data with a total token count below the maximum limit.

        Raises:
            ValueError: Raised if an invalid direction is specified (not 'front' or 'back').
            ValueError: Raised if the input data is reduced to a size where it becomes empty or loses its meaningful context, 
                indicating that the initial data exceeded the maximum token limit by a significant margin.
        """
        output_data = input_data.copy()
        if isinstance(output_data, PromptTemplate):
            while self.token_counter(str(output_data.get_prompt())) >= self.max_tokens:
                if len(output_data.get_prompt()) <= 2:
                    raise ValueError(
                        "The combined length of the new input exceeds the max tokens allowed. "
                        "Please increase the max_tokens or reduce the length of the input and output."
                    )
                # Remove the oldest conversation pair
                for _ in range(2):
                    output_data.remove_message(0)
        else:
            tokens_str = str(output_data)
            if direction == "front":
                while self.token_counter(tokens_str) >= self.max_tokens and output_data:
                    if len(output_data) <= 1:
                        raise ValueError(
                            "The combined length of the new input exceeds the max tokens allowed. "
                            "Please increase the max_tokens or reduce the length of the input and output."
                        )
                    output_data.pop(0)
                    tokens_str = str(output_data)
            elif direction == "back":
                while self.token_counter(tokens_str) >= self.max_tokens and output_data:
                    if len(output_data) <= 1:
                        raise ValueError(
                            "The combined length of the new input exceeds the max tokens allowed. "
                            "Please increase the max_tokens or reduce the length of the input and output."
                        )
                    output_data.pop()
                    tokens_str = str(output_data)
            else:
                raise ValueError("Invalid direction. Use 'front' or 'back'.")
            
        return output_data