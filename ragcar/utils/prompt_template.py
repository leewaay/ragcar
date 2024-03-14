from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Union, List, Dict


class PromptTemplate:
    
    
    def __init__(
        self, 
        prompt: Optional[Union[str, List[Dict[str, str]]]] = None,
        user_prefix: Optional[str] = None,
        assistant_prefix: Optional[str] = None
    ):
        """
        Initialize a PromptTemplate instance.
        
        Args:
            prompt (Optional[Union[str, List[Dict[str, str]]]]): The initial conversation prompt. This can be:
                - A single string representing the initial message or scenario.
                - A list of dictionaries, where each dictionary represents an individual message with
                  'role' (e.g., 'user' or 'assistant') and 'content' (the message text).
            user_prefix (Optional[str]): A prefix to prepend to user messages, helping to identify them.
            assistant_prefix (Optional[str]): A prefix to prepend to assistant messages, helping to identify them.
        """
        self.prompt = prompt if isinstance(prompt, (str, list)) else []
        self.user_prefix = user_prefix
        self.assistant_prefix = assistant_prefix

    def say_as(self, role: str, message: str) -> PromptTemplate:
        """
        Adds a message with an associated role to the prompt.

        Args:
            role (str): The role of the speaker, indicating who is the source of the message. This could be any descriptive string like 'user', 'assistant', etc.
            message (str): The content of the message being added to the prompt. This is the actual text that the speaker is said to be conveying.

        Returns:
            PromptTemplate: This instance of PromptTemplate is returned to enable method chaining. This allows for the sequential addition of multiple messages 
                to the prompt by calling this method repeatedly.
        """
        self.prompt.append({"role": role, "content": message})
        return self

    def system(self, message: str) -> PromptTemplate:
        """Shortcut for say_as with 'system' as role"""
        return self.say_as("system", message)

    def user(self, message: str) -> PromptTemplate:
        """Shortcut for say_as with 'user' as role"""
        if self.user_prefix:
            return self.say_as("user", f"{self.user_prefix}: {message}")
        else:
            return self.say_as("user", f"{message}")

    def assistant(self, message: str) -> PromptTemplate:
        """Shortcut for say_as with 'assistant' as role"""
        if self.user_prefix:
            return self.say_as("assistant", f"{self.assistant_prefix}: {message}")
        else:
            return self.say_as("assistant", f"{message}")

    def remove_message(self, index: int) -> PromptTemplate:
        """
        Remove a specific message from the prompt by its index.

        Args:
            index (int): The index of the message to remove.

        Returns:
            PromptTemplate: The current instance to allow method chaining.

        Raises:
            IndexError: If the index is out of the range for the prompt list.
        """
        if 0 <= index < len(self.prompt):
            del self.prompt[index]
        else:
            raise IndexError(f"Index {index} is out of range for the prompt list.")
        return self
    
    def get_prompt(self) -> Union[str, List[Dict[str, str]]]:
        """
        Retrieves the current value of the prompt.

        Returns:
            Union[str, List[Dict[str, str]]]: The current value of `prompt`, which may be a single string or a list of dictionaries with string values.
        """
        return self.prompt
    
    def format_text(self, **kwargs: str) -> str:
        """
        Formats the prompt content into a single string using provided formatting parameters.

        Args:
            **kwargs (str): Keyword arguments for string formatting.

        Returns:
            str: A formatted string.
        """
        if isinstance(self.prompt, str):  # If the prompt is a string, format it directly
            return self.prompt.strip().format(**kwargs)
        
        formatted_messages = []
        for message in self.prompt:
            content = message["content"].strip().format(**kwargs)
            if message["role"] == "system":
                formatted_messages.append(f"{content}\n\n")
            else:
                formatted_messages.append(f"{content}\n")

        return ''.join(formatted_messages).strip()

    def format_chat(self, **kwargs: str) -> List[Dict[str, str]]:
        """
        Formats the stored prompt as a list of message dictionaries, optionally using provided string formatting parameters.

        Args:
            **kwargs: Arbitrary keyword arguments used for string formatting of messages. This allows dynamic insertion of content into the prompt messages.

        Returns:
            List[Dict[str, str]]: A formatted list of message dictionaries. Each dictionary contains 'role' and 'content' keys, 
                where 'role' indicates the speaker ('user' or 'assistant') and 'content' is the formatted message text.
        """
        if isinstance(self.prompt, str):  # If the prompt is a string, consider it as a user message
            return [{"role": "user", "content": self.prompt.strip().format(**kwargs)}]
        
        formatted_messages = []
        for message in self.prompt:
            role = message["role"]
            content = message["content"].strip().format(**kwargs)
            formatted_messages.append({"role": role, "content": content})

        # Check if the last message's role is "user" and append the assistant_prefix
        if formatted_messages[-1]["role"] == "user" and self.assistant_prefix:
            formatted_messages[-1]["content"] += f"\n{self.assistant_prefix}: "
        
        # Memory insertion logic
        if "memory" in kwargs and isinstance(kwargs["memory"], PromptTemplate):
            memory_msgs = kwargs["memory"].get_prompt()
            
            memory_position = next(
                (i for i, m in reversed(list(enumerate(formatted_messages))) if m["role"] == "user"),
                None  # Default if no such pattern is found
            )
            
            if memory_position is not None:
                for memory_msg in memory_msgs:
                    memory_content = memory_msg["content"].strip().format(**kwargs)
                    formatted_messages.insert(memory_position, {"role": memory_msg["role"], "content": memory_content})
                    memory_position += 1  # Update position since we've added new messages
        
        return formatted_messages
    
    def save(self, file_path: str) -> None:
        """
        Saves the current prompt template to a specified file in JSON format. This method is useful for persisting conversation templates or configurations.

        Args:
            file_path (str): The file path where the prompt template should be saved. The file must have a '.json' extension.

        Raises:
            ValueError: If the provided file path does not have a '.json' extension, indicating that the file format is not supported.
        """
        if isinstance(file_path, str):
            file_path = Path(file_path)

        file_path.parent.mkdir(parents=True, exist_ok=True)

        if file_path.suffix == ".json":
            with file_path.open("w", encoding="utf-8") as f:
                json.dump(self.prompt, f, indent=4, ensure_ascii=False)
        else:
            raise ValueError(f"{file_path} must be a .json file")

    @classmethod
    def load(cls, file_path: str) -> PromptTemplate:
        """
        Loads a prompt template from a specified JSON file and creates a PromptTemplate instance from it.

        Args:
            file_path (str): The file path from which to load the prompt template. The file must have a '.json' extension.

        Returns:
            PromptTemplate: An instance of PromptTemplate initialized with the data loaded from the specified JSON file.

        Raises:
            ValueError: If the provided file path does not have a '.json' extension, indicating that the file format is not supported.
        """
        if isinstance(file_path, str):
            file_path = Path(file_path)

        if file_path.suffix == ".json":
            with file_path.open("r") as f:
                prompt = json.load(f)
        else:
            raise ValueError(f"{file_path} must be a .json file")

        return cls(prompt)