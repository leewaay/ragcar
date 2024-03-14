import logging
import json
import uuid
from typing import Optional, Union, Tuple, Generator, AsyncGenerator, Dict

import requests
import aiohttp
import openai


logger = logging.getLogger(__name__)


class ClovaBase:
    """Base class for HyperCLOVA models"""
    
    # Define the cost per token for each model outside the function
    COST_PER_TOKEN = {
        'HCX-003': 0.005,
        'HCX-003-tuning': 0.03,
        'HCX-002': 0.005,
        'HCX-002-tuning': 0.03,
        'LK-D': 0.04,
        'LK-D-tuning': 0.12,
        'LK-C': 0.015,
        'LK-C-tuning': 0.045,
        'LK-B': 0.0025,
        'LK-B-tuning': 0.0075,
    }
    
    def __init__(self, api_url: str, api_key: str, app_key: str, stream: Optional[bool] = False):
        self.api_url = api_url
        
        parts = api_url.split('/')
        if 'tasks' in api_url:  # HyperCLOVA tuning model
            last_two = parts[-2:]

            if last_two[-1] == 'completions':
                last_two[-1] = 'LK-D'
            elif last_two[-1] == 'chat-completions':
                last_two[-1] = 'HCX-003'

            self.model_n = '/'.join(last_two)
        else:
            self.model_n = parts[-1]
        
        self.headers = {
            'Content-Type': 'application/json; charset=utf-8',
            'X-NCP-APIGW-API-KEY': api_key,
            'X-NCP-CLOVASTUDIO-API-KEY': app_key,
            # 'X-NCP-CLOVASTUDIO-REQUEST-ID': request_id
        }
        
        if stream:
            self.headers["Accept"] = "text/event-stream"
        
        self.stream = stream
    
    def _to_camel(self, snake_str: str) -> str:
        """
        Convert a snake_case string to camelCase.
        
        Args:
            snake_str (str): The snake_case string to be converted.

        Returns:
            str: The converted camelCase string.
        """
        components = snake_str.split('_')
        return components[0] + ''.join(x.title() for x in components[1:])
    
    def _get_model_type(self, model_n: str) -> str:
        """
        Extract the model type from Clova engine string.

        Args:
            model_n (str): The model name string from which to extract the model type

        Returns:
            str: The model type extracted from the engine string.

        Raises:
            ValueError: If the engine string format is invalid.
        """
        try:
            return f"{model_n.split('/')[-1]}-tuning" if "/" in model_n else model_n
        except IndexError:
            raise ValueError("Invalid url string format. Unable to extract model type.")

    def _calculate_charge_per_request(self, model_type: str, total_tokens: int) -> Optional[float]:
        """
        Calculate the total cost of a request based on the number of input and output tokens and the engine used.

        Args:
            model_type (str): The type of the model.
            total_tokens (int): The total number of tokens used in a request.

        Returns:
            Optional[float]: The total cost of the request in KRW or None if the model type is not recognized.
        """
        if model_type not in self.COST_PER_TOKEN:
            return None

        cost_per_input_token = self.COST_PER_TOKEN[model_type]
        total_cost_in_krw = total_tokens * cost_per_input_token

        return total_cost_in_krw
    
    def format_response(
        self, 
        response_data,
        request_id: str,
        response_time: float
    ) -> Dict[str, Union[str, int, float, Dict]]:
        """
        Format the response from the API.

        Args:
            response_data (Dict): The raw response from the API, expected to contain result details.
            request_id (str): The unique identifier of the request.
            response_time (float): The time (in seconds) it took to get the response from the API.

        Returns:
            Dict[str, Union[str, int, float, Dict]]: A dictionary containing formatted response details, including:
                - id (str): The request identifier.
                - model (str): The model name used for the request.
                - content (str): The main content message from the response.
                - finish_reason (str): The reason why the operation was finished.
                - input_tokens (int): The number of tokens in the input.
                - output_tokens (int): The number of tokens in the output.
                - total_tokens (int): The total number of tokens processed.
                - predicted_cost (Union[str, float]): The predicted cost of the operation, or "Unknown" if not calculable.
                - response_time (float): The response time of the API call.
                - ai_filter (Dict): A dictionary with AI filter scores, if applicable.
        """
        result = response_data.get('result', {})
        
        content = result.get('message', {}).get('content', '').strip()
        
        finish_reason = result.get('stopReason')
        
        input_tokens = result.get('inputLength')
        output_tokens = result.get('outputLength')
        total_tokens = (input_tokens or 0) + (output_tokens or 0)
        
        model_type = self._get_model_type(self.api_url)
        predicted_cost = self._calculate_charge_per_request(model_type, total_tokens)
        
        ai_filter = result.get('aiFilter')
        if ai_filter:
            ai_filter = {item['name']: int(item['score']) for item in ai_filter}
        
        formatted_data = {
            "id": request_id,
            "model": self.model_n,
            "content": content,
            "finish_reason": finish_reason,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "predicted_cost": predicted_cost if predicted_cost is not None else "Unknown",
            "response_time": response_time,
            "ai_filter": ai_filter
        }
            
        return formatted_data
    
    def fetch(self, **kwargs: dict) -> dict:
        """
        Send a POST request to the API.

        Args:
            kwargs (dict): The parameters for the request, which are passed as keyword arguments.

        Returns:
            dict: The response from the API.
        
        Raises:
            RuntimeError: If the API does not respond with a '20000' status code.
        """
        request_id = f"clova-{str(uuid.uuid4())}"
        
        # Convert snake_case keys to camelCase
        kwargs = {self._to_camel(k): v for k, v in kwargs.items()}
        
        logger.info(
            json.dumps(
                {
                    "id": request_id,
                    "model": self.api_url,
                    "parameters": kwargs
                }, ensure_ascii=False, indent=4
            )
        )
        
        response = requests.post(
            self.api_url,
            json=kwargs, 
            headers=self.headers,
            stream=self.stream
        )
        
        logger.info(
            json.dumps(
                {
                    "id": request_id,
                    "message": "Request completed successfully"
                }, ensure_ascii=False, indent=4
            )
        )
        
        if self.stream:
            return response, request_id
        
        parsed_data = response.json()
        
        if parsed_data['status']['code'] == '20000':
            return parsed_data, request_id
        
        raise RuntimeError((f"Request failed with status code: {parsed_data['status']}"))

    async def afetch(self, **kwargs: dict) -> dict:
        """
        Send an asynchronous POST request to the API.

        Args:
            kwargs (dict): The parameters for the request, which are passed as keyword arguments.

        Returns:
            dict: The response from the API.
        
        Raises:
            RuntimeError: If the API does not respond with a '20000' status code.
        """
        request_id = f"clova-{str(uuid.uuid4())}"
        
        # Convert snake_case keys to camelCase
        kwargs = {self._to_camel(k): v for k, v in kwargs.items()}
        
        logger.info(
            json.dumps(
                {
                    "id": request_id,
                    "model": self.api_url,
                    "parameters": kwargs
                }, ensure_ascii=False, indent=4
            )
        )
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.api_url, 
                data=json.dumps(kwargs), 
                headers=self.headers
            ) as response:
                logger.info(
                    json.dumps(
                        {
                            "id": request_id,
                            "response": "Request completed successfully"
                        }, ensure_ascii=False, indent=4
                    )
                )
                
                if self.stream:
                    return await response.text(), request_id
                
                data = await response.text()
                parsed_data = json.loads(data)
                
                if parsed_data['status']['code'] == '20000':
                    return parsed_data, request_id
                
                raise RuntimeError((f"Request failed with status code: {parsed_data['status']}"))


class OpenaiBase:
    """Base class for OpenAI models"""
    
    # Define the cost per token for each model outside the function
    # https://openai.com/pricing
    COST_PER_TOKEN = {
        'gpt-4-base': {
            'input': 0.03 / 1000,
            'output': 0.06 / 1000
        },
        'gpt-4-large': {
            'input': 0.06 / 1000,
            'output': 0.12 / 1000
        },
        'gpt-4-turbo': {
            'input': 0.01 / 1000,
            'output': 0.03 / 1000
        },
        'gpt-3.5': {
            'input': 0.0010 / 1000,
            'output': 0.0020 / 1000
        },
        'gpt-3.5-instruct': {
            'input': 0.0015 / 1000,
            'output': 0.0020 / 1000
        },
        'gpt-3.5-tuning': {
            'input': 0.0030 / 1000,
            'output': 0.0060 / 1000
        },
        'davinci': {
            'input': 0.0020 / 1000,
            'output': 0.0020 / 1000
        },
        'davinci-tuning': {
            'input': 0.0120 / 1000,
            'output': 0.0120 / 1000
        },
        'babbage': {
            'input': 0.0004 / 1000,
            'output': 0.0004 / 1000
        },
        'babbage-tuning': {
            'input': 0.0016 / 1000,
            'output': 0.0016 / 1000
        }
    }
    
    def __init__(self, model_n: str, api_key: str):
        self.model_n = model_n
        openai.api_key = api_key

    def _get_model_type(self, model_n: str) -> str:
        """
        Extract the model type from OpenAI engine string.

        Args:
            model_n (str): The model name string from which to extract the model type.

        Returns:
            str: The model type extracted from the engine string.

        Raises:
            ValueError: If the engine string format is invalid.
        """
        if model_n.startswith('text-'):
            model_n = model_n.replace('text-', '', 1)

        if 'gpt-3.5' in model_n:
            if 'instruct' in model_n:
                return 'gpt-3.5-instruct'
            elif model_n.startswith('ft:'):
                return 'gpt-3.5-turbo-tuning'
            else:
                return 'gpt-3.5'
        elif 'gpt-4' in model_n:
            if '32k' in model_n:
                return 'gpt-4-large'
            elif '1106' in model_n:
                return 'gpt-4-turbo'
            else:
                return 'gpt-4-base'
        else:
            try:
                if model_n.startswith('ft:'):
                    return model_n.split('-')[0].replace('ft:', '') + "-tuning"
                else:
                    return model_n.split('-')[0]
            except IndexError:
                raise ValueError("Invalid model name string format. Unable to extract model type.")

    def _calculate_charge_per_request(self, model_type: str, input_tokens: int, output_tokens: int) -> Optional[float]:
        """
        Calculate the total cost of a request based on the number of input and output tokens and the engine used.

        Args:
            model_type (str): The type of the model.
            input_tokens (int): The input number of tokens used in a request.
            output_tokens (int): The output number of tokens used in a request.

        Returns:
            Optional[float]: The total cost of the request in KRW or None if the model type is not recognized.
        """
        if model_type not in self.COST_PER_TOKEN:
            return None

        total_cost_in_usd = 0

        if input_tokens:
            cost_per_input_token = self.COST_PER_TOKEN[model_type]['input']
            total_cost_in_usd += input_tokens * cost_per_input_token

        if output_tokens:
            cost_per_output_token = self.COST_PER_TOKEN[model_type]['output']
            total_cost_in_usd += output_tokens * cost_per_output_token

        return total_cost_in_usd

    def format_response(
        self,
        response_data,
        request_id: str,
        response_time: float
    ) -> Dict[str, Union[str, int, float]]:
        """
        Format the response from the API.

        Args:
            response_data (dict): The raw response data from the API.
            request_id (str): The unique identifier for the request.
            response_time (float): The duration it took to receive the response, in seconds.

        Returns:
            Dict[str, Union[str, int, float]]: A dictionary containing the formatted response data. This includes:
                - id (str): The request ID.
                - model (str): The model name used for the request.
                - content (str): The main content returned by the API.
                - finish_reason (str): The reason provided by the API for the request's completion.
                - input_tokens (int): The number of tokens used in the input.
                - output_tokens (int): The number of tokens generated as output.
                - total_tokens (int): The total number of tokens used (input + output).
                - predicted_cost (Union[str, float]): The predicted cost of the operation, or "Unknown" if not calculable.
                - response_time (float): The response time for the API call.
        """
        choice = response_data.get('choices', [{}])[0]
        
        model_type = self._get_model_type(self.model_n)
        
        if model_type.startswith('gpt'):
            message = choice.get('message', {})
            content = message if message.get('function_call') else message.get('content', '').strip()
        else:
            content = choice.get('text', '').strip()
        
        finish_reason = choice.get('finish_reason')
        
        usage = response_data.get('usage', {})
        input_tokens = usage.get('prompt_tokens')
        output_tokens = usage.get('completion_tokens')
        total_tokens = (input_tokens or 0) + (output_tokens or 0)
        
        predicted_cost = self._calculate_charge_per_request(model_type, input_tokens, output_tokens)
        
        formatted_data = {
            "id": request_id,
            "model": self.model_n,
            "content": content,
            "finish_reason": finish_reason,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "predicted_cost": predicted_cost if predicted_cost is not None else "Unknown",
            "response_time": response_time
        }
            
        return formatted_data

    def fetch(self, create_fn, **kwargs) -> Union[Tuple[dict, str], Tuple[Generator[dict, None, None], str]]:
        """
        Calls an external service and logs the request and response.

        Args:
            create_fn (Callable): The function to call the external service.
            **kwargs: Arbitrary keyword arguments passed to the external service call.

        Returns:
            Union[Tuple[dict, str], Tuple[Generator[dict, None, None], str]]: A tuple containing the response 
                from the external service and the request ID.
        """
        request_id = f"openai-{str(uuid.uuid4())}"
        
        logger.info(
            json.dumps(
                {
                    "id": request_id,
                    "model": self.model_n,
                    "parameters": kwargs
                }, ensure_ascii=False, indent=4
            )
        )
        
        response = create_fn(model=self.model_n, **kwargs)
        
        logger.info(
            json.dumps(
                {
                    "id": request_id,
                    "message": "Request completed successfully"
                }, ensure_ascii=False, indent=4
            )
        )
        
        return response, request_id
    
    async def afetch(self, create_fn, **kwargs) -> Union[Tuple[dict, str], Tuple[AsyncGenerator[dict, None], str]]:
        """
        Asynchronously calls an external service and logs the request and response.

        Args:
            create_fn (Callable): The function to call the external service.
            **kwargs: Arbitrary keyword arguments passed to the external service call.

        Returns:
            Union[Tuple[dict, str], Tuple[AsyncGenerator[dict, None], str]]: A tuple containing the response 
                from the external service and the request ID.
        """
        request_id = f"openai-{str(uuid.uuid4())}"
        
        logger.info(
            json.dumps(
                {
                    "id": request_id,
                    "model": self.model_n,
                    "parameters": kwargs
                }, ensure_ascii=False, indent=4
            )
        )
        
        response = await create_fn(model=self.model_n, **kwargs)
        
        logger.info(
            json.dumps(
                {
                    "id": request_id,
                    "message": "Request completed successfully"
                }, ensure_ascii=False, indent=4
            )
        )
        
        return response, request_id