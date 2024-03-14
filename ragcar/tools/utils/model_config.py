import os
import json
import zipfile
from dataclasses import dataclass
from dataclasses_json import dataclass_json
from typing import Optional, Union

import gdown
from dotenv import load_dotenv


@dataclass_json
@dataclass
class TransformerConfig:
    """Configuration for transformer-based model fine-tuned for the STS Task"""
    model_n: str
    token: Optional[str] = None


@dataclass_json
@dataclass
class GoogledriveInfo:
    """Information for a transformer-based model fine-tuned for the STS Task, stored in Google Drive"""
    model_n: str
    model_url: str


@dataclass_json
@dataclass
class OpenaiConfig:
    """Configuration for using GPT-based models with OpenAI"""
    model_n: str
    api_key: str


@dataclass_json
@dataclass
class ClovastudioConfig:
    """Configuration for using HyperCLOVA-based models with NAVER Clova Studio"""
    model_n: str
    api_key: str
    app_key: str


@dataclass
class EncoderInfo:
    """Information for a transformer-based model fine-tuned for the STS Task"""
    src: str
    model: Union[str, dict]


@dataclass_json
@dataclass
class ElasticsearchConfig:
    """Configuration for Elasticsearch as a vector database"""
    encoder_key: EncoderInfo
    host_n: str
    http_auth: Optional[tuple] = None
    scheme: Optional[str] = "http"
    verify_certs: Optional[bool] = True
    timeout: Optional[int] = 5
    max_retries: Optional[int] = 2
    retry_on_timeout: Optional[bool] = True


def get_save_dir(save_dir: str = None) -> str:
    """
    Get a save directory for models or files.

    If a save directory is specified by the user, it returns that directory.
    Otherwise, it defaults to '~/.cache/gdown/models', creating the directory if it does not exist.
    
    Args:
        save_dir(str, optional): User-defined save directory; defaults to None.
    
    Returns:
        str: Set save directory.
    """
    if save_dir:  # If user wants to manually define save directory
        os.makedirs(save_dir, exist_ok=True)
        return save_dir

    # Default save directory: ~/.cache/gdown/models
    default_save_dir = os.path.expanduser('~/.cache/gdown/models')
    os.makedirs(default_save_dir, exist_ok=True)
    
    return default_save_dir

def download_from_url(info: GoogledriveInfo, root_dir: str) -> str:
    """
    Downloads a model from Google Drive and extracts it to the specified directory.

    Args:
        info (GoogledriveInfo): An instance of GoogledriveInfo containing the model's name
            and Google Drive URL. This class provides structured information to identify
            and locate the model to be downloaded.
        root_dir (str): The root directory where the model should be saved. If this directory
            does not exist, it will be created during the download process. The model's contents
            will be extracted into this directory.

    Returns:
        str: The path to the downloaded and extracted model. This path is constructed by 
             combining the root directory and the model's name, providing a direct reference
             to the extracted model's location.
    """
    model_path = os.path.join(root_dir, info.model_n)
    
    if not os.path.exists(model_path):
        gdown.download(url=info.model_url, output="{}.zip".format(info.model_n), quiet=False, fuzzy=True)
        working_dir = os.getcwd()
        source = os.path.join(working_dir, "{}.zip".format(info.model_n))
        source_file = zipfile.ZipFile(source)
        source_file.extractall(root_dir)
        os.remove(source)
        
    return TransformerConfig(model_n=model_path)

def download_or_load(
    src: str,
    model_info: Union[str, dict]
) -> TransformerConfig:
    """
    Downloads or loads a transformer-based model, depending on the source and model information provided.
    
    Args:
        src (str): Specifies the source of the model.
        model_info (Union[str, dict]): Information about the model to be downloaded or loaded. Structure depends on the value of `src`:
            - When `src` is 'model_name_or_path', `model_info` can be:
                - A string specifying the Hugging Face model name or the local path to the model.
                - A dictionary for private Hugging Face models, containing:
                    - 'model_n': The name of the model.
                    - 'token': The Hugging Face access token for private models.
            - When `src` is 'googledrive', `model_info` should be a dictionary with:
                - 'model_n': Name of the model, used for naming the downloaded file.
                - 'model_url': Google Drive URL of the source model file to download.
        save_dir (str, optional): Custom directory path for saving the model. If not provided, a default cache directory is used.
    
    Returns:
        TransformerConfig: An object containing the configuration for the model, including the model name (`model_n`)
                           and, if applicable, the Hugging Face access token (`token`) for private models.
    """
    if src == "model_name_or_path":  # Hugging Face model or local model
        if isinstance(model_info, dict):  # private models
            return TransformerConfig.from_json(json.dumps(model_info))
        
        return TransformerConfig(model_n=model_info)
    
    # Google Drive
    info = GoogledriveInfo.from_json(json.dumps(model_info))
    root_dir = get_save_dir()  # Default save directory
    return download_from_url(info, root_dir)

def get_openai_config(model_info: Union[str, dict]) -> OpenaiConfig:
    """
    Constructs an OpenaiConfig object from given model information or environment variables.

    Args:
        model_info (Union[str, dict]): A dictionary containing the model name ('model_n') and API key ('api_key'),
                                       or a string specifying the model name. If a string is provided, the API key
                                       is fetched from the 'OPENAI_API_KEY' environment variable.

    Returns:
        OpenaiConfig: Configuration object containing the model name and API key for OpenAI API access.
    """
    if isinstance(model_info, dict):
        return OpenaiConfig.from_json(json.dumps(model_info))
    else:
        load_dotenv()
        
        api_key = os.getenv('OPENAI_API_KEY')
        if api_key is None:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        
        return OpenaiConfig(model_n=model_info, api_key=api_key)

def get_clova_config(model_info: Union[str, dict]) -> ClovastudioConfig:
    """
    Constructs a ClovastudioConfig object from given model information or environment variables.

    Args:
        model_info (Union[str, dict]): A dictionary containing the model name ('model_n'), API key ('api_key'), and application key ('app_key'), 
                                       or a string specifying the model name. The 'api_key' corresponds to the original variable 'X-NCP-APIGW-API-KEY' and 
                                       'app_key' corresponds to 'X-NCP-CLOVASTUDIO-API-KEY'. If a string is provided, the API and app keys are fetched from 
                                       the environment variables, respectively.

    Returns:
        ClovastudioConfig: Configuration object containing the model name, API key, and app key for NAVER Clova Studio API access.
    """
    load_dotenv()
    
    if isinstance(model_info, dict):
        return ClovastudioConfig.from_json(json.dumps(model_info))
    else:
        clovastudio_api_key = os.getenv('X-NCP-APIGW-API-KEY')
        if clovastudio_api_key is None:
            raise ValueError("X-NCP-APIGW-API-KEY environment variable is not set")
        
        clovastudio_app_key = os.getenv('X-NCP-CLOVASTUDIO-API-KEY')
        if clovastudio_app_key is None:
            raise ValueError("X-NCP-CLOVASTUDIO-API-KEY environment variable is not set")
        
        return ClovastudioConfig(
            model_n=model_info, 
            api_key=clovastudio_api_key, 
            app_key=clovastudio_app_key
        )

def get_es_config(host_info: dict) -> ElasticsearchConfig:
    """
    Constructs an ElasticsearchConfig object from a given host information dictionary.

    Args:
        host_info (dict): A dictionary containing the Elasticsearch configuration details, including the encoder key
                          ('encoder_key'), host name ('host_n'), HTTP authentication credentials ('http_auth' - optional),
                          connection scheme ('scheme' - optional), certificate verification ('verify_certs' - optional),
                          connection timeout ('timeout' - optional), maximum retries on failure ('max_retries' - optional),
                          and whether to retry on timeout ('retry_on_timeout' - optional).

    Returns:
        ElasticsearchConfig: Configuration object containing the connection information for Elasticsearch.
    """
    return ElasticsearchConfig.from_json(json.dumps(host_info))