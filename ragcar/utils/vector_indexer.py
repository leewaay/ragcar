import logging
import uuid
import time
from functools import reduce
from typing import Optional, Union, Any, Dict, List

from tqdm import tqdm
import numpy as np
from elasticsearch import Elasticsearch, helpers
from elasticsearch.exceptions import ConnectionTimeout, ConnectionError, NotFoundError

from ragcar.tools.utils.base import RagcarToolBase


logger = logging.getLogger(__name__)


class ElasticsearchVectorIndexer:
    """Class for indexing sentence embeddings in Elasticsearch using Sentence Transformers."""

    def __init__(
        self, 
        host: str, 
        http_auth=None, 
        scheme="http", 
        verify_certs=False, 
        timeout=5, 
        max_retries=2, 
        retry_on_timeout=True
    ):
        """
        Initializes an Elasticsearch client with specified configurations.

        Args:
            host (str): The host address of the Elasticsearch server.
            http_auth (optional): Authentication credentials (username, password) if required.
            scheme (str, optional): The protocol scheme, 'http' or 'https'. Defaults to 'http'.
            verify_certs (bool, optional): Whether to verify SSL certificates. Defaults to False.
            timeout (int, optional): Timeout for performing the operations in seconds. Defaults to 5.
            max_retries (int, optional): Maximum number of retries before an operation fails. Defaults to 2.
            retry_on_timeout (bool, optional): Whether to retry on timeout. Defaults to True.

        Raises:
            ValueError: If an invalid host is provided.
            ConnectionError: If unable to connect to the Elasticsearch instance.
        """
        self.client = Elasticsearch(
            host,
            http_auth=http_auth,
            scheme=scheme,
            verify_certs=verify_certs,
            timeout=timeout,
            max_retries=max_retries,
            retry_on_timeout=retry_on_timeout
        )
        
        if self.client.ping():
            logger.info("Connected to Elasticsearch.")
        else:
            logger.error("Could not connect to Elasticsearch.")

    def get_data(
        self, 
        index_name: str, 
        field_to_embed: Union[str, List[str]], 
        query_size: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        Retrieves documents from an Elasticsearch index using the Scroll API for efficient large-scale data retrieval.

        Args:
            index_name (str): The name of the Elasticsearch index to fetch documents from.
            field_to_embed (Union[str, List[str]]): The path to the text field(s) in the documents for embedding purposes.
            query_size (int, optional): The number of documents to retrieve per scroll batch. Defaults to 1000.

        Returns:
            List[Dict[str, Any]]: A list of documents retrieved from Elasticsearch.

        Raises:
            ValueError: If an invalid field path is provided.
        """
        logger.info(f"Fetching documents from index: {index_name} using Scroll API")

        query = {
            "size": query_size,
            "track_total_hits": True, 
            "_source": ".".join(field_to_embed),
            "query": {
                "match_all": {}
            }
        }

        # Initialize the scroll
        res = self.client.search(index=index_name, body=query, scroll='1m')  # '1m' means each scroll request lasts 1 minute
        total = res['hits']['total']['value']
        scroll_id = res['_scroll_id']
        
        total_docs = []

        # Using tqdm for progress bar
        with tqdm(total=total, desc="Fetching from Elasticsearch", unit="docs") as pbar:
            while len(res['hits']['hits']):
                total_docs.extend(res['hits']['hits'])
                pbar.update(len(res['hits']['hits']))  # Update the progress bar
                
                # Scroll to next batch of data
                res = self.client.scroll(scroll_id=scroll_id, scroll='1m')
                scroll_id = res['_scroll_id']

        logger.info(f"Retrieved {len(total_docs)} documents from index: {index_name}")

        return total_docs
    
    def embed_data(
        self, 
        embedder: RagcarToolBase,
        metadata: List[Dict[str, Any]], 
        field_to_embed: Union[str, List[str]] = 'text', 
    ) -> List[List[float]]:
        """
        Generates embeddings for specified text data using the given embedding model.

        Args:
            embedder (RagcarToolBase): The embedding model instance to use for generating embeddings.
            metadata (List[Dict[str, Any]]): A list of dictionaries containing the data for which embeddings will be generated.
            field_to_embed (Union[str, List[str]]): The key(s) in the metadata dict pointing to the text data for embedding.

        Returns:
            List[List[float]]: A list containing the generated embeddings, with each embedding being a list of floats.

        Raises:
            ValueError: If the provided metadata does not match the expected format or if there's an issue with accessing the data to embed.
        """
        sentences = []
        for item in metadata:
            item = item.get('_source', item)
            
            if isinstance(metadata[0], dict):
                if isinstance(field_to_embed, str):
                    sentences.append(item[field_to_embed])
                else:
                    sentences.append(reduce(lambda d, key: d[key], field_to_embed, item))
            else:
                raise ValueError("Provided metadata is not in the expected Elasticsearch format.")
            
        logger.info(f"Generating embeddings for {len(sentences)} sentences")
        embeddings = embedder(sentences)
        logger.info("Embeddings generated successfully")
            
        return embeddings
    
    def _create_index_with_mapping(
        self, 
        index_name: str, 
        vector_field: str , 
        dims: int
    ) -> None:
        """
        Creates an Elasticsearch index with a specific mapping for embedding vectors.

        Args:
            index_name (str): The name of the Elasticsearch index to be created.
            vector_field (str): The name of the field where embedding vectors will be stored.
            dims (int): The dimensions of the embedding vectors.

        Returns:
            None
        """
        logger.info(f"Creating index with mapping for index: {index_name}")
        if not self.client.indices.exists(index=index_name):
            mapping = {
                "mappings": {
                    "properties": {
                        vector_field: {
                            "type": "dense_vector",
                            "dims": dims
                        }
                    }
                }
            }
            self.client.indices.create(index=index_name, body=mapping)
            logger.info(f"Created index: {index_name} successfully")
    
    def backup_index(
        self, 
        index_name: str, 
        backup_name: Optional[str] = None,
        overwrite: bool = True
    ) -> None:
        """
        Creates a backup of an Elasticsearch index by reindexing its data into a new index.

        Args:
            index_name (str): The name of the index to backup.
            backup_name (Optional[str]): The name of the backup index. If not provided, a default name is generated.
            overwrite (bool): Whether to overwrite the backup index if it exists. Defaults to True.

        Returns:
            None

        Raises:
            ConnectionTimeout: If the connection to Elasticsearch times out during the backup process.
            ConnectionError: If there is a failure in connecting to Elasticsearch.
            NotFoundError: If the original index does not exist.
            Exception: For any other unexpected errors during the operation.
        """
        if backup_name is None:
            backup_name = f"{index_name}_backup_{str(uuid.uuid4())[:8]}"

        if not self.client.indices.exists(index=index_name):
            logger.error(f"Index {index_name} does not exist. Cannot create backup.")
            return
        
        if self.client.indices.exists(index=backup_name):
            if overwrite:
                self.client.indices.delete(index=backup_name)
                logger.info(f"Deleted existing index {backup_name}.")
            else:
                logger.warning(f"Index {backup_name} already exists. Use 'overwrite=True' to overwrite the existing index.")
                return

        body = {
            "source": {
                "index": index_name
            },
            "dest": {
                "index": backup_name
            }
        }

        try:
            # Try to execute reindex and get task_id
            reindex_response = self.client.reindex(body=body, wait_for_completion=False)
            task_id = reindex_response["task"]
            
            # Check task completion
            task = self.client.tasks.get(task_id=task_id)
            while not task['completed']:
                time.sleep(5)  # Wait for 5 seconds before checking again
                task = self.client.tasks.get(task_id=task_id)
            
            if task['response']:  # Replace this condition with the appropriate way to check task success
                logger.info(f"Backup for index {index_name} created as {backup_name}")
            else:
                logger.error("Backup task failed.")
        except ConnectionTimeout:
            logger.warning("Connection timed out. The task is still running. Please check manually.")
        except ConnectionError:
            logger.error("Failed to connect to Elasticsearch. Please check your connection.")
        except NotFoundError:
            logger.error(f"Index {index_name} not found.")
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")

    def restore_from_backup(
        self, 
        backup_name: str, 
        index_name: str, 
        overwrite: bool = True
    ) -> None:
        """
        Restores data from a backup index to a target index in Elasticsearch.

        Args:
            backup_name (str): The name of the backup index from which data will be restored.
            index_name (str): The name of the target index to restore data into.
            overwrite (bool): If True, the target index will be deleted and recreated if it already exists. Defaults to True.

        Returns:
            None

        Raises:
            ConnectionTimeout: If the connection to Elasticsearch times out during the restore operation.
            ConnectionError: If there is a failure in connecting to Elasticsearch.
            NotFoundError: If the specified backup index does not exist.
            Exception: For any other unexpected errors during the operation.
        """
        if not self.client.indices.exists(index=backup_name):
            logger.error(f"Backup index {backup_name} does not exist. Cannot restore data.")
            return

        if self.client.indices.exists(index=index_name):
            if overwrite:
                self.client.indices.delete(index=index_name)
                logger.info(f"Deleted existing index {index_name}.")
            else:
                logger.warning(f"Index {index_name} already exists. Use 'overwrite=True' to overwrite the existing index.")
                return
        
        body = {
            "source": {
                "index": backup_name
            },
            "dest": {
                "index": index_name
            }
        }

        try:
            # Try to execute reindex and get task_id
            reindex_response = self.client.reindex(body=body, wait_for_completion=False)
            task_id = reindex_response["task"]
            
            # Check task completion
            task = self.client.tasks.get(task_id=task_id)
            while not task['completed']:
                time.sleep(5)  # Wait for 5 seconds before checking again
                task = self.client.tasks.get(task_id=task_id)
            
            if task['response']:  # Replace this condition with the appropriate way to check task success
                logger.info(f"Data restored to index {index_name} from backup {backup_name}")
            else:
                logger.error("Restore task failed.")
        except ConnectionTimeout:
            logger.warning("Connection timed out. The task is still running. Please check manually.")
        except ConnectionError:
            logger.error("Failed to connect to Elasticsearch. Please check your connection.")
        except NotFoundError:
            logger.error(f"Index {backup_name} not found.")
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
    
    def delete_data(self, index_name: str, query: Optional[Dict[str, Any]] = None) -> None:
        """
        Deletes documents from an Elasticsearch index based on a provided query or all documents if no query is provided.

        Args:
            index_name (str): The name of the Elasticsearch index from which documents will be deleted.
            query (Optional[Dict[str, Any]]): A dictionary representing the Elasticsearch query to specify which documents to delete. 
                If None, all documents in the index will be deleted.

        Returns:
            None

        Raises:
            Warning: If the specified index does not exist or no documents matched the query.
        """
        if not self.client.indices.exists(index=index_name):
            logger.warning(f"Index {index_name} does not exist.")
            return

        if query is None:
            # If no query is provided, delete all documents from the index.
            query = {
                "query": {
                    "match_all": {}
                }
            }

        # Use the delete by query API of Elasticsearch
        response = self.client.delete_by_query(index=index_name, body=query)
        
        # Log results
        if response.get('deleted', 0) > 0:
            logger.info(f"Deleted {response['deleted']} documents from index: {index_name}")
        else:
            logger.warning(f"No documents matched the query in index: {index_name}")
    
    def index_data(
        self, 
        index_name: str, 
        metadata: Union[List[str], List[Dict[str, Any]]], 
        embeddings: List[List[float]], 
        decimal_precision: int = 5, 
        text_field: str = 'text', 
        vector_field: str = 'vector', 
        field_to_index: Optional[Union[str, List[str]]] = None
    ) -> None:
        """
        Indexes or updates documents in an Elasticsearch index with provided embeddings and metadata.

        Args:
            index_name (str): Name of the Elasticsearch index to target.
            metadata (Union[List[str], List[Dict[str, Any]]]): Metadata associated with embeddings, can be a list of strings or dictionaries.
            embeddings (List[List[float]]): Numeric vectors representing embeddings to be indexed.
            decimal_precision (int, optional): Number of decimal places to keep for embedding vectors. Defaults to 5.
            text_field (str): Field name for the text data within the metadata. Defaults to 'text'.
            vector_field (str): Field name where embeddings will be stored. Defaults to 'vector'.
            field_to_index (Optional[Union[str, List[str]]], optional): Specific field(s) to update in the document, 
                if None, updates or indexes the whole document.

        Returns:
            None

        Raises:
            ConnectionTimeout: If the connection to Elasticsearch times out during indexing.
            ConnectionError: If there is a failure in connecting to Elasticsearch.
            Exception: For any other unexpected errors during the operation.
        """
        logger.info(f"Indexing data into index: {index_name}")
        actions = []

        # Determine operation type based on index existence
        op_type = "update" if self.client.indices.exists(index=index_name) else "index"
        if op_type == "index":
            self._create_index_with_mapping(index_name, vector_field, len(embeddings[0]))

        for i, item in enumerate(metadata):
            item = item.get('_source', item)
            doc_id = item.get('_id', str(uuid.uuid4()))
            
            action_data = {
                vector_field: np.round(embeddings[i].tolist(), decimal_precision).tolist()
            }
            
            if field_to_index:
                if isinstance(field_to_index, list):
                    for field in field_to_index:
                        action_data[field] = item.get(field)
                else:
                    action_data[field_to_index] = item.get(field_to_index)
            else:
                if isinstance(item, dict):
                    action_data[text_field] = item[text_field]
                else: # If it's just a list of strings
                    action_data[text_field] = item
            
            action = {
                "_op_type": op_type,
                "_index": index_name,
                "_id": doc_id
            }
            
            if op_type == "index":
                action["_source"] = action_data
            else:
                action["doc"] = action_data
                action["doc_as_upsert"] = True
                
            actions.append(action)

        try:
            helpers.bulk(self.client, actions)
            logger.info(f"Indexed data into index: {index_name} successfully")
        except ConnectionTimeout:
            logger.warning("Connection timed out. Some documents may not have been indexed. Please check manually.")
        except ConnectionError:
            logger.error("Failed to connect to Elasticsearch. Please check your connection.")
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")