import logging
from typing import Optional, Union, List

import torch
import asyncio
from sentence_transformers import util

from ragcar.tools.utils.base import RagcarFactoryBase, RagcarBiencoderBase, RagcarAsyncBiencoderBase
from ragcar.tools.utils.model_config import get_es_config


logger = logging.getLogger(__name__)


class RagcarSemanticSearchFactory(RagcarFactoryBase):
    """
    Semantic search using Elasticsearch or Bi-Encoder, returns top relevant matches.
    
    Examples:
        >>> retriever = Ragcar(tool="semantic_search", src="model_name_or_path")
        >>> retriever(
        ... ["`예술인 기회소득` 청신호… `경기국제공항` 난기류", "한국vs크로아티아 경기 주심은 왜 눈물을 닦았나[VNL]"], 
        ... ['고준호 경기도의원 "파주 5000번 버스 경기 공공버스로 전환"', "강호 독일 상대로 8경기 만에 마침내 1세트 따낸 女배구"]
        ... )
        [
            [
                {'corpus_id': 0, 'score': 0.2568105459213257}, 
                {'corpus_id': 1, 'score': 0.07774988561868668}
            ], 
            [
                {'corpus_id': 1, 'score': 0.22476400434970856}, 
                {'corpus_id': 0, 'score': 0.16569660604000092}
            ]
        ]
    """
    
    def __init__(
        self, 
        tool: str, 
        src: str, 
        model: Optional[Union[str, dict]],
        device: Optional[str] = None,
        use_async: bool = False
    ):
        super().__init__(tool, src, model, device)
        self.use_async = use_async
    
    @staticmethod
    def get_available_srcs() -> List[str]:
        return [
            "model_name_or_path",
            "googledrive",
            "openai",
            "clova",
            "elasticsearch",
        ]
    
    @staticmethod
    def get_available_models() -> dict:
        return {
            "model_name_or_path": [
                "leewaay/kpf-bert-base-klueNLI-klueSTS-MSL512",
                "leewaay/klue-roberta-base-klueNLI-klueSTS-MSL512",
                "leewaay/klue-roberta-large-klueNLI-klueSTS-MSL512",
                "MODELS_SUPPORTED(https://huggingface.co/models?pipeline_tag=sentence-similarity)"
            ],
            "googledrive": [
                "YOUR_MODEL"
            ],
            "openai": [
                "text-embedding-3-large",
                "text-embedding-3-small",
                "text-embedding-ada-002",
                "MODELS_SUPPORTED(https://platform.openai.com/docs/models)"
            ],
            "clova": [
                "YOUR_MODEL(https://www.ncloud.com/product/aiService/clovaStudio)"
            ],
            "elasticsearch": [
                "YOUR_MODEL"
            ],
        }
    
    def load(self, device: str) -> object:
        """
        Loads and initializes the model based on configuration settings.

        Args:
            device (str): The computing device ('cpu' or 'cuda') where the model will be loaded.

        Returns:
            object: The initialized model ready for use.
        """
        from ragcar.tools.sentence_embedding import RagcarSentenceFactory
        
        if self.config.src == "elasticsearch":
            es_conf = get_es_config(self.config.model_info)
            
            encoder = RagcarSentenceFactory(
                tool="sentence_embedding",
                src=es_conf.encoder_key.src,
                model=es_conf.encoder_key.model
            ).load(device)
            
            if self.use_async:
                from elasticsearch import AsyncElasticsearch
            
                es = AsyncElasticsearch(
                    es_conf.host_n, 
                    http_auth=es_conf.http_auth, 
                    scheme=es_conf.scheme,
                    verify_certs=es_conf.verify_certs,
                    timeout=es_conf.timeout, 
                    max_retries=es_conf.max_retries, 
                    retry_on_timeout=es_conf.retry_on_timeout, 
                )
                
                return RagcarAsyncElasticSemanticSearch(
                    self.config, 
                    encoder=encoder, 
                    db=es
                )
            else:
                from elasticsearch import Elasticsearch
            
                es = Elasticsearch(
                    es_conf.host_n, 
                    http_auth=es_conf.http_auth, 
                    scheme=es_conf.scheme,
                    verify_certs=es_conf.verify_certs,
                    timeout=es_conf.timeout, 
                    max_retries=es_conf.max_retries, 
                    retry_on_timeout=es_conf.retry_on_timeout, 
                )
                
                return RagcarElasticSemanticSearch(
                    self.config, 
                    encoder=encoder, 
                    db=es
                )
        
        encoder = RagcarSentenceFactory(
            tool="sentence_embedding",
            src=self.config.src,
            model=self.config.model_info,
        ).load(device)
        
        return RagcarSemanticSearch(
            self.config, 
            encoder=encoder
        )
        

class RagcarSemanticSearch(RagcarBiencoderBase):
    
    def __init__(
        self, 
        config,
        encoder
    ):
        super().__init__(config)
        self._encoder = encoder
    
    def _retrieve_relevant_entries(
        self, 
        query_embedding: torch.tensor, 
        corpus_embeddings: torch.tensor, 
        top_k: int,
        min_score: float
    ):
        """
        Retrieves the most relevant entries from a corpus of embeddings based on a query embedding,
        considering only entries with a score above a minimum threshold.
        
        Args:
            query_embedding (torch.tensor): The embedding of the query to search for.
            corpus_embeddings (torch.tensor): The embeddings of the corpus to search within.
            top_k (int): The number of top relevant results to return.
            min_score (float): The minimum score a result must have to be considered relevant.
        
        Returns:
            list: A list of hits, each a list of dictionaries with 'score' and potentially other metadata,
                filtered by the minimum score threshold.
        """
        query_embedding = query_embedding.cuda()  # Move query embedding to GPU for faster computation
        corpus_embeddings = corpus_embeddings.cuda()  # Move corpus embeddings to GPU
        
        hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=top_k)  # Perform semantic search
        
        if min_score > 0:  # Filter hits by minimum score if specified
            hits = [[entry for entry in hit if entry['score'] >= min_score] for hit in hits]
        
        return hits
    
    def predict(
        self,
        query,
        corpus,
        **kwargs,
    ) -> list:
        """
        Predicts the most relevant entries from a corpus based on a query using sentence embeddings.
        
        Args:
            query (str, list, or torch.Tensor): The query to be matched. Can be a single sentence, a list of sentences,
                                                or a precomputed tensor of embeddings.
            corpus (str, list, or torch.Tensor): The corpus to be searched. Can be a single sentence, a list of sentences,
                                                a list of paragraphs/documents, or a precomputed tensor of embeddings.
            top_k (int, optional): The number of top matching entries to retrieve. Defaults to 5.
            min_score (float, optional): The minimum score threshold for an entry to be considered relevant. Defaults to 0.
        
        Returns:
            List[dict]: A list of dictionaries representing the relevant entries. Each entry contains metadata such as
                        the similarity score.
        """
        top_k = kwargs.get("top_k", 5)
        min_score = kwargs.get("min_score", 0)
        
        if isinstance(query, torch.Tensor):
            query_embeddings = query
        else:
            query_embeddings = self._encoder.predict(query, convert_to_tensor=True)
        
        if isinstance(corpus, torch.Tensor):
            corpus_embeddings = corpus
        else:
            corpus_embeddings = self._encoder.predict(corpus, convert_to_tensor=True)
        
        outputs = self._retrieve_relevant_entries(query_embeddings, corpus_embeddings, top_k, min_score)
        
        return outputs


class RagcarElasticSemanticSearch(RagcarBiencoderBase):
    
    def __init__(
        self, 
        config,
        encoder,
        db,
    ):
        super().__init__(config)
        self._encoder = encoder
        self._client = db
    
    def _get_embedding_from_es(self, doc_id: str, vector_field:str, index_n:str) -> Optional[torch.Tensor]:
        """
        Retrieves the embedding vector from Elasticsearch for a given document ID and vector field.

        Args:
            doc_id (str): The unique identifier of the document in the Elasticsearch index.
            vector_field (str): The name of the field in the document that contains the embedding vector.
            index_n (str): The name of the Elasticsearch index where the document is stored.
        
        Returns:
            torch.Tensor: The embedding vector as a PyTorch tensor, or None if the document or vector field does not exist.
        
        Raises:
            ValueError: If no document is found for the given ID, if the document does not have the specified vector field,
                        or if an error occurs during the retrieval process.
        """
        try:
            response = self._client.search(index=index_n, body={"query": {"term": {"_id": doc_id}}})
            hits = response['hits']['hits']
            
            if not hits:
                raise ValueError(f"No documents found for id {doc_id} in index {self.index_n}")
            
            result = hits[0]['_source']
            vector_value = result.get(vector_field)
            
            if vector_value is None:
                raise ValueError(f"No vector found for field {vector_field} in document {doc_id}")
            
            return torch.Tensor(vector_value)
        except Exception as e:
            raise ValueError(f"Error occurred while retrieving document {doc_id} from Elasticsearch: {str(e)}")
    
    def _retrieve_relevant_entries_from_es(
        self, 
        query_embedding: torch.Tensor,
        vector_field: str,
        index_n: str,
        knn: bool,
        top_k: int,
        min_score: float,
        source_fields: list,
        filter: list,
        must_not: list
    ) -> list:
        """
        Retrieves relevant documents from Elasticsearch based on cosine similarity between the query embedding and document vectors using either k-NN or script score query.
        
        Args:
            query_embedding (torch.Tensor): The query embedding vector.
            vector_field (str): The name of the document field that contains the embedding vector to compare against.
            index_n (str): The name of the Elasticsearch index to search in.
            knn (bool): If True, use k-NN search. If False, use script score query.
            top_k (int): The number of top documents to retrieve.
            min_score (float): The minimum score documents must have to be included in the results.
            source_fields (list): The list of source fields to include in the returned documents.
            filter (list): The filter criteria for the query.
            must_not (list): The criteria that documents must not match.
        
        Returns:
            list: A list of documents that match the query criteria, each as a dictionary, sorted by their relevance based on cosine similarity.
        """
        body = {
            "from": 0,
            "size": top_k,
            "min_score": min_score,
            "_source": source_fields,
        }
        
        if knn:
            body["query"] = {
                "bool": {
                    "filter": filter,
                    "must_not": must_not,
                    "must": [
                        {
                            "elastiknn_nearest_neighbors": {
                                "field": vector_field,
                                "model": "lsh",
                                "similarity": "cosine",
                                "candidates": top_k*10,
                                "vec": {
                                    "values": query_embedding.cpu().numpy()
                                }
                            }
                        }
                    ]
                }
            }
        else:
            body["query"] = {
                "script_score": {
                    "query" : {
                        "bool": {
                            "filter": filter,
                            "must_not": must_not,
                        }
                    },
                    "script": {
                        "source": f"cosineSimilarity(params.query_vector, '{vector_field}')+1.0",
                        "params": {
                            "query_vector": query_embedding.cpu().numpy()
                        }
                    }
                }
            }
        
        response = self._client.search(index=index_n, body=body, request_timeout=5)
        return response['hits']['hits']
    
    def predict(
        self,
        query,
        corpus,
        **kwargs,
    ) -> list:
        """
        Conducts sentence embedding and retrieves relevant entries from Elasticsearch based on the provided query.
        
        Args:
            query (str or list or Tensor): The input query or list of queries, or their embeddings.
            corpus (str): The target corpus or the name of the Elasticsearch dense_vector field.
            index_n (str): The name of the Elasticsearch index to search.
            top_k (int, optional): The number of top matching entries to retrieve (default: 5).
            min_score (float, optional): The minimum similarity score for entries to be considered relevant (default: 0).
            doc_id (bool, optional): If True, treats the input query as an Elasticsearch document ID for direct retrieval (default: False).
            knn (bool, optional): If True, uses the k-nearest neighbors algorithm for retrieval (default: False).
            source_fields (list, optional): Specifies which fields to include in the returned documents (default: []).
            filter (list, optional): Specifies filter criteria for the search, each filter as a dictionary (default: []).
            must_not (list, optional): Specifies conditions for documents to be excluded, each condition as a dictionary (default: []).

        Returns:
            List[dict]: A list of dictionaries, each representing a relevant entry retrieved from Elasticsearch.
        """
        index_n = kwargs.get("index_n")
        top_k = kwargs.get("top_k", 5)
        min_score = kwargs.get("min_score", 0) + 1.0
        doc_id = kwargs.get("doc_id", False)
        knn = kwargs.get("knn", False)
        source_fields = kwargs.get("source_fields", [])
        filter = [{"exists": {"field": corpus}}]
        filter.extend(kwargs.get("filter", []))
            
        if doc_id:
            if isinstance(query, str):
                query_embeddings = self._get_embedding_from_es(query, corpus, index_n)
            elif isinstance(query, list):
                query_embeddings = torch.stack([self._get_embedding_from_es(q, corpus, index_n) for q in query])
        else:
            if isinstance(query, torch.Tensor):
                query_embeddings = query
            else:
                query_embeddings = self._encoder.predict(query, convert_to_tensor=True)
        
        if query_embeddings.dim() > 1:
            outputs = []
            for i, query_embedding in enumerate(query_embeddings):
                must_not = kwargs.get("must_not", [])
                
                if doc_id:
                    must_not.append({"match": {"_id": query[i]}})
                
                outputs.append(
                    self._retrieve_relevant_entries_from_es(
                        query_embedding=query_embedding,
                        vector_field=corpus,
                        index_n=index_n,
                        knn=knn,
                        top_k=top_k,
                        min_score=min_score,
                        source_fields=source_fields,
                        filter=filter,
                        must_not=must_not
                    )
                )
        else:
            must_not = kwargs.get("must_not", [])

            if doc_id:
                must_not.append({"match": {"_id": query}})
                
            outputs = self._retrieve_relevant_entries_from_es(
                query_embedding=query_embeddings,
                vector_field=corpus,
                index_n=index_n,
                knn=knn,
                top_k=top_k,
                min_score=min_score,
                source_fields=source_fields,
                filter=filter,
                must_not=must_not
            )
        
        return outputs


class RagcarAsyncElasticSemanticSearch(RagcarAsyncBiencoderBase):
    
    def __init__(
        self, 
        config,
        encoder,
        db,
    ):
        super().__init__(config)
        self._encoder = encoder
        self._client = db
    
    async def _get_embedding_from_es(self, doc_id: str, vector_field: str, index_n: str) -> Optional[torch.Tensor]:
        """
        Asynchronously retrieves the embedding vector from Elasticsearch for a given document ID and vector field.

        Args:
            doc_id (str): The unique identifier of the document in the Elasticsearch index.
            vector_field (str): The name of the field in the document that contains the embedding vector.
            index_n (str): The name of the Elasticsearch index where the document is stored.
        
        Returns:
            Optional[torch.Tensor]: The embedding vector as a PyTorch tensor, or None if the document or vector field does not exist.
        
        Raises:
            ValueError: If no document is found for the given ID, if the document does not have the specified vector field,
                        or if an error occurs during the retrieval process.
        """
        try:
            response = await self._client.search(index=index_n, body={"query": {"term": {"_id": doc_id}}})
            hits = response['hits']['hits']
            
            if not hits:
                raise ValueError(f"No documents found for id {doc_id} in index {index_n}")
            
            result = hits[0]['_source']
            vector_value = result.get(vector_field)
            
            if vector_value is None:
                raise ValueError(f"No vector found for field {vector_field} in document {doc_id}")
            
            return torch.Tensor(vector_value)
        except Exception as e:
            raise ValueError(f"Error occurred while retrieving document {doc_id} from Elasticsearch: {str(e)}")
    
    async def _retrieve_relevant_entries_from_es(
        self, 
        query_embedding: torch.Tensor,
        vector_field: str,
        index_n: str,
        knn: bool,
        top_k: int,
        min_score: float,
        source_fields: list,
        filter: list,
        must_not: list
    ) -> list:
        """
        Asynchronously retrieves relevant documents from Elasticsearch based on cosine similarity between the query embedding and document vectors using either k-NN or script score query.
        
        Args:
            query_embedding (torch.Tensor): The query embedding vector.
            vector_field (str): The name of the document field that contains the embedding vector to compare against.
            index_n (str): The name of the Elasticsearch index to search in.
            knn (bool): If True, use k-NN search. If False, use script score query.
            top_k (int): The number of top documents to retrieve.
            min_score (float): The minimum score documents must have to be included in the results.
            source_fields (list): The list of source fields to include in the returned documents.
            filter (list): The filter criteria for the query.
            must_not (list): The criteria that documents must not match.
        
        Returns:
            list: A list of documents that match the query criteria, each as a dictionary, sorted by their relevance based on cosine similarity.
        """
        body = {
            "from": 0,
            "size": top_k,
            "min_score": min_score,
            "_source": source_fields,
        }
        
        if knn:
            body["query"] = {
                "bool": {
                    "filter": filter,
                    "must_not": must_not,
                    "must": [
                        {
                            "elastiknn_nearest_neighbors": {
                                "field": vector_field,
                                "model": "lsh",
                                "similarity": "cosine",
                                "candidates": top_k*10,
                                "vec": {
                                    "values": query_embedding.cpu().numpy()
                                }
                            }
                        }
                    ]
                }
            }
        else:
            body["query"] = {
                "script_score": {
                    "query" : {
                        "bool": {
                            "filter": filter,
                            "must_not": must_not,
                        }
                    },
                    "script": {
                        "source": f"cosineSimilarity(params.query_vector, '{vector_field}')+1.0",
                        "params": {
                            "query_vector": query_embedding.cpu().numpy()
                        }
                    }
                }
            }
        
        response = await self._client.search(index=index_n, body=body, request_timeout=5)
        return response['hits']['hits']
    
    async def predict(
        self,
        query,
        corpus,
        **kwargs,
    ) -> list:
        """
        Asynchronously conducts sentence embedding and retrieves relevant entries from Elasticsearch based on the provided query.
        
        Args:
            query (str or list or Tensor): The input query or list of queries, or their embeddings.
            corpus (str): The target corpus or the name of the Elasticsearch dense_vector field.
            index_n (str): The name of the Elasticsearch index to search.
            top_k (int, optional): The number of top matching entries to retrieve (default: 5).
            min_score (float, optional): The minimum similarity score for entries to be considered relevant (default: 0).
            doc_id (bool, optional): If True, treats the input query as an Elasticsearch document ID for direct retrieval (default: False).
            knn (bool, optional): If True, uses the k-nearest neighbors algorithm for retrieval (default: False).
            source_fields (list, optional): Specifies which fields to include in the returned documents (default: []).
            filter (list, optional): Specifies filter criteria for the search, each filter as a dictionary (default: []).
            must_not (list, optional): Specifies conditions for documents to be excluded, each condition as a dictionary (default: []).

        Returns:
            List[dict]: A list of dictionaries, each representing a relevant entry retrieved from Elasticsearch.
        """
        index_n = kwargs.get("index_n")
        top_k = kwargs.get("top_k", 5)
        min_score = kwargs.get("min_score", 0)
        min_score = min_score + 1.0
        doc_id = kwargs.get("doc_id", False)
        knn = kwargs.get("knn", False)
        source_fields = kwargs.get("source_fields", [])
        filter = [{"exists": {"field": corpus}}]
        filter.extend(kwargs.get("filter", []))
            
        if doc_id:
            if isinstance(query, str):
                query_embeddings = await self._get_embedding_from_es(query, corpus, index_n)
            elif isinstance(query, list):
                tasks = [self._get_embedding_from_es(q, corpus, index_n) for q in query]
                query_embeddings = torch.stack(await asyncio.gather(*tasks))
        else:
            if isinstance(query, torch.Tensor):
                query_embeddings = query
            else:
                query_embeddings = self._encoder.predict(query, convert_to_tensor=True)
        
        if query_embeddings.dim() > 1:
            tasks = []
            for i, query_embedding in enumerate(query_embeddings):
                must_not = kwargs.get("must_not", [])
                
                if doc_id:
                    must_not.append({"match": {"_id": query[i]}})
                
                tasks.append(
                    self._retrieve_relevant_entries_from_es(
                        query_embedding=query_embedding,
                        vector_field=corpus,
                        index_n=index_n,
                        knn=knn,
                        top_k=top_k,
                        min_score=min_score,
                        source_fields=source_fields,
                        filter=filter,
                        must_not=must_not
                    )
                )
            
            outputs = await asyncio.gather(*tasks)
        else:
            must_not = kwargs.get("must_not", [])

            if doc_id:
                must_not.append({"match": {"_id": query}})
                
            outputs = await self._retrieve_relevant_entries_from_es(
                query_embedding=query_embeddings,
                vector_field=corpus,
                index_n=index_n,
                knn=knn,
                top_k=top_k,
                min_score=min_score,
                source_fields=source_fields,
                filter=filter,
                must_not=must_not
            )
        
        return outputs