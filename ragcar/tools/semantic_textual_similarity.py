from abc import abstractmethod
from typing import Optional, Union, List

import torch
from sentence_transformers import util

from ragcar.tools.utils.base import RagcarFactoryBase, RagcarBiencoderBase
from ragcar.tools.utils.model_config import download_or_load


class RagcarStsFactory(RagcarFactoryBase):
    """
    Measure semantic textual similarity between sentences
    
    Examples:
        >>> sts = Ragcar(tool="sentence_similarity", src="model_name_or_path")
        >>> sts(["흡연", "외도"], ["바람피면 죽는다", "담배피면 죽는다", "라이터", "간통"], sorted=True)
        [
            [tensor(0.5466), 1, 3],
            [tensor(0.5263), 0, 1],
            [tensor(0.4776), 0, 2],
            [tensor(0.4251), 0, 3],
            [tensor(0.2208), 1, 2],
            [tensor(0.1471), 0, 0],
            [tensor(0.1330), 1, 1],
            [tensor(0.1021), 1, 0]
        ]
        >>> sts("저는 경기도에 살고 있어요", ["경기도에 거주하고 있습니다", "경기 시작 10분 전 입니다.", "서울 근교에 집이 있어요"])
        tensor([[0.9404, 0.1166, 0.4393]])
        
        >>> sts = Ragcar(tool="sentence_similarity", src="model_name_or_path", model="leewaay/klue-roberta-large-klueSTS-cross", encoder_type="cross")
        >>> sts("저는 경기도에 살고 있어요", ["경기도에 거주하고 있습니다", "경기 시작 10분 전 입니다.", "서울 근교에 집이 있어요"])
        tensor([[0.9115, 0.0061, 0.1801]])
    """
    
    def __init__(
        self, 
        tool: str, 
        src: str, 
        model: Optional[Union[str, dict]],
        device: Optional[str] = None,
        encoder_type: str = 'bi'
    ):
        super().__init__(tool, src, model, device)
        self.encoder_type = encoder_type
    
    @staticmethod
    def get_available_srcs():
        return [
            "model_name_or_path",
            "googledrive",
            "openai",
            "clova",
        ]
    
    @staticmethod
    def get_available_models():
        return {
            "model_name_or_path": [
                "leewaay/kpf-bert-base-klueNLI-klueSTS-MSL512",
                "leewaay/klue-roberta-base-klueNLI-klueSTS-MSL512",
                "leewaay/klue-roberta-large-klueNLI-klueSTS-MSL512",
                "leewaay/kpf-bert-base-klueSTS-cross",
                "leewaay/klue-roberta-large-klueSTS-cross",
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
            ]
        }
    
    def load(self, device: str):
        """
        Loads and initializes the model based on configuration settings.

        Args:
            device (str): The computing device ('cpu' or 'cuda') where the model will be loaded.

        Returns:
            object: The initialized model ready for use.
        """
        if any(src in self.config.src for src in ['model_name_or_path', 'googledrive']) and self.encoder_type == "cross":
            model_conf = download_or_load(
                self.config.src, 
                self.config.model_info
            )
            
            from sentence_transformers.cross_encoder import CrossEncoder
        
            model = CrossEncoder(model_conf.model_path)
        
            return RagcarCrossSts(self.config, model)
        
        from ragcar.tools.sentence_embedding import RagcarSentenceFactory
        
        model = RagcarSentenceFactory(
            tool="sentence_embedding",
            src=self.config.src,
            model=self.config.model_info
        ).load(device)
        
        return RagcarSts(self.config, model)


class RagcarStsBase(RagcarBiencoderBase):
    
    def get_sorted_similarity(self, scores: List[int]) -> List[List[int]]:
        """
        Sorts cosine similarity scores between sentence pairs in descending order.

        Args:
            scores (List[int]): A matrix of similarity scores between sentence pairs.

        Returns:
            List[List[int]]: A list of [score, index1, index2] lists sorted by score in descending order, 
                where 'index1' and 'index2' represent the positions of the sentence pairs in the original input list.
        """
        #Find the pairs with the highest cosine similarity scores
        pairs = []
        for i in range(len(scores)):
            for j in range(len(scores[i])):
                pairs.append([scores[i][j], i, j])
        
        #Sort scores in decreasing order
        pairs = sorted(pairs, key=lambda x: x[0], reverse=True)
        
        return pairs


class RagcarSts(RagcarStsBase):
    
    def __init__(self, config, model):
        super().__init__(config)
        self._model = model
    
    def predict(self, sent_a, sent_b, **kwargs):
        """
        Calculates the semantic similarity between two sets of sentences using a Bi-Encoder architecture.

        Args:
            sent_a (str, list, or torch.Tensor): The first sentence or list of sentences.
            sent_b (str, list, torch.Tensor, or None): The second sentence or list of sentences. If None, calculates self-similarity for `sent_a`.
            sorted (bool): If True, returns similarity scores in descending order.
            batch_size (int): Batch size for processing when handling multiple sentences.
            convert_to_tensor (bool): If True, converts the embeddings to PyTorch tensors.

        Returns:
            If `sent_b` is not None and 'sorted' is False, returns a torch.Tensor containing similarity scores.
            If 'sorted' is True, or if `sent_a` is processed for self-similarity, returns a list of lists containing similarity scores and indices from `sent_a` and `sent_b`.

        Raises:
            ValueError: If the inputs are not both strings, lists of strings, or tensors.
        """
        # If sent_a is a string, wrap it in a list
        if isinstance(sent_a, str):
            sent_a = [sent_a]
            
        # If sent_b is a string, wrap it in a list
        if isinstance(sent_b, str):
            sent_b = [sent_b]

        # If inputs are single list
        if isinstance(sent_a, list) and sent_b is None:
            if len(sent_a) < 10000:
                embeddings_a = self._model.predict(sent_a, convert_to_tensor=True)
                embeddings_b = embeddings_a
                cosine_scores = util.cos_sim(embeddings_a, embeddings_b)
                outputs = []
                for i in range(len(cosine_scores)-1):
                    for j in range(i+1, len(cosine_scores)):
                        outputs.append([cosine_scores[i][j], i, j])
            else:
                outputs = util.paraphrase_mining(self._model, sent_a)
                
            return outputs

        # If inputs are list of strings, encode them into embeddings
        if isinstance(sent_a, list) and isinstance(sent_b, list):
            embeddings_a = self._model.predict(sent_a, convert_to_tensor=True)
            embeddings_b = self._model.predict(sent_b, convert_to_tensor=True)
        # If inputs are tensors, just use them directly
        elif isinstance(sent_a, torch.Tensor) and isinstance(sent_b, torch.Tensor):
            embeddings_a = sent_a
            embeddings_b = sent_b
        else:
            raise ValueError("Inputs should be both strings, lists of strings, or tensors.")
            
        scores = util.cos_sim(embeddings_a, embeddings_b)
        
        if "sorted" in kwargs and kwargs["sorted"] is True:
            outputs = self.get_sorted_similarity(scores)
        else:
            outputs = scores

        return outputs
    
    
class RagcarCrossSts(RagcarStsBase):
    
    def __init__(self, config, model):
        super().__init__(config)
        self._model = model
    
    def predict(self, sent_a, sent_b, **kwargs):
        """
        Computes semantic textual similarity scores between two sets of sentences using a cross-encoder model.

        Args:
            sent_a (str or list): The first sentence or list of sentences to be encoded.
            sent_b (str or list): The second sentence or list of sentences to be encoded.
            **kwargs: Arbitrary keyword arguments, including 'sorted' (bool) to specify whether to return the similarity scores in descending order.

        Returns:
            A torch.Tensor of similarity scores if 'sorted' is False. If 'sorted' is True, returns a list of lists containing similarity scores, and indices from `sent_a` and `sent_b`.

        Raises:
            ValueError: If the inputs are not both strings or lists of strings.
        """
        # If sent_a is a string, wrap it in a list
        if isinstance(sent_a, str):
            sent_a = [sent_a]
            
        # If sent_b is a string, wrap it in a list
        if isinstance(sent_b, str):
            sent_b = [sent_b]

        # If inputs are list of strings, encode them into embeddings
        if isinstance(sent_a, list) and isinstance(sent_b, list):
            scores = []
            
            for a in sent_a:
                #Concatenate the query and all passages and predict the scores for the pairs [query, ]
                pairs = [[a, b] for b in sent_b]
                scores.append(self._model.predict(pairs))
            
            scores = torch.Tensor(scores)
        else:
            raise ValueError("Inputs should be both strings, lists of strings, or tensors.")
            
        if "sorted" in kwargs and kwargs["sorted"] is True:
            outputs = self.get_sorted_similarity(scores)
        else:
            outputs = scores

        return outputs