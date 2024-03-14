# RAGCAR: Retrieval-Augmented Generative Companion for Advanced Research

RAGCAR🚛는 카카오브레인의 자연어 처리 라이브러리 [PORORO](https://github.com/kakaobrain/pororo) 아키텍처를 기반으로 구축하여, 대규모 언어 모델(Large Language Models, LLM) OpenAI의 [GPT](https://github.com/openai/openai-python)와 NAVER의 [HyperCLOVA X](https://www.ncloud.com/product/aiService/clovaStudio) API 기능을 추가하고 RAG(Retrieval-Augmented Generation)에 필요한 도구들을 쉽게 사용할 수 있도록 지원합니다.

## Installation

- `python>=3.8` 환경에서 정상적으로 동작합니다.

- 아래 커맨드를 통해 패키지를 설치하실 수 있습니다.

```console
pip install ragcar
```

- 혹은 아래와 같이 **로컬 환경**에서 설치를 하실 수도 있습니다.

```console
git clone https://github.com/leewaay/ragcar.git
cd ragcar
pip install -e .
```

<br>

## Usage

다음과 같은 명령어로 `Ragcar` 를 사용할 수 있습니다.

- 먼저, `Ragcar` 를 임포트하기 위해서는 다음과 같은 명령어를 실행하셔야 합니다:

```python
>>> from ragcar import Ragcar
```

<br>

- 임포트 이후에는, 다음 명령어를 통해 현재 `Ragcar` 에서 지원하고 있는 Task를 확인하실 수 있습니다.

```python
>>> from ragcar import Ragcar
>>> Ragcar.available_tools()
"Available tools are ['tokenization', 'sentence_embedding', 'sentence_similarity', 'semantic_search', 'text_generation', 'text_segmentation']"
```

<br>

- Task 별로 어떠한 모델이 지원되는지 확인하기 위해서는 아래 과정을 거치시면 됩니다.

```python
>>> from ragcar import Ragcar
>>> Ragcar.available_models("text_generation")
'Available models for text_generation are ([src]: openai, [model]: gpt-4-turbo-preview, gpt-4, gpt-3.5-turbo, MODELS_SUPPORTED(https://platform.openai.com/docs/models)), ([src]: clova, [model]: YOUR_MODEL(https://www.ncloud.com/product/aiService/clovaStudio))'
```

<br>

- 특정 Task를 수행하고자 하실 때에는, `tool` 인자에 앞서 살펴본 **도구명**과 `src` 인자에 **모델 종류**를 넣어주시면 됩니다.

```python
>>> from ragcar import Ragcar
>>> from ragcar.utils import PromptTemplate
>>> prompt_template = PromptTemplate("사용자: {input} 수도는?\nAI:")

>>> generator = Ragcar(tool="text_generation", src="openai", prompt_template=prompt_template)
```

<br>

- 객체 생성 이후에는, 다음과 같이 입력 값을 넘겨주는 방식으로 사용이 가능합니다. 자세한 사용방법은 [examples](https://github.com/leewaay/ragcar/tree/main/examples)에서 각 Task 예제를 참고해주세요.

```python
>>> generator(input="대한민국")
'대한한국의 수도는 서울특별시입니다.'
```

<br>

### ⚠️ 환경변수 설정 방법

특정 `src`는 보안과 유지보수가 필요한 환경변수(ex. **API Key**)를 요구하며, 다음의 3가지 방법 중 하나로 설정할 수 있습니다:

1. [`.env` 파일](https://velog.io/@joahkim/%ED%83%90%EB%82%98bnb.env): 프로젝트 최상위 루트에 .env 파일을 생성하고 필요한 환경 변수 값을 입력합니다.

<br>

2. export: 터미널에서 필요한 환경변수를 직접 선언합니다.

```bash
export OPENAI_API_KEY='sk-...'
```

<br>

3. `model` 인자 값: 필요한 환경변수를 model 인자 값으로 직접 입력합니다. (**기본 제공되는 `model` 외에 추가가 필요한 경우에도 동일하게 적용**)

```python
>>> Ragcar.available_customizable_src("text_generation")
"Available customizable src for text_generation are ['clova', 'openai']"

>>> Ragcar.available_model_fields("clova")
'Available fields for clova are ([field]: model_n, [type]: str), ([field]: api_key, [type]: str), ([field]: app_key, [type]: str)'
```

```python
>>>generator = Ragcar(
    tool="text_generation", 
    src="clova", 
    model={
        "model_n": "YOUR_API_URL", 
        "api_key": "YOUR_APIGW-API-KEY",
        "app_key": "YOUR_CLOVASTUDIO-API-KEY"
    }, 
    prompt_template=prompt_template
)
```

<br>

- 보다 상세한 활용 방법은 [examples](https://github.com/leewaay/ragcar/tree/main/examples)를 확인해 주세요!

<br>

### ⚠️ Clova & Clovax `src` 사용 시 주의사항

**text_generation** `tool`을 **clova** `src`와 함께 사용할 때, 공식 Parameter 대비 변경된 사항에 주의해야 합니다:

- **파라미터 명 변경**:
  - `top_k` 대신 `presence_penalty`를 사용해주세요.
  - `repeat_penalty` 대신 `frequency_penalty`를 사용해주세요.

<br>

- **파라미터 값 범위**:
  - `0.0 < temperature < 1.0`
  - `0.0 < top_p < 1.0`
  - `0 < presence_penalty < 128`
  - `0.0 < frequency_penalty < 10.0`

<br>

### ⚠️ 구글 드라이브 모델 업로드 방법

[sentence_embedding example](https://github.com/leewaay/ragcar/blob/main/examples/sentence_embedding.ipynb) 확인

<br>

## Documentation

궁금한 사항이나 의견 등이 있으시다면 [이슈](https://github.com/leewaay/ragcar/-/issues)를 남겨주세요.

<br>

## Contributors

[이원석](https://github.com/leewaay)

<br>

## Acknowledgements 

* [sentence-transformers](https://github.com/UKPLab/sentence-transformers)
```bibtex 
@misc{pororo,
  author       = {Heo, Hoon and Ko, Hyunwoong and Kim, Soohwan and
                  Han, Gunsoo and Park, Jiwoo and Park, Kyubyong},
  title        = {PORORO: Platform Of neuRal mOdels for natuRal language prOcessing},
  howpublished = {\url{https://github.com/kakaobrain/pororo}},
  year         = {2021},
}
```

* [pororo](https://github.com/kakaobrain/pororo)
```bibtex 
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```