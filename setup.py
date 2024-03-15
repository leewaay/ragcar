from setuptools import setup, find_packages

setup(
    name='ragcar', 
    version='0.1.1', 
    url='https://github.com/leewaay/ragcar.git', 
    author='Wonseok Lee', 
    author_email='wonsuklee7020@gmail.com', 
    description='RAGCAR: Retrieval-Augmented Generative Companion for Advanced Research', 
    packages=find_packages(), 
    python_requires='>=3.8',
    install_requires=[
        'torch==2.0.1',
        'python-dateutil',
        'dataclasses_json',
        'python-dotenv',
        'tqdm',
        'gdown',
        'tiktoken',
        'kiwipiepy',
        'elasticsearch==7.13.1',
        'aiohttp',
        'sentence-transformers>=2.4.0',
        'openai==0.28.1',
        'pytorch-lightning',
        'fsspec'
    ],
)