# KNOW기반 직업 추천 알고리즘 경진대회

## Introduction
본 문서는 **KNOW기반 직업 추천 알고리즘 경진대회** Private 2nd 코드 및 점수 재현 방법에 대해 서술하고 있습니다.

## Prerequisites
본 코드는 후술할 한국어 임베딩의 버전 충돌 문제로 인해 가상환경을 사용하고 있습니다.
- base
    - numpy
    - omegaconf
    - pandas
    - pytorch_lightning
    - scikit_learn
    - torch==1.10.1
    - transformers
    - wandb
- transformers_4_15_0
    - numpy
    - pandas
    - pytorch_lightning
    - scikit_learn
    - torch==1.10.1
    - transformers==4.15.0
    - tqdm
- transformers_2_8_0
    - boto3
    - gluonnlp >= 0.6.0
    - mxnet >= 1.4.0
    - onnxruntime == 1.8.0
    - sentencepiece >= 0.1.6
    - torch >= 1.7.0
    - transformers == 2.8.0
    - tqdm

`transformers_4_15_0`와 `transformers_2_8_0` 환경은 후술할 데이터셋 작업 문단에서 더 자세하게 설명하고 있습니다. 사전 처리가 완료된 데이터셋 파일이 같이 포함되어 있으며, 이를 이용할 경우 해당 가상환경 설정을 건너뛸 수 있습니다.

본 프로젝트는 `conda`를 통한 가상환경 제어를 사용하여 일련의 작업을 수행하는 유틸리티 스크립트를 포함하고 있습니다. 원활한 작업을 위해 `conda` 사용을 권장드립니다.

## Preprocessing Dataset
설문조사 데이터에 포함된 자연어 항목을 효과적으로 처리하기 위해, 본 프로젝트는 한국어 임베딩 모델을 사용합니다.
- SimCSE (https://github.com/BM-K/KoSimCSE-SKT)
- Averaged BERT Input Embeddings
한국어 KoSimCSE 모델을 사용하여 문장 임베딩을 생성하기 위해, 다음의 명령어를 통해 환경설정을 진행해 주십시오.
```bash
$ conda activate transformers_2_8_0
$ git clone https://github.com/BM-K/KoSimCSE.git
$ cd KoSimCSE
$ git clone https://github.com/SKTBrain/KoBERT.git
$ cd KoBERT
$ pip install -r requirements.txt
$ pip install .
$ cd ..
$ pip install -r requirements.txt
```
명령어 실행이 완료되었다면 [Prerequisites](#prerequisites) 문단에서 명시된 라이브러리를 설치해 주시기 바랍니다. 이 때 `transformers` 라이브러리의 버전은 2.8.0을 만족해야 합니다. 버전이 일치하지 않을 경우 해당 라이브러리 삭제 후 재설치를 권장드립니다.

`transformers` 버전이 서로 상이하기 때문에, `KoBERT`와 `KoSimCSE`가 정상적으로 실행되지 않습니다. `KoSimCSE/KoBERT/kobert/pytorch_bert.py`의 28번째 줄
```python
bertmodel = BertModel.from_pretrained(model_path), return_dict=False)
```
를
```python
bertmodel = BertModel.from_pretrained(model_path))
```
로 수정하여 주십시오.

이후 [해당 레포지토리의 README 문서](https://github.com/BM-K/KoSimCSE-SKT)에서 사전학습 모델을 다운로드받아 주시기 바랍니다. 다운로드된 파일 (nli_checkpoint.pt)은 KoSimCSE 폴더 내에 위치해 있어야 합니다.

이제 대회 데이터셋을 다운로드해 주십시오. 다운로드된 파일과 `KoSimCSE` 폴더를 모두 프로젝트 경로의 `res` 폴더의 하위로 옮겨주시기 바랍니다. 이후 다음의 명령어를 수행해 주십시오.
```bash
$ bash utilities/preprocess.sh
```
혹시 환경설정에 오류가 발생할 시, `utilities/preprocess.sh` 파일의 첫 줄에 위치한
```bash
source ~/anaconda3/etc/profile.d/conda.sh
```
명령어에서 현재 시스템의 아나콘다 경로를 수정하여 주십시오.

실행이 완료되었다면, 다음의 파일이 생성되었는지 확인해 주세요. 해당 파일을 모두 `res` 폴더로 옮기면 데이터셋 전처리는 완료됩니다.
- KNOW_2017.pkl
- KNOW_2018.pkl
- KNOW_2019.pkl
- KNOW_2020.pkl
- KNOW_2017_test.pkl
- KNOW_2018_test.pkl
- KNOW_2019_test.pkl
- KNOW_2020_test.pkl

혹은 첨부된, 전처리가 완료된 위의 파일들을 사용할 수 있습니다. 해당 파일들은 위의 과정과 동일하게 진행되어 생성된 파일들입니다.

## Train the Models
모델을 학습하기 위해 다음의 명령어를 입력하십시오.
```bash
$ python src/train.py config/sid-512d-1tb-1ab-18l.yaml data.filename=res/KNOW_2017.pkl data.fold_index=0 data.num_folds=5 train.random_seed=42
```
연도 데이터, KFold 갯수 및 random seed를 변경하기 위해서 위의 명령어에 명시된 값을 조절할 수 있습니다. 혹은 전체 데이터에 대한 5-fold 학습을 위해 다음의 명령어를 사용하십시오.
```bash
$ bash utilities/train.sh data.random_seed=42
```
본 프로젝트와 동일한 결과를 위해 다음의 random seed에 대한 학습을 진행해 주시기 바랍니다. 물론 GPU의 random generation이 상이하여 다른 결과를 얻을 수 있음을 고려해야 합니다.
- data.random_seed=0
- data.random_seed=1
- data.random_seed=2
- data.random_seed=3
- data.random_seed=4
- data.random_seed=20
- data.random_seed=24
- data.random_seed=42
- data.random_seed=777
- data.random_seed=1111
- data.random_seed=1234
- data.random_seed=2022
- data.random_seed=9876
- data.random_seed=9999
- data.random_seed=65535

각 random seed별 모델 가중치가 프로젝트 폴더 최상단에 위치하게 됩니다. 각 seed별 완료된 모델을 새로운 폴더 (e.g. `rs42`)에 격리해 주십시오.

## Predict the KNOW Codes
학습이 완료된 모델, 혹은 동봉된 가중치 파일들이 격리되어 있는 경로를 사용하여 다음과 같이 테스트 데이터셋에 대한 예측 파일을 생성합니다.
```bash
$ bash utilities/predict-test.sh ./rs42
```
모든 random seed에 대해 명령어를 실행한 뒤, 폴더 내에 예측된 `.csv` 파일들이 생성되었는지 확인합니다. 이들을 하나로 결합하기 위해 다음의 명령어를 추가적으로 실행합니다.
```bash
$ python utilities/create_submission.py **/*.csv --merge --ensemble
```
모든 과정이 완료되었다면, `submission-ensemble.csv` 파일이 생성되었는지 확인하십시오. 해당 파일이 제출에 사용된 submission입니다.