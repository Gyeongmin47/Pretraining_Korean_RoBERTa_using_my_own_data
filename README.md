# Pretraining Korean RoBERTa using my own data

BERT, ELECTRA 는 한국어로 사전학습한 모델이 많지만, [RoBERTa](https://arxiv.org/pdf/1907.11692.pdf) 는 아직까지도 공개된 없는것 같아 이렇게 작성하게 되었습니다. 

본 프로젝트에서는 데이터를 전처리하는 from scratch 부터, facebook의 [Pretraining RoBERTa using your own data](https://github.com/facebookresearch/fairseq/blob/main/examples/roberta/README.pretraining.md) 를 토대로 한국어에 맞게 최적화하여 작성하는 점 참고 부탁드리겠습니다.

[RoBERTa](https://arxiv.org/pdf/1907.11692.pdf) 는 기존 [BERT](https://aclanthology.org/N19-1423.pdf) 와 비교하여 학습 과정에서 _Next Sentence Prediction (NSP) 제거, dynamic masking_ 을 수행했다는 점에 차이가 있고, 결과적으로 BERT 와 비교하여 좋은 성능을 보였습니다.

[ELECTRA](https://arxiv.org/pdf/2003.10555.pdf) 와 비교하면 상대적으로 비슷하거나 낮은 성능을 기록하지만, RoBERTa 또한 충분히 고려될 수 있는 언어모델이라는 점을 감안하여 봐주시면 됩니다. :)

KoRoBERTa는 약 **750MB의 한국어 text**로 [Sentencepiece](https://github.com/google/sentencepiece) 토크나이저로 학습했습니다. huggingface에 맞추기 위해 중간 과정을 필요로 하지만, 천천히 따라하시면 누구나 RoBERTa-base 모델을 만들 수 있도록 작성하고자 했습니다.

또한, 이미 학습된 언어모델을 huggingface hub로 공유하여, 누구나 쉽게 사용할 수 있도록 제공하고자 하였습니다.



## Fairseq Environment Setup for RoBERTa

---
1. Conda & CUDA Setup
```
# Conda
conda create -n fairseq-roberta python=3.7

# CUDA의 경우 본인 GPU 환경에 맞게 세팅
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
```

2. 추가 라이브러리 설치
```
pip install gluonnlp kss sklearn konlpy sentencepiece seqeval setproctitle soynlp datasets fire cmake tqdm parmap
```

3. Fairseq 설치
```
# fairseq 사이트 참고
1. git clone https://github.com/pytorch/fairseq
2. cd fairseq
3. pip install --editable ./
```

4. Apex 설치 (시간이 꽤 소요되며, apex 사이트에서도 동일하게 받을 수 있습니다.)
```
# apec 사이트 참고
1. git clone https://github.com/NVIDIA/apex
2. cd apex
3. pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" \
  --global-option="--deprecated_fused_adam" --global-option="--xentropy" \
  --global-option="--fast_multihead_attn" ./
4. 설치 시 deprecated가 엄청 뜨는데, 그리고 나서 설치됩니다  
(참고: https://github.com/pytorch/fairseq/issues/2372)
```


## Sentencepiece 모델 만들기

---

자료는 [cchyun](https://paul-hyun.github.io/vocab-with-sentencepiece/) 님의 자료를 참고하였습니다. :)

한국어 말뭉치는 _[위키미디어 다운로드](https://dumps.wikimedia.org/kowiki/)_ 에서 다운받을 수 있으며, 여기에선 20220201 데이터를 사용합니다.

여기서 **pages-articles.xml.bz2** 파일을 다운받아서 전처리 해야하는데, [cchyun](https://paul-hyun.github.io/about) 님의 web-crawler 를 사용해서 쉽게 처리 가능합니다. (위키데이터 라인 별 .txt 문장으로 만들어 전처리)

```
git clone https://github.com/paul-hyun/web-crawler.git
cd web-crawler
pip install tqdm
pip install pandas
pip install bs4
pip install wget
pip install pymongo

python kowiki.py
```

위 소스에서 filename 변수만 다운받은 위키데이터로 변경합니다.
```
filename = "kowiki-20220201-pages-articles.xml.bz2"
```

위 명령어를 실행하면 kowiki에 실제 wiki 데이터 형식는 다음과 같이 되어있으며, 이 중 “text”: 에 해당하는 부분을 가져옵니다.
```
{"id": "5", "url": "https://ko.wikipedia.org/wiki?curid=5", "title": "지미 카터", "text": "지미 카터\n\n제임스 얼 카터 주니어(, 1924년 10월 1일 ~ )는 ... 별명이 붙기도 했다.\n\n\n\n"}
```

### SPM build

Google의 [Sentencepiece](https://github.com/google/sentencepiece) 를 사용하여 spm 모델을 만듭니다. 이 과정에서 BOS, EOC 위치에 PAD, UNK, CLS, SEP로 변경해서 사용했는데, 이건 토크나이저의 목적성에 따라 다르게 변경할 수 있습니다 :)


아래 명령어를 실행하면 .model, .vocab이 생성됩니다.

```
$ python build_sentencepiece_model.py
```


대용량 corpus로부터 생성하는 경우 OOM이 발생할 수 있는데요, 아래 argument를 추가해주시기 바랍니다.

```
"--input_sentence_size=10000000 "
"--shuffle_input_sentence=true "
"--train_extremely_large_corpus=true "
```


build_sentencepiece_model 에서 vocab_size,



## Citation

If you find our work useful, please read and cite the following paper:

```bibtex
@article{kim2021enhancing,
  title={Enhancing Korean Named Entity Recognition With Linguistic Tokenization Strategies},
  author={Kim, Gyeongmin and Son, Junyoung and Kim, Jinsung and Lee, Hyunhee and Lim, Heuiseok},
  journal={IEEE Access},
  volume={9},
  pages={151814--151823},
  year={2021},
  publisher={IEEE}
}
```

## Reference

--- 
- [Pretraining RoBERTa using your own data](https://github.com/facebookresearch/fairseq/blob/main/examples/roberta/README.pretraining.md)
- [Sentencepiece](https://github.com/google/sentencepiece)
- 
