# Pretraining Korean RoBERTa using my own data

BERT, ELECTRA 는 한국어로 사전학습한 모델이 많지만, [RoBERTa](https://arxiv.org/pdf/1907.11692.pdf) 는 아직까지도 공개된 없는것 같아 이렇게 작성하게 되었습니다. 

본 프로젝트에서는 데이터를 전처리하는 from scratch 부터, facebook의 [Pretraining RoBERTa using your own data](https://github.com/facebookresearch/fairseq/blob/main/examples/roberta/README.pretraining.md) 를 토대로 한국어에 맞게 최적화하여 작성하는 점 참고 부탁드리겠습니다.

[RoBERTa](https://arxiv.org/pdf/1907.11692.pdf)는 기존 [BERT](https://aclanthology.org/N19-1423.pdf) 와 비교하여 학습 과정에서 _Next Sentence Prediction (NSP) 제거, dynamic masking_ 을 수행했다는 점에 차이가 있고, 결과적으로 BERT 와 비교하여 좋은 성능을 보였습니다.

ELECTRA와 비교하면 상대적으로 비슷하거나 낮은 성능을 기록하지만, RoBERTa 또한 충분히 고려될 수 있는 언어모델이라는 점을 감안하여 봐주시면 감사하겠습니다. :)

KoRoBERTa는 약 **17GB의 한국어 text**로 학습하였고, [Sentencepiece](https://github.com/google/sentencepiece) 토크나이저로 학습했습니다. huggingface에 맞추기 위해 중간 과정을 필요로 하지만, 천천히 따라하시면 누구나 RoBERTa-base 모델을 만들 수 있도록 작성하고자 했습니다.

또한, 이미 학습된 언어모델을 huggingface hub로 공유하여, 누구나 쉽게 사용할 수 있도록 제공하고자 하였습니다.



## Fairseq Environment Setup for RoBERTa

---


## Prerequisites

---


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
