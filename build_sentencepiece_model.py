import os
import json
import pickle

import sentencepiece as spm
from unicodedata import normalize

def check_dir(dir):
    cur_dir = ''
    dir_list = dir.split('/')
    if len(dir_list[0]) == 0:
        cur_dir = '/'
    for d in dir_list:
        cur_dir = os.path.join(cur_dir, d)
        if not os.path.exists(cur_dir):
            os.mkdir(cur_dir)

def save_vocab(vocab: dict, output_dir):
    print("Save vocabs {}...".format(output_dir))
    f = open(output_dir, 'w', encoding='utf-8')
    reverse_vocab = dict() # key: index, value: str
    for key, value in vocab.items():
        reverse_vocab[value] = key
    for i in range(len(reverse_vocab)):
        f.write(reverse_vocab[i] + '\n')
    print("Done!")

def train_spm(
        vocab_size=52000,
        character_coverage=0.9995,
        # absolute path = "/mnt/raid6/totoro4007/myrepo/sentencepiece_model/web-crawler/corpus/corpus.txt"
        data_dir='web-crawler/corpus/corpus.txt',
        # absolute path = "/mnt/raid6/totoro4007/myrepo/sentencepiece_model/spm_model"
        output_dir='spm_model',
        tokenization_type='original',
        control_symbols="[PAD],[CLS],[SEP]",
        user_defined_symbols="[MASK]",
        mode_sub_char=False,
):
    check_dir(output_dir)

    if mode_sub_char:
        # sub-char인 경우, "identity" option으로 둔다. Corpus가 이미 NFKD normalize 되어있음.
        normalization_rule_name = "identity"
    else:
        # See: https://github.com/google/sentencepiece/blob/master/doc/normalization.md
        normalization_rule_name = "nmt_nfkc"  # Default spm normalization rule is 'nmt-nfkc'

    model_prefix = tokenization_type + '-spm'
    output_dir = os.path.join(output_dir, model_prefix)

    spm.SentencePieceTrainer.Train(
        "--input={} "
        "--input_format=text "  # one-sentence-per-line raw corpus file
        "--model_prefix={} "
        "--vocab_size={} "
        "--character_coverage={} "
        "--model_type={} "
        "--control_symbols={} "
        "--user_defined_symbols={} "
        "--pad_id=0 "
        "--pad_piece=[PAD] "
        "--unk_id=1 "
        "--unk_piece=[UNK] "
        "--bos_id=2 "
        "--bos_piece=[CLS] "
        "--eos_id=3 "
        "--eos_piece=[SEP] "    
        "--max_sentence_length=2048 "
        "--input_sentence_size=10000000 "
        "--shuffle_input_sentence=true "
        "--train_extremely_large_corpus=true "
        "--normalization_rule_name={}".format(
            data_dir,
            output_dir,
            vocab_size,
            character_coverage,
            "unigram",  # model_type="unigram"
            control_symbols,
            user_defined_symbols,
            normalization_rule_name
        )
    )

if __name__ == "__main__":
    train_spm()