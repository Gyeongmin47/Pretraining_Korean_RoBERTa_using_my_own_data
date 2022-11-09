import os
import json
import pickle

import sentencepiece as spm
from unicodedata import normalize
from sklearn.model_selection import train_test_split


def clean_text(input_file, output_file):

    with open(input_file, 'r', encoding='utf-8') as f_r:
        with open(output_file + 'train.txt', 'w', encoding='utf-8') as f_w_tr:
            with open(output_file + 'valid.txt', "w", encoding='utf-8') as f_w_v:
                with open(output_file + 'test.txt', "w", encoding='utf-8') as f_w_te:
                    dataset = f_r.readlines()

                    train, _valid = train_test_split(dataset, test_size=0.2, random_state=42, shuffle=True)
                    valid, test = train_test_split(_valid, test_size=0.5, random_state=42, shuffle=True)

                    for i, sent in enumerate(train):
                        sent = sent.strip()
                        f_w_tr.write(sent)
                        f_w_tr.write("\n")

                    for i, sent in enumerate(valid):
                        sent = sent.strip()
                        f_w_v.write(sent)
                        f_w_v.write("\n")

                    for i, sent in enumerate(test):
                        sent = sent.strip()
                        f_w_te.write(sent)
                        f_w_te.write("\n")

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


# -*- coding: utf-8 -*-


def train_spm(
        vocab_size=32000,
        character_coverage=0.9995,
        # input path = "<set your path>/web-crawler/kowiki/corpus.txt"
        data_dir='web-crawler/kowiki/corpus.txt',
        # output path = "<set output path>/spm_model"
        output_dir='./spm_model',
        control_symbols="[PAD],[CLS],[SEP]",
        user_defined_symbols="[MASK]",
):
    check_dir(output_dir)

    input_file = "web-crawler/kowiki/corpus.txt"
    output_file = "web-crawler/kowiki/corpus_"

    clean_text(input_file, output_file)

    # See: https://github.com/google/sentencepiece/blob/master/doc/normalization.md
    normalization_rule_name = "nmt_nfkc"  # Default spm normalization rule is 'nmt-nfkc'

    model_prefix = 'spm'
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