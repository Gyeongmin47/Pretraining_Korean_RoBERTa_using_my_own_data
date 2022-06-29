# -*- coding: utf-8 -*-


def clean_text(input_file, output_file):

    leng_check = []

    with open(input_file, 'r', encoding='utf-8') as f_r:
        with open(output_file + 'train', 'w', encoding='utf-8') as f_w_tr:
            with open(output_file + 'valid', "w", encoding='utf-8') as f_w_v:
                with open(output_file + 'test', "w", encoding='utf-8') as f_w_te:
                    dataset = f_r.readlines()

                    total_line = len(dataset)

                    partition = total_line // 20
                    train_part = partition*18
                    valid_part = partition*18 + partition
                    test_part = partition*18 + partition + partition

                    for i, sent in enumerate(dataset):

                        sent = sent.strip()
                        length = len(sent.split(" "))

                        leng_check.append(length)

                        #if length >= 1000:
                        #    continue

                        # chr(9679) or chr(9604) or chr(9605) or chr(9606) or chr(9607) or chr(9608)
                        if chr(9604) in sent:
                            continue
                        elif chr(9605) in sent:
                            continue
                        elif chr(9606) in sent:
                            continue
                        elif chr(9607) in sent:
                            continue
                        elif chr(9608) in sent:
                            continue

                        if i < train_part:
                            f_w_tr.write(sent)
                            f_w_tr.write("\n")

                        elif i < valid_part:
                            f_w_v.write(sent)
                            f_w_v.write("\n")

                        else:
                            f_w_te.write(sent)
                            f_w_te.write("\n")


                    #print(len(extra_id))
                    # print(max(leng_check))
                    # print(len(leng_check))

#input_file = "/mnt/raid6/totoro4007/fairseq-roberta/crypto_only_roberta/corpus/crypto_corpus/clean_pretrain_data_v2.txt"
input_file = "/mnt/raid6/totoro4007/fairseq-roberta/crypto_only_roberta/corpus/kowiki/kowiki.txt"
#output_file = "/mnt/raid6/totoro4007/fairseq-roberta/crypto_only_roberta/corpus/crypto_corpus/clean_crypto_corpus_0318.txt"
output_file = "/mnt/raid6/totoro4007/fairseq-roberta/crypto_only_roberta/corpus/kowiki/kowiki_.txt"

clean_text(input_file, output_file)