for SPLIT in train valid test; do
    python -m multiprocessing_spm_encoder \
    --spm_model ./spm_model/spm.model \
    --inputs ./web-crawler/kowiki/corpus_${SPLIT}.txt \
    --outputs ./web-crawler/kowiki/corpus_${SPLIT}.bpe \
    --output_format id \
    --proctitle SPM_modeling \
    --workers 100;
done


TEXT=./web-crawler/kowiki
fairseq-preprocess \
    --only-source \
    --trainpref $TEXT/corpus_train.bpe \
    --validpref $TEXT/corpus_valid.bpe \
    --testpref $TEXT/corpus_test.bpe   \
    --destdir data-bin/kowiki \
    --workers 100 \
    --padding-factor 20 \
    --nwordssrc 32000; # 사용자의 코퍼스 사이즈 따라 다르게 설정