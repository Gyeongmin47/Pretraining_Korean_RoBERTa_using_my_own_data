TOTAL_UPDATES=100000    # Total number of training steps
WARMUP_UPDATES=6000     # Warmup the learning rate over this many updates
PEAK_LR=0.0007          # Peak learning rate, adjust as needed
TOKENS_PER_SAMPLE=512   # Max sequence length
MAX_POSITIONS=512       # Num. positional embeddings (usually same as above)
MAX_SENTENCES=64        # Number of sequences per batch (batch size)
UPDATE_FREQ=32          # Increase the batch size 16x
DATA_DIR=data-bin/kowiki
SAVE_DIR=checkpoints/roberta-spm-unigram-kowiki-only
SAVE_INTERVAL_UPDATES=5000
CUDA_VISIBLE_DEVICES=0,1,2,3 fairseq-train --fp16 $DATA_DIR \
    --task masked_lm --criterion masked_lm \
    --arch roberta_base --sample-break-mode complete --tokens-per-sample $TOKENS_PER_SAMPLE \
    --optimizer adam --adam-betas '(0.9,0.98)' --adam-eps 1e-6 --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $PEAK_LR --warmup-updates $WARMUP_UPDATES --total-num-update $TOTAL_UPDATES \
    --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
    --batch-size $MAX_SENTENCES --update-freq $UPDATE_FREQ \
    --max-update $TOTAL_UPDATES --log-format simple --log-interval 1 \
    --save-dir $SAVE_DIR --save-interval-updates $SAVE_INTERVAL_UPDATES \
    --skip-invalid-size-inputs-valid-test --disable-validation 2>&1 \
| tee fairseq-spm-unigram-normal.log