# fwd: x . w^T
# bwd1: e . w
# bwd2: e^T . x

# had options are none, left, right, both
# quant options are none, global, row_wise, col_wise, block32

bash train_fft.sh DATASET=viggo LR=8e-5 SEED=1 \
    FWD_X_HAD=right \
    FWD_W_HAD=right \
    BWD1_E_HAD=left \
    BWD1_W_HAD=none \
    BWD2_E_HAD=none \
    BWD2_X_HAD=right \
    FWD_X_QUANT=global \
    FWD_W_QUANT=global \
    BWD1_E_QUANT=global \
    BWD1_W_QUANT=global \
    BWD2_E_QUANT=global \
    BWD2_X_QUANT=global