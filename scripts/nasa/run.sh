#!/bin/bash

export TENSORBOARD_DIR="../exps/tb"

exp_name="ae-1"
train_cells=""
valid_cells=""
window_length=32768
window_step=32768
latent_size=256
nj=8
data_dir="../data/nasa"
epochs=100
lr=0.001
stage=0
use_ae_dir=

echo "$0 $@"

. cmd.sh
. parse_options.sh

exp_dir=../exps/$exp_name

mkdir -p $exp_dir

# source ../../.venv/bin/activate
which python

if [ -n "${use_ae_dir}" ]; then
    ln -sr $use_ae_dir $exp_dir/ae
fi

ae_out_dir=$exp_dir/ae
dcap_out_dir=$exp_dir/dcap
mkdir -p $ae_out_dir $dcap_out_dir $exp_dir/.info

if [ $stage -le -1 ]; then
    mkdir -p $data_dir
    python local/data_prep.py --nasa-data-dir "$NASA_DATA_DIR" \
            --data-dir $data_dir --normalized-time-step $time_step \
            --window-length $window_length \
            --window-overlap $window_overlap \
            --nj $nj >> $log_file.data_prep 2>&1 || exit 1;
fi

if [ $stage -le 0 ]; then
    python ./train_ae.py --exp-name $exp_name \
                --out-dir $ae_out_dir \
                --data-dir $data_dir \
                --train-cells "$train_cells" \
                --valid-cells "$valid_cells" \
                --window-length $window_length \
                --latent-size $latent_size \
                --epochs $epochs \
                --lr $lr \
                --window-step $window_step --nj $nj \
                >> $ae_out_dir/log.ae 2>$ae_out_dir/log.ae.err || exit 1
fi

if [ $stage -le 1 ]; then
    python ./train_dcap.py --exp-name $exp_name \
                --out-dir $dcap_out_dir \
                --ae-dir $ae_out_dir \
                --data-dir $data_dir \
                --train-cells "$train_cells" \
                --valid-cells "$valid_cells" \
                --window-length $window_length --window-step $window_step \
                --latent-size $latent_size \
                --epochs $epochs \
                --lr $lr \
                --nj $nj >> $dcap_out_dir/log.dcap 2>$dcap_out_dir/log.dcap.err || exit 1

    cat $dcap_out_dir/best-mape-model.txt | sort -t "," -k 2 -n | head
fi

if [ $stage -le 2 ]; then
    python ./infer_dcap.py --exp-name $exp_name \
                --out-dir $dcap_out_dir/preds \
                --dcap-dir $dcap_out_dir \
                --data-dir $data_dir \
                --train-cells "$train_cells" \
                --valid-cells "$valid_cells" \
                --window-length $window_length --window-step $window_length \
                --latent-size $latent_size \
                --nj $nj >> $dcap_out_dir/log.dcap.pred 2>$dcap_out_dir/log.dcap.pred.err || exit 1

fi