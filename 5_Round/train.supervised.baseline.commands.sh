export SHOT=10 #1, 3, 5, 10, 100000
export BATCHSIZE=20
export EPOCHSIZE=30
export LEARNINGRATE=1e-6



CUDA_VISIBLE_DEVICES=0 python -u train.supervised.baseline.py \
    --task_name rte \
    --do_train \
    --do_lower_case \
    --num_train_epochs $EPOCHSIZE \
    --train_batch_size $BATCHSIZE \
    --eval_batch_size 32 \
    --learning_rate $LEARNINGRATE \
    --max_seq_length 128 \
    --seed 42 \
    --round_name base > log.base.txt 2>&1 &

CUDA_VISIBLE_DEVICES=1 python -u train.supervised.baseline.py \
    --task_name rte \
    --do_train \
    --do_lower_case \
    --num_train_epochs $EPOCHSIZE \
    --train_batch_size $BATCHSIZE \
    --eval_batch_size 32 \
    --learning_rate $LEARNINGRATE \
    --max_seq_length 128 \
    --seed 16 \
    --round_name r1 > log.r1.txt 2>&1 &

CUDA_VISIBLE_DEVICES=2 python -u train.supervised.baseline.py \
    --task_name rte \
    --do_train \
    --do_lower_case \
    --num_train_epochs $EPOCHSIZE \
    --train_batch_size $BATCHSIZE \
    --eval_batch_size 32 \
    --learning_rate $LEARNINGRATE \
    --max_seq_length 128 \
    --seed 32 \
    --round_name r2 > log.r2.txt 2>&1 &

CUDA_VISIBLE_DEVICES=3 python -u train.supervised.baseline.py \
    --task_name rte \
    --do_train \
    --do_lower_case \
    --num_train_epochs $EPOCHSIZE \
    --train_batch_size $BATCHSIZE \
    --eval_batch_size 32 \
    --learning_rate $LEARNINGRATE \
    --max_seq_length 128 \
    --seed 64 \
    --round_name r3 > log.r3.txt 2>&1 &


CUDA_VISIBLE_DEVICES=6 python -u train.supervised.baseline.py \
    --task_name rte \
    --do_train \
    --do_lower_case \
    --num_train_epochs $EPOCHSIZE \
    --train_batch_size $BATCHSIZE \
    --eval_batch_size 32 \
    --learning_rate $LEARNINGRATE \
    --max_seq_length 128 \
    --seed 128 \
    --round_name r4 > log.r4.txt 2>&1 &

CUDA_VISIBLE_DEVICES=5 python -u train.supervised.baseline.py \
    --task_name rte \
    --do_train \
    --do_lower_case \
    --num_train_epochs $EPOCHSIZE \
    --train_batch_size $BATCHSIZE \
    --eval_batch_size 32 \
    --learning_rate $LEARNINGRATE \
    --max_seq_length 128 \
    --seed 128 \
    --round_name r5 > log.r5.txt 2>&1 &
