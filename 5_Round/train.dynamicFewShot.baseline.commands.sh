export BATCHSIZE=32
export EPOCHSIZE=50
export SEED=42 #42, 16, 32
export LEARNINGRATE=1e-6



CUDA_VISIBLE_DEVICES=0 python -u train.dynamicFewShot.baseline.py \
    --task_name rte \
    --do_train \
    --do_lower_case \
    --num_train_epochs $EPOCHSIZE \
    --train_batch_size $BATCHSIZE \
    --eval_batch_size 64 \
    --learning_rate $LEARNINGRATE \
    --max_seq_length 64 \
    --seed $SEED \
    --round_name base > log.dynamicFewShot.base.seed.$SEED.debug.txt 2>&1 &

CUDA_VISIBLE_DEVICES=1 python -u train.dynamicFewShot.baseline.py \
    --task_name rte \
    --do_train \
    --do_lower_case \
    --num_train_epochs $EPOCHSIZE \
    --train_batch_size $BATCHSIZE \
    --eval_batch_size 64 \
    --learning_rate $LEARNINGRATE \
    --max_seq_length 64 \
    --seed $SEED \
    --round_name r1 > log.dynamicFewShot.r1.seed.$SEED.debug.txt 2>&1 &

CUDA_VISIBLE_DEVICES=2 python -u train.dynamicFewShot.baseline.py \
    --task_name rte \
    --do_train \
    --do_lower_case \
    --num_train_epochs $EPOCHSIZE \
    --train_batch_size $BATCHSIZE \
    --eval_batch_size 64 \
    --learning_rate $LEARNINGRATE \
    --max_seq_length 64 \
    --seed $SEED \
    --round_name r2 > log.dynamicFewShot.r2.seed.$SEED.debug.txt 2>&1 &

CUDA_VISIBLE_DEVICES=3 python -u train.dynamicFewShot.baseline.py \
    --task_name rte \
    --do_train \
    --do_lower_case \
    --num_train_epochs $EPOCHSIZE \
    --train_batch_size $BATCHSIZE \
    --eval_batch_size 64 \
    --learning_rate $LEARNINGRATE \
    --max_seq_length 64 \
    --seed $SEED \
    --round_name r3 > log.dynamicFewShot.r3.seed.$SEED.debug.txt 2>&1 &


CUDA_VISIBLE_DEVICES=4 python -u train.dynamicFewShot.baseline.py \
    --task_name rte \
    --do_train \
    --do_lower_case \
    --num_train_epochs $EPOCHSIZE \
    --train_batch_size $BATCHSIZE \
    --eval_batch_size 64 \
    --learning_rate $LEARNINGRATE \
    --max_seq_length 64 \
    --seed $SEED \
    --round_name r4 > log.dynamicFewShot.r4.seed.$SEED.debug.txt 2>&1 &

CUDA_VISIBLE_DEVICES=6 python -u train.dynamicFewShot.baseline.py \
    --task_name rte \
    --do_train \
    --do_lower_case \
    --num_train_epochs $EPOCHSIZE \
    --train_batch_size $BATCHSIZE \
    --eval_batch_size 64 \
    --learning_rate $LEARNINGRATE \
    --max_seq_length 64 \
    --seed $SEED \
    --round_name r5 > log.dynamicFewShot.r5.seed.$SEED.debug.txt 2>&1 &
