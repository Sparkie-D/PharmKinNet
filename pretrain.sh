source activate decision


CUDA_VISIBLE_DEVICES=0 python main.py \
        --model-size XS \
        --task pretrain \
        --train-path data/skeleton_train.csv \
        --valid-path data/skeleton_valid.csv 

wait