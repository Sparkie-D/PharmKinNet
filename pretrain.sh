source activate decision


model=MLP
CUDA_VISIBLE_DEVICES=0 python main.py \
        --model-type ${model} \
        --model-size XS \
        --task pretrain \
        --train-path data/skeleton_train.csv \
        --valid-path data/skeleton_valid.csv \
wait