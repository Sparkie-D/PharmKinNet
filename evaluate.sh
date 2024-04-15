source activate decision


for id in 0 1 2 3 4
do
    CUDA_VISIBLE_DEVICES=0 python evaluate/metrics.py \
        --log_path log/finetune/RNN\&model_size=XS\&log_id=35-$id &
done

wait