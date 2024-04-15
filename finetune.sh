source activate decision


for id in 0 1 2 3 4
do
    CUDA_VISIBLE_DEVICES=0 python main.py \
        --log-id 0408-mpa/35-$id \
        --train-path data/data_kfold/35/35_train_data_$id.csv \
        --valid-path data/data_kfold/35/35_valid_data_$id.csv \
        --normalizer-path log/pretrain/RNN\&model_size=XS\&log_id=0329/model/normalizer.pkl \
        --model-path log/pretrain/RNN\&model_size=XS\&log_id=0329/model/models_60.pth &
done

wait