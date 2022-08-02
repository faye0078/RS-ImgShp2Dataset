CUDA_VISIBLE_DEVICES=2 python predict.py \
 --infer_batch_size 12 --dataset "Guangdong" --num_worker 8\
 --model_name 'PIDNet' --nclass 3\
 --resume "" --checkname 'PIDNet'