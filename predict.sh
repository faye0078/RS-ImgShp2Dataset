CUDA_VISIBLE_DEVICES=2 python predict.py \
 --infer_batch_size 12 --dataset "Guangdong_train" --num_worker 8\
 --model_name 'hrnet' --nclass 3\
 --resume "/media/dell/DATA/wy/LightRS/run/Guangdong_train/hrnet/model_best.pth.tar" --checkname 'hrnet/predict' --mode 'split'