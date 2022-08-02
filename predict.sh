CUDA_VISIBLE_DEVICES=2 python predict.py \
 --infer_batch_size 12 --dataset Guangdong --checkname 'PIDNet' --resize 512 --crop_size 512 --num_worker 8\
 --epochs 100 --model_name 'hrnet' --nclass 3\
 --resume "/media/dell/DATA/wy/Seg_NAS/run/GID/retrain/GID15_hrnet/experiment_0/epoch89_checkpoint.pth.tar"