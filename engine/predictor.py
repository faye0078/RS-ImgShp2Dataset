import os
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import cv2
import torch
from utils.loss import SegmentationLosses
from data import make_predict_split_loader, make_predict_concat_loader
from data.concat import recons_prob_map
# from decoder import Decoder
from utils.lr_scheduler import LR_Scheduler
from utils.saver import Saver
from utils.evaluator import Evaluator
from model.UNet import U_Net
from utils.copy_state_dict import copy_state_dict
from model.HRNet import get_HRNet_model
from model.make_fast_nas import fastNas
from model.PIDNet import get_PID_model

import sys
sys.path.append("./apex")
import sys
sys.path.append("..")
try:
    from apex import amp
    APEX_AVAILABLE = True
except ModuleNotFoundError:
    APEX_AVAILABLE = False
class Predictor(object):
    def __init__(self, args):
        self.args = args
        # 定义保存
        self.saver = Saver(args)
        self.saver.save_experiment_config()
        # 使用amp
        self.use_amp = True if (APEX_AVAILABLE and args.use_amp) else False
        self.opt_level = args.opt_level
        # 定义dataloader
        self.kwargs = {'num_workers': args.num_worker, 'pin_memory': True, 'drop_last':True}
        self.test_loader = None
        self.nclass = args.nclass
        self.criterion = SegmentationLosses(weight=None, cuda=args.cuda).build_loss(mode=args.loss_type)
        torch.cuda.empty_cache()
        # 定义网络
        if args.model_name == 'hrnet':
            model = get_HRNet_model(args)
        elif args.model_name == 'fast-nas':
            model = fastNas()
        elif args.model_name == 'PIDNet':
            model = get_PID_model(args)
        self.model = model
        # Define Evaluator
        self.evaluator = Evaluator(self.nclass)

        if args.cuda:
            self.model = self.model.cuda()
        # 使用apex支持混合精度分布式训练
        if self.use_amp and args.cuda:
            keep_batchnorm_fp32 = True if (self.opt_level == 'O2' or self.opt_level == 'O3') else None

            # fix for current pytorch version with opt_level 'O1'
            if self.opt_level == 'O1' and torch.__version__ < '1.3':
                for module in self.model.modules():
                    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                        # Hack to fix BN fprop without affine transformation
                        if module.weight is None:
                            module.weight = torch.nn.Parameter(
                                torch.ones(module.running_var.shape, dtype=module.running_var.dtype,
                                           device=module.running_var.device), requires_grad=False)
                        if module.bias is None:
                            module.bias = torch.nn.Parameter(
                                torch.zeros(module.running_var.shape, dtype=module.running_var.dtype,
                                            device=module.running_var.device), requires_grad=False)

            # print(keep_batchnorm_fp32)
            self.model = amp.initialize(
                self.model, opt_level=self.opt_level,
                keep_batchnorm_fp32=keep_batchnorm_fp32, loss_scale="dynamic")
            print('cuda finished')

        # 加载模型
        self.best_pred = 0.0
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
            checkpoint = torch.load(args.resume)
            self.start_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['state_dict'],  strict=False)
            self.best_pred = checkpoint['best_pred']
            print(self.best_pred)
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))

    def split_predict(self):
        self.test_loader = make_predict_split_loader(self.args, **self.kwargs)
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.test_loader, desc='\r', ncols=80)
        test_loss = 0.0
        with torch.no_grad():
            for i, sample in enumerate(tbar):
                image, target = sample['image'], sample['mask']
                if self.args.cuda:
                    image, target = image.cuda().float(), target.cuda().float()
                output = self.model(image)
                loss = self.criterion(output, target)
                test_loss += loss.item()
                tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
                pred = output.data.cpu().numpy()
                target = target.cpu().numpy()
                pred = np.argmax(pred, axis=1)
                pred = np.squeeze(pred)
                lut = get_GID_vege_lut()
                img = lut[pred]
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                name = sample["name"][0]
                # 黑色填充
                # squeeze_target = target.squeeze()
                # img[squeeze_target == 255] = (0, 0, 0)
                self.saver.save_img(img, name)

    def concat_predict(self):
        # 推理过程主循
        self.test_loader = make_predict_concat_loader(self.args, **self.kwargs)
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.test_loader, desc='\r', ncols=80)
        with torch.no_grad():
            for sample in tbar:
                if self.args.cuda:
                    image = sample[0].cuda().float()
                    if len(sample) == 3:
                        target = sample[1].cuda().float()

                shape = image.shape
                self.args.origin_size = shape[2:] # TODO
                pred = torch.zeros(size=(shape[0], 3, *shape[2:]))
                
                for i in range(0, shape[0], self.args.infer_batch_size):
                    pred[i:i+self.args.infer_batch_size] = self.model(image[i:i+self.args.infer_batch_size])
                pred = pred.data.cpu().numpy()

                if len(sample) == 3:
                    target = target.cpu().numpy()
                pred = recons_prob_map(pred, self.args.origin_size, self.args.crop_size, self.args.stride)
                pred = np.argmax(pred, axis=0)
                pred = np.squeeze(pred)
                lut = get_GID_vege_lut()
                img = lut[pred]
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                # 黑色填充
                # squeeze_target = target.squeeze()
                # img[squeeze_target == 255] = (0, 0, 0)
                # self.saver.save_img(img, name[0])

def get_GID_vege_lut():
    lut = np.zeros((512,3), dtype=np.uint8)
    lut[0] = [0,255,0]
    lut[1] =  [255, 0, 0]
    lut[2] =  [153,102,51]
    lut[3] =  [0,0,0]
    return lut