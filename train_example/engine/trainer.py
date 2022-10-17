import os
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
import cv2
import torch
from collections import OrderedDict
from utils.loss import SegmentationLosses
from data import make_train_loader
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

class Trainer(object):
    def __init__(self, args):
        self.args = args

        # 定义保存
        self.saver = Saver(args)
        self.saver.save_experiment_config()
        # 使用amp
        self.use_amp = True if (APEX_AVAILABLE and args.use_amp) else False
        self.opt_level = args.opt_level

        # 定义dataloader
        kwargs = {'num_workers': args.num_worker, 'pin_memory': True, 'drop_last':True}
        self.train_loader, self.val_loader, self.test_loader = make_train_loader(args, **kwargs)
        self.nclass = args.nclass
        weight = torch.tensor([3,2,1])
        self.criterion = SegmentationLosses(weight=weight, cuda=args.cuda).build_loss(mode=args.loss_type)

        torch.cuda.empty_cache()
        # 定义网络
        if args.model_name == 'hrnet':
            model = get_HRNet_model(args)
        elif args.model_name == 'fast-nas':
            model = fastNas()
        elif args.model_name == 'PIDNet':
            model = get_PID_model(args)

        optimizer = torch.optim.SGD(
                model.parameters(),
                args.lr,
                momentum=args.momentum,
                weight_decay=args.weight_decay
            )


        self.model, self.optimizer = model, optimizer

        # Define Evaluator
        self.evaluator = Evaluator(self.nclass)
        # Define lr scheduler
        self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr,
                                      args.epochs, 1000, min_lr=args.min_lr)

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
            self.model, [self.optimizer] = amp.initialize(
                self.model, [self.optimizer], opt_level=self.opt_level,
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


            copy_state_dict(self.optimizer.state_dict(), checkpoint['optimizer'])
            self.best_pred = checkpoint['best_pred']
            self.best_pred = 0.7
            print(self.best_pred)
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            self.start_epoch = 0


    def training(self, epoch):
        train_loss = 0.0
        # try:
        #     self.train_loaderA.dataset.set_stage("train")
        # except AttributeError:
        #     self.train_loaderA.dataset.dataset.set_stage("train")  # for subset
        self.model.train()
        tbar = tqdm(self.train_loader, ncols=80)

        for i, sample in enumerate(tbar):
            image = sample["image"]
            target = sample["mask"]
            # image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda().float(), target.cuda().float()
            self.scheduler(self.optimizer, i, epoch, self.best_pred)
            self.optimizer.zero_grad()
            output = self.model(image)
            loss = self.criterion(output, target)
            if self.use_amp:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))


    def validation(self, epoch):
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.val_loader, desc='\r', ncols=80)
        test_loss = 0.0

        true_list = []

        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['mask']
            if self.args.cuda:
                image, target = image.cuda().float(), target.cuda().float()
            with torch.no_grad():
                output = self.model(image)
            loss = self.criterion(output, target)
            test_loss += loss.item()
            tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            # Add batch sample into evaluator
            self.evaluator.add_batch(target, pred)

        # Fast test during the training
        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU, IoU = self.evaluator.Mean_Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()

        print('Validation:')
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}, IoU:{}".format(Acc, Acc_class, mIoU, FWIoU, IoU))
        print('Loss: %.3f' % test_loss)
        new_pred = mIoU
        is_best = False

        state_dict = self.model.state_dict()
        self.saver.save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': state_dict,
            'optimizer': self.optimizer.state_dict(),
            'best_pred': self.best_pred,
        }, is_best, 'current_checkpoint.pth.tar')

        if new_pred > self.best_pred:
            is_best = True
            self.test_model(epoch)
            self.best_pred = new_pred
            if torch.cuda.device_count() > 1:
                state_dict = self.model.module.state_dict()
            else:
                state_dict = self.model.state_dict()
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': state_dict,
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best, 'epoch{}_checkpoint.pth.tar'.format(str(epoch + 1)))
        # self.saver.save_checkpoint({
        #     'epoch': epoch + 1,
        #     'state_dict': state_dict,
        #     'optimizer': self.optimizer.state_dict(),
        #     'best_pred': self.best_pred,
        # }, is_best, 'epoch{}_checkpoint.pth.tar'.format(str(epoch + 1)))

        self.saver.save_train_info(test_loss, epoch, Acc, mIoU, FWIoU, IoU, is_best)

    def test_model(self, epoch):
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.test_loader, desc='\r', ncols=80)
        test_loss = 0.0

        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['mask']
            if self.args.cuda:
                image, target = image.cuda().float(), target.cuda().float()
            with torch.no_grad():
                output = self.model(image)
            loss = self.criterion(output, target)
            test_loss += loss.item()
            tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            # Add batch sample into evaluator
            self.evaluator.add_batch(target, pred)

        # Fast test during the training
        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU, IoU = self.evaluator.Mean_Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()

        print('Test:')
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}, IoU:{}".format(Acc, Acc_class, mIoU, FWIoU, IoU))
        print('Loss: %.3f' % test_loss)

        self.saver.save_test_info(test_loss, epoch, Acc, mIoU, FWIoU, IoU)

    def predict(self):
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.test_loader, desc='\r', ncols=80)
        test_loss = 0.0

        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['mask']
            if self.args.cuda:
                image, target = image.cuda().float(), target.cuda().float()
            with torch.no_grad():
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

            squeeze_target = target.squeeze()
            img[squeeze_target == 255] = (0, 0, 0)

            self.saver.save_img(img, name)
    def select_true_data(self):
        self.model.eval()
        tbar = tqdm(self.val_loader, desc='\r', ncols=80)
        test_loss = 0.0
        true_list = []

        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['mask']
            if self.args.cuda:
                image, target = image.cuda().float(), target.cuda().float()
            with torch.no_grad():
                output = self.model(image)
            loss = self.criterion(output, target)
            test_loss += loss.item()
            tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            # Add batch sample into evaluator
            # 筛选合适的影像
            evaluator_patch = Evaluator(self.nclass)
            evaluator_patch.add_batch(target, pred)
            mIoU, IoU = evaluator_patch.Mean_Intersection_over_Union()
            if IoU[0] > 0.4 or IoU[1] > 0.4:
                if IoU[0] < 0.25:
                    continue
                for label_name in sample["name"]:
                    name = label_name.replace('/label', '/image').replace('_label', '_img')
                    true_list.append(name + '\t' + label_name)
        df = pd.DataFrame(true_list, columns=['one'])
        val_df = df.sample(frac=0.5, random_state=0, axis=0)
        test_df = df[~df.index.isin(val_df.index)]
        val_df.to_csv("F:/WHU_WY/LightRS/data/list/split/guangdong_val_true.lst", columns=['one'], index=False,
                      header=False)
        test_df.to_csv("F:/WHU_WY/LightRS/data/list/split/guangdong_test_true.lst", columns=['one'], index=False,
                       header=False)
        print("all true val has {}".format(str(len(true_list))))

    def select_confidence_data(self):
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.train_loader, desc='\r', ncols=80)
        test_loss = 0.0
        true_list = []

        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['mask']
            if self.args.cuda:
                image, target = image.cuda().float(), target.cuda().float()
            with torch.no_grad():
                output = self.model(image)
            loss = self.criterion(output, target)
            test_loss += loss.item()
            tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
            output = torch.softmax(output, dim=1)
            pred = output.data.cpu().numpy()
            confidence = np.max(pred, axis=1)
            m_confidence = sum(sum(confidence.squeeze())) / (400 * 400)
            if m_confidence > 0.75:
                for label_name in sample["name"]:
                    name = label_name.replace('/label', '/image').replace('_label', '_img')
                    true_list.append(name + '\t' + label_name)
                    print("{}/{}".format(str(len(true_list)), str(i)))
        df = pd.DataFrame(true_list, columns=['one'])
        df.to_csv("F:/WHU_WY/LightRS/data/list/split/guangdong_train_confi1.lst", columns=['one'], index=False,
                       header=False)
        print("all true val has {}".format(str(len(true_list))))
def get_GID_vege_lut():
    lut = np.zeros((512,3), dtype=np.uint8)
    lut[0] = [0,255,0]
    lut[1] =  [255, 0, 0]
    lut[2] =  [153,102,51]
    lut[3] =  [0,0,0]
    return lut