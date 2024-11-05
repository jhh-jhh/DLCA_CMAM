import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
from lib.convnext_cmam import *
from utils.test_dataloader import test_dataset

import cv2

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot'    ,default=None)
    parser.add_argument('--testsize', type=int, default=352, help='testing size')
    # todo Change the path to our model
    parser.add_argument('--pth_path', type=str, default="./model_pth/convnext_cmam_color/checkpoint.pth")
    opt = parser.parse_args()
    model=PolypPVT(32)
    model.load_state_dict(torch.load(opt.pth_path, map_location='cuda:0'))
    model.cuda()
    model.eval()
    for _data_name in ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']:

        ##### put data_path here #####
        data_path = './dataset/TestDataset/{}'.format(_data_name)
        ##### save_path #####
        save_path = './result_map/polyp/{}/'.format(_data_name)

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        image_root = '{}/images/'.format(data_path)
        gt_root = '{}/masks/'.format(data_path)
        num1 = len(os.listdir(gt_root))
        test_loader = test_dataset(image_root, gt_root, 352)
        for i in range(num1):
            image, gt, name = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()
            P1,P2 = model(image)
            res = F.upsample(P1+P2, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            cv2.imwrite(save_path+name, res*255)
        print(_data_name, 'Finish!')
