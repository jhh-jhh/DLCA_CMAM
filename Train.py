import torch
from torch.autograd import Variable
import os
import argparse
from datetime import datetime
from lib.convnext_cmam import PolypPVT
from utils.dataloader import get_loader, test_dataset
from utils.utils import clip_gradient, adjust_lr, AvgMeter
import torch.nn.functional as F
import numpy as np
import logging
import pandas as pd

import matplotlib.pyplot as plt

# Initialize the variables
best = 0.85
global_epoch=1
list_loss=[]
list_wighted_mean_dice=[]

def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    return (wbce + wiou).mean()


def test(model, path, dataset,size):

    data_path = os.path.join(path, dataset)
    image_root = '{}/images/'.format(data_path)
    gt_root = '{}/masks/'.format(data_path)
    model.eval()
    num1 = len(os.listdir(gt_root))
    test_loader = test_dataset(image_root, gt_root, size)
    DSC = 0.0
    with torch.no_grad():
        for i in range(num1):
            image, gt, name = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()

            res, res1  = model(image)
            # eval Dice
            res = F.upsample(res+res1 , size=gt.shape[2:], mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            input = res
            target = np.array(gt)
            N = gt.shape
            smooth = 1
            input_flat = np.reshape(input, (-1))
            target_flat = np.reshape(target, (-1))
            intersection = (input_flat * target_flat)
            dice = (2 * intersection.sum() + smooth) / (input.sum() + target.sum() + smooth)
            dice = '{:.4f}'.format(dice)
            dice = float(dice)
            DSC = DSC + dice

    return DSC / num1



def train(train_loader, model, optimizer, epoch, test_path):
    model.train()
    global best,global_epoch,list_loss,list_wighted_mean_dice
    size_rates = [ 0.75,1] 
    loss_P2_record = AvgMeter()
    for i, pack in enumerate(train_loader, start=1):
        for rate in size_rates:
            optimizer.zero_grad()
            # ---- data prepare ----
            images, gts = pack
            images = Variable(images).cuda()
            gts = Variable(gts).cuda()
            # ---- rescale ----
            trainsize = int(round(opt.trainsize * rate / 32) * 32)
            if rate != 1:
                images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
            # ---- forward ----
            P1, P2= model(images)
            # ---- loss function ----
            loss_P1 = structure_loss(P1, gts)
            loss_P2 = structure_loss(P2, gts)
            loss = loss_P1 + loss_P2 
            list_loss.append(loss.data)
            # ---- backward ----
            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            # ---- recording loss ----
            if rate == 1:
                loss_P2_record.update(loss_P2.data, opt.batchsize)
        # ---- train visualization ----
        if i % 20 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                  ' lateral-5: {:0.4f}]'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step,
                         loss_P2_record.show()))
    # save model 
    save_path = (opt.train_save)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # torch.save(model.state_dict(), save_path +str(epoch)+ 'PolypPVT.pth')
    torch.save(model.state_dict(), save_path + 'PolypPVT.pth')
    # choose the best model

    global dict_plot

    # todo Change test1path to local test dataset path
    test1path = "/dataset/TestDataset/"
    if (epoch + 1) % 1 == 0:
        dice_list = []
        for dataset in ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']:
            dataset_dice = test(model, test1path, dataset,opt.trainsize)
            logging.info('epoch: {}, dataset: {}, dice: {}'.format(epoch, dataset, dataset_dice))
            print(dataset, ': ', dataset_dice)
            dice_list.append(dataset_dice)
            dict_plot[dataset].append(dataset_dice)
        weighted_avg_dice = dice_list[0] * 0.07519 + dice_list[1] * 0.07769 + dice_list[2] * 0.12531 + dice_list[3] * 0.47619 + \
                   dice_list[4] * 0.24561
        avg_dice = (dice_list[0] + dice_list[1] + dice_list[2] + dice_list[3] + dice_list[4]) / 5
        dict_plot['test'].append(weighted_avg_dice)
        list_wighted_mean_dice.append(weighted_avg_dice)
        print('##############################################################################', epoch, weighted_avg_dice)
        print('##############################################################################', 'avg_dice', avg_dice)
        if weighted_avg_dice > best:
            #best = weighted_avg_dice
            torch.save(model.state_dict(), save_path + str(global_epoch) + 'PolypPVT-best.pth')
            logging.info(
                '##############################################################################{}:{}'.format(epoch,weighted_avg_dice))
    global_epoch+=1

def plot_train(dict_plot=None, name = None):
    color = ['red', 'lawngreen', 'lime', 'gold', 'm', 'plum', 'blue']
    line = ['-', "--"]
    for i in range(len(name)):
        plt.plot(dict_plot[name[i]], label=name[i], color=color[i], linestyle=line[(i + 1) % 2])
        transfuse = {'CVC-300': 0.902, 'CVC-ClinicDB': 0.918, 'Kvasir': 0.918, 'CVC-ColonDB': 0.773,'ETIS-LaribPolypDB': 0.733, 'test':0.83}
        plt.axhline(y=transfuse[name[i]], color=color[i], linestyle='-')
    plt.xlabel("epoch")
    plt.ylabel("dice")
    plt.title('Train')
    plt.legend()
    plt.savefig('eval.png')
    # plt.show()


if __name__ == '__main__':
    dict_plot = {'CVC-300':[], 'CVC-ClinicDB':[], 'Kvasir':[], 'CVC-ColonDB':[], 'ETIS-LaribPolypDB':[], 'test':[]}
    name = ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB', 'test']
    ##################model_name#############################
    model_name = 'convnext_cmam_color'
    ###############################################
    parser = argparse.ArgumentParser()

    parser.add_argument('--epoch', type=int,
                        default=200, help='epoch number')

    parser.add_argument('--lr', type=float,
                        default=1e-4, help='learning rate')

    parser.add_argument('--optimizer', type=str,
                        default='AdamW', help='choosing optimizer AdamW or SGD')

    parser.add_argument('--augmentation',
                        default=True, help='choose to do random flip rotation')
                        
    parser.add_argument('--color_augmentation',
                        default=True, help='choose to do random flip rotation')

    parser.add_argument('--batchsize', type=int,
                        default=1, help='training batch size')

    parser.add_argument('--trainsize', type=int,
                        default=352, help='training dataset size')

    parser.add_argument('--num_worker', type=int,
                        default=1, help='training dataset size')

    parser.add_argument('--clip', type=float,
                        default=0.5, help='gradient clipping margin')

    parser.add_argument('--decay_rate', type=float,
                        default=0.1, help='decay rate of learning rate')

    parser.add_argument('--decay_epoch', type=int,
                        default=50, help='every n epochs decay learning rate')

    # todo Change train_path to local train dataset path
    parser.add_argument('--train_path', type=str,
                        default="./dataset/TrainDataset/",
                        help='path to train dataset')

    # todo Change train_path to local test dataset path
    parser.add_argument('--test_path', type=str,
                        default="./dataset/TestDataset/",
                        help='path to testing Kvasir dataset')

    # todo Change path to save model
    parser.add_argument('--train_save', type=str,
                        default='./model_pth/'+model_name+'/')

    opt = parser.parse_args()
    logging.basicConfig(filename='train_log.log',
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')

    #torch.backends.cudnn.enabled = False
    # ---- build models ----
    torch.cuda.set_device(0)  # set your gpu device
    model = PolypPVT(32).cuda()

    params = model.parameters()

    if opt.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(params, opt.lr, weight_decay=1e-4)
    else:
        optimizer = torch.optim.SGD(params, opt.lr, weight_decay=1e-4, momentum=0.9)

    print(optimizer)
    image_root = '{}/images/'.format(opt.train_path)
    gt_root = '{}/masks/'.format(opt.train_path)

    train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize,
                              augmentation=opt.augmentation,num_workers=opt.num_worker,color_aug=opt.color_augmentation)
    total_step = len(train_loader)

    print("#" * 20, "Start Training", "#" * 20)

    model_path = opt.train_save+"checkpoint.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print("Successfully loaded network model, continue training ...")
    else:
        print("No reloadable network model, retrain ...")
    
    for epoch in range(1, opt.epoch + 1):
        adjust_lr(optimizer, opt.lr, epoch, 0.1, 200)
        train(train_loader, model, optimizer, epoch, opt.test_path)
        
    list_loss_cpu = [tensor.cpu().numpy() for tensor in list_loss]  
        
    df=pd.DataFrame(list_loss_cpu)
    df.to_csv(opt.train_save+"loss.csv", index=False)
    
    df1=pd.DataFrame(list_wighted_mean_dice)
    df1.to_csv(opt.train_save+"weighted_mean_dice.csv", index=False)