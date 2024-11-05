import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import random
import torch


class PolypDataset(data.Dataset):
    """
    dataloader for polyp segmentation tasks
    """
    def __init__(self, image_root, gt_root, trainsize, augmentations,color_aug):
        self.trainsize = trainsize
        self.augmentations = augmentations
        self.color_augmentation=color_aug
        print(self.augmentations)
        print(self.color_aug)
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.filter_files()
        self.size = len(self.images)
        if self.augmentations == True:
            print('Using RandomRotation, RandomFlip')
            self.img_transform = transforms.Compose([
                transforms.RandomRotation(90, resample=False, expand=False, center=None, fill=None),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])])
            self.gt_transform = transforms.Compose([
                transforms.RandomRotation(90, resample=False, expand=False, center=None, fill=None),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor()])
            
        else:
            print('no augmentation')
            self.img_transform = transforms.Compose([
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])])
            
            self.gt_transform = transforms.Compose([
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor()])
            

    def __getitem__(self, index):
        
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        
        seed = np.random.randint(2147483647) # make a seed with numpy generator 
        random.seed(seed) # apply this seed to img tranfsorms
        torch.manual_seed(seed) # needed for torchvision 0.7
        if self.img_transform is not None:
            if self.color_augmentation == True:
                #print("use color aug.")
                image = self.img_transform(self.color_aug(image))
            else:
                image = self.img_transform(image)
            
        random.seed(seed) # apply this seed to img tranfsorms
        torch.manual_seed(seed) # needed for torchvision 0.7
        if self.gt_transform is not None:
            gt = self.gt_transform(gt)
        return image, gt

    def filter_files(self):
        assert len(self.images) == len(self.gts)
        images = []
        gts = []
        for img_path, gt_path in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
        self.images = images
        self.gts = gts

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            # return img.convert('1')
            return img.convert('L')

    def resize(self, img, gt):
        assert img.size == gt.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST)
        else:
            return img, gt

    def color_aug(self,image):
        # B=[55.016,12.802,41.095,10.215]
        # G=[76.799,13.811,53.016,8.671]
        # R=[126.660,28.400,75.946,10.117]
        # H=[83.047,15.892,55.816,8.074]
        # S=[136.260,5.467,7.410,2.578]
        # V=[98.834,9.466,18.022,3.918]
        B_mean=np.random.normal(55.016, 12.802, 1)
        B_std = np.random.normal(41.095, 10.215, 1)
        G_mean=np.random.normal(76.799, 13.811, 1)
        G_std = np.random.normal(53.016, 8.671, 1)
        R_mean = np.random.normal(126.660, 28.400, 1)
        R_std = np.random.normal(75.946, 10.117, 1)
 
        transform_color_mean=np.array([[[R_mean[0],G_mean[0],B_mean[0]]]])
        transform_color_std = np.array([[[R_std[0], G_std[0], B_std[0]]]])
        # H_mean = np.random.normal(83.047, 15.892, 1)
        # H_std = np.random.normal(55.816, 8.074, 1)
        # S_mean = np.random.normal(136.260, 5.467, 1)
        # S_std = np.random.normal(7.410, 2.578, 1)
        # V_mean = np.random.normal(98.834, 9.466, 1)
        # V_std = np.random.normal(18.022, 3.918, 1)
        image=np.array(image,dtype=np.float32)
 
        img_mean = image.mean(axis=(0, 1), keepdims=True).squeeze()
        img_std = image.std(axis=(0, 1), keepdims=True).squeeze()
 
        image = (image - img_mean) / img_std * transform_color_std + transform_color_mean
        image = 255 * (image - np.min(image)) / (np.max(image) - np.min(image))
 
        img = Image.fromarray(np.uint8(image))
        return img


    def __len__(self):
        return self.size


def get_loader(image_root, gt_root, batchsize, trainsize, shuffle=True, num_workers=4, pin_memory=False, augmentation=False,color_aug=False):

    dataset = PolypDataset(image_root, gt_root, trainsize, augmentation,color_aug)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader


class test_dataset:
    def __init__(self, image_root, gt_root, testsize):
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.tif') or f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor()])
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)
        gt = self.gt_transform(self.binary_loader(self.gts[self.index])).unsqueeze(0)
        name = self.images[self.index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        return image, gt, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
