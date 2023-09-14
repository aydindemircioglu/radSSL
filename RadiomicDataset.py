# Written by Whalechen
# https://github.com/Tencent/MedicalNet/blob/18c8bb6cd564eb1b964bffef1f4c2283f1ae6e7b/datasets/brains18.py

import math
import os
import random
import pandas as pd
from glob import glob
import torch.nn as nn

import SimpleITK as sitk
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import nibabel
from scipy import ndimage

#import albumentations as A


class RadiomicDataset (Dataset):
    def __init__(self, root_dir, df, input_D, input_H, input_W, transforms = None):
        self.input_D = input_D
        self.input_H = input_H
        self.input_W = input_W
        self.img_list = df
        self.root_dir = root_dir
        # if transforms is None:
        #     self.transforms = A.Compose([
        #         A.RandomCrop3D(self.input_D, self.input_H, self.input_W),
        #         A.RandomBrightnessContrast(p=0.2),
        #         A.Normalize(),
        #     ])
        # else:
        #     self.transforms = transforms


    def __len__(self):
        return len(self.img_list)


    def __sitk2tensorarray__(self, sitk_image):
        image_array = sitk.GetArrayFromImage(sitk_image).astype("float32")
        image_array = np.transpose(image_array, (2, 1, 0))  # Transpose to match depth, height, width
        return image_array


    def __crop_to_lesion__(self, volume, label):
        non_zero_mask = label != 0

        z_indices = np.where(non_zero_mask.any(axis=(1, 2)))[0]
        h_indices = np.where(non_zero_mask.any(axis=(0, 2)))[0]
        w_indices = np.where(non_zero_mask.any(axis=(0, 1)))[0]

        min_z, max_z = z_indices[0], z_indices[-1]
        min_h, max_h = h_indices[0], h_indices[-1]
        min_w, max_w = w_indices[0], w_indices[-1]

        return volume[min_z:max_z + 1, min_h:max_h + 1, min_w:max_w + 1], label[min_z:max_z + 1, min_h:max_h + 1, min_w:max_w + 1]


    def __resize_data__(self, data):
        [depth, height, width] = data.shape
        scale = [self.input_D*1.0/depth, self.input_H*1.0/height, self.input_W*1.0/width]
        data = ndimage.zoom(data, scale, order=3)
        return data


    def __itensity_normalize__(self, volume, mask):
        pixels = volume[mask > 0]
        mean = pixels.mean()
        std = pixels.std()
        out = (volume - mean) / std
        return out


    def __getitem__(self, idx):
        patID = self.img_list.iloc[idx]["Patient"]
        img_name = glob(os.path.join(self.root_dir, str(patID) +"*image_1*.nii.gz") )[0]
        label_name = glob(os.path.join(self.root_dir, str(patID) +"*segmentation_1*.nii.gz") )[0]
        assert os.path.isfile(img_name)
        assert os.path.isfile(label_name)

        # read image and mask ,we always have both.
        img = sitk.ReadImage(img_name)
        assert img is not None
        mask = sitk.ReadImage(label_name)
        assert mask is not None

        img = self.__sitk2tensorarray__(img)
        mask = self.__sitk2tensorarray__(mask)

        img, mask = self.__crop_to_lesion__(img, mask)

        img = self.__resize_data__(img)
        mask = self.__resize_data__(mask)
        assert img.shape ==  mask.shape, "img shape:{} is not equal to mask shape:{}".format(img_array.shape, mask_array.shape)

        img = self.__itensity_normalize__(img, mask)
        img = img[np.newaxis, ...]
        return img



if __name__ == '__main__':
    from parameters import *
    import medicalnet

    # take a radiomics dataset for testing
    fsID = "641351e6bea78b47fd5ddbc23a229816"


            # just laod all the shit to see if it works
            # for r in range(len(rdata)):
            #     print(rdata.img_list.iloc[r]["Patient"])
            #     print(rdata[r].shape)
            # gulash


    # generate models
    for r in [10,18,34,50]:
        mpath = f"{basedir}/pretrained/medicalimagenet/resnet_{r}_23dataset.pth"
        W = 112; H = 112; D = 56
        model = medicalnet.generate_model(r, False, 'B', None, W, H, D, mpath)

        for dataID in dList:
            df = pd.read_csv(os.path.join(featuresPath, dataID, f"{fsID}_{dataID}_0_0_train.csv"))
            print (f"Loaded {dataID} with shape {df.shape}")
            rdata = RadiomicDataset (cachePath, df, D, H, W)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            batch_size = 4
            dataloader = DataLoader(rdata, batch_size=batch_size, shuffle=False, num_workers=16)
            _ = model.eval()

            fvs = []
            global_max_pooling = nn.AdaptiveMaxPool3d((1, 1, 1))
            for batch in dataloader:
                inputs = batch.to(device)
                predictions = model(inputs)
                pooled_predictions = global_max_pooling(predictions)
                pooled_predictions = pooled_predictions.view(predictions.size(0), -1)
                predictions = pooled_predictions.detach().cpu().numpy()
                fvs.append(predictions)

            fvs = np.vstack(fvs)


#
