import torch
import numpy as np
import pandas as pd
from glob import glob
import torch.nn as nn

from RadiomicDataset import *
from parameters import *


class MedicalImageNetFeatureExtractor:
    def __init__(self, model, D, H, W):
        self.model = model
        self.D = D
        self.H = H
        self.W = W

    def __call__(self, subdf):
        # image_path will only be a patID number

        # create its own dataset
        rdata = RadiomicDataset (cachePath, subdf, self.D, self.H, self.W)
        dataloader = DataLoader(rdata, batch_size=1, shuffle=False, num_workers=16)

        fvs = []
        global_max_pooling = nn.AdaptiveMaxPool3d((1, 1, 1))
        for batch in dataloader:
            inputs = batch.cuda()
            predictions = self.model(inputs)
            pooled_predictions = global_max_pooling(predictions)
            pooled_predictions = pooled_predictions.view(predictions.size(0), -1)
            predictions = pooled_predictions.detach().cpu().numpy()
            fvs.append(predictions)
        fvs = np.vstack(fvs)
        fvs = pd.DataFrame(fvs)
        fvs = fvs.max(axis = 0) # stupid fix to convert
        return fvs

#


if __name__ == "__main__":
    import medicalnet
    fsID = "641351e6bea78b47fd5ddbc23a229816"
    for dataID in dList:
        df = pd.read_csv(os.path.join(featuresPath, dataID, f"{fsID}_{dataID}_0_0_train.csv"))

        for j in range(len(df)):
            subdf = pd.DataFrame(df.iloc[j]).T
            W = 112; H = 112; D = 56
            rdata = RadiomicDataset (cachePath, subdf, D, H, W)
            dataloader = DataLoader(rdata, batch_size=1, shuffle=False, num_workers=16)

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
