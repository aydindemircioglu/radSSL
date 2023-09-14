import torch
import numpy as np
from glob import glob
import pandas as pd
import torch.nn as nn

from mmpretrain import FeatureExtractor, get_model

from parameters import *


class MMPretrainFeatureExtractor:
    def __init__(self, model, stage, agg = "max"):
        self.model = model
        self.stage = stage
        self.agg = agg
        self.extractor = FeatureExtractor(self.model, stage = self.stage)

    def __call__(self, image_paths):
        sfv = self.extractor (image_paths)
        slFeats = []
        for s in sfv:
            # avg pooling missing?
            if len(s[-1].shape) > 1:
                m = nn.AdaptiveMaxPool2d((1, 1))
                z = m(s[-1]).squeeze()
                slFeats.append(z.detach().cpu().numpy())
            else:
                # if we have a torch tensor we need to do something
                if torch.is_tensor(s[-1]) == True:
                    slFeats.append(s[-1].detach().cpu().numpy())
                else:
                    slFeats.append(s[-1])
        slFeats = pd.DataFrame(slFeats)
        if self.agg == "max":
            slFeats = slFeats.max(axis = 0)
        elif self.agg == "mean":
            slFeats = slFeats.mean(axis = 0)
        return slFeats


if __name__ == "__main__":
    pass

#
