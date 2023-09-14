import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.python.keras.applications import inception_resnet_v2
from glob import glob
import torch

from parameters import *


class KerasFeatureExtractor:
    def __init__(self, model, layer_name, transformation, batch_size = 32, agg = "max"):
        self.model = model
        self.layer_name = layer_name
        self.transformation = transformation
        self.feature_extractor = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
        self.batch_size = batch_size
        self.agg = agg

    def process_image(self, image_path):
        img = load_img(image_path)
        img = img_to_array(img)
        img = self.transformation(img)
        img = tf.expand_dims(img, axis=0)
        feature_vector = self.feature_extractor(img)
        return feature_vector.numpy()

    def apply_global_max_pooling(self, features):
        if len(features.shape) > 2:
            features = GlobalMaxPooling2D()(features)
        return features

    def __call__(self, image_paths):
        num_images = len(image_paths)
        feature_vectors = []

        for i in range(0, num_images, self.batch_size):
            batch_paths = image_paths[i:i + self.batch_size]
            batch_features = []

            for image_path in batch_paths:
                feature_vector = self.process_image(image_path)
                feature_vector = self.apply_global_max_pooling(feature_vector)
                batch_features.append(feature_vector)

            batch_features = np.vstack(batch_features)
            feature_vectors.append(batch_features)

        sfv = [np.vstack(feature_vectors)]

        slFeats = []
        for s in sfv:
            # avg pooling missing? # this is stupid, but because its copy&paste,
            # and i wont change it now, muahahaha
            slFeats.append(s[-1])
        slFeats = pd.DataFrame(slFeats)
        agg = "max"
        if agg == "max":
            slFeats = slFeats.max(axis = 0)
        elif agg == "mean":
            slFeats = slFeats.mean(axis = 0)
        return slFeats


if __name__ == "__main__":
    def getSlices (patID, dataID):
        allSlices = glob(os.path.join(slicesPath, dataID, f"*{patID}*.png"), recursive = True)
        return allSlices

    def transform_image(img):
        img = tf.image.resize(img, (224, 224))
        img = tf.image.central_crop(img, central_fraction=1.0)
        return img

    models = []
    models.append(f"{basedir}/pretrained/radimagenet/RadImageNet-ResNet50_notop.h5")
    models.append(f"{basedir}/pretrained/radimagenet/RadImageNet-DenseNet121_notop.h5")
    models.append(f"{basedir}/pretrained/radimagenet/RadImageNet-InceptionV3_notop.h5")
    models.append(f"{basedir}/pretrained/radimagenet/RadImageNet-IRV2_notop.h5")

    for m in models:
        print ("\n\n\n\n\n", m)
        model = tf.keras.models.load_model(m)
        layer_name = model.layers[-1].name
        print (model.summary())
        processor = KerasFeatureExtractor(model, layer_name, transform_image)
        image_files = getSlices("Liver-076", "Liver")
        feature_vectors = processor(image_files)
        print(feature_vectors.shape)  # Print the shape of the extracted features

#
