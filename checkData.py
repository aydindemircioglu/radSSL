#

import pandas as pd
import logging
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt

from helpers import *
from parameters import *


def gatherInfos(blackList, dataID, metas):
    rx, ry, rz = [], [], []
    sx, sy, sz = [], [], []
    data = getData (dataID, dropPatID = False, useBlacklist = False, imagePath = radDBPath)

    metas[dataID] = {}
    for i, (idx, row) in enumerate(data.iterrows()):
        if row["Patient"] in blackList:
            continue
        reader = sitk.ImageFileReader()
        reader.SetFileName(row["Image"])
        reader.LoadPrivateTagsOn()
        reader.ReadImageInformation()
        rx.append(float(reader.GetMetaData("pixdim[1]")))
        ry.append(float(reader.GetMetaData("pixdim[2]")))
        rz.append(float(reader.GetMetaData("pixdim[3]")))
        sx.append(float(reader.GetMetaData("dim[1]")))
        sy.append(float(reader.GetMetaData("dim[2]")))
        sz.append(float(reader.GetMetaData("dim[3]")))
        metas[dataID][row["Patient"]] = rz[-1]
    return rx, ry, rz, sx, sy, sz, metas


def getStr (rx):
    s = str(np.round(np.median(rx), 1)) + " (" + str( np.round(np.min(rx),1) ) + " - " + str(np.round(np.max(rx),1)) + ")"
    return s


def renderSlice (patID, bestSlice, gamma, dataID):
    fvol = glob(os.path.join(cachePath, patID +"*image_1*.nii.gz") )
    fmask = glob(os.path.join(cachePath, patID +"*segmentation_1*.nii.gz") )
    volITK = sitk.ReadImage(fvol)
    volMaskITK = sitk.ReadImage(fmask)
    vol = sitk.GetArrayFromImage(volITK)[0,:,:,:]
    volMask = sitk.GetArrayFromImage(volMaskITK)[0,:,:,:]

    img = vol[bestSlice,:,:].copy()
    mask = volMask[bestSlice,:,:].copy().astype(np.uint8)
    mask = (mask > 0).astype(np.uint8)

    # small fix for HN
    if dataID == "HN":
        h, w = img.shape
        img = img[h//4:3*h//4, h//4:3*h//4]
        mask = mask[h//4:3*h//4, h//4:3*h//4]

    vol = img
    if dataID in CT_datasets:
        vol[vol < -1024] = -1024
        vol[vol > 2048] = 2048
        vol = (vol - np.min(vol))/(np.max(vol) - np.min(vol))
        vol = np.asarray(255*vol, dtype = np.uint8)
    else:
        vol = (vol - np.min(vol))/(np.max(vol) - np.min(vol))
        vol = np.asarray(255*vol, dtype = np.uint8)
    img = vol
    gamma_corrected = np.power(img / 255.0, gamma)
    img = (gamma_corrected * 255).astype(np.uint8)

    overlay_color = (16, 0, 112)
    a = 90
    cimg = cv2.merge([img, img, img])
    cmask = cimg.copy() * 0 + (1,1,1)
    cmask = (cmask*overlay_color*cv2.merge([mask, mask, mask])).astype(np.uint8)
    overlay = cv2.addWeighted(cimg, 1 - a / 255, cmask, a / 255, 0)
    pngimgName = os.path.join("./slices/", f'{dataID}___{patID}_{bestSlice}_image.png')
    cv2.imwrite(pngimgName, overlay)


def generateSampleImages (dList):
    slices = [("CRLM", "CRLM-061", 292, 1.2), ("C4KCKiTS", "KiTS-00104", 337, 1.2),
            ("Desmoid", "Desmoid-077", 54, 0.5), ("GBM", "GBM-76-6282", 79, 1.0),
            ("GIST", "GIST-148", 372, 1.1), ("HN", "HN1461", 208, 1.0),
            ("ISPY1", "ISPY1-1168", 82, 0.4), ("Lipo", "Lipo-002", 188, 1.0),
            ("Liver", "Liver-092", 113, 1.0), ("Melanoma", "Melanoma-062", 103, 1.0)]
    for dataID, patID, bestSlice, gamma in slices:
        renderSlice (patID, bestSlice, gamma, dataID)





def joinImages ():
    image_directory = 'slices'
    image_files = [file for file in os.listdir(image_directory) if file.endswith('.png')]
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.dpi = 300
    image_files = sorted(image_files)
    for i, image_file in enumerate(image_files):
        # Extract DATAID from the image filename
        data_id = image_file.split('_')[0]

        # Read the image using OpenCV
        img = cv2.imread(os.path.join(image_directory, image_file))

        # Get the original image dimensions
        original_height, original_width, _ = img.shape

        # Define the target width and height for resizing (adjust as needed)
        target_width, target_height = 300, 300

        # Calculate the aspect ratio
        aspect_ratio = original_width / original_height

        # Resize the image while preserving the aspect ratio
        if aspect_ratio > 1:
            new_width = target_width
            new_height = int(target_width / aspect_ratio)
        else:
            new_height = target_height
            new_width = int(target_height * aspect_ratio)

        img = cv2.resize(img, (new_width, new_height))

        # Create a blank canvas with the target dimensions
        canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)

        # Calculate the position to place the resized image on the canvas
        x_offset = (target_width - new_width) // 2
        y_offset = (target_height - new_height) // 2

        # Paste the resized image onto the canvas
        canvas[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = img

        # Display the image and DATAID
        axes[i // 5, i % 5].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[i // 5, i % 5].set_title(data_id, fontsize=20)  # Double the font size
        axes[i // 5, i % 5].axis('off')

    # Adjust spacing between subplots
    plt.subplots_adjust(wspace=0.1, hspace=0.3)

    # Save the figure as a PNG file
    plt.savefig('paper/Figure2.png', bbox_inches='tight', dpi = 300)



if __name__ == "__main__":
    print ("Hi.")

    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.WARNING)
    
    blackList = []
    metas = {}
    for dataID in dList:
        _, _, _, _, _, _, metas = gatherInfos(blackList, dataID, metas)
    
    # generate blackList
    for dataID in dList:
        data = getData (dataID, dropPatID = False, useBlacklist = False, imagePath = radDBPath)
        rmedian = np.median(list(metas[dataID].values()))
        #print (dataID, rmedian)
        for k in metas[dataID]:
            if metas[dataID][k] > 2*rmedian:
                blackList.append(k)
    
    # write blacklist
    pd.DataFrame(blackList).to_csv("./data/blacklist.csv", index = False)
    
    # gather without those
    dTable = []
    for dataID in dList:
        rx, ry, rz, sx, sy, sz, metas = gatherInfos(blackList, dataID, metas)
        if dataID in CT_datasets:
            mod = "CT"
        else:
            mod = "MR"
        dTable.append({"data": dataID, "Modality": mod, "N": len(rx), "Inplane Resolution": getStr(rx),  "Slice Thickness": getStr(rz)})
        #print(dataID, sorted(rz))
    dTable = pd.DataFrame(dTable)
    dTable.to_excel("./results/Data.xlsx")
    print (dTable)

    # generate a figure with all images
    generateSampleImages (dList)
    joinImages()

#
