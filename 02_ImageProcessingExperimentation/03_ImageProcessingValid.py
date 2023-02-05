"""
Written by: Rupesh Kumar Dey
Summary:
    - Python function script to run Image Processing Enhancement on the Images (validation set)
    - Creates Datasets 2 - 4. Dataset Set 5 Created Manually
    - Performs Image Enhancement of CLAHE, MULTISCALE and RAYLEIGH and saves image in updated folder
    - Calculates Metrics of the enhanced image compared to the original image and tabulates the data onto a CSV file.
"""
# importing required libraries
from CLAHECOTS import CLAHE
from MultiScaleFusionCOTS import MultiScaleFusion
from RayleighEnhancementPackageCOTS import RayleighHistogramStretching
import RayleighEnhancementPackageCOTS

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import peak_signal_noise_ratio as psnr

import cv2
import numpy
from PIL import Image
import os
import natsort
import matplotlib.pyplot as plt
from statistics import mean
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

"""
Initialize the image processing functions
"""
clahe_function = CLAHE()
multiscaleFusion_function = MultiScaleFusion()
rayleigh_function = RayleighHistogramStretching()

# Reading in raw data
df = pd.read_csv("valDataAnalysis.csv")
df["PSNR_CLAHE"] = 0
df["SSIM_CLAHE"] = 0
df["MSE_CLAHE"] = 0

df["PSNR_MULTISCALE"] = 0
df["SSIM_MULTISCALE"] = 0
df["MSE_MULTISCALE"] = 0

df["PSNR_RAYLEIGH"] = 0
df["SSIM_RAYLEIGH"] = 0
df["MSE_RAYLEIGH"] = 0

# print(df.head())

# defined project base path and path of raw image data
basePath = "C:\\Users\\User\\Documents\\Masters in AI\\GITractClassification" + "\\05_Dataset\\OriginalDatasetSplitted\\val\\"
path_RAW_normal = basePath + "0_normal\\"
path_RAW_ulcerative = basePath + "1_ulcerative_colitis\\"
path_RAW_polyp = basePath + "2_polyps\\"
path_RAW_esoph = basePath + "3_esophagitis\\"

# Initialize counter to count number of images processed to keep track
counter = 0

# Get the image id from the df
for image_id in df["image_id"]:
    # Get index
    index = df.index[(df["image_id"]) == image_id][0]

    # Original image_id
    baseName = image_id
    # image_id of processed image
    imageNameClahe = "CLAHE_" + image_id
    imageNameMulti = "MULTISCALE_" + image_id
    imageNameRayleigh = "RAYLEIGH_" + image_id

    # Check if Raw image exists
    image_path = basePath + df["path"][index]
    if not os.path.exists(image_path):
        continue

    # Read raw image and convert to RGB format
    raw_image = cv2.imread(image_path)
    raw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)

    # Process Image using enhancement functions
    claheImage = clahe_function.imageCLAHE(image_path, input_type = 2)
    multiscaleFusionImage = multiscaleFusion_function.multiScaleFusion(image_path, input_type = 2)
    rayleighImage = rayleigh_function.RayleighStretch(image_path, input_type=2)

    # PSNR Calculation
    psnr_clahe = psnr(raw_image,claheImage)
    psnr_multiscale = psnr(raw_image, multiscaleFusionImage)
    psnr_rayleigh = psnr(raw_image, rayleighImage)

    # MSE Calculation
    mse_clahe = mse(raw_image, claheImage)
    mse_multiscale = mse(raw_image, multiscaleFusionImage)
    mse_rayleigh = mse(raw_image, rayleighImage)

    # SSIM Calculation
    ssim_clahe = []
    ssim_multiscale = []
    ssim_rayleigh = []
    for i in range(3):
        ssim_clahe.append(ssim(raw_image[:,:,i],claheImage[:,:,i]))
        ssim_multiscale.append(ssim(raw_image[:,:,i],multiscaleFusionImage[:,:,i]))
        ssim_rayleigh.append(ssim(raw_image[:, :, i], rayleighImage[:, :, i]))
    ssim_clahe = mean(ssim_clahe)
    ssim_multiscale = mean(ssim_multiscale)
    ssim_rayleigh = mean(ssim_rayleigh)

    # Appending Results to Dataframe
    df["PSNR_CLAHE"][index] = psnr_clahe
    df["PSNR_MULTISCALE"][index] = psnr_multiscale
    df["PSNR_RAYLEIGH"][index] = psnr_rayleigh

    df["MSE_CLAHE"][index] = mse_clahe
    df["MSE_MULTISCALE"][index] = mse_multiscale
    df["MSE_RAYLEIGH"][index] = mse_rayleigh

    df["SSIM_CLAHE"][index] = ssim_clahe
    df["SSIM_MULTISCALE"][index] = ssim_multiscale
    df["SSIM_RAYLEIGH"][index] = ssim_rayleigh
#     
    # Convert np images into PIL savable format:
    PIL_raw_image = Image.fromarray(raw_image)
    PIL_claheImage = Image.fromarray(claheImage)
    PIL_multiscaleFusionImage = Image.fromarray(multiscaleFusionImage)
    PIL_rayleighImage = Image.fromarray(rayleighImage)

    # # Save enhanced Images in corresponding folders.
    """
    # Commented as to not interrupt already available images in project folder. Uncomment to rerun and process images again.
    PIL_claheImage.save(basePath + df["class"][index] + "_processed\\CLAHE\\" + imageNameClahe)
    PIL_multiscaleFusionImage.save(basePath + df["class"][index] + "_processed\\Multiscale\\" + imageNameMulti)
    PIL_rayleighImage.save(basePath + df["class"][index] + "_processed\\Rayleigh\\" + imageNameRayleigh)
    """
    # Creating a subplot to compare all 3 image enhancement techniques against the original image
    """
    # Commented as to not interrupt already availabe images in project folder. Uncomment to rerun and process image again
    plt.figure(figsize=[128, 128])
    plt.rcParams.update({"font.size": 100})
    plt.subplot(2, 2, 1)
    plt.imshow(Image.fromarray(raw_image))
    plt.title("ORIGINAL IMAGE")
    plt.subplot(2, 2, 2)
    plt.imshow(Image.fromarray(claheImage))
    plt.title("CLAHE")
    plt.subplot(2, 2, 3)
    plt.imshow(Image.fromarray(multiscaleFusionImage))
    plt.title("MULTISCALE")
    plt.subplot(2, 2, 4)
    plt.imshow(Image.fromarray(rayleighImage))
    plt.title("RAYLEIGH")
    plt.savefig(basePath + "merged\\compare_" + baseName)
    plt.cla()
    plt.clf()
    plt.close()
    """

    # Add 1 to counter
    counter = counter + 1

    # Print counter to check update
    print(counter)
    print("DONE FOR IMAGE: " + image_path + " Index no: " + str(index))

# # Save results to csv file
# df.to_csv('valDataAnalysisUpdated.csv',index = False)
print("The End")