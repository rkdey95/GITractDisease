"""
Written by: Rupesh Kumar Dey
Introduction: Python class object to perform CLAHE image processing
"""

# Importing required libraries
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import math
import copy
from PIL import Image, ImageCms
from skimage.color import rgb2lab, lab2rgb

class CLAHE:
    # Initialization
    def __init__(self):
        self.path = None

    # function to perform CLAHE enhancment
    def imageCLAHE(self,input, input_type):
        """
        Function that Performs CLAHE image enhancement
        Inputs:
            a) input - Image array or Image file location
            b) input_type - Image type 1 - Image array, 2 - Image File Location
        """
        assert input_type in [1, 2], "Please specify input type of either ""1 - Image"" or ""2 - File location"""

        # If input type is image
        if input_type == 1:
            # CV2 image BGR format
            input_image8 = input

        # If input type is File Location
        elif input_type == 2:
            # Image path location
            input_image8 = cv2.imread(input)

        # Convert image from BGR to RGB
        input_image8 = cv2.cvtColor(input_image8, cv2.COLOR_BGR2RGB)

        # Clip Limit
        Ncl = 0.002
        m,n = input_image8.shape[0], input_image8.shape[1]
        shape = m * n

        # Create a copy of the image
        clahe_image8 = copy.deepcopy(input_image8)

        # Loop through each colour channel and perform CLAHE enhancement on each channel
        for i in range(3):
            temp_avg = clahe_image8[:,:,i].flatten()
            temp_avg = np.unique(temp_avg).shape[0]
            Navg = shape / temp_avg
            Cl = Ncl * Navg
            clahe = cv2.createCLAHE(clipLimit = int(Cl), tileGridSize = (8,8))
            clahe_image8[:,:,i] = clahe.apply(clahe_image8[:,:,i])

        # Return CLAHE enhanced image.
        return clahe_image8