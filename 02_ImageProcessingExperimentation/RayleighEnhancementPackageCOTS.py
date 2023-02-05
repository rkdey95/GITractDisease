"""
Written by: Rupesh Kumar Dey
Introduction:
    Python script for the Rayleigh Histogram stretching class.

    Classes:
        a) pixelNode - class that stores attributes of a pixel node
        b) pixelNodeLower - class that stores attributes of a pixel node in the lower region of the image
        c) RayleighHistogramStretching - class that performs Rayleigh histogram stretching
"""

# Importing required libraries
import os
import natsort
import numpy as np
import datetime
import cv2
from skimage.color import rgb2hsv, hsv2rgb
from PIL import Image
import matplotlib.pyplot as plt
import math

class pixelNode(object):
    # Initialization
    def __init__(self,height_idx,width_idx,pixel):
        self.x = height_idx
        self.y = width_idx
        self.value = pixel

    # Print node data
    def displayNodeData(self):
        print(self.x, self.y, self.value)

# Class to store node pixel
class pixelNodeLower(object):
    def __init__(self,height_idx,width_idx, pixel):
        self.x = height_idx
        self.y = width_idx
        self.value = pixel

    def displayNodeData(self):
        print(self.x,self.y, self.value)

# Class that performs Rayleigh Histogram Stretching
class RayleighHistogramStretching:
    def __init__(self):
        self.path = None
        self.files = None
        self.height = None
        self.width = None
        self.esp = 2.2204e-16


    def RayleighStretch(self,input, input_type = 1, output_path = None):
        """
        # Function that performs the Rayleigh Image Enhancement
        Inputs:
            a) input - inputs of either an image, filename or a folder of images
            b) input_type - 1) Image itself, 2) Image_filename itself 3)  folder of files with images.
            c) output_path - Path of where to save the enhanced image (if applicable)
        """
        # Validating inputs
        assert input_type in [1,2,3], "Please specify input type of either ""1 - Image"" or ""2 - File location"" or ""3 - Folder"" "

        # If input is an image
        if input_type == 1:
            input_image = input
            # Check that input image is a np.array
            assert isinstance(input_image,np.ndarray), "Please insert image that is read using cv2.imread(). Image should be in np.ndarray in BGR"
            # Convert format from BGR to RGB
            input_image = cv2.cvtColor(input_image,cv2.COLOR_BGR2RGB)
            # Image dimensions
            self.height = input_image.shape[0]
            self.width = input_image.shape[1]
            # Perform Rayleigh image processing enhancement
            processed_image = self.equalizeRGB(input_image)
            processed_image = self.histogram_stretching(processed_image)
            processedImageLower, processedImageUpper = self.RayleighStretching(processed_image)
            processed_image = 0.5 * (processedImageLower.astype(np.float64) + processedImageUpper.astype(np.float64))
            processed_image = self.HSVEqualization(processed_image)
            return processed_image

        # If filename  / path of image is given
        elif input_type == 2:
            # Read image and perform enhancement
            input_image = cv2.imread(input)
            input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
            self.height = input_image.shape[0]
            self.width = input_image.shape[1]
            processed_image = self.equalizeRGB(input_image)
            processed_image = self.histogram_stretching(processed_image)
            processedImageLower, processedImageUpper = self.RayleighStretching(processed_image)
            processed_image = 0.5 * (processedImageLower.astype(np.float64) + processedImageUpper.astype(np.float64))
            processed_image = self.HSVEqualization(processed_image)
            return processed_image

        # If input given is a folder of images
        elif input_type == 3:
            assert os.path.isdir(input) == True, "Please ensure that the folder location of images are specified."
            # assert output_path == None, "Please specify an output path"

            # Array to store images in the folder
            compiled_image = []

            # Initialize parameters
            self.path = input
            self.files = os.listdir(self.path)
            self.files = natsort.natsorted(self.files)
            valid_extension = [".jpg"]

            # Loop through each image in the folder and perform image enhancement
            for idx in range(len(self.files)):
                file = self.files[idx]
                filepath = self.path + "/" + file
                ext = os.path.splitext(filepath)
                if ext[1] not in valid_extension:
                    continue
                if os.path.isfile(filepath):
                    print('********    file   ********',file)
                    image = cv2.imread(filepath)
                    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                    self.height =  image.shape[0]
                    self.width = image.shape[1]
                    processed_image = self.equalizeRGB(image)
                    processed_image = self.histogram_stretching(processed_image)
                    processedImageLower, processedImageUpper = self.RayleighStretching(processed_image)
                    processed_image = 0.5 * (processedImageLower.astype(np.float64) + processedImageUpper.astype(np.float64))
                    processed_image = self.HSVEqualization(processed_image)
                    processed_image = Image.fromarray(processed_image)

                    compiled_image.append(processed_image)

            return compiled_image

    def equalizeRGB(self, image):
        """
        Function to perform Histogram Equalization
        Inputs:
            a) Image

        Outputs:
            b) processed image
        """
        # Convert image to 32 bit and split to individual channels
        image = np.float32(image)
        red, green, blue = cv2.split(image)

        # calculate mean value for each channel
        mean_red = np.mean(red)
        mean_green = np.mean(green)
        mean_blue =  np.mean(blue)

        # calculate the max, min and median between the 3 color channels' average values
        color_avg = np.array((mean_red, mean_green, mean_blue))
        color_max = np.max(color_avg)
        color_min = np.min(color_avg)
        color_median = np.median(color_avg)

        # Calculate gain factors
        A_val = color_median / color_min
        B_val = color_median / color_max

        # Multiply gain to corresponding min and max colour channels
        if color_min == mean_red:
            red *= A_val
        elif color_min == mean_green:
            green *= A_val
        else:
            blue *= A_val

        if color_max == mean_red:
            red *= B_val
        elif color_max == mean_green:
            green *= B_val
        else:
            blue *= B_val

        # Create a blank image to replace
        output_image = np.zeros(image.shape, 'float64')

        # Replace channels of output_image with the corresponding processing RGB channels
        temp_store = [red, green, blue]
        for i in range(0,3):
            output_image[:,:,i] = temp_store[i]
        output_image = np.clip(output_image, 0, 255)

        return output_image

    def histogram_stretching(self,image):
        # Get image dimensions
        height = image.shape[0]
        width =  image.shape[1]

        # For extract the maximum and minimum pixel from each channel
        for chan in range(3):
            max_pixel = np.max(image[:,:,chan])
            min_pixel = np.min(image[:,:,chan])

            # perform stretching on each of the pixels in the image
            for h in range(height):
                for w in range(width):
                    image[h,w,chan] = (((image[h,w,chan] - min_pixel) * (255 - 0)) / (max_pixel - min_pixel)) + 0
        return image

    def RayleighStretching(self,image):
        # Get image dimensions
        height = image.shape[0]
        width = image.shape[1]

        # Split channels
        red_channel = image[:,:,0]
        green_channel = image[:,:,1]
        blue_channel = image[:,:,2]

        # Split each channel histogram into lower and upper regions based on calculated mean
        redLower, redUpper = self.splitRayleighRegions(red_channel, height, width)
        greenLower, greenUpper = self.splitRayleighRegions(green_channel, height, width)
        blueLower, blueUpper = self.splitRayleighRegions(blue_channel, height, width)

        # fit lower histogram pixels into a temporary imageLower image
        imageLower = np.zeros(image.shape)
        imageLower[:, :, 0] = redLower
        imageLower[:, :, 1] = greenLower
        imageLower[:, :, 2] = blueLower
        imageLower = imageLower.astype(np.uint8)

        # Fit the upper histogram pixels into a temporary imageUpper image
        imageUpper = np.zeros(image.shape)
        imageUpper[:, :, 0] = redUpper
        imageUpper[:, :, 1] = greenUpper
        imageUpper[:, :, 2] = blueUpper
        imageUpper = imageUpper.astype(np.uint8)

        return imageLower, imageUpper


    def splitRayleighRegions(self, color_channel, height, width):
        """
        Function to split image channel into lower and upper regions
        Inputs:
            a) color_channel - image channel
            b) height - height of image
            c) width - width of image

        Outputs:
            a) colourLower - lower region of the split image
            b) colourUpper - upper region of the split image
        """

        # Calculate image parameters
        numPixels = height * width
        maxPixel = np.max(color_channel)
        minPixel = np.min(color_channel)
        averagePixel = np.mean(color_channel)

        # Define arrays to store images of the lower and upper nodes
        lowerNode = []
        upperNode = []

        # Loop through each pixel and store nodes in lower and upper nodes respectively
        for h in range(height):
            for w in range(width):
                tempNode = pixelNode(h,w,color_channel[h,w])
                tempLowerNode = pixelNodeLower(h,w,color_channel[h,w])

                lowerNode.append(tempLowerNode)
                upperNode.append(tempNode)

        # Sort node pixels
        lowerNode = sorted(lowerNode, key = lambda node: node.value, reverse = False)
        upperNode = sorted(upperNode, key = lambda node: node.value, reverse = False)

        # Setting cutoff value for the pixels
        for pix in range(numPixels):
            if upperNode[pix].value >= averagePixel:
                cutOff = pix
                break

        # convert pixel value into integers from decimal
        for pix in range(numPixels):
            lowerNode[pix].value = int(lowerNode[pix].value)
            upperNode[pix].value = int(upperNode[pix].value)

        # Individual stretch the lower and upper regions of the histograms
        lowerSplit = self.RayleighLowerStretch(lowerNode, height, width, cutOff)
        upperSplit = self.RayleighUpperStretch(upperNode, height, width, cutOff)

        # Temporary variables to replace with procesed pixels
        colorLower = np.zeros((height,width))
        colorUpper = np.zeros((height,width))

        # Assign values for lower and upper regions.
        for pix in range(numPixels):
            if pix > cutOff:
                colorLower[lowerSplit[pix].x, lowerSplit[pix].y] = 255
                colorUpper[upperSplit[pix].x, upperSplit[pix].y] = upperSplit[pix].value
            else:
                colorLower[lowerSplit[pix].x, lowerSplit[pix].y] = lowerSplit[pix].value
                colorUpper[upperSplit[pix].x, upperSplit[pix].y] = 0

        return colorLower, colorUpper

    def RayleighLowerStretch(self,node, height, width, cutOff):
        """
        # Function to perform Rayleigh Histogram stretching on the lower region histogram of the image
        Inputs:
            a) node - lower pixel node
            b) height
            c) width
            d) cutOff - pixel cutoff value

        Outputs:
            a) node - stretched lower pixel node
        """
        alpha = 0.4                             # Rayleigh distribution parameter
        pixelRange = [0,255]                    # Range of pixels 0 - 255 pixels
        pixelCount = np.zeros(256)              # NumPixel in the image (used as a counter for the frequency of pixels in the image)
        tempNormalizationStore = np.zeros(256)  # Temporary variable to be updated with the normalized values of the image
        e = np.e                                # exponent

        # Count the frequency of pixels
        for pix in range(cutOff):
            pixelCount[node[pix].value] += 1

        # Calculate probability of pixel occurence
        pixelProb = pixelCount / cutOff
        # Calculate pixel cumulative frequency
        pixelCumulative = np.cumsum(pixelProb)
        # Calculate range of pixels from (255)
        pixelRange = pixelRange[1] - pixelRange[0]

        # Calculating output processed pixel after Rayleigh image processing
        alphaConst = 2 * alpha ** 2
        valMax = 1 - e ** (-1 / alphaConst)
        val = np.array(valMax * pixelCumulative)

        for pix in range(256):
            if val[pix] >= 1:
                val[pix] = val[pix] - self.esp
        for pix in range(256):
            tempNormalizationStore[pix] = np.sqrt(-alphaConst * math.log((1 - val[pix]),e))
            normalizedVal = tempNormalizationStore[pix] * pixelRange

            # Set threshold output at 255
            if normalizedVal > 255:
                pixelCumulative[pix] = 255
            else:
                pixelCumulative[pix] = normalizedVal

        for pix in range(cutOff):
            node[pix].value = pixelCumulative[node[pix].value]

        return node

    def RayleighUpperStretch(self,node, height, width, cutOff):
        """
          Function to perform Rayleigh Histogram stretching on the upper region histogram of the image
          Inputs:
              a) node - upper pixel node
              b) height
              c) width
              d) cutOff - pixel cutoff value

          Outputs:
              a) node - stretched upper pixel node
       """
        e = np.e                                # Exponent
        numPixels = height * width              # Image dimensions
        alpha = 0.4                             # Stretching parameter
        pixelRange = [0, 255]                   # Pixel range
        pixelCount = np.zeros(256)              # NumPixel in the image
        tempNormalizationStore = np.zeros(256)  # Temporary variable to store processed pixel

        # count frequency of pixels
        for pix in range(cutOff, numPixels):
            pixelCount[node[pix].value] += 1

        # Calculate probability distribution of pixels
        pixelProb = pixelCount / (numPixels - cutOff)
        pixelCumulative = np.cumsum(pixelProb)
        pixelRange = pixelRange[1] - pixelRange[0]

        # Perform Rayleigh histogram stretching
        alphaConst = 2 * alpha ** 2
        valMax = 1 - e ** (-1 / alphaConst)
        val = np.array(valMax * pixelCumulative)

        for pix in range(256):
            if val[pix] >= 1:
                val[pix] = val[pix] - self.esp
        for pix in range(256):
            tempNormalizationStore[pix] = np.sqrt(-alphaConst * math.log((1 - val[pix]), e))
            normalizedVal = tempNormalizationStore[pix] * pixelRange
            if normalizedVal > 255:
                pixelCumulative[pix] = 255
            else:
                pixelCumulative[pix] = normalizedVal
        for pix in range(cutOff, numPixels):
            node[pix].value = pixelCumulative[node[pix].value]

        return node

    def HSVEqualization(self, image):
        """
        Function that performs HSV colour equalization
        Inputs:
            a) image - input image in RGB format (uint8)

        Outputs:
            a) image - HSV equalized output image in RGB format (uint8)
        """
        # Get image dimensions
        height = image.shape[0]
        width = image.shape[1]

        # clip image pixels that are not in  0 -255 range as precaution
        image = np.clip(image, 0, 255).astype(np.uint8)
        # Convert image to hsv format
        image_hsv = rgb2hsv(image)

        # Perform stretching on the S and V channels of the image
        image_hsv[:, :, 1] = self.HSVStretching(image_hsv[:, :, 1], height, width)
        image_hsv[:, :, 2] = self.HSVStretching(image_hsv[:, :, 2], height, width)

        # Reconvert the image back to RGB
        image = hsv2rgb(image_hsv) * 255
        # Clip values over the 0 - 255 range
        image = np.clip(image, 0, 255)
        # Convert image type back to uint 8
        image = image.astype(np.uint8)
        # Return HSV stretched image
        return image

    def HSVStretching(self,channel, height, width):
        """
        Function that performs stretching of the S and V channels
        Inputs:
            a) channel - image colour channel
            b) height - height of the image
            c) width - width of the image
        """
        # get dimension of channel
        numPixels = height * width

        # Append pixel values to R_array for processing
        R_array = []
        for h in range(height):
            for w in range(width):
                R_array.append(channel[h][w])

        # Create a copy of channel
        R_array2 = channel.copy
        # Sort pixels in R_array
        R_array.sort()

        # Get the minimum and maximum pixel values
        I_min = R_array[int(numPixels/100)]
        I_max =  R_array[-int(numPixels/100)]

        # Perform stretching for the particular channel
        array_Global_histogram_stretching = np.zeros((height,width))

        for h in range(height):
            for w in range(width):
                if channel[h][w] < I_min:
                    p_out = channel[h][w]
                    array_Global_histogram_stretching[h][w] = p_out

                elif channel[h][w] > I_max:
                    p_out = channel[h][w]
                    array_Global_histogram_stretching[h][w] = p_out

                else:
                    p_out = (channel[h][w] - I_min) * ((1) / (I_max - I_min))
                    array_Global_histogram_stretching[h][w] = p_out
        # Return final output
        return array_Global_histogram_stretching




















