"""
Written by: Rupesh Kumar Dey
Contains: MultiScaleFusion (type - Class)
    a) A class object package that performs image preprocessing on the input image and associated helper functions

Inspired by the Color Balance and Fusion for Underwater Image Enhancement
DOI: 10.1109/TIP.2017.2759252
"""

# Importing required python libraries and helper functions
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import math
import copy
from PIL import Image, ImageCms
from skimage.color import rgb2lab, lab2rgb


# Defining the MultiScaleFusion class object
class MultiScaleFusion:
    """
    Class object that performs multiscale image enhancement
    """
    # Initialization
    def __init__(self):
        self.path = None


    def white_balance(self,img8):
        """
        - Function that performs Image White Balancing on the input image
        Input:
            a) img8 - 3 channel RGB image in 8 bit format (0 - 255)
        Output:
            b) result8 - white balanced image in 8 bit format
        Reference: https://stackoverflow.com/questions/40586276/unexpected-output-while-converting-rgb-image-to-lab-image
        """

        result8 = cv2.cvtColor(img8, cv2.COLOR_RGB2LAB)    # Converts image from RGB to LAB
        result32 = result8.astype(np.float32)              # Convert pixels ino 32 bit data
        # Calculate average of channel A and B
        avg_a32 = np.average(result32[:, :, 1])
        avg_b32 = np.average(result32[:, :, 2])
        # Perform white balancing
        result32[:, :, 1] = result32[:, :, 1] - ((avg_a32 - 128) * (result32[:, :, 0] / 255.0) * 1.1)
        result32[:, :, 2] = result32[:, :, 2] - ((avg_b32 - 128) * (result32[:, :, 0] / 255.0) * 1.1)
        # Reconvert image back to uint8
        result8 = result32.astype(np.uint8)
        # Reconvert image from LAB to RGB
        result8 = cv2.cvtColor(result8, cv2.COLOR_LAB2RGB)
        # Return processed image
        return result8

    def gammaCorrection(self, src, gamma):
        """
        Function that performs Gamma Correction on an image to correct the contrast of the image
        Input:
            a) src - Image
            b) gamma - gamma correction value
        """
        invGamma = 1 / gamma
        table = [((i / 255) ** invGamma) * 255 for i in range(256)]
        table = np.array(table, np.uint8)
        return cv2.LUT(src, table)

    def imadjust(self, img32, min_in, max_in, min_out, max_out, gamma=1):
        """
        # https://stackoverflow.com/questions/39767612/what-is-the-equivalent-of-matlabs-imadjust-in-python
        - Function similar to the MATLAB imadjust function.
        - Converts an image range from (min_in,max_in) to (min_out, max_out)
        Input:
            a) img32 - 32 bit normalized input image (3 channel)
            b) min_in - minimum input image pixel value
            c) max_in - maximum input image pixel value
            d) min_out - minimum output image pixel value
            e) max_out - maximum output image pixel value
            f) gamma - parameter describing relationship of input and output values

        Output:
            a) out32 - processed output image in 32 bit.
        """
        out32 = (((img32 - min_in) / (max_in - min_in)) ** gamma) * (max_out - min_out) + min_out
        return out32

    def saliency_detection(self, image8):
        """
        Saliency level estimator function.
        Input:
            a) image8 - 8 bit input iamge (0-255)

        Output:
            b) sm32 - return saliency matrics in 32 bit normalized format
        """
        # Perform GaussianBlurring on the image
        imageGaussian8 = cv2.GaussianBlur(image8, (3, 3), 0)
        # Convert image from RGB to LAB. Convert to 32 bit data and normalize
        lab_im32 = rgb2lab(imageGaussian8).astype(np.float32) / 255
        # Extract individual L, A and B channels
        l32 = lab_im32[:, :, 0];
        a32 = lab_im32[:, :, 1];
        b32 = lab_im32[:, :, 2]
        # Calculate mean of L, A and B channels
        lm32 = np.mean(l32);
        am32 = np.mean(a32);
        bm32 = np.mean(b32)
        # Calculate saliency matrix
        sm32 = (l32 - lm32) ** 2 + (a32 - am32) ** 2 + (b32 - bm32) ** 2
        # Return value / output
        return sm32

    def gaussian_pyramid(self, image32, level):
        """
        Gaussian Pyramid Function that performs Gaussian filtering on n-level of the processed image
        Input:
            a) image32 - 32 bit normalized image
            b) level - number of levels ie. iterations of image outputs to produce

        Output:
            b) out - an array of processed output image (in 32 bit normalized) in various sizes for n_levels

        References:
            # https://www.analyticsvidhya.com/blog/2021/08/implementing-convolution-as-an-image-filter-using-opencv/
            # https://www.geeksforgeeks.org/python-opencv-filter2d-function/
        """
        # Define Gaussian Filter kernel
        h = 1 / 16 * np.array([1, 4, 6, 4, 1]).reshape(1, 5)
        filter = np.matmul(h.T, h)

        # Output array to store pyramid of output image results
        out = []

        # Convert image back to uint8 format
        temp_image32 = copy.deepcopy(image32)
        image8 = (image32 * 255).astype(np.uint8)

        # Perform filtering using filter kernel and reconvert back to 32bit for first level
        temp_out8 = cv2.filter2D(src=image8, ddepth=-1, kernel=filter)
        temp_out32 = temp_out8.astype(np.float32) / 255

        # Append first processed image back to out array
        out.append(temp_out32)

        # For each level, perform filter on the image from the previous level
        for i in range(1, level):
            temp_image32 = temp_image32[0::2, 0::2]
            temp_image8 = (temp_image32 * 255).astype(np.uint8)
            temp_out8 = cv2.filter2D(src=temp_image8, ddepth=-1, kernel=filter)
            temp_out32 = temp_out8.astype(np.float32) / 255
            out.append(temp_out32)

        # Return output array of pyramid processed images
        return out

    def laplacianPyramid(self, image32, level):
        """
        Laplacian Pyramid function for n_level
        Input:
            a) image32 - 32 bit normalized image to be processed
            b) level -  number of levels ie. iterations of image outputs to produce

        Output:
            a) out - output 32 bit normalized image
        """
        # Define filter kernel
        h = 1 / 16 * np.array([1, 4, 6, 4, 1]).reshape(1, 5)

        # Output array to store data
        out = []
        out.append(image32)
        temp_image32 = copy.deepcopy(image32)

        # Perform Laplacian filtering for each level of the pyramid as defined by user
        for i in range(1, level):
            temp_image32 = temp_image32[0::2, 0::2]
            out.append(temp_image32)

        # Calculate DoG
        for i in range(level - 1):
            m, n = out[i].shape
            outPlus1_8 = (out[i + 1] * 255).astype(np.uint8)
            outPlus1_8 = cv2.resize(outPlus1_8, (n, m))
            outPlus1_32 = outPlus1_8.astype(np.float32) / 255
            out[i] = out[i] - outPlus1_32
        return out

    def pyramidReconstruct(self, pyramid, level):
        """
        Final Image pyramid reconstruction
        Input:
            a) pyramid - an array of images for n_levels
            b) level - number of images in the array.

        Output:
            a) pyramid[0] - 32 bit normalized image, the first image in the pyramid after processing.
        """
        # For each level in the image pyramid stored array
        for i in range(level - 1, 0, -1):
            # get shape of image on each pyramid level
            m, n = pyramid[i - 1].shape

            # Convert image back to 32 bit
            pyramid_i8 = (pyramid[i] * 255).astype(np.uint8)
            pyramid_i8 = cv2.resize(pyramid_i8, (n, m))
            pyramid_i32 = pyramid_i8.astype(np.float32) / 255

            # Add the previous level image to the present
            pyramid[i - 1] = pyramid[i - 1] + pyramid_i32

        # Return the first / highest level ie. where the size of the image is as per the original image size.
        return pyramid[0]

    def multiScaleFusion(self, input, input_type = 1):
        """
        The main function for performing multiscale fusion enhancement
        Input:
            a) input - either an image from cv2.imread (BGR format) or image file location ("path.jpg" format.
            b) input_type - 1 for image np.ndarray or 2 for file path location

        Output:
            a) final_image - final_processed image.
        """

        """
        Stage 1: Load Images and Split channels
        """
        assert input_type in [1, 2], "Please specify input type of either ""1 - Image"" or ""2 - File location"""
        if input_type == 1:
            # CV2 image BGR format
            input_image = input
        elif input_type == 2:
            # Image path location
            input_image = cv2.imread(input)

        # Convert BGR image to RGB format and convert to grayscale
        inputImageColor = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        inputImageGray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)


        # Convert to float 32 format and normalize the image
        normInputImageColor32 = inputImageColor.astype(np.float32) / 255
        normInputImageGray32 = inputImageGray.astype(np.float32) / 255

        # Extract RGB channels
        I_red32 = normInputImageColor32[:,:,0]
        I_green32 = normInputImageColor32[:,:,1]
        I_blue32 = normInputImageColor32[:,:,2]

        # Obtain mean of RGB channels
        I_red_mean32 = np.mean(I_red32)
        I_green_mean32 = np.mean(I_green32)
        I_blue_mean32 = np.mean(I_blue32)

        # print(I_red_mean, I_green_mean, I_blue_mean)

        """
        Perform Color Compensation
        """
        # Color compensation parameter for red and clue channel
        alpha_red = 0.1
        alpha_blue = 0.0  # Optional. To turn off set to 0

        I_RedBalanced32 = I_red32 + alpha_red * (I_green_mean32 - I_red_mean32)
        I_BlueBalanced32 = I_blue32 + alpha_blue * (I_green_mean32 - I_blue_mean32)

        """
        Performing white balancing
        """
        # Image after performing color compensation
        processed_image32 = np.zeros(input_image.shape)
        processed_image32[:,:,0] = I_RedBalanced32
        processed_image32[:,:,1] = I_green32
        processed_image32[:,:,2] = I_BlueBalanced32

        # Perform white balancing
        processed_image8 = (processed_image32 * 255).astype(np.uint8)
        processed_image8 = self.white_balance(processed_image8)
        whiteBalancedImage8 = copy.deepcopy(processed_image8)

        # Perform gamma correction on the image
        processed_image32 = processed_image8.astype(np.float32) / 255
        processed_image32 = self.imadjust(processed_image32, 0,1,0,1, gamma = 2)
        processed_image8 = (processed_image32 * 255).astype(np.uint8)
        gammaImage8 = copy.deepcopy(processed_image8)
        gammaImage32 = processed_image32

        sigma = 20
        filter_size = 2*math.ceil(2*sigma)+1
        N_iter = 30
        IGauss8 = copy.deepcopy(whiteBalancedImage8)

        # Perform Gaussian blur after gamma correction
        for i in range(N_iter):
            # Reference: https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_filtering/py_filtering.html
            IGauss8 = cv2.GaussianBlur(IGauss8,(filter_size,filter_size),sigma)
            IGauss8 = np.minimum(IGauss8, whiteBalancedImage8)

        gain = 1
        whiteBalancedImage32 = whiteBalancedImage8.astype(np.float32) / 255
        IGauss32 = (IGauss8.astype(np.float32) / 255)

        Norm32 = (whiteBalancedImage32 - gain * IGauss32)
        Norm8 = (Norm32 * 255).astype(np.uint8)

        # Perform histogram equalization on each image channel
        for i in range(3):
            Norm8[:,:,i] = cv2.equalizeHist(Norm8[:,:,i])

        Norm8 = Norm8
        Norm32 = Norm8.astype(np.float32) / 255

        Isharp32 = (whiteBalancedImage32 + Norm32) / 2
        Isharp8 = (Isharp32*255).astype(np.uint8)

        """
        Weight Calculation
        """
        Isharp_lab8 = cv2.cvtColor(Isharp8, cv2.COLOR_RGB2LAB)
        Igamma_lab8 = cv2.cvtColor(gammaImage8, cv2.COLOR_RGB2LAB)

        Isharp_lab32 = Isharp_lab8.astype(np.float32) / 255
        Igamma_lab32 = Igamma_lab8.astype(np.float32) / 255

        """
        Weight Input 1
        """
        # Calculate laplacian contrast weight
        R1_32 = Isharp_lab32[:, :, 0] / 255
        WC1_32 = np.sqrt(((Isharp32[:,:,0] - R1_32) ** 2 + (Isharp32[:,:,1] - R1_32) ** 2 + (Isharp32[:,:,2] - R1_32) ** 2) / 3)
        WC1_8 = (WC1_32*255).astype(np.uint8)

        # Calculate Saliency weight
        WS1_32 = self.saliency_detection(Isharp8)
        WS1_32 = WS1_32 / np.max(WS1_32)
        WS1_8 = (WS1_32 * 255).astype(np.float32)

        # Calculate the saturation weight
        WSAT1_32 = np.sqrt(1/3 * ((Isharp32[:,:,0] - R1_32) ** 2 + (Isharp32[:,:,1] - R1_32) ** 2 + (Isharp32[:,:,2] - R1_32) ** 2))
        WSAT1_8 = (WSAT1_32 * 255).astype(np.uint8)

        """
        Weight Input 2
        """
        # Calculate laplacian contrast weight
        R2_32 = Igamma_lab32[:, :, 0] / 255
        WC2_32 = np.sqrt(((gammaImage32[:,:,0] - R2_32) ** 2 + (gammaImage32[:,:,1] - R2_32) ** 2 + (gammaImage32[:,:,2] - R2_32) ** 2) / 3)
        WC2_8 = (WC2_32*255).astype(np.uint8)

        # Calculate Saliency weight
        WS2_32 = self.saliency_detection(gammaImage8)
        WS2_32 = WS2_32 / np.max(WS2_32)
        WS2_8 = (WS2_32 * 255).astype(np.uint8)

        # Calculate the saturation weight
        WSAT2_32 = np.sqrt(1/3 * ((gammaImage32[:,:,0] - R1_32) ** 2 + (gammaImage32[:,:,1] - R1_32) ** 2 + (gammaImage32[:,:,2] - R1_32) ** 2))
        WSAT2_8 = (WSAT2_32 * 255).astype(np.uint8)

        """
        Calculate normalized weight
        """
        W1_32 = (WC1_32 + WS1_32 + WSAT1_32+0.1) / (WC1_32 + WS1_32 + WSAT1_32 + WC2_32 + WS2_32 + WSAT2_32 + 0.2)
        W2_32 = (WC2_32 + WS2_32 + WSAT2_32 + 0.1) / (WC1_32 + WS1_32 + WSAT1_32 + WC2_32 + WS2_32 + WSAT2_32 + 0.2)

        W1_8 = (W1_32 * 255).astype(np.uint8)
        W2_8 = (W2_32 * 255).astype(np.uint8)

        """
        Perform Naive Fusion
        """
        R_32 = np.zeros(Isharp32.shape)
        for i in range(3):
            R_32[:,:,i] = np.multiply(W1_32, Isharp32[:,:,i]) + np.multiply(W2_32 , gammaImage32[:,:,i])

        R_8 = (R_32 * 255).astype(np.uint8)
        naive_fusion32 = copy.deepcopy(R_32)
        naive_fusion8 = copy.deepcopy(R_8)

        """
        Perform Multi scale Fusion
        """
        # Calculate Gaussian Pyramid
        level = 10
        # Single channels
        weight1_arr32 = self.gaussian_pyramid(W1_32, level)
        weight1_arr8 = []
        for image in weight1_arr32:
            temp_image = (image*255).astype(np.uint8)
            weight1_arr8.append(temp_image)

        weight2_arr32 = self.gaussian_pyramid(W2_32, level)
        weight2_arr8 = []

        for image in weight2_arr32:
            temp_image = (image*255).astype(np.uint8)
            weight2_arr8.append(temp_image)

        """
        Calculate Laplacian Pyramid
        """
        # Input 1
        R1_32 = self.laplacianPyramid(Isharp32[:,:,0],10)
        G1_32 = self.laplacianPyramid(Isharp32[:,:,1],10)
        B1_32 = self.laplacianPyramid(Isharp32[:,:,2],10)

        # Input 2
        R2_32 = self.laplacianPyramid(gammaImage32[:,:,0],10)
        G2_32 = self.laplacianPyramid(gammaImage32[:,:,1],10)
        B2_32 = self.laplacianPyramid(gammaImage32[:,:,2],10)

        # Temporary array of zeros of len(n_level)
        Rr_32 = [0] * level
        Rg_32 = [0] * level
        Rb_32 = [0] * level

        # Fusion of channels
        for i in range(level):
            Rr_32[i] = weight1_arr32[i] * R1_32[i] + weight2_arr32[i] * R2_32[i]
            Rg_32[i] = weight1_arr32[i] * G1_32[i] + weight2_arr32[i] * G2_32[i]
            Rb_32[i] = weight1_arr32[i] * B1_32[i] + weight2_arr32[i] * B2_32[i]

        """
        Pyramid reconstruction based on Gaussian and Laplacian Pyramid
        """
        R_32 = self.pyramidReconstruct(Rr_32,level)
        G_32 = self.pyramidReconstruct(Rg_32,level)
        B_32 = self.pyramidReconstruct(Rb_32,level)

        R_8 = (R_32 * 255).astype(np.uint8)
        G_8 = (G_32 * 255).astype(np.uint8)
        B_8 = (B_32 * 255).astype(np.uint8)

        """
        Reconstruct final processed image
        """
        final_image = np.zeros(inputImageColor.shape, "uint8")
        final_image[:,:,0] = R_8
        final_image[:,:,1] = G_8
        final_image[:,:,2] = B_8

        # Switch to determine the output from Naive Fusion or Multiscale Fusion
        final_image = naive_fusion8

        return final_image