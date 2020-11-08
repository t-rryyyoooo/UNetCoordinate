import numpy as np
import os
import SimpleITK as sitk
import tensorflow as tf
import matplotlib.pyplot as plt
from decimal import Decimal, ROUND_HALF_UP
import re

def getSizeFromString(string, digit=3):
    matchobj = re.match("([0-9]+)-" * (digit - 1) + "([0-9]+)", string)
    if matchobj is None:
        print("[ERROR] Invalid size : {}.".format(string))

    size = np.array([int(s) for s in matchobj.groups()])
    return size


def setWindowSize(image_array, min_value=-110, max_value=250):
    image_array = np.where(image_array < min_value, min_value, image_array)
    image_array = np.where(image_array > max_value, max_value, image_array)
    
    return image_array

def rounding(number, digit):
    """
    This function rounds number at digit.
    number : float
    digit : 0.1 or 0.01 ... max 1
    if digit equals 1, return int, else float.

    """
    x = Decimal(str(number)).quantize(Decimal(str(digit)), ROUND_HALF_UP)
    if str(digit) == "1":
        x = int(x)
    else:
        x = float(x)

    return x


def getImageWithMeta(imageArray, refImage, spacing=None, origin=None, direction=None):
    image = sitk.GetImageFromArray(imageArray)
    if spacing is None:
        spacing = refImage.GetSpacing()
    if origin is None:
        origin = refImage.GetOrigin()
    if direction is None:
        direction = refImage.GetDirection()

    image.SetSpacing(spacing)
    image.SetOrigin(origin)
    image.SetDirection(direction)

    return image


def croppingForNumpy(image_array, lower_crop_size, upper_crop_size):
    assert image_array.ndim == len(lower_crop_size) == len(upper_crop_size)
    
    slices = []
    for lower, upper, size in zip(lower_crop_size, upper_crop_size, image_array.shape):
        slices.append(slice(lower, size - upper))
        
    slices = tuple(slices)
    
    cropped_image_array = image_array[slices]
    
    return cropped_image_array

def cropping(image, lower_crop_size, upper_crop_size):
    crop_filter = sitk.CropImageFilter()
    crop_filter.SetLowerBoundaryCropSize(lower_crop_size)
    crop_filter.SetUpperBoundaryCropSize(upper_crop_size)
    cropped_image = crop_filter.Execute(image)
    cropped_image.SetOrigin(image.GetOrigin())

    return cropped_image

def paddingForNumpy(image_array, lower_pad_size, upper_pad_size, mirroring=False):
    assert image_array.ndim == len(lower_pad_size) == len(upper_pad_size)
    
    pad_width = []
    for lower, upper in zip(lower_pad_size, upper_pad_size):
        pad_width.append((lower, upper))

    if mirroring:
        mode = "symmetric"
        constant_values = None
    else:
        mode = "constant"
        constant_values = image_array.min()

    padded_image_array = np.pad(
            image_array, 
            pad_width = pad_width,
            mode=mode, 
            constant_values=constant_values
            )

    return padded_image_array

def padding(image, lower_pad_size, upper_pad_size, mirroring = False):
    pad_filter = sitk.MirrorPadImageFilter() if mirroring else sitk.ConstantPadImageFilter()
    if not mirroring:
        minmax = sitk.MinimumMaximumImageFilter()
        minmax.Execute(image)
        minval = minmax.GetMinimum()
        pad_filter.SetConstant(minval)

    pad_filter.SetPadLowerBound(lower_pad_size)
    pad_filter.SetPadUpperBound(upper_pad_size)
    padded_image = pad_filter.Execute(image)
    padded_image.SetOrigin(image.GetOrigin())

    return padded_image

def clippingForNumpy(image_array, lower_clip_indices, upper_clip_indices):
    slices = []
    for lower, upper in zip(lower_clip_indices, upper_clip_indices):
        s = slice(lower, upper)
        slices.append(s)

    slices = tuple(slices)

    clipped_image_array = image_array[slices]

    return clipped_image_array

def clipping(image, lower_clip_index, upper_clip_index):
    z_slice = slice(lower_clip_index[0], upper_clip_index[0])
    y_slice = slice(lower_clip_index[1], upper_clip_index[1])
    x_slice = slice(lower_clip_index[2], upper_clip_index[2])

    clipped_image = image[z_slice, y_slice, x_slice]
    clipped_image.SetOrigin(image.GetOrigin())

    return clipped_image


def caluculatePaddingSize(image_size, image_patch, label_patch, slide):
    just = (image_size % label_patch) != 0
    label_pad_size = just * (label_patch - (image_size % label_patch)) + (label_patch - slide)
    image_pad_size = label_pad_size + (image_patch - label_patch)

    lower_pad_size_label = label_pad_size // 2
    upper_pad_size_label = (label_pad_size + 1) // 2
    lower_pad_size_image = image_pad_size // 2
    upper_pad_size_image = (image_pad_size + 1) // 2
    lower_pad_size = np.array([lower_pad_size_image, lower_pad_size_label])
    upper_pad_size = np.array([upper_pad_size_image, upper_pad_size_label])
    return lower_pad_size, upper_pad_size


def DICE(trueLabel, result):
    intersection=np.sum(np.minimum(np.equal(trueLabel,result),trueLabel))
    union = np.count_nonzero(trueLabel)+np.count_nonzero(result)
    dice = 2 * intersection / (union + 10**(-9))
   
    return dice

def createParentPath(filepath):
    head, _ = os.path.split(filepath)
    if len(head) != 0:
        os.makedirs(head, exist_ok = True)

# 3D -> 3D or 2D -> 2D
def resampleSize(image, newSize, is_label = False):
    originalSpacing = image.GetSpacing()
    originalSize = image.GetSize()

    if image.GetNumberOfComponentsPerPixel() == 1:
        minmax = sitk.MinimumMaximumImageFilter()
        minmax.Execute(image)
        minval = minmax.GetMinimum()
    else:
        minval = None


    newSpacing = [osp * os / ns for osp, os, ns in zip(originalSpacing, originalSize, newSize)]
    newOrigin = image.GetOrigin()
    newDirection = image.GetDirection()

    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(newSize)
    resampler.SetOutputOrigin(newOrigin)
    resampler.SetOutputDirection(newDirection)
    resampler.SetOutputSpacing(newSpacing)
    if minval is not None:
        resampler.SetDefaultPixelValue(minval)
    if is_label:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)

    resampled = resampler.Execute(image)

    return resampled

def resampleSpacing(img, spacing, is_label=False):
      # original shape
      input_shape = img.GetSize()
      input_spacing = img.GetSpacing()
      new_shape = [ int(ish * isp / osp) for ish, isp, osp in zip(input_shape, input_spacing, spacing)]

      if img.GetNumberOfComponentsPerPixel() == 1:
          minmax = sitk.MinimumMaximumImageFilter()
          minmax.Execute(img)
          minval = minmax.GetMinimum()
      else:
          minval = None

      resampler = sitk.ResampleImageFilter()
      resampler.SetSize(new_shape)
      resampler.SetOutputOrigin(img.GetOrigin())
      resampler.SetOutputDirection(img.GetDirection())
      resampler.SetOutputSpacing(spacing)

      if minval is not None:
          resampler.SetDefaultPixelValue(minval)
      if is_label:
          resampler.SetInterpolator(sitk.sitkNearestNeighbor)

      resampled = resampler.Execute(img)

      return resampled

def advancedSettings(xlabel, ylabel, fontsize=20):
    #plt.figure(figsize=(10,10))
    plt.xlabel(xlabel, fontsize=fontsize)
    plt.ylabel(ylabel, fontsize=fontsize)
    #plt.xticks(left + width/2,left)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.show()
    
    return 

def caluculateAVG(num):
    if len(num) == 0:
        return -1
    else:
        nsum = 0
        for i in range(len(num)):
            nsum += num[i]

        return nsum / len(num)
