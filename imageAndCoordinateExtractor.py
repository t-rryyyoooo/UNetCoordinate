import sys
import SimpleITK as sitk
import numpy as np
from pathlib import Path
from tqdm import tqdm
from coordinateArrayCreater import CoordinateArrayCreater
from itertools import product
from functions import paddingForNumpy, croppingForNumpy, clippingForNumpy, caluculatePaddingSize, getImageWithMeta, createParentPath

class ImageAndCoordinateExtractor():
    """
    Class which Clips the input image and label to patch size.
    In 13 organs segmentation, unlike kidney cancer segmentation,
    SimpleITK axis : [sagittal, coronal, axial] = [x, y, z]
    numpy axis : [axial, coronal, saggital] = [z, y, x]
    Mainly numpy!
    
    """
    def __init__(self, image, label, center=(0, 0, 0), mask=None, image_array_patch_size=[16, 48, 48], label_array_patch_size=[16, 48, 48], overlap=1):
        """
        image : original CT image
        label : original label image
        mask : mask image of the same size as the label
        image_patch_size : patch size for CT image.
        label_patch_size : patch size for label image.
        slide : When clippingForNumpy, shit the clip position by slide

        """
        
        self.org = image
        self.image_array = sitk.GetArrayFromImage(image)
        self.label_array = sitk.GetArrayFromImage(label)
        if mask is not None:
            self.mask_array = sitk.GetArrayFromImage(mask)
        else:
            self.mask_array = None

        self.center = center

        """ patch_size = [z, y, x] """
        self.image_array_patch_size = np.array(image_array_patch_size)
        self.label_array_patch_size = np.array(label_array_patch_size)

        self.overlap = overlap
        self.slide = self.label_array_patch_size // overlap

    def execute(self):
        """ Clip image and label. """

        """ Caluculate each paddingForNumpy size for label and image to clip correctly. """
        self.lower_pad_size, self.upper_pad_size = caluculatePaddingSize(np.array(self.label_array.shape), self.image_array_patch_size, self.label_array_patch_size, self.slide)

        """ Pad image and label. """
        self.image_array = paddingForNumpy(self.image_array, self.lower_pad_size[0].tolist(), self.upper_pad_size[0].tolist())
        self.label_array = paddingForNumpy(self.label_array, self.lower_pad_size[1].tolist(), self.upper_pad_size[1].tolist())
        if self.mask_array is not None:
            self.mask_array = paddingForNumpy(self.mask_array, self.lower_pad_size[1].tolist(), self.upper_pad_size[1].tolist())


        """ If self.center is not None, get coordinate array. """
        cac = CoordinateArrayCreater(
                image_array = self.image_array,
                center = self.center
                )
        cac.execute()
        coordinate_array = cac.getCoordinate(kind="relative")

        """ Clip the image and label to patch size. """
        self.image_array_patch_list = self.makePatch(self.image_array, self.image_array_patch_size, self.slide)
        self.label_array_patch_list = self.makePatch(self.label_array, self.label_array_patch_size, self.slide)
        if self.mask_array is not None:
            self.mask_array_patch_list = self.makePatch(self.mask_array, self.label_array_patch_size, self.slide)

        """ Make patch size and slide 4 dimention because coordinate array has 4 dimention. """
        ndim = self.image_array.ndim
        coordinate_array_patch_size = np.array([ndim] + self.image_array_patch_size.tolist())
        coordinate_slide = np.array([ndim] + self.slide.tolist())

        self.coordinate_array_patch_list = self.makePatch(coordinate_array, coordinate_array_patch_size, coordinate_slide)

        """ Confirm if makePatch runs correctly. """
        assert len(self.image_array_patch_list) == len(self.label_array_patch_list) == len(self.coordinate_array_patch_list)

        if self.mask_array is not None:
            assert len(self.image_array_patch_list) == len(self.mask_array_patch_list)

        """ Check mask. """
        self.masked_indices = []
        for i in range(len(self.image_array_patch_list)):
            if self.mask_array is not None:
                if (self.mask_array_patch_list[i] == 0).all():
                    continue

            self.masked_indices.append(i)

    def makePatch(self, image_array, patch_size, slide):
        size = np.array(image_array.shape) - patch_size 
        indices = []
        for i in range(image_array.ndim):
            r = range(0, size[i] + 1, slide[i])
            indices.append(r)
        indices = [i for i in product(*indices)]

        patch_list = []
        with tqdm(total=len(indices), desc="Clipping images...", ncols=60) as pbar:
            for index in indices:
                lower_clip_size = np.array(index)
                upper_clip_size = lower_clip_size + patch_size

                patch = clippingForNumpy(image_array, lower_clip_size, upper_clip_size)
                patch_list.append(patch)

                pbar.update(1)

        return patch_list


    def output(self):
        return np.array(self.image_array_patch_list)[self.masked_indices, ...], np.array(self.label_array_patch_list)[self.masked_indices, ...], np.array(self.coordinate_array_patch_list)[self.masked_indices, ...]

    def save(self, save_path):
        save_path = Path(save_path)
        save_image_path = save_path / "dummy.npy"

        if not save_image_path.parent.exists():
            createParentPath(str(save_image_path))

        with tqdm(total=len(self.masked_indices), desc="Saving image, label and coordinate...", ncols=60) as pbar:
            for i in self.masked_indices:
                save_image_path = save_path / "image_{:04d}.npy".format(i)
                save_label_path = save_path / "label_{:04d}.npy".format(i)
                save_coordinate_path = save_path / "coordinate_{:04d}.npy".format(i)

                np.save(str(save_image_path), self.image_array_patch_list[i])
                np.save(str(save_label_path), self.label_array_patch_list[i])
                np.save(str(save_coordinate_path), self.coordinate_array_patch_list[i])
                pbar.update(1)



    def restore(self, predict_array_list):
        predict_array = np.zeros_like(self.label_array)

        size = np.array(self.label_array.shape) - self.label_array_patch_size 
        indices = []
        for i in range(self.label_array.ndim):
            r = range(0, size[i] + 1, self.slide[i])
            indices.append(r)
        indices = np.array([i for i in product(*indices)])

        with tqdm(total=len(predict_array_list), desc="Restoring image...", ncols=60) as pbar:
            for pre_array, idx in zip(predict_array_list, indices[self.masked_indices]): 
                slices = []
                for i in range(self.label_array.ndim):
                    s = slice(idx[i], idx[i] + self.label_array_patch_size[i])
                    slices.append(s)
                slices = tuple(slices)

                predict_array[slices] = pre_array
                pbar.update(1)


        predict_array = croppingForNumpy(predict_array, self.lower_pad_size[1].tolist(), self.upper_pad_size[1].tolist())
        predict = getImageWithMeta(predict_array, self.org)
        predict.SetOrigin(self.org.GetOrigin())
        

        return predict






