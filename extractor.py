import sys
import re
import SimpleITK as sitk
import numpy as np
from pathlib import Path
from tqdm import tqdm
from itertools import product
from functions import padding, cropping, clipping, caluculatePaddingSize, getImageWithMeta, createParentPath
import re

class extractor():
    """
    Class which Clips the input image and label to patch size.
    In 13 organs segmentation, unlike kidney cancer segmentation,
    SimpleITK axis : [sagittal, coronal, axial] = [x, y, z]
    numpy axis : [axial, coronal, saggital] = [z, y, x]
    In this class we use simpleITK to clip mainly. Pay attention to the axis.
    
    """
    def __init__(self, image, label, mask=None, image_patch_size=[48, 48, 16], label_patch_size=[48, 48, 16], overlap=1):
        """
        image : original CT image
        label : original label image
        mask : mask image of the same size as the label
        image_patch_size : patch size for CT image.
        label_patch_size : patch size for label image.
        slide : When clipping, shit the clip position by slide
        phase : train -> For training model, seg -> For segmentation

        """
        
        self.org = image
        self.image = image
        self.label = label
        self.mask = mask

        """ patch_size = [z, y, x] """
        self.image_patch_size = np.array(image_patch_size)
        self.label_patch_size = np.array(label_patch_size)

        self.overlap = overlap
        self.slide = self.label_patch_size // overlap

    def execute(self):
        """ Clip image and label. """

        """ Caluculate each padding size for label and image to clip correctly. """
        self.lower_pad_size, self.upper_pad_size = caluculatePaddingSize(np.array(self.label.GetSize()), self.image_patch_size, self.label_patch_size, self.slide)

        """ Pad image and label. """
        self.image = padding(self.image, self.lower_pad_size[0].tolist(), self.upper_pad_size[0].tolist(), mirroring=True)
        self.label = padding(self.label, self.lower_pad_size[1].tolist(), self.upper_pad_size[1].tolist())
        if self.mask is not None:
            self.mask = padding(self.mask, self.lower_pad_size[1].tolist(), self.upper_pad_size[1].tolist())

        """ Clip the image and label to patch size. """
        image_patch_list = self.makePatch(self.image, self.image_patch_size, self.slide)
        label_patch_list = self.makePatch(self.label, self.label_patch_size, self.slide)
        if self.mask is not None:
            mask_patch_list = self.makePatch(self.mask, self.label_patch_size, self.slide)

        assert len(image_patch_list) == len(label_patch_list)
        if self.mask is not None:
            assert len(image_patch_list) == len(mask_patch_list)

        """ Check mask. """
        self.image_patch_list = [] 
        self.image_patch_array_list = []
        self.label_patch_list = []
        self.label_patch_array_list = []
        for i in range(len(image_patch_list)):
            if self.mask is not None:
                mask_patch_array = sitk.GetArrayFromImage(mask_patch_list[i])
                if (mask_patch_array == 0).all():
                    continue

            image_patch_array = sitk.GetArrayFromImage(image_patch_list[i])
            label_patch_array = sitk.GetArrayFromImage(label_patch_list[i])
            self.image_patch_list.append(image_patch_list[i])
            self.label_patch_list.append(label_patch_list[i])
            self.image_patch_array_list.append(image_patch_array)
            self.label_patch_array_list.append(label_patch_array)



    def makePatch(self, image, patch_size, slide):
        size = np.array(image.GetSize()) - patch_size 
        indices = [i for i in product(range(0, size[0] + 1, self.slide[0]), range(0, size[1] +  1, self.slide[1]), range(0, size[2] + 1, self.slide[2]))]

        patch_list = []
        with tqdm(total=len(indices), desc="Clipping images...", ncols=60) as pbar:
            for index in indices:
                lower_clip_size = np.array(index)
                upper_clip_size = lower_clip_size + patch_size

                patch = clipping(image, lower_clip_size, upper_clip_size)
                patch_list.append(patch)

                pbar.update(1)

        return patch_list


    def output(self, kind = "Array"):
        if kind == "Array":
            return self.image_patch_array_list, self.label_patch_array_list
        elif kind == "Image":
            return self.image_patch_list, self.label_patch_list
        else:
            print("[ERROR] Invalid kind : {}.".format(kind))
            sys.exit()

    def save(self, save_path):
        save_path = Path(save_path)
        save_image_path = save_path / "dummy.mha"

        if not save_image_path.parent.exists():
            createParentPath(str(save_image_path))

        with tqdm(total=len(self.image_patch_list), desc="Saving image and label...", ncols=60) as pbar:
            for i, (image, label) in enumerate(zip(self.image_patch_list, self.label_patch_list)):
                save_image_path = save_path / "image_{:04d}.mha".format(i)
                save_label_path = save_path / "label_{:04d}.mha".format(i)

                sitk.WriteImage(image, str(save_image_path), True)
                sitk.WriteImage(label, str(save_label_path), True)
                pbar.update(1)



    def restore(self, predict_array_list):
        predict_array = np.zeros_like(sitk.GetArrayFromImage(self.label))

        size = np.array(self.label.GetSize()) - self.label_patch_size 
        indices = [i for i in product(range(0, size[0] + 1, self.slide[0]), range(0, size[1] +  1, self.slide[1]), range(0, size[2] + 1, self.slide[2]))]

        with tqdm(total=len(predict_array_list), desc="Restoring image...", ncols=60) as pbar:
            for pre_array, idx in zip(predict_array_list, indices): 
                x_slice = slice(idx[0], idx[0] + self.label_patch_size[0])
                y_slice = slice(idx[1], idx[1] + self.label_patch_size[1])
                z_slice = slice(idx[2], idx[2] + self.label_patch_size[2])


                predict_array[z_slice, y_slice, x_slice] = pre_array
                pbar.update(1)


        predict = getImageWithMeta(predict_array, self.label)
        predict = cropping(predict, self.lower_pad_size[1].tolist(), self.upper_pad_size[1].tolist())
        predict.SetOrigin(self.label.GetOrigin())
        

        return predict






