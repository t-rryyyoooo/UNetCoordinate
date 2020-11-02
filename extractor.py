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
    def __init__(self, image, label, mask=None, image_patch_size=[48, 48, 16], label_patch_size=[48, 48, 16], overlap=None, phase="train"):
        """
        image : original CT image
        label : original label image
        mask : mask image of the same size as the label
        image_patch_size : patch size for CT image.
        label_patch_size : patch size for label image.
        slide : When clipping, shit the clip position by slide
        phase : train -> For training model, seg -> For segmentation

        """
        
        self.image = image
        self.label = label
        if phase == "train":
            self.label = label
            self.mask = mask

        elif phase == "segmentation":
            self.mask = mask
            if mask is not None:
                self.label = mask

        else:
            print("[ERROR] Invalid phase : {}".format(phase))
            sys.exit()

        self.phase = phase

        """ patch_size = [z, y, x] """
        self.image_patch_size = np.array(image_patch_size)
        self.label_patch_size = np.array(label_patch_size)

        """ Check slide size is correct."""
        if overlap is None:
            self.slide = self.label_patch_size
            self.overlap = 1
        else:
            self.slide = self.label_patch_size // overlap
            self.overlap = overlap



    def execute(self):
        """
        Clip image and label.

        """

        """ For restoration. """
        self.meta = {}

        """ Caluculate each padding size for label and image to clip correctly. """
        lower_pad_size, upper_pad_size = caluculatePaddingSize(np.array(self.label.GetSize()), self.image_patch_size, self.label_patch_size, self.slide)

        self.meta["lower_padding_size"] = lower_pad_size[1].tolist()
        self.meta["upper_padding_size"] = upper_pad_size[1].tolist()
        
        """ Pad image and label. """
        padded_image = padding(self.image, lower_pad_size[0].tolist(), upper_pad_size[0].tolist(), mirroring=True)
        padded_label = padding(self.label, lower_pad_size[1].tolist(), upper_pad_size[1].tolist())
        if self.mask is not None:
            padded_mask = padding(self.mask, lower_pad_size[1].tolist(), upper_pad_size[1].tolist())

        self.meta["padded_label"] = padded_label

        self.meta["patch_size"] = self.label_patch_size 

        """ Clip the image and label to patch size. """
        self.image_list = [] 
        self.image_array_list = []
        self.label_list = []
        self.label_array_list = []

        ixsize, iysize, izsize = np.array(padded_image.GetSize()) - self.image_patch_size
        total_image_patch_idx = [i for i in product(range(0, ixsize + 1, self.slide[0]), range(0, iysize + 1, self.slide[1]), range(0, izsize + 1, self.slide[2]))]

        lxsize, lysize, lzsize = np.array(padded_label.GetSize()) - self.label_patch_size

        total_label_patch_idx = [i for i in product(range(0, lxsize + 1, self.slide[0]), range(0, lysize + 1, self.slide[1]), range(0, lzsize + 1, self.slide[2]))]

        self.meta["total_patch_idx"] = total_label_patch_idx

        if len(total_image_patch_idx) != len(total_label_patch_idx):
            print("[ERROR] The number of clliped image and label is different.")
            sys.exit()

        with tqdm(total=len(total_image_patch_idx), desc="Clipping image and label...", ncols=60) as pbar:
            for ix, lx in zip(range(0, ixsize + 1, self.slide[0]), range(0, lxsize + 1, self.slide[0])):
                for iy, ly in zip(range(0, iysize + 1, self.slide[1]), range(0, lysize + 1, self.slide[1])):
                    for iz, lz in zip(range(0, izsize + 1, self.slide[2]), range(0, lzsize + 1, self.slide[2])):

                        """ Set the lower and upper clip index """
                        image_lower_clip_index = np.array([ix, iy, iz])
                        image_upper_clip_index = image_lower_clip_index + self.image_patch_size
                        label_lower_clip_index = np.array([lx, ly, lz])
                        label_upper_clip_index = label_lower_clip_index + self.label_patch_size
                        if self.mask is not None:
                            """ Clip mask image to label patch size. """
                            clipped_mask = clipping(padded_mask, label_lower_clip_index, label_upper_clip_index)
                            clipped_mask.SetOrigin(self.mask.GetOrigin())

                            clipped_mask_array = sitk.GetArrayFromImage(clipped_mask)

                            """ If you feed mask image, you check if the image contains the masked part. If not, skip and set False to the check_mask array"""
                            if self.phase == "train" and (clipped_mask_array == 0).all(): 
                                pbar.update(1)
                                continue
                      
                        """ Clip label to label patch size. """
                        clipped_label = clipping(padded_label, label_lower_clip_index, label_upper_clip_index)
                        clipped_label.SetOrigin(self.label.GetOrigin())

                        clipped_label_array = sitk.GetArrayFromImage(clipped_label)

                        self.label_list.append(clipped_label)
                        self.label_array_list.append(clipped_label_array)

                        clipped_image = clipping(padded_image, image_lower_clip_index, image_upper_clip_index)
                        clipped_image.SetOrigin(self.label.GetOrigin())
                        clipped_image_array = sitk.GetArrayFromImage(clipped_image)
                        self.image_list.append(clipped_image)
                        self.image_array_list.append(clipped_image_array)

                        pbar.update(1)



    def output(self, kind = "Array"):
        if kind == "Array":
            return self.image_array_list, self.label_array_list
        elif kind == "Image":
            return self.image_list, self.label_list
        else:
            print("[ERROR] Invalid kind : {}.".format(kind))
            sys.exit()

    def save(self, save_path):
        save_path = Path(save_path)
        save_image_path = save_path / "dummy.mha"

        if not save_image_path.parent.exists():
            createParentPath(str(save_image_path))

        with tqdm(total=len(self.image_list), desc="Saving image and label...", ncols=60) as pbar:
            for i, (image, label) in enumerate(zip(self.image_list, self.label_list)):
                save_image_path = save_path / "image_{:04d}.mha".format(i)
                save_label_path = save_path / "label_{:04d}.mha".format(i)

                sitk.WriteImage(image, str(save_image_path), True)
                sitk.WriteImage(label, str(save_label_path), True)
                pbar.update(1)



    def restore(self, predict_array_list):
        predict_array = np.zeros_like(sitk.GetArrayFromImage(self.meta["padded_label"]))

        with tqdm(total=len(predict_array_list), desc="Restoring image...", ncols=60) as pbar:
            for pre_array, idx in zip(predict_array_list, self.meta["total_patch_idx"]):
                x_slice = slice(idx[0], idx[0] + self.meta["patch_size"][0])
                y_slice = slice(idx[1], idx[1] + self.meta["patch_size"][1])
                z_slice = slice(idx[2], idx[2] + self.meta["patch_size"][2])


                predict_array[z_slice, y_slice, x_slice] = pre_array
                pbar.update(1)


        predict = getImageWithMeta(predict_array, self.label)
        predict = cropping(predict, self.meta["lower_padding_size"], self.meta["upper_padding_size"])
        predict.SetOrigin(self.label.GetOrigin())
        

        return predict






