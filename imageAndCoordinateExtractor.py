import sys
import SimpleITK as sitk
import numpy as np
from pathlib import Path
from tqdm import tqdm
from coordinateArrayCreater import CoordinateArrayCreater
from itertools import product
from functions import padding, cropping, clipping, caluculatePaddingSize, getImageWithMeta, createParentPath

class ImageAndCoordinateExtractor():
    """
    Class which Clips the input image and label to patch size.
    In 13 organs segmentation, unlike kidney cancer segmentation,
    SimpleITK axis : [sagittal, coronal, axial] = [x, y, z]
    numpy axis : [axial, coronal, saggital] = [z, y, x]
    In this class we use simpleITK to clip mainly. Pay attention to the axis.
    
    """
    def __init__(self, image, label, center=(0, 0, 0), mask=None, image_patch_size=[48, 48, 16], label_patch_size=[48, 48, 16], overlap=1):
        """
        image : original CT image
        label : original label image
        mask : mask image of the same size as the label
        image_patch_size : patch size for CT image.
        label_patch_size : patch size for label image.
        slide : When clipping, shit the clip position by slide

        """
        
        self.org = image
        self.image = image
        self.label = label
        self.mask = mask
        self.center = center

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


        """ If self.center is not None, get coordinate array. """
        cac = CoordinateArrayCreater(
                image_array = sitk.GetArrayFromImage(self.image),
                center = self.center
                )
        cac.execute()
        coordinate_array = cac.getCoordinate(kind="relative")

        """ Split coordinate and convert splited coordinate to image for making patch. """
        coordinate_part = []
        for i in range(self.image.GetDimension()):
            coord_part = getImageWithMeta(coordinate_array[i, ...], self.image)
            coordinate_part.append(coord_part)

        """ Clip the image and label to patch size. """
        image_patch_list = self.makePatch(self.image, self.image_patch_size, self.slide)
        label_patch_list = self.makePatch(self.label, self.label_patch_size, self.slide)
        if self.mask is not None:
            mask_patch_list = self.makePatch(self.mask, self.label_patch_size, self.slide)

        coordinate_part_patch_list = []
        for coord_part in coordinate_part:
            coord_part_patch_list = self.makePatch(coord_part, self.image_patch_size, self.slide)
            coordinate_part_patch_list.append(coord_part_patch_list)

        """ Confirm if makePatch runs correctly. """
        assert len(image_patch_list) == len(label_patch_list)
        for coordinate_part_patch in coordinate_part_patch_list:
            assert len(image_patch_list) == len(coordinate_part_patch)

        if self.mask is not None:
            assert len(image_patch_list) == len(mask_patch_list)

        """ Check mask. """
        self.image_patch_array_list = []
        self.label_patch_array_list = []
        self.coordinate_patch_array_list = []
        for i in range(len(image_patch_list)):
            if self.mask is not None:
                mask_patch_array = sitk.GetArrayFromImage(mask_patch_list[i])
                if (mask_patch_array == 0).all():
                    continue

            image_patch_array = sitk.GetArrayFromImage(image_patch_list[i])
            label_patch_array = sitk.GetArrayFromImage(label_patch_list[i])

            """ Integrate splited coordinate array."""
            coordinate_patch_array = []
            for coord_part_patch_list in coordinate_part_patch_list:
                coord_part_patch_array = sitk.GetArrayFromImage(coord_part_patch_list[i])
                coordinate_patch_array.append(coord_part_patch_array)
            coordinate_patch_array = np.stack(coordinate_patch_array)

            self.image_patch_array_list.append(image_patch_array)
            self.label_patch_array_list.append(label_patch_array)
            self.coordinate_patch_array_list.append(coordinate_patch_array)

    def makePatch(self, image, patch_size, slide):
        size = np.array(image.GetSize()) - patch_size 
        indices = [i for i in product(range(0, size[0] + 1, slide[0]), range(0, size[1] +  1, slide[1]), range(0, size[2] + 1, slide[2]))]

        patch_list = []
        with tqdm(total=len(indices), desc="Clipping images...", ncols=60) as pbar:
            for index in indices:
                lower_clip_size = np.array(index)
                upper_clip_size = lower_clip_size + patch_size

                patch = clipping(image, lower_clip_size, upper_clip_size)
                patch_list.append(patch)

                pbar.update(1)

        return patch_list


    def output(self):
        return self.image_patch_array_list, self.label_patch_array_list, self.coordinate_patch_array_list

    def save(self, save_path):
        save_path = Path(save_path)
        save_image_path = save_path / "dummy.npy"

        if not save_image_path.parent.exists():
            createParentPath(str(save_image_path))

        with tqdm(total=len(self.image_patch_array_list), desc="Saving image, label and coordinate...", ncols=60) as pbar:
            for i, (image_array, label_array, coordinate_array) in enumerate(zip(self.image_patch_array_list, self.label_patch_array_list, self.coordinate_patch_array_list)):
                save_image_path = save_path / "image_{:04d}.npy".format(i)
                save_label_path = save_path / "label_{:04d}.npy".format(i)
                save_coordinate_path = save_path / "coordinate_{:04d}.npy".format(i)

                np.save(str(save_image_path), image_array)
                np.save(str(save_label_path), label_array)
                np.save(str(save_coordinate_path), coordinate_array)
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






