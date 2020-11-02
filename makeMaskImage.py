import argparse
import SimpleITK as sitk
import numpy as np
from functions import getImageWithMeta, createParentPath

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("label_path", help="~/Desktop/data/kidney/case_00000/segmentation.nii.gz")
    parser.add_argument("save_path", help="~/Desktop/data/kidney/case_00000/mask.mha")
    parser.add_argument("--mask_number", default=-1, type=int)

    args = parser.parse_args()

    return args

def main(args):
    label = sitk.ReadImage(args.label_path)
    label_array = sitk.GetArrayFromImage(label)

    if args.mask_number < 0:
        mask_array = (label_array > 0).astype(np.int)

    else:
        mask_array = (label_array == args.mask_number).astype(np.int)

    mask = getImageWithMeta(mask_array, label)
    createParentPath(args.save_path)
    print("Saving mask image to {} ...".format(args.save_path))
    sitk.WriteImage(mask, args.save_path, True)
    print("Done")

if __name__ == "__main__":
    args = parseArgs()
    main(args)
