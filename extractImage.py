import argparse
from pathlib import Path
import SimpleITK as sitk
from extractor import extractor as extor
from functions import getImageWithMeta
import re

args = None

def ParseArgs():
    parser = argparse.ArgumentParser()

    parser.add_argument("image_path", help="$HOME/Desktop/data/kits19/case_00000/imaging.nii.gz")
    parser.add_argument("label_path", help="$HOME/Desktop/data/kits19/case_00000/segmentation.nii.gz")
    parser.add_argument("save_slice_path", help="$HOME/Desktop/data/slice/hist_0.0/case_00000", default=None)
    parser.add_argument("--mask_path", help="$HOME/Desktop/data/kits19/case_00000/label.mha")
    parser.add_argument("--image_patch_size", help="48-48-16", default="48-48-16")
    parser.add_argument("--label_patch_size", help="48-48-16", default="48-48-16")
    parser.add_argument("--overlap", help="1", type=int, default=1)

    args = parser.parse_args()
    return args

def main(args):
    """ Read image and label. """
    label = sitk.ReadImage(args.label_path)
    image = sitk.ReadImage(args.image_path)
    if args.mask_path is not None:
        mask = sitk.ReadImage(args.mask_path)
    else:
        mask = None

    """ Get the patch size from string."""
    matchobj = re.match("([0-9]+)-([0-9]+)-([0-9]+)", args.image_patch_size)
    if matchobj is None:
        print("[ERROR] Invalid patch size : {}.".fotmat(args.image_patch_size))
        sys.exit()

    image_patch_size = [int(s) for s in matchobj.groups()]
    
    """ Get the patch size from string."""
    matchobj = re.match("([0-9]+)-([0-9]+)-([0-9]+)", args.label_patch_size)
    if matchobj is None:
        print("[ERROR] Invalid patch size : {}.".fotmat(args.label_patch_size))
        sys.exit()

    label_patch_size = [int(s) for s in matchobj.groups()]


    extractor = extor(
            image = image, 
            label = label,
            mask = mask,
            image_patch_size = image_patch_size, 
            label_patch_size = label_patch_size, 
            overlap = args.overlap, 
            )

    extractor.execute()
    """
    il, ll = extractor.output(kind="Array")
    p = extractor.restore(ll)
    pa = sitk.GetArrayFromImage(p)
    la = sitk.GetArrayFromImage(label)
    from functions import DICE
    dice = DICE(la, pa)
    print(dice)
    """
    extractor.save(args.save_slice_path)


if __name__ == '__main__':
    args = ParseArgs()
    main(args)
