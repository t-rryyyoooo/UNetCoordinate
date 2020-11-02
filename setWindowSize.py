import argparse
from pathlib import Path
from functions import setWindowSize, getImageWithMeta
import SimpleITK as sitk

def parseArgs():
    parser = argparse.ArgumentParser()

    parser.add_argument("image_path")
    parser.add_argument("save_path")
    parser.add_argument("--min_value", default=-110, type=float)
    parser.add_argument("--max_value", default=250, type=float)

    args = parser.parse_args()

    return args

def main(args):
    image = sitk.ReadImage(args.image_path)
    image_array = sitk.GetArrayFromImage(image)
    
    image_array_win = setWindowSize(
            image_array = image_array, 
            min_value = args.min_value, 
            max_value = args.max_value
            )

    image_win = getImageWithMeta(image_array_win, image)

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    print("Saving images to {} ...".format(str(save_path)))
    sitk.WriteImage(image_win, str(save_path), True)
    print("Done.")

if __name__ == "__main__":
    args = parseArgs()
    main(args)
    


    

