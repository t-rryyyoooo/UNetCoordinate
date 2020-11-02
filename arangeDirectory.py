from pathlib import Path
import shutil
import os

"""
Arange directory from originl directory structure donloaded from web site to the new one which extractImage.py assumes.
"""
def main():
    path = Path("/home/vmlab/Desktop/data/Abdomen/RawData/Training")
    save_path = Path("/home/vmlab/Desktop/data/Abdomen")

    images = sorted((path / "img").glob("img*"))
    labels = sorted((path / "label").glob("label*"))
    if len(images) != len(labels):
        print("[ERROR] the number of images and the one of labels have to be the same.")
        sys.exit()

    for i, (image, label) in enumerate(zip(images, labels)):
        save = save_path / ("case_" + str(i).zfill(2))
        os.makedirs(str(save), exist_ok=True)
        save_image_path = save / "imaging.nii.gz"
        save_label_path = save / "segmentation.nii.gz"
        print("From {} to {}".format(str(image), str(save_image_path)))
        print("From {} to {}".format(str(image), str(save_label_path)))
        shutil.move(str(image), str(save_image_path))
        shutil.move(str(label), str(save_label_path))

if __name__ == "__main__":
    main()


