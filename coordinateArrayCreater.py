import numpy as np
import SimpleITK as sitk
import sys

class CoordinateArrayCreater():
    def __init__(self, image_array, center=(0, 0, 0)):
        assert len(center) == len(image_array.shape)
        self.image_array = image_array
        self.center = np.tile(center, image_array.shape)
        self.center = self.center.reshape(-1, *image_array.shape)


    def execute(self):
        lines = [np.arange(s) for s in self.image_array.shape]
        coordinate = np.meshgrid(*lines, indexing="ij")
        self.coordinate = np.stack(coordinate)

        self.relative_coordinate = self.coordinate - self.center

    def getCoordinate(self, kind="relative"):
        if kind == "relative":
            return self.relative_coordinate

        elif kind == "original":
            return self.coordinate
        else:
            print("[ERROR] Kind must be original/relative")
            sys.exit()


def main():
    image = sitk.ReadImage("/home/vmlab/Desktop/data/Abdomen/case_00/imaging_resampled.nii.gz")
    image_array = sitk.GetArrayFromImage(image)
    image_array = np.array([[0, 1, 2], [3, 4, 5]])

    cac = CoordinateArrayCreater(image_array, center=(0,0))
    cac.execute()

if __name__ == "__main__":
    main()
