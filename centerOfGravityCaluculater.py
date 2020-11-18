import SimpleITK as sitk
import numpy as np

class CenterOfGravityCaluculater():
    def __init__(self, image):
        self.image_array = sitk.GetArrayFromImage(image)

    def execute(self):
        coordinates = np.where(self.image_array == 1)
        center = [np.average(c) for c in coordinates]

        return np.array(center)

if __name__ == "__main__":
    image = sitk.ReadImage("/home/vmlab/Desktop/data/Abdomen/case_00/liver_resampled.mha")
    a = np.array([[0, 1, 1], [1, 0, 0]])
    print(a)
    print(a.shape)
    a_ = sitk.GetImageFromArray(a)

    cogc = CenterOfGravityCaluculater(a_)
    c = cogc.execute()

    print(c)
    """
    a__ = np.meshgrid(*[np.arange(s) for s in a.shape], indexing="ij")
    a__ = np.stack(a__)
    print(a__.shape)
    c_ = np.tile(c, a.shape).reshape(-1, *a.shape)
    print(c_.shape)
    print(c_)
    print(a__ - c_)
    """
