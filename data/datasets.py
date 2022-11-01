from PIL import Image
import os
from torch.utils.data import Dataset
from skimage.io import imread

    
class PatchPathDataset(Dataset):
    def __init__(self, patch_labels, image_folder, predicting_var, transform=None, patch_folder='Patches'):
        self.patch_labels = patch_labels
        self.image_folder = image_folder
        self.predicting_var = predicting_var
        self.transform = transform
        self.patch_folder = patch_folder

    def __len__(self):
        return len(self.patch_labels)

    def __getitem__(self, idx):
        patch_info = self.patch_labels.loc[idx]
        patch_path = os.path.join(self.image_folder, patch_info.cohort, self.patch_folder, patch_info.slide,
                                  patch_info.magnification, patch_info.patch)
        img = imread(patch_path)
        pil_img = Image.fromarray(img)

        if self.transform is not None:
            pil_img = self.transform(pil_img)

        target = patch_info[self.predicting_var]

        return pil_img, target, patch_path