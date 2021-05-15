import SimpleITK as sitk
import numpy as np
from torch.utils.data import Dataset as BaseDataset
from torch.utils.data import DataLoader
from os import listdir
from os.path import isfile, join
import os
import uuid
import zipfile
from django.conf import settings
import cv2
import shutil

SIZE = 160


def resize_image(img, size):
    method = cv2.INTER_AREA
    if img.shape[0] < size or img.shape[1] < size:
        method = cv2.INTER_CUBIC
    return cv2.resize(img, (size, size), interpolation=method)


class Dataset(BaseDataset):
    def __init__(
            self,
            images,
            sitk_image
    ):
        self.images = images
        self.ct_obj = sitk_image

    def __getitem__(self, i):
        # read data
        real_image = self.images[i]

        check = np.array(real_image)
        if len(check.shape) > 2:
            drop_index = -1
            for i in range(len(check.shape)):
                if check.shape[i] == 1:
                    drop_index = i
                    break
            real_image = check.squeeze(drop_index)

        image = resize_image(real_image, SIZE)
        image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        s = image
        image = np.rollaxis(s[..., np.newaxis], 2, 0)

        image = np.array(image, np.float32)

        return image, real_image

    @staticmethod
    def resize_image(img):
        return resize_image(img, SIZE)

    def get_sitk_object(self):
        return self.ct_obj

    def __len__(self):
        return len(self.images)


def handle_uploaded_file(f):
    is_dir = False
    uuid_str = str(uuid.uuid1())
    temp_name = uuid_str + "-" + f.name
    path = os.path.join(settings.BASE_DIR, 'temp', temp_name)
    with open(path, 'wb+') as destination:
        for chunk in f.chunks():
            destination.write(chunk)
    if f.name.endswith('.zip'):
        temp_dir = os.path.join(settings.BASE_DIR, 'temp', temp_name.replace(".zip", ""))
        with zipfile.ZipFile(path, 'r') as zip_ref:
            zip_ref.extractall(os.path.join(settings.BASE_DIR, 'temp', temp_dir))
        is_dir = True
        os.remove(path)
        path = temp_dir
    return path, is_dir


def load_as_dir(dir_path):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dir_path)
    reader.SetFileNames(dicom_names)
    input_image = reader.Execute()
    return sitk.GetArrayFromImage(input_image), input_image


def load_as_file(file_path):
    input_image = sitk.ReadImage(file_path)
    return sitk.GetArrayFromImage(input_image),input_image


def get_dataloader(path, is_dir):
    original_slices = []
    sitk_obj = None
    if is_dir:
        original_slices,sitk_obj = load_as_dir(path)
    else:
        original_slices,sitk_obj = load_as_file(path)
    user_dataset = Dataset(images=original_slices, sitk_image=sitk_obj)
    user_dataloader = DataLoader(user_dataset, batch_size=5, shuffle=False)
    if is_dir:
        shutil.rmtree(path)
    else:
        os.remove(path)
    return user_dataloader, user_dataset


def full_load(file):
    path, is_dir = handle_uploaded_file(file)
    return getattr(path, is_dir)
