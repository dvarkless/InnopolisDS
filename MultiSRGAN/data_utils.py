import logging
import logging.handlers as log_handlers
from pathlib import Path
from random import shuffle

import albumentations as A
import cv2
import numpy as np
from alive_progress import alive_bar, alive_it
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import (CenterCrop, Compose, Resize, ToPILImage,
                                    ToTensor)


def is_image_file(filename):
    return (filename.suffix in ['.png', '.jpg', '.jpeg',
            '.PNG', '.JPG', '.JPEG'])


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


def get_transformed_pair_plain(hr_img, crop_size):
    hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2RGB)
    hr_img = A.RandomCrop(crop_size, crop_size,
                          always_apply=True)(image=hr_img)
    # Apply bicubic interpolation:
    lr_img = A.Resize(crop_size // 2, crop_size // 2,
                      interpolation=2, always_apply=True)(image=hr_img['image'])
    lr_img = lr_img['image']
    hr_img = hr_img['image']
    if lr_img.dtype == 'float32':
        lr_img *= 255  # or any coefficient
        lr_img = lr_img.astype(np.uint8)
    if hr_img.dtype == 'float32':
        hr_img *= 255  # or any coefficient
        hr_img = hr_img.astype(np.uint8)
    return ToTensor()(lr_img), ToTensor()(hr_img)


def get_transformed_pair_extended(hr_img, crop_size):
    hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2RGB)
    hr_img = A.RandomCrop(crop_size, crop_size, always_apply=True)(
        image=hr_img)
    hr_img = A.RandomBrightnessContrast()(image=hr_img['image'])
    hr_img = A.RandomRotate90()(image=hr_img['image'])
    hr_img = hr_img['image']
    # Apply bicubic interpolation:
    lr_img = A.Resize(crop_size // 2, crop_size // 2,
                      interpolation=2, always_apply=True)(image=hr_img)
    lr_img = lr_img['image']
    if lr_img.dtype == 'float32':
        lr_img *= 255  # or any coefficient
        lr_img = lr_img.astype(np.uint8)
    if hr_img.dtype == 'float32':
        hr_img *= 255  # or any coefficient
        hr_img = hr_img.astype(np.uint8)
    return ToTensor()(lr_img), ToTensor()(hr_img)


def get_transformed_pair_photo(hr_img, crop_size):
    hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2RGB)
    hr_img = A.RandomCrop(crop_size, crop_size,
                          always_apply=True)(image=hr_img)
    hr_img = A.RandomBrightnessContrast()(image=hr_img['image'])
    hr_img = A.RandomRotate90()(image=hr_img['image'])
    hr_img = hr_img['image']
    # Apply bicubic interpolation:
    lr_img = A.Resize(crop_size // 2, crop_size // 2,
                      interpolation=2, always_apply=True)(image=hr_img)
    lr_img = lr_img['image']
    if lr_img.dtype == 'float32':
        lr_img *= 255  # or any coefficient
        lr_img = lr_img.astype(np.uint8)
    lr_img = A.ISONoise(p=0.5)(image=lr_img)
    lr_img = A.JpegCompression(40, 70, p=0.5)(image=lr_img['image'])
    lr_img = lr_img['image']
    if hr_img.dtype == 'float32':
        hr_img *= 255  # or any coefficient
        hr_img = hr_img.astype(np.uint8)
    return ToTensor()(lr_img), ToTensor()(hr_img)


def get_transformed_pair_game(hr_img, crop_size):
    hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2RGB)
    hr_img = A.RandomCrop(crop_size, crop_size,
                          always_apply=True)(image=hr_img)
    hr_img = A.RandomBrightnessContrast()(image=hr_img['image'])
    hr_img = A.RandomRotate90()(image=hr_img['image'])
    hr_img = hr_img['image']
    # Apply bicubic interpolation:
    lr_img = A.Resize(crop_size // 2, crop_size // 2,
                      interpolation=2, always_apply=True)(image=hr_img)
    lr_img = A.JpegCompression(40, 70, p=0.5)(image=lr_img['image'])
    lr_img = lr_img['image']
    if lr_img.dtype == 'float32':
        lr_img *= 255  # or any coefficient
        lr_img = lr_img.astype(np.uint8)
    if hr_img.dtype == 'float32':
        hr_img *= 255  # or any coefficient
        hr_img = hr_img.astype(np.uint8)
    return ToTensor()(lr_img), ToTensor()(hr_img)


def get_transformed_pair_video(hr_img, crop_size):
    hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2RGB)
    hr_img = A.RandomCrop(crop_size, crop_size,
                          always_apply=True)(image=hr_img)
    hr_img = A.RandomBrightnessContrast()(image=hr_img['image'])
    hr_img = A.RandomRotate90()(image=hr_img['image'])
    hr_img = A.MotionBlur(blur_limit=(3, 11))(image=hr_img['image'])
    hr_img = hr_img['image']
    # Apply bicubic interpolation:
    lr_img = A.Resize(crop_size // 2, crop_size // 2,
                      interpolation=2, always_apply=True)(image=hr_img)
    lr_img = lr_img['image']
    if lr_img.dtype == 'float32':
        lr_img *= 255  # or any coefficient
        lr_img = lr_img.astype(np.uint8)
    lr_img = A.ISONoise(p=0.5)(image=lr_img)
    lr_img = A.JpegCompression(50, 75, p=0.5)(image=lr_img['image'])
    lr_img = lr_img['image']
    if hr_img.dtype == 'float32':
        hr_img *= 255  # or any coefficient
        hr_img = hr_img.astype(np.uint8)
    return ToTensor()(lr_img), ToTensor()(hr_img)


def get_logging_handler():
    formatter = logging.Formatter(
        '%(asctime)s - [%(levelname)s] - [%(module)s] - "%(message)s"')
    logging_handler = log_handlers.TimedRotatingFileHandler(
        'log.log', when='D', interval=2, backupCount=3)
    logging_handler.setFormatter(formatter)
    return logging_handler


def display_transform():
    return Compose([
        Resize(1080),
        CenterCrop(600),
    ])


class TrainDatasetFromFolder(Dataset):
    __available_transforms = {
        'plain': get_transformed_pair_plain,
        'extended': get_transformed_pair_extended,
        'photo': get_transformed_pair_photo,
        'game': get_transformed_pair_game,
        'video': get_transformed_pair_video,
    }

    def __init__(self, dataset_dir, crop_size, transform='plain'):
        super().__init__()
        dataset_dir = Path(dataset_dir)
        self.image_filenames = [x for x in dataset_dir.iterdir()
                                if is_image_file(x)]
        if crop_size % 2 > 0:
            crop_size -= 1
        self.crop_size = crop_size

        if transform not in self.__available_transforms.keys():
            my_lst = self.__available_transforms.keys()
            msg = f'Choose parameter transform={transform} from {my_lst}'
            raise ValueError(msg)
        self.pair_transform = self.__available_transforms[transform]

    def __getitem__(self, index):
        image_path = self.image_filenames[index]
        hr_image = cv2.imread(str(image_path))
        if hr_image is None:
            raise FileNotFoundError(
                f'cannot open hr image at path "{image_path}"')
        lr_image, hr_image = self.pair_transform(
            hr_image,
            self.crop_size)
        return lr_image, hr_image

    def __len__(self):
        return len(self.image_filenames)


class ValDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir):
        super().__init__()
        self.image_filenames = [x for x in dataset_dir.iterdir()
                                if is_image_file(x)]

    def __getitem__(self, index):
        image_path = self.image_filenames[index]
        hr_image = cv2.imread(str(image_path))
        hr_image = cv2.cvtColor(hr_image, cv2.COLOR_BGR2RGB)
        if hr_image is None:
            raise FileNotFoundError(
                f'cannot open hr image at path "{image_path}"')

        w, h = hr_image.shape[0:2]
        crop_size = min(w, h)
        if crop_size % 2 > 0:
            crop_size -= 1
        lr_scale = A.Resize(crop_size // 2, crop_size // 2, interpolation=2)
        hr_scale = A.Resize(crop_size, crop_size, interpolation=2)
        hr_image = A.CenterCrop(crop_size, crop_size)(image=hr_image)['image']
        lr_image = lr_scale(image=hr_image)['image']
        hr_restore_img = hr_scale(image=lr_image)['image']
        return ToTensor()(lr_image), ToTensor()(hr_restore_img), \
            ToTensor()(hr_image)

    def __len__(self):
        return len(self.image_filenames)


class TestDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir):
        super().__init__()
        dataset_dir = Path(dataset_dir)
        self.lr_path = dataset_dir / 'SRF_2/data/'
        self.hr_path = dataset_dir / 'SRF_2/target/'
        self.lr_filenames = [x for x in self.lr_path.iterdir()
                             if is_image_file(x)]
        self.hr_filenames = [x for x in self.hr_path.iterdir()
                             if is_image_file(x)]

    def __getitem__(self, index):
        image_name = self.lr_filenames[index].stem
        lr_image = cv2.imread(str(self.lr_filenames[index]))
        if lr_image is None:
            raise FileNotFoundError(f'cannot open lr image "{image_name}"')
        w, h = lr_image.shape[0:2]
        hr_image = cv2.imread(str(self.hr_filenames[index]))
        if hr_image is None:
            raise FileNotFoundError(f'cannot open hr image "{image_name}"')
        hr_scale = A.Resize(2 * h, 2 * w, interpolation=2)
        hr_restore_img = hr_scale(image=lr_image)['image']
        return image_name, ToTensor()(lr_image), ToTensor()(hr_restore_img), \
            ToTensor()(hr_image)

    def __len__(self):
        return len(self.lr_filenames)


def train_test_split_save(dir, percentage=0.1):
    dir = Path(dir)
    lst_dir = [path for path in dir.iterdir()
               if path.is_file()]
    test_len = int(len(lst_dir)*percentage)
    shuffle(lst_dir)
    test_lst, train_lst = lst_dir[:test_len], lst_dir[test_len:]
    test_dir = dir / Path('valid')
    train_dir = dir / Path('train')
    test_dir.mkdir(exist_ok=True)
    train_dir.mkdir(exist_ok=True)
    for item in test_lst:
        item.rename(test_dir / item.name)
    for item in train_lst:
        item.rename(train_dir / item.name)


class VideoSlicer:
    __formats = ['.mp4', '.mp3', '.mkv', '.avi']

    def __init__(self, expected_output, output_dir='sliced_data') -> None:
        self.expected_output = expected_output
        self.output_dir = Path(output_dir)
        if not self.output_dir.exists():
            self.output_dir.mkdir()

    def _estimated_length(self, path_lst):
        video_len = 0
        for video_path in path_lst:
            capture = cv2.VideoCapture(str(video_path))
            video_len += capture.get(cv2.CAP_PROP_FRAME_COUNT)
        return int(video_len)

    def _full_capture_read(self, path_lst):
        for video_path in path_lst:
            print(f'Opening file "{video_path.name}"')
            capture = cv2.VideoCapture(str(video_path))
            success = True
            while success:
                success, frame = capture.read()
                yield success, frame

    def _get_list_files(self, path):
        path = Path(path)
        if path.exists():
            if path.is_dir():
                return [file for file in path.iterdir()
                        if file.suffix in self.__formats]
            else:
                if path.suffix in self.__formats:
                    return [path]
                else:
                    return []
        return []

    def slice(self, file_path):
        files = self._get_list_files(file_path)
        if not files:
            msg = f'Cannot find any valid video file at path "{file_path}"'
            raise FileNotFoundError(msg)

        frames_count = self._estimated_length(files)
        save_divisor = frames_count // self.expected_output
        assert save_divisor > 0
        count = 0
        with alive_bar(frames_count, dual_line=True) as progress:
            for i, (success, frame) in enumerate(self._full_capture_read(files)):
                if i == 0:
                    progress.title('Slicing video...')
                if success:
                    if i % save_divisor == 0:
                        if count > self.expected_output:
                            break
                        if frame.mean() < 5:
                            continue
                        img_path = self.output_dir / f'{count}.png'
                        cv2.imwrite(str(img_path), frame)
                        count += 1
                progress.text(f'Images so far: ({count})')
                progress()

        msg = f'Slicing is done, got {count} images'
        print(msg)

    def __call__(self, file_path, /):
        self.slice(file_path)


def main():
    # path = "/run/media/dvarkless/WindowsData/Video/"
    # VideoSlicer(800, output_dir='data/no_mans_sky_1080')(path)
    train_test_split_save('data/no_mans_sky_1080/')


if __name__ == '__main__':
    main()
