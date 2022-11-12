import subprocess
import sys
import time
from math import log
from operator import mod, xor
# import matplotlib.pyplot as plt
# import logging
from pathlib import Path

import cv2
import moviepy.editor as mpe
import numpy as np
import torch
import torchvision.transforms as transforms
from alive_progress import alive_it
# import torchvision.transforms as transforms
from PIL import Image, ImageShow
from torch.autograd import Variable
from torchvision.transforms import (InterpolationMode, Resize, ToPILImage,
                                    ToTensor)

import pytorch_ssim
from decorators import no_grad, timer
from model import Generator


class ModelTester:
    def __init__(self, model_path, use_gpu=True, model_tag=None) -> None:
        self.logger = None
        self.use_gpu = use_gpu
        self.last_runtime = 0
        self.model_runtime = 0
        self.root_dir = Path('test_output')
        self.input_dir = Path('test_input')
        self.model_tag = '' if model_tag is None else model_tag

        if not self.root_dir.exists():
            self.root_dir.mkdir()

        if self.use_gpu:
            if not torch.cuda.is_available():
                raise OSError('CUDA is not available')

        self.model = Generator()
        if self.use_gpu:
            self.model.cuda()

        model_path = Path(model_path)
        if model_path.exists():
            self.model.load_state_dict(torch.load(model_path))
            # self.model = torch.load(model_path)
            # self.model.cuda()
        else:
            raise FileNotFoundError(
                f'Model at path "{model_path.absolute().resolve()}" is not found')

        self.model.eval()

    def __image_type_change(self, img, none_is_ok=False, compress=False):
        if img is None:
            if none_is_ok:
                return None
        if isinstance(img, str):
            img = Path(img)
        if isinstance(img, Path):
            img = Image.open(img)
            img = Resize((img.size[1]+mod(img.size[1], 2),
                          img.size[0]+mod(img.size[0], 2)),
                         InterpolationMode.BICUBIC)(img)
            if compress:
                img = Resize((img.size[1]//2, img.size[0]//2),
                             InterpolationMode.BICUBIC)(img)
        if isinstance(img, Image.Image):
            img = Variable(ToTensor()(img)).unsqueeze(0)
        if not isinstance(img, torch.Tensor):
            raise TypeError(
                f'The type of input image of "{type(img)}" is unacceptable')
        return img

    @timer
    @no_grad
    def run_on_image(self,
                     lr_image: Image.Image | torch.Tensor | Path | str | None = None,
                     hr_image: Image.Image | torch.Tensor | Path | str | None = None,
                     output_mode: str | bool = False, save_bicubic=False,
                     ) -> None:
        imgs = [lr_image, hr_image]
        title = 'Provided image'
        for i, image in enumerate(imgs):
            if isinstance(image, str | Path):
                title = Path(image).name
                imgs[i] = image
                for suf in ['.png', '.jpg']:
                    title = title.removesuffix(suf)
        lr_image, hr_image = imgs

        if not xor(lr_image is None, hr_image is None):
            raise ValueError('Please provide exactly one image')

        lr_image = self.__image_type_change(lr_image, True)
        compressed_image = self.__image_type_change(hr_image, True,
                                                    compress=True)
        hr_image = self.__image_type_change(hr_image, True)
        image = compressed_image if lr_image is None else lr_image
        # thing to shut typechecker up:
        if not isinstance(image, torch.Tensor):
            print(image)
            raise ValueError('Something went wrong with image tensor, lol')
        if self.use_gpu:
            if isinstance(hr_image, torch.Tensor):
                hr_image = hr_image.cuda()
            image = image.cuda()
        start_time = time.time()
        try:
            out_image = self.model(image)
        except RuntimeError as e:
            print(e)
            print('=====================================================')
            print(
                f'Input image of size {tuple(image.shape[2:])} is too big for your GPU, try a smaller one')
            sys.exit(1)

        image_size = out_image.shape
        self.model_runtime = time.time() - start_time

        if isinstance(hr_image, torch.Tensor):
            image_psnr, image_ssim = self.get_metrics(
                out_image, hr_image)
        else:
            image_psnr, image_ssim = None, None
        out_image = ToPILImage()(out_image[0].data.cpu())

        # another type converter
        if isinstance(output_mode, str) or output_mode is None:
            if output_mode == 'show':
                output_mode = True
            elif output_mode == 'save':
                output_mode = False
            else:
                raise ValueError(
                    f'Unknown value of argument output_mode = "{output_mode}" \
                    chosen from ("save", "show") or bool value')

        # save or show our image
        if save_bicubic:
            bicubic_img = ToPILImage()(image[0].data.cpu())
            transform = Resize(
                image_size[2:], interpolation=InterpolationMode.BICUBIC)
            bicubic_img = transform(bicubic_img)
            bicubic_img.save(self.root_dir / f'{title}_bicubic.png')

        if output_mode:
            if isinstance(hr_image, torch.Tensor):
                title = f'{title} with metrics: psnr \
                         ={image_psnr:.2f}db, ssim={image_ssim:.2f}'
            ImageShow.show(out_image, title=title)
        else:
            if isinstance(hr_image, torch.Tensor):
                filename = f'upscaled_{self.model_tag}_{title}_psnr_{image_psnr:.2f}db_ssim_{image_ssim:.2f}.png'
            else:
                filename = f'upscaled_{self.model_tag}_{title}.png'
            out_image.save(self.root_dir / filename)

    @timer
    @no_grad
    def run_on_video(self, vid_path: str | Path, video_type: str):
        if video_type not in ('upscale', 'comparison'):
            raise ValueError(
                'please choose argument "video_type" from ("upscale", "comparison")')
        if not isinstance(vid_path, str | Path):
            raise TypeError(
                f'Type of method input is {type(vid_path)},\
                  should be [str | pathlib.Path]')
        vid_path = Path(vid_path)

        videoCapture = cv2.VideoCapture(str(vid_path))
        fps = videoCapture.get(cv2.CAP_PROP_FPS)
        frame_numbers = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
        w = videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)
        h = videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.sr_video_size = (int(w * 2), int(h * 2))

        filename = vid_path.stem
        if video_type == 'upscale':
            filename = f'upscaled_{vid_path.stem}'
            self.sr_writer = self.sr_writer_builder(filename, fps)
        else:
            self.sr_writer = None
        if video_type == 'comparison':
            filename = f'lr_vs_sr_for_{vid_path.stem}'
            self.compared_writer = self.compared_writer_builder(filename, fps)
        else:
            self.compared_writer = None

        self.__save_audio_from_video(vid_path)
        conversion_progress = alive_it(range(int(frame_numbers)))

        for i in conversion_progress:
            success, frame = videoCapture.read()
            if success:
                image = Variable(ToTensor()(frame)).unsqueeze(0)
                if self.use_gpu:
                    image = image.cuda()

                sr_tensor = self.model(image)
                sr_tensor = sr_tensor.cpu()
                sr_array = sr_tensor.data[0].numpy()
                sr_array *= 255.0
                sr_array = (np.uint8(sr_array)).transpose((1, 2, 0))
                if video_type == 'upscale':
                    self.sr_writer.write(sr_array)
                if video_type == 'comparison':
                    sr_img = self.compared_frame_assemble(sr_array, 'sr')
                    lr_img = self.compared_frame_assemble(frame, 'lr')
                    final_image = np.concatenate(
                        (np.asarray(lr_img), np.asarray(sr_img)), axis=1)
                    self.compared_writer.write(final_image)
            else:
                print(f'frame N{i}\'s dropped')
        vid_name = filename + '.avi'
        self.__insert_saved_audio(self.root_dir / vid_name)

    def get_metrics(self, sr_image: torch.Tensor,
                    hr_image: torch.Tensor) -> tuple[float, float]:
        mse = torch.nn.functional.mse_loss(sr_image, hr_image)
        mse = mse.item()
        psnr = 10 * log((hr_image.max().item()**2) / mse)
        ssim = pytorch_ssim.ssim(sr_image, hr_image).item()
        return psnr, ssim

    def __save_audio_from_video(self, vid_path: str | Path):
        if Path(vid_path).exists():
            video_clip = mpe.VideoFileClip(str(vid_path))
        else:
            raise FileNotFoundError(f'Cannot find file "{vid_path}"')

        audio_clip = video_clip.audio
        if audio_clip is None:
            print('This video has no audio')
            return
        audio_clip.write_audiofile(str(self.root_dir / 'temp_audio.wav'))

    def __insert_saved_audio(self, vid_path):
        vid_path = Path(vid_path)
        if not vid_path.exists():
            raise FileNotFoundError(f'Cannot find file "{vid_path}"')
        cmd = f'ffmpeg -i {vid_path} -i {self.root_dir}/temp_audio.wav \
               -map 0:v -map 1:a -c:v copy -shortest \
               {self.root_dir}/{vid_path.stem}.mp4'
        subprocess.call(cmd, shell=True)

    def sr_writer_builder(self, filename, fps):
        filename = filename + '.avi'
        return cv2.VideoWriter(str(self.root_dir / filename),
                               cv2.VideoWriter_fourcc(*'MPEG'), fps, self.sr_video_size)

    def compared_writer_builder(self, filename, fps):
        filename = filename + '.avi'
        compared_video_size = (
            int(self.sr_video_size[0] * 2 + 10), int(self.sr_video_size[1] * 2))
        return cv2.VideoWriter(str(self.root_dir / filename),
                               cv2.VideoWriter_fourcc(*'MPEG'), fps, compared_video_size)

    def compared_frame_assemble(self, frame: np.ndarray, frame_type: str):
        frame = ToPILImage()(frame)
        if frame_type == 'lr':
            frame = transforms.Resize(
                (self.sr_video_size[1], self.sr_video_size[0]),
                interpolation=InterpolationMode.BICUBIC)(frame)
            return transforms.Pad(padding=(0, 0, 5, 5))(frame)
        if frame_type == 'sr':
            return transforms.Pad(padding=(5, 0, 0, 5))(frame)


if __name__ == '__main__':
    tree_image = 'tree.png'
    vegetaples_image = 'vegatables.png'
    sky_image = 'skyscraper.png'
    laptop_image = 'laptop.png'
    mse_model = '/run/media/dvarkless/LinuxData/Files/Учеба/Data_Science_Course/SRGAN/models/favorites/mse_vs_gan_vs_full/mse_only_120.pt'
    gan_model = '/run/media/dvarkless/LinuxData/Files/Учеба/Data_Science_Course/SRGAN/models/favorites/mse_vs_gan_vs_full/gan_only_120.pt'
    full_model = '/run/media/dvarkless/LinuxData/Files/Учеба/Data_Science_Course/SRGAN/models/favorites/augmentations_full/Generator_2022-11-06_epoch220_photo.pt'
    model_names = ['mse', 'gan', 'full']
    models = [mse_model, gan_model, full_model]
    images = [tree_image, vegetaples_image, sky_image, laptop_image]
    for name, model_path in zip(model_names, models):
        tester = ModelTester(model_path, model_tag=name)
        for image_path in images:
            tester.run_on_image(hr_image=image_path, save_bicubic=True)
    print('Done!')
