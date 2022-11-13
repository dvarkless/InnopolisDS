import logging
from datetime import date
from math import log10
from pathlib import Path

import pandas as pd
import pytorch_ssim
import shutup
import torch
import torch.optim as optim
import torchvision.utils as utils
from alive_progress import alive_it
from data_utils import (TrainDatasetFromFolder, ValDatasetFromFolder,
                        display_transform, get_logging_handler)
from loss import AdversarialLoss, FullLoss, MSELoss
from matplotlib import pyplot as plt
from model import Discriminator, Generator
from torch.autograd import Variable
# from torch.autograd.anomaly_mode import set_detect_anomaly
from torch.utils.data import DataLoader

shutup.please()


class Trainer:
    """Класс для тренировки модели SRGAN."""

    __model_types = ['mse', 'gan', 'full']
    __loss_types = {'mse': MSELoss,
                    'gan': AdversarialLoss,
                    'full': FullLoss}

    def __init__(self, crop_size=120, epochs=100,
                 gen_optimizer=None, disc_optimizer=None,
                 gen_optimizer_params=None, disc_optimizer_params=None,
                 verbose_logs=False, gen_model_name=None,
                 disc_model_name=None, model_type='full',
                 save_interval=10, loss_coeffs=None,
                 colab=True) -> None:
        """
    crop_size - размер вырезаемого при обучении окна из изображения.
    Влияет на занимаемую память при обучении.

    epochs - количество эпох обучения

    gen_optimizer - класс оптимизатора генератора из PyTorch
    по умолчанию - Adam

    disc_optimizer - класс оптимизатора генератора из PyTorch
    по умолчанию - Adam

    gen_optimizer_params - параметры для передачи в инициализируемый
    класс оптимизатора генератора

    disc_optimizer_params - параметры для передачи в инициализируемый
    класс оптимизатора дискриминатора

    verbose_logs - не используется и не будет использоваться, как
    я полагаю

    gen_model_name - задать имя генератора при сохранении, полезно
    при автоматизации

    disc_model_name - задать имя дискриминатора при сохранении, полезно
    при автоматизации

    model_type - тип модели - только с метрикой mse, только с метрикой gan,
    полная ("mse", "gan", "full")

    save_interval - интервал сохранения модели в эпохах

    loss_coeffs: Tuple[float, float, float] - коэффициенты функций потерь:
    MSE, GAN, ResNet50 (1, 0.01, 0.05)

    colab: Bool - поставьте False если запускаете в колабе
        """

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG if verbose_logs else logging.INFO)
        self.logger.addHandler(get_logging_handler())
        self.logger.info('======= TRAINER STARTED ======')

        self.cuda = torch.cuda.is_available()
        if not self.cuda:
            msg = f'Could not detect GPU on board, aborting \
                    (torch.cuda.is_available={self.cuda})'
            self.logger.error(msg)
            raise OSError(msg)

        self.model_type = model_type
        assert self.model_type in self.__model_types
        self.has_disc = True if self.model_type != 'mse' else False

        self.save_interval = save_interval
        self.colab = colab

        self.gen_model = Generator()
        self.disc_model = Discriminator()

        if gen_model_name and disc_model_name:
            self.logger.info(
                f'Loading models from files "{gen_model_name}" \
                  and "{disc_model_name}"')
            self.gen_model.load_state_dict(torch.load(gen_model_name))
            self.disc_model.load_state_dict(torch.load(disc_model_name))
        else:
            self.logger.info('Initializing new models')

        gen_optimizer_params = gen_optimizer_params if gen_optimizer_params else dict()
        disc_optimizer_params = disc_optimizer_params if disc_optimizer_params else dict()
        gen_optimizer_params['params'] = self.gen_model.parameters()
        disc_optimizer_params['params'] = self.disc_model.parameters()

        gen_optimizer = gen_optimizer if gen_optimizer else optim.Adam
        disc_optimizer = disc_optimizer if disc_optimizer else optim.Adam
        self.logger.info(
            f'G_optimizer is "{gen_optimizer.__name__}", \
              params = {gen_optimizer_params}')
        if self.has_disc:
            self.logger.info(
                f'D_optimizer is "{disc_optimizer.__name__}", \
                  params = {disc_optimizer_params}')
        self.gen_optimizer = gen_optimizer(**gen_optimizer_params)
        self.disc_optimizer = disc_optimizer(**disc_optimizer_params)

        if loss_coeffs is None:
            self.gen_criterion = self.__loss_types[self.model_type]()
        else:
            self.gen_criterion = self.__loss_types[self.model_type](
                *loss_coeffs)

        if self.model_type == 'mse':
            self._step_batch = self._step_batch_mse
        if self.model_type == 'gan':
            self._step_batch = self._step_batch_gan
        if self.model_type == 'full':
            self._step_batch = self._step_batch_full

        self.epochs = epochs
        self.crop_size = crop_size

        creation_time_str = date.today().isoformat()
        self.out_path = Path(f'training_results/SRF_{creation_time_str}')
        self.logger.debug(f'writing data into "{self.out_path}"')
        if not self.out_path.exists():
            self.logger.debug(f'Creating directory "{self.out_path}"')
            self.out_path.mkdir()

        self.trainer_results = []

        # print('# generator parameters:', sum(param.numel() for param in self.gen_model.parameters()))
        # print('# discriminator parameters:', sum(param.numel() for param in self.disc_model.parameters()))
        self.gen_model.cuda()
        self.disc_model.cuda()
        self.gen_criterion.cuda()
        self.logger.debug('CUDA ok')

        if not self.has_disc:
            del self.disc_model
            del self.disc_optimizer

    def fit(self, train_hr_dir, eval_hr_dir, batch_size=32,
            data_augmentation_type='plain', model_tag=None,
            save_g_as=None, save_d_as=None):
        """
           Запуск обучения.

           train_hr_dir - путь к тренировочному датасету

           eval_hr_dir - путь к проверочному датасету

           batch_size - размер выборки

           data_augmentation_type - тип аугментаций изображении

           model_tag - вставлять это в названия релевантных файлов

           save_g_as - имя генератора при сохранении

           save_d_as - имя дискриминатора при сохранении
        """
        if model_tag:
            self.model_tag = model_tag
        train_hr_dir = Path(train_hr_dir)
        eval_hr_dir = Path(eval_hr_dir)
        train_set = TrainDatasetFromFolder(
            train_hr_dir, crop_size=self.crop_size,
            transform=data_augmentation_type)
        eval_set = ValDatasetFromFolder(eval_hr_dir)

        train_loader = DataLoader(
            dataset=train_set, num_workers=4, batch_size=batch_size, shuffle=True)
        eval_loader = DataLoader(
            dataset=eval_set, num_workers=4, batch_size=1, shuffle=False)

        bar = alive_it(range(self.epochs), self.epochs,
                       calibrate=0.5, force_tty=self.colab,
                       dual_line=True)
        epoch = 0
        for epoch in bar:
            if hasattr(self, 'model_tag'):
                bar.title(f'SRGAN_{model_tag}.fit()')
            else:
                bar.title('SRGAN.fit()')
            if epoch % self.save_interval == 0 and epoch > 0:
                bar.text('-> Saving models and history...')
                self._save_models(epoch)

            bar.text(f'-> Training epoch {epoch+1}/{self.epochs}')
            gen_loss, disc_loss, gen_score, disc_score = self._train_epoch(
                train_loader)

            print(f'============= {epoch+1}/{self.epochs} =============')
            print(
                f'Training losses: Generator: {gen_loss:.4f}, Discriminator: {disc_loss:.4f}')
            print(f'Generator score:     ({gen_score:.3f})')
            print(f'Discriminator score: ({disc_score:.3f})')
            print('Separate losses:')

            img_l, adv_l, perc_l = self.gen_criterion.get_losses()

            print(f'Image loss = {img_l:.3f};', end=' ')
            print(f'Adversarial loss = {adv_l:.3f};', end=' ')
            print(f'Perception loss = {perc_l:.3f}')
            bar.text(f'-> Evaluating epoch {epoch+1}/{self.epochs}')

            results = self._eval_epoch(eval_loader, epoch)

            print(f'Evaluation results:')
            for name, val in results.items():
                print(f'{name} = {val:.4f}')

            results['gen_loss'] = gen_loss
            results['disc_loss'] = disc_loss
            results['gen_score'] = gen_score
            results['disc_score'] = disc_score
            self.trainer_results.append(results.copy())
            del results

        self._save_models(epoch+1, save_g_as, save_d_as)

    def _save_models(self, epoch: int = 0, g_name=None, d_name=None):
        msg = 'saving models and history...'
        self.logger.info(msg)

        models_path = Path('models')
        if not models_path.exists():
            models_path.mkdir()
        if g_name is None:
            g_name = f'Generator_{date.today().isoformat()}'\
                + f'_epoch{epoch}'*bool(epoch)
            if hasattr(self, 'model_tag'):
                g_name += f'_{self.model_tag}'
        if d_name is None:
            d_name = f'Discriminator_{date.today().isoformat()}'\
                + f'_epoch{epoch}'*bool(epoch)
            if hasattr(self, 'model_tag'):
                d_name += f'_{self.model_tag}'

        torch.save(self.gen_model.state_dict(), models_path / f'{g_name}.pt')
        if self.has_disc:
            torch.save(self.disc_model.state_dict(),
                       models_path / f'{d_name}.pt')

        metric_table_name = f'Training_metrics_{date.today().isoformat()}'
        if hasattr(self, 'model_tag'):
            metric_table_name += f'_{self.model_tag}'
        self._save_metric_data(metric_table_name, epoch)

    def _step_batch_mse(self, lr_img, hr_img, *args):
        self.gen_model.zero_grad()
        sr_img = self.gen_model(lr_img)
        self.logger.debug(f'sr_img info: shape = {sr_img.shape}')
        gen_loss = self.gen_criterion(sr_img, hr_img)
        gen_loss.backward()
        self.gen_optimizer.step()

        return (
            gen_loss.item(),
            0,
            1,
            0,
        )

    def _step_batch_gan(self, lr_img, hr_img, hr_aug):
        sr_img = self.gen_model(lr_img)
        self.logger.debug(f'sr_img info: shape = {sr_img.shape}')

        self.disc_model.zero_grad()

        real_out = self.disc_model(hr_aug).mean()
        fake_out = self.disc_model(sr_img).mean()
        self.logger.debug(f'real_out = {real_out}')
        self.logger.debug(f'fake_out = {fake_out}')
        disc_loss = -torch.log(1-fake_out) - torch.log(real_out)
        self.logger.debug(f'disc_loss = {disc_loss}')
        disc_loss.backward()
        self.disc_optimizer.step()

        self.gen_model.zero_grad()
        sr_img = self.gen_model(lr_img)
        fake_out = self.disc_model(sr_img).mean()

        gen_loss = self.gen_criterion(fake_out, sr_img, hr_img)
        gen_loss.backward()
        self.gen_optimizer.step()

        return (
            gen_loss.item(),
            disc_loss.item(),
            fake_out.item(),
            real_out.item(),
        )

    def _step_batch_full(self, lr_img, hr_img, hr_aug):
        sr_img = self.gen_model(lr_img)
        self.logger.debug(f'sr_img info: shape = {sr_img.shape}')

        self.disc_model.zero_grad()

        real_out = self.disc_model(hr_aug).mean()
        fake_out = self.disc_model(sr_img).mean()
        self.logger.debug(f'real_out = {real_out}')
        self.logger.debug(f'fake_out = {fake_out}')
        disc_loss = -torch.log(1-fake_out) - torch.log(real_out)
        self.logger.debug(f'disc_loss = {disc_loss}')
        disc_loss.backward()
        self.disc_optimizer.step()

        self.gen_model.zero_grad()
        sr_img = self.gen_model(lr_img)
        fake_out = self.disc_model(sr_img).mean()

        gen_loss = self.gen_criterion(fake_out, sr_img, hr_img)
        gen_loss.backward()
        self.gen_optimizer.step()

        return (
            gen_loss.item(),
            disc_loss.item(),
            fake_out.item(),
            real_out.item(),
        )

    def _train_epoch(self, loader):
        if not loader:
            msg = 'DataLoader is not defined'
            self.logger.error(msg)
            raise ValueError(msg)

        self.gen_model.train()
        if self.has_disc:
            self.disc_model.train()
        self.gen_criterion.clear_losses()

        gen_epoch_loss, disc_epoch_loss = 0, 0
        gen_epoch_score, disc_epoch_score = 0, 0
        i = 0
        for i, data in enumerate(loader):
            self.logger.debug(f'========= BATCH N{i} ==========')
            lr_img, hr_img, hr_aug = data
            hr_aug = Variable(hr_img).cuda()
            hr_img = Variable(hr_img).cuda()
            lr_img = Variable(lr_img).cuda()

            self.logger.debug(f'hr_img info: shape = {hr_img.shape}')
            self.logger.debug(f'lr_img info: shape = {lr_img.shape}')
            self.logger.debug(f'hr_aug info: shape = {hr_aug.shape}')

            gen_loss, disc_loss, gen_score, disc_score = self._step_batch(
                lr_img,
                hr_img,
                hr_aug,
            )
            gen_epoch_loss += gen_loss
            disc_epoch_loss += disc_loss
            gen_epoch_score += gen_score
            disc_epoch_score += disc_score
            self.logger.debug(f'gen_loss = {gen_loss}')
            self.logger.debug(f'disc_loss = {disc_loss}')
            self.logger.debug(f'gen_score = {gen_score}')
            self.logger.debug(f'disc_score = {disc_score}')
        i += 1

        gen_epoch_score /= i
        disc_epoch_score /= i
        gen_epoch_loss /= i
        disc_epoch_loss /= i
        return (gen_epoch_loss, disc_epoch_loss,
                gen_epoch_score, disc_epoch_score)

    def _eval_epoch(self, loader, epoch):
        self.gen_model.eval()

        with torch.no_grad():
            evaling_results = {'mse': 0.0, 'psnr': 0.0, 'ssim': 0.0}
            eval_images = []
            i = 0
            for i, (eval_lr, eval_restored, eval_hr) in enumerate(loader):
                eval_lr = eval_lr.cuda()
                eval_hr = eval_hr.cuda()

                eval_sr = self.gen_model(eval_lr)
                single_mse = torch.nn.functional.mse_loss(eval_sr, eval_hr)
                evaling_results['mse'] += single_mse.item()

                evaling_results['psnr'] += (eval_hr.max().item()
                                            ** 2) / single_mse.item()
                evaling_results['ssim'] += pytorch_ssim.ssim(
                    eval_sr, eval_hr).item()
                eval_images.extend(
                    [display_transform()(eval_restored.squeeze(0)),
                     display_transform()(eval_hr.data.cpu().squeeze(0)),
                     display_transform()(eval_sr.data.cpu().squeeze(0))])
            i += 1
            evaling_results['mse'] /= i
            evaling_results['psnr'] /= i
            evaling_results['psnr'] = 10 * log10(evaling_results['psnr'])
            evaling_results['ssim'] /= i

            for name, val in evaling_results.items():
                self.logger.debug(f'==={name} = {val:.3f}===')
            eval_images = torch.stack(eval_images)
            eval_images = torch.chunk(eval_images, eval_images.size(0) // 15)
            index = 1
            for image in eval_images:
                image = utils.make_grid(image, nrow=3, padding=5)
                image_name = f'epoch_{epoch}_N{index}_psnr-'
                image_name += f'{evaling_results["psnr"]:.3f}db_ssim-'
                image_name += f'{evaling_results["ssim"]:.3f}'
                if hasattr(self, 'model_tag'):
                    image_name += f'_{self.model_tag}'
                image_name += '.png'

                utils.save_image(
                    image,
                    self.out_path / image_name,
                    padding=5)
                index += 1
                if index > 5:
                    break

        return evaling_results

    def _save_metric_data(self, name, epoch):
        out_path = Path('statistics/')
        data_frame = pd.DataFrame(
            data={'Loss_D': self.get_metric_list('disc_loss'),
                  'Loss_G': self.get_metric_list('gen_loss'),
                  'Score_D': self.get_metric_list('disc_score'),
                  'Score_G': self.get_metric_list('gen_score'),
                  'PSNR': self.get_metric_list('psnr'),
                  'SSIM': self.get_metric_list('ssim')},
            index=range(0, epoch))
        data_frame.to_csv(out_path / f'{name}.csv', index_label='Epoch')

    def get_metric_list(self, metric_name):
        """
        Получить список с метрикой обучения.

        metric_name - имя метрики
        """
        out_data = []
        if not (metric_name in self.trainer_results[0].keys()):
            msg = f'Wrong metric name: "{metric_name}"'
            self.logger.error(msg)
            print(msg)
            raise KeyError(msg)

        for item in self.trainer_results:
            item = item.cpu() if isinstance(item, torch.Tensor) else item
            out_data.append(item[metric_name])

        return out_data
