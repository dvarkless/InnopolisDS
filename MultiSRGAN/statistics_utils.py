from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class PlotProcessor:
    def __init__(self, csv_path) -> None:
        self.data = self._get_from_csv(csv_path)
        sns.set_style("darkgrid")

    def _show(self):
        pass

    def _get_from_csv(self, csv_path: str | Path) -> pd.DataFrame:
        csv_path = Path(csv_path)
        if not csv_path.exists():
            msg = f'Could not open file "{csv_path}"'
            raise FileNotFoundError(msg)
        return pd.read_csv(csv_path)

    def _get_slice(self, pos, first_half=True):
        pass

    def add_data(self, csv_path, before=None, after=None):
        pass

    def show_gan_score(self, markers=False, title=None):
        gan_data = self.data.loc[:, ["Epoch", "Score_G", "Score_D"]]
        sns.lineplot(x='Epoch', y='value', hue='variable',
                     data=pd.melt(gan_data, ['Epoch']),
                     markers=markers)
        if title:
            plt.title(title)
            plt.ylabel('G_vs_D')
        plt.show()

    def show_image_score(self, metric='ssim', markers=False, title=None):
        if metric == 'psnr':
            metric = 'PSNR (db)'
            image_data = self.data.loc[:, ["Epoch", "PSNR"]]
        elif metric == 'ssim':
            metric = 'SSIM'
            image_data = self.data.loc[:, ["Epoch", "SSIM"]]
        else:
            raise KeyError(metric)

        g = sns.lineplot(x='Epoch', y='value', hue='variable',
                         data=pd.melt(image_data, ['Epoch']),
                         markers=markers)
        if metric == 'psnr':
            g.set(yscale='log')
            ticks = [18, 20, 22, 24, 26, 28, 30]
            g.set_yticks(ticks)
            g.set_yticklabels(ticks)
        if title:
            plt.title(title)
            plt.ylabel(metric)

        plt.show()


if __name__ == '__main__':
    stats_mse = '/run/media/dvarkless/LinuxData/Files/Учеба/Data_Science_Course/SRGAN/statistics/models_full/Training_metrics_2022-11-09_mse.csv'
    stats_gan = '/run/media/dvarkless/LinuxData/Files/Учеба/Data_Science_Course/SRGAN/statistics/models_full/Training_metrics_2022-11-09_gan.csv'
    stats_full = '/run/media/dvarkless/LinuxData/Files/Учеба/Data_Science_Course/SRGAN/statistics/models_full/Training_metrics_2022-11-09_full.csv'
    stats = [stats_mse, stats_gan, stats_full]
    titles = ['MSE model', 'GAN model', 'Full model']
    for title, stat in zip(titles, stats):
        pltproc = PlotProcessor(stat)
        pltproc.show_image_score(
            metric='ssim', title=title, markers=True)
        pltproc.show_image_score(
            metric='psnr', title=title, markers=True)
        pltproc.show_gan_score(title=title, markers=True)
