from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class PlotProcessor:
    """Средство вывода графиков сохраненных при обучении."""

    def __init__(self, csv_path) -> None:
        """csv_path - путь к сохраненным данным."""
        self.data = self._get_from_csv(csv_path)
        sns.set_style("darkgrid")

    def _get_from_csv(self, csv_path: str | Path) -> pd.DataFrame:
        csv_path = Path(csv_path)
        if not csv_path.exists():
            msg = f'Could not open file "{csv_path}"'
            raise FileNotFoundError(msg)
        return pd.read_csv(csv_path)

    def show_gan_score(self, markers=False, title=None):
        """
        Вывести на экран G_score vs D_score

        markers - показать маркеры на графика (настройка seaborn)

        title - название графика
        """
        gan_data = self.data.loc[:, ["Epoch", "Score_G", "Score_D"]]
        sns.lineplot(x='Epoch', y='value', hue='variable',
                     data=pd.melt(gan_data, ['Epoch']),
                     markers=markers)
        if title:
            plt.title(title)
            plt.ylabel('G_vs_D')
        plt.show()

    def show_image_score(self, metric='ssim', markers=False, title=None):
        """
        Вывести график потерь по изображению.

        metric - название метрики для вывода ("psnr", "ssim")

        markers - показать маркеры на графика (настройка seaborn)

        title - название графика
        """
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

    def get_last_vals(self) -> pd.DataFrame:
        """Получить последние значения из файла в формате pandas DataFrame"""
        return self.data.iloc[-1]
