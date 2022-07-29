import cv2 as cv
import numpy as np


def get_features(img):
    """
        Функция получает из изображения полезную визуальную информацию:
        количество и расположение углов, кругов и линий.

        Алгоритм нахождения углов - Harris Corner detector
        Кругов - Hough Circles
        Линий - Hough Lines

        Все настройки подобраны на получение максимально возможных
        деталей.

        input: img: np.ndarray, shape = (784,) from emnist-letters dataset

        output: img_features: np.ndarray shape = (13,) values in range(0, Inf)
    """
    img = np.resize(img, (28, 28)).astype(np.uint8).T
    corners = cv.goodFeaturesToTrack(
        img, 25, 0.05, 4, useHarrisDetector=True, k=-0.09)

    try:
        corners = np.int0(corners)
    except TypeError:
        corners = np.array([])
        corn_xm, corn_ym = 0, 0
        corn_xs, corn_ys = 0, 0
    else:
        corners.resize((corners.shape[0], 2))
        corn_xm, corn_ym = corners.mean(axis=0)
        corn_xs, corn_ys = corners.std(axis=0)
    num_corners = len(corners)

    circles = cv.HoughCircles(
        img, cv.HOUGH_GRADIENT, 1, 5, param1=20, param2=8, minRadius=5, maxRadius=20
    )
    try:
        num_circles = len(circles[0, :])
    except TypeError:
        num_circles = 0
        circ_m = 0
        circ_s = 0
    else:
        circ_m = np.array(circles[0][:, 0:1]).mean()
        circ_s = np.array(circles[0][:, 2]).mean()

    edges = cv.Canny(img, 20, 180, apertureSize=3)
    lines = cv.HoughLines(edges, 0.5, np.pi / 90, 10)

    try:
        num_lines = len(lines)
    except TypeError:
        num_lines = 0
        rho_m, rho_s = 0, 0
        theta_m, theta_s = 0, 0
    else:
        lines.resize((lines.shape[0], 2))
        rho, theta = lines.T
        rho_m = np.array(rho).mean()
        theta_m = np.array(theta).mean()
        rho_s = np.array(rho).std()
        theta_s = np.array(theta).std()

    return np.array(
        [
            num_corners,
            corn_xm,
            corn_ym,
            corn_xs,
            corn_ys,
            num_circles,
            circ_m,
            circ_s,
            num_lines,
            rho_m,
            rho_s,
            theta_m,
            theta_s,
        ]  # len() = 13
    )

def convert_to_emnist(img: np.ndarray):
    """
        Конвертирует изображение, чтобы оно соответствовало таковым из
        датасета emnist-letters
        input: np.ndarray - shape(N,)
        output: np.ndarray - shape(N,)
    """
    img = np.resize(img, (28, 28)).astype(np.uint8).T
    return get_plain_data(img)


def get_plain_data(img):
    """
        Функция выравнивает полученный массив и конвертирует его значения в 8 битные (0-255)

        input:
            img - input array to convert
        output:
            out - converted array (np.ndarray) (type: np.uint8)

    """
    return img.flatten().astype(np.uint8)


def threshold_mid(img):
    """
        Функция применяет пороговую функцию к каждому значению в массиве:
            >> threshold = 127
            >> pixel_value = 0 if pixel_value <= threshold else 255

        input:
            img - input array to convert
        output:
            out - converted array (np.ndarray) (type: np.uint8)

    """
    img = get_plain_data(img)
    img[img > 127] = 255
    img[img <= 127] = 0
    return img


def threshold_low(img):
    """
         Функция применяет пороговую функцию к каждому значению в массиве:
             >> threshold = 50
             >> pixel_value = 0 if pixel_value <= threshold else 255

         input:
             img - input array to convert
         output:
             out - converted array (np.ndarray) (type: np.uint8)

     """
    img = get_plain_data(img)
    img[img > 50] = 255
    img[img <= 50] = 0
    return img


def threshold_high(img):
    """
        Функция применяет пороговую функцию к каждому значению в массиве:
            >> threshold = 200
            >> pixel_value = 0 if pixel_value <= threshold else 255

        input:
            img - input array to convert
        output:
            out - converted array (np.ndarray) (type: np.uint8)

    """
    img = get_plain_data(img)
    img[img > 200] = 255
    img[img <= 200] = 0
    return img


class PCA_transform:
    """
        Применяет Метод Главных Компонент (PCA) к массиву данных для уменьшения его размерности

        Для задания вектора коэффициентов надо вызвать конструктор класса со значением
        желаемой выходной размерности и вызвать метод .fit(dataset) передавая в него
        тренировочный набор данных.

        Далее можно использовать экземпляр класса как функцию для конвертации отдельных точек датасета.
        ref: "https://stackoverflow.com/questions/58666635/implementing-pca-with-numpy"

        use case:
            >> transformer = PCA_transform(3) # we want our output dataset to have 3 dims
            >> transformer.fit(dataset) # dataset.shape = (N, k) N - sample number, k - dimensions
            >> # transformer = PCA_transform(3).fit(dataset) # same thing
            >> 
            >> converted_ds1 = np.vectorize(transformer)(dataset)
            >> converted_ds2 = np.vectorize(transformer)(test_data) # use it for validation dataset too!

    """

    def __init__(self, n_dims) -> None:
        self.n_dims = n_dims

    def __call__(self, img):
        img = get_plain_data(img)
        return img @ self.PCA_vector

    def __repr__(self):
        return f'<PCA_transform({self.n_dims});vector:{self.PCA_vector.shape}>'

    def fit(self, dataset, answer_column=True):
        """
            Подстройка конвертера под передаваемые данные.
            
            Расчитывает и запоминает трансформационный вектор
            на основе переданного набора данных.

            Обязательный для вызова метод.

            input:
                dataset: np.ndarray - input dataset
                answer_column: bool - True if there is a column with answer
                labels in the dataset

            output - PCA_transform instance
        """
        if answer_column:
            dataset = dataset[:, 1:]
        dataset = np.array(list(map(get_plain_data, dataset)))
        dataset_cov = np.cov(dataset.T)
        e_values, e_vectors = np.linalg.eigh(dataset_cov)

        e_ind_order = np.flip(e_values.argsort())
        e_values = e_values[e_ind_order]
        self.PCA_vector = e_vectors[:, e_ind_order]
        self.PCA_vector = self.PCA_vector[:, :self.n_dims]

        return self
