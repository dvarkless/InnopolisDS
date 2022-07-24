import cv2 as cv
import numpy as np
from image_feature_detector import get_features, get_plain_data
from model_base import BaseModel


class KnnModel(BaseModel):
    def __init__(self, custom_params=None) -> None:
        super().__init__(custom_params)
        self.k = getattr(self, 'k', 1)

    def fit(self, train_data):
        self.ans, self.train_data = self._splice_data(train_data)
        self.n = self.train_data.shape[0]

        return self

    def predict(self, new_points):
        self.data = new_points
        predictions = np.zeros(self.data.shape[0])
        # Loop over each new point to classify
        for i, point in enumerate(self.data):
            distances = self._calculate_distances(point)
            # Finds the k smallest distances and the corresponding points' labels
            label_neighbors = self.ans[np.argpartition(distances, self.k)[
                : self.k]]

            # Finds the majority of k nearest neighbors
            predictions[i] = np.bincount(
                label_neighbors.astype("int64")).argmax()
        return predictions

    def _calculate_distances(self, new_point):
        # Expand by repeating to increase speed (avoid looping)
        new_point = np.resize(new_point, (self.n, new_point.shape[0]))
        # euclidean_distance = (self.trained_input - new_point)**2
        euclidean_distance = np.sum((self.train_data - new_point) ** 2, axis=1)
        return euclidean_distance


if __name__ == "__main__":
    hp = {
        'data_converter': get_plain_data,
        'k': 3,
    }

    my_data = np.genfromtxt("datasets/light-train.csv",
                            delimiter=",", filling_values=0)
    my_knn = KnnModel(custom_params=hp).fit(my_data)

    eval_data = np.genfromtxt("datasets/light-test.csv",
                              delimiter=",", filling_values=0)
    predict = my_knn.predict(eval_data[:, 1:])
    print(predict)
