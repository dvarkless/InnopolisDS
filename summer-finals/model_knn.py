import numpy as np
import cv2 as cv
from model_base import BaseModel
from image_feature_detector import get_features, get_plain_data


class KnnModel(BaseModel):
    def __init__(self, k=3, data_converter=None) -> None:
        super().__init__(data_converter=data_converter)
        self.k = k

    def fit(self, train_data):
        self.ans, self.data = self._splice_data(train_data)
        self.n = self.data.shape[0]
        self.train_data = self.data
        print(f"data_shape {self.train_data.shape}")

    def predict(self, new_points):
        _, new_points = self._splice_data(new_points)
        predictions = np.zeros(new_points.shape[0])
        # Loop over each new point to classify
        for i, point in enumerate(new_points):
            print(point)
            distances = self._calculate_distances(point)
            # Finds the k smallest distances and the corresponding points' labels
            label_neighbors = self.ans[np.argpartition(distances, self.k)[: self.k]]

            # Finds the majority of k nearest neighbors
            predictions[i] = np.bincount(label_neighbors.astype("int64")).argmax()

        return predictions

    def _calculate_distances(self, new_point):
        # Expand by repeating to increase speed (avoid looping)
        new_point = np.resize(new_point, (self.n, new_point.shape[0]))
        print(f"new_point_shape {new_point.shape}")
        # euclidean_distance = (self.data - new_point)**2
        euclidean_distance = np.sum((self.train_data - new_point) ** 2, axis=1)
        print(euclidean_distance)
        print(f"euclid_dist_shape{euclidean_distance.shape}")
        return euclidean_distance


if __name__ == "__main__":
    my_knn = KnnModel(3, data_converter=get_features)
    my_data = np.genfromtxt("light-train.csv", delimiter=",")
    print(my_data)
    my_knn.fit(my_data)

    eval_data = np.genfromtxt("light-test.csv", delimiter=",")
    predict = my_knn.predict(eval_data)
    print(predict)
