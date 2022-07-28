import numpy as np
from model_base import BaseModel


class KnnModel(BaseModel):
    def __init__(self, custom_params=None) -> None:
        super().__init__(custom_params)
        self.assert_have(['k'])

    def fit(self, train_data):
        self.answer, self.train_data = self._splice_data(train_data)
        self.n = self.train_data.shape[0]

        return self

    def predict(self, new_points):
        self.data = new_points
        predictions = np.zeros(self.data.shape[0])
        # Loop over each new point to classify
        k = getattr(self, 'k')
        for i, point in enumerate(self.data):
            distances = self._calculate_distances(point)
            # Finds the k smallest distances and the corresponding points' labels
            label_neighbors = self.answer[np.argpartition(distances, k)[:k]]

            # Finds the majority of k nearest neighbors
            predictions[i] = np.bincount(
                label_neighbors.astype("int64")).argmax()

            if hasattr(self, '_tick'):
                self._tick()
        return predictions

    def _calculate_distances(self, new_point):
        # Expand by repeating to increase speed (avoid looping)
        new_point = np.resize(new_point, (self.n, new_point.shape[0]))
        # euclidean_distance = (self.trained_input - new_point)**2
        euclidean_distance = np.sum((self.train_data - new_point) ** 2, axis=1)
        return euclidean_distance
