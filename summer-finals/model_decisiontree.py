from itertools import combinations
from typing import Any

import numpy as np
from image_feature_detector import *
from metrics import *
from model_base import BaseModel
from model_runner import ModelRunner


class Question:
    def __init__(self, level: int, feature: int, question_val: float) -> None:
        self.level = level
        self.feature = feature
        self.question_val = question_val
        self.is_true_mask = np.array([])
        self.right_len = 0
        self.left_len = 0

    def __call__(self, row: np.ndarray):
        if row[self.feature] > self.question_val:
            return self.right_branch(row)
        else:
            return self.left_branch(row)

    def __name__(self):
        return f'Question "#{self.feature} > {self.question_val:.3f}?" on level {self.level}'

    def __repr__(self):
        return f'{self.level}) Is var #{self.feature} > {self.question_val:.3f}?\n \
                (G={self.gini:.2f})((L({self.left_len})->{self.left_mean:.3f})R({self.right_len})->{self.right_mean:.3f})'

    def split_by_question(self, data: np.ndarray):
        is_true = data[:, self.feature] > self.question_val
        is_false = ~is_true
        self.right_len = data[is_true].shape[0]
        self.right_mean = data[is_true][:, -1].mean()

        self.left_len = data[is_false].shape[0]
        self.left_mean = data[is_false][:, -1].mean()

        return data[is_false], data[is_true]

    def my_gini(self, data: np.ndarray, min_samples: int = 0, balance_coeff: float = 1):
        self.gini = DecisionTreeClassifier.get_gini(
            *self.split_by_question(data), min_samples, balance_coeff)
        return self.gini

    def hold_branches(self, left_branch, right_branch):
        self.left_branch = left_branch
        self.right_branch = right_branch

        return self


class Leaf:
    def __init__(self, level: int, data: np.ndarray) -> None:
        self.level = level
        self.prediction = data[:, -1].mean()

    def __call__(self, row: Any):
        return self.prediction

    def __repr__(self):
        return f'{self.level}) return {self.prediction:.3f}'


class DecisionTreeClassifier(BaseModel):
    def __init__(self, custom_params=None) -> None:
        self.must_have_params = [
            'sample_len',
            'num_classes',
            'window_size',
            'min_samples',
            'max_depth',
            'tree_type',
        ]

        super().__init__(custom_params)
        self.tree_root = []

        if getattr(self, 'balance_coeff', 1) <= 0:
            raise ValueError(
                f'Balance coefficient could not be negative or zero, got {getattr(self,"balance_coeff")}')

    def fit(self, dataset: np.ndarray):
        self.y, self.x = self._splice_data(dataset)
        if getattr(self, 'tree_type') in ('binary', 'multilabel'):
            self.y = self.y[:, np.newaxis]
        elif getattr(self, 'tree_type') == 'multilabel_ovo':
            self.y = self._ovo_split(self.y)
            if not hasattr(self, 'balance_coeff'):
                self.balance_coeff = getattr(self, 'num_classes')//2
        elif getattr(self, 'tree_type') == 'multilabel_ovr':
            self.y = self.get_probabilities(self.y)
            if not hasattr(self, 'balance_coeff'):
                self.balance_coeff = getattr(self, 'num_classes')
        else:
            raise AssertionError(
                f"Unknown tree type: '{getattr(self, 'tree_type')}'")

        self.tree_root = []
        self.question_list = self._return_ranges(
            self.x.min(), self.x.max(), self.x.mean(), getattr(self, 'sample_len'))

        for i in range(self.y.shape[1]):
            data = np.hstack((self.x, self.y[:, i][:, np.newaxis]))

            self.tree_root.append(self.build_tree(0, data))
            if hasattr(self, '_tick'):
                self._tick()

    def _return_ranges(self, min_val: float, max_val: float, mean_val: float, num: int):
        num = (num+1)//2
        right = np.geomspace(mean_val, max_val, num=num)
        if min_val < 0:
            left = -(np.geomspace(mean_val, -min_val +
                     mean_val*2, num=num) - mean_val*2)[::-1]
        else:
            left = np.geomspace(mean_val, min_val+mean_val/100, num=num)[::-1]

        return np.hstack((left, right[1:]))

    def build_tree(self, level: int, data: np.ndarray):
        window_size = getattr(self, 'window_size')
        # Округляем до ближайшего нечетного
        # Чтобы работало вычисление текущих координат
        window_size = round(window_size+0.5)-1

        my_question = self.create_node(data, level, window_size)

        if my_question.my_gini(data, getattr(self, 'min_samples'), getattr(self, 'balance_coeff', 1)) == 0:
            return Leaf(level, data)
        if level == getattr(self, 'max_depth', 10):
            return Leaf(level, data)

        left, right = my_question.split_by_question(data)

        left_branch = self.build_tree(level+1, left)
        right_branch = self.build_tree(level+1, right)

        return my_question.hold_branches(left_branch, right_branch)

    def predict(self, input_data: np.ndarray):
        self.data = input_data
        out = np.zeros((input_data.shape[0], self.y.shape[1]))
        for i, tree in enumerate(self.tree_root):
            out[:, i] = np.array(list(map(tree, self.data)))

        if getattr(self, 'tree_type') in ('binary', 'multilabel'):
            return out
        elif getattr(self, 'tree_type') == 'multilabel_ovo':
            return self._ovo_choose(out)
        elif getattr(self, 'tree_type') == 'multilabel_ovr':
            return self._ovr_choose(out)

    def _ovr_choose(self, x: np.ndarray):
        """
        Возвращаем позицию наибольшего значения

        input:
            x - input array, 2 dims

        output - np.ndarray, 1 dim
        """
        return np.argmax(x, axis=1) + 1

    def _ovo_split(self, x: np.ndarray):
        num_count = getattr(self, 'num_classes')
        pairs = list(combinations(range(1, num_count+1), 2))
        out = np.zeros((x.shape[0], len(pairs)))
        for i, vals in enumerate(pairs):
            val1, val2 = vals
            out[:, i] = -x
            out[:, i][out[:, i] == -val1] = 1
            out[:, i][out[:, i] == -val2] = -1
            out[:, i][out[:, i] < 0] = 0

        return out

    def _ovo_choose(self, x: np.ndarray):
        num_count = getattr(self, 'num_classes')
        out = np.zeros((x.shape[0], num_count))
        for i, vals in enumerate(combinations(range(num_count), 2)):
            val1, val2 = vals
            out[:, val1] += x[:, i]
            out[:, val2] -= x[:, i]

        return self._ovr_choose(out)

    def _array_window(self, arr, x, y, size):
        ''' Given a 2D-array, returns an nxn array whose "center" element is arr[x,y]'''
        arr = np.roll(np.roll(arr, shift=-x+1, axis=0), shift=-y+1, axis=1)
        return arr[:size, :size]

    def create_node(self, data: np.ndarray, level: int, window_size: int = -1):
        feature_len = self.x.shape[1]

        curr_x = feature_len//2
        curr_y = len(self.question_list)//2

        possible_questions = np.ones(
            (feature_len, len(self.question_list))) * self.question_list
        feature_array = np.ones(
            (feature_len, len(self.question_list))) * np.arange(feature_len)[:, np.newaxis]
        feature_array = feature_array.astype(np.int64)

        for _ in range(25):
            if window_size > 0:
                question_window = self._array_window(
                    possible_questions, curr_x, curr_y, window_size)
                feature_window = self._array_window(
                    feature_array, curr_x, curr_y, window_size)
            else:
                question_window = possible_questions
                feature_window = feature_array

            v_gini = np.vectorize(self.calculate_gini)
            v_gini.excluded.add(0)
            local_gini = v_gini(
                data, feature_window, question_window)
            # print(f'max local_gini = {local_gini.max()}')
            if local_gini.max() == 0:
                break

            max_ind = np.unravel_index(
                np.argmax(local_gini, axis=None), local_gini.shape)
            if window_size == -1:
                curr_x, curr_y = [int(val) for val in max_ind]
                break

            max_ind_global = (max_ind[0] + curr_x - window_size//2,
                              max_ind[1] + curr_y - window_size//2)
            curr_x, curr_y = [int(max(pos, 0)) for pos in max_ind_global]
            # print(f'max_global = {max_ind_global}')
            # print(f'current = {(curr_x, curr_y)}')
            if max_ind_global == (curr_x, curr_y):
                break

        return Question(level, curr_x, self.question_list[curr_y])

    def calculate_gini(self, data: np.ndarray, feature: int, question_val: float):
        my_question = Question(0, feature, question_val)

        left, right = my_question.split_by_question(data)
        return self.get_gini(left, right, getattr(self, 'min_samples', 0),
                             getattr(self, 'balance_coeff', 1))

    @staticmethod
    def get_gini(left: np.ndarray, right: np.ndarray, min_samples: float = 0, balance_coeff: float = 1):
        if balance_coeff >= 1:
            bal_true = balance_coeff
            bal_false = 1
        else:
            bal_true = 1
            bal_false = 1/balance_coeff

        l_index, l_counts = np.unique(left[:, -1], return_counts=True)
        r_index, r_counts = np.unique(right[:, -1], return_counts=True)

        len_L = l_counts.sum()
        len_R = r_counts.sum()

        l_counts[l_index!=0] *= bal_true
        l_counts[l_index==0] *= bal_false

        r_counts[r_index!=0] *= bal_true
        r_counts[r_index==0] *= bal_false

        if len_L <= min_samples or len_R <= min_samples:
            return 0
        return np.sum(np.power(l_counts, 2))/len_L + np.sum(np.power(r_counts, 2))/len_R

    def print_tree(self):
        print('========DecisionTreeModel=======')
        print(f'Parameters are: ')
        for name in self.must_have_params:
            val = getattr(self, name)
            print(f'{name} = {val}')
        print('===Tree structure:===')
        for i, branch in enumerate(self.tree_root):
            print(f'===================={i}=======================')
            self.__print_tree(branch)

    def __print_tree(self, node, spacing=""):
        """World's most elegant tree printing function."""

        # Print the question at this node
        print(spacing + str(node))

        if isinstance(node, Leaf):
            return
        # Call this function recursively on the true branch
        print(spacing + '--> True:')
        self.__print_tree(node.right_branch, spacing + "  ")

        # Call this function recursively on the false branch
        print(spacing + '--> False:')
        self.__print_tree(node.left_branch, spacing + "  ")


if __name__ == "__main__":
    training_data = np.genfromtxt("datasets/light-train.csv",
                                  delimiter=",", filling_values=0)
    evaluation_data = np.genfromtxt(
        "datasets/light-test.csv", delimiter=",", filling_values=0)

    evaluation_input = evaluation_data[:, 1:]
    evaluation_answers = evaluation_data[:, 0]

    # my_metrics = [return_accuracy, return_recall, return_precision, return_f1]
    my_metrics = [return_accuracy, predictions_mean, predictions_std]
    PCA_severe = PCA_transform(49).fit(
        training_data, answer_column=True)  # x16
    PCA_lossy = PCA_transform(12).fit(
        training_data, answer_column=True)  # x16

    hp = {
        'data_converter': PCA_severe,
        'sample_len': 16,
        'num_classes': 26,
        'window_size': -1,
        'min_samples': 4,
        'max_depth': 5,
        'tree_type': 'multilabel_ovo',

    }

    TreeRunner = ModelRunner(DecisionTreeClassifier,
                             defaults=hp, metrics=my_metrics, responsive_bar=True)
    params_to_change = {
        'data_converter': [PCA_lossy]
    }
    TreeRunner.run(training_data, evaluation_input,
                   evaluation_answers, params_to_change, one_vs_one=False)
    # TreeRunner.model.print_tree()
    input()
