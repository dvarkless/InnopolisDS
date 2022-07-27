from itertools import combinations

import numpy as np
from model_base import BaseModel
from model_runner import ModelRunner

from metrics import *
from image_feature_detector import *


class Question:
    def __init__(self, level, feature, question_val) -> None:
        self.level = level
        self.feature = feature
        self.question_val = question_val
        self.is_true_mask = np.array([])

    def __call__(self, row):
        if row[self.feature] > self.question_val:
            return self.right_branch(row)
        else:
            return self.left_branch(row)

    def __name__(self):
        return f'Question #{self.feature}>{self.question_val}? on level {self.level}'

    def __repr__(self):
        return f'{self.level}) Is var #{self.feature}>{self.question_val}? (gini={self.gini})'

    def split_by_question(self, data):
        is_true = data[:, self.feature] > self.question_val
        is_false = ~is_true

        return is_false@data, is_true@data

    def my_gini(self, data, min_samples=0):
        self.gini = DecisionTreeClassifier.get_gini(
            *self.split_by_question(data), min_samples)
        return self.gini

    def hold_branches(self, left_branch, right_branch):
        self.left_branch = left_branch
        self.right_branch = right_branch

        return self


class Leaf:
    def __init__(self, level, data) -> None:
        self.level = level
        self.prediction = data[:, -1].mean()

    def __call__(self, row):
        return self.prediction

    def __repr__(self):
        return f'{self.level}) return {self.prediction}'


class DecisionTreeClassifier(BaseModel):
    def __init__(self, custom_params=None) -> None:
        self.must_have_params = [
            'decrease_accuracy',
            'min_val',
            'max_val',
            'num_classes',
            'leap_size',
            'min_samples',
            'max_depth',
            'tree_type',
        ]

        super().__init__(custom_params)
        self.tree_root = []

    def fit(self, dataset):
        self.y, self.x = self._splice_data(dataset)
        if getattr(self, 'tree_type') in ('binary', 'multilabel'):
            self.y = self.y[:, np.newaxis]
        elif getattr(self, 'tree_type') == 'multilabel_ovo':
            self.y = self._ovo_split(self.y)
        elif getattr(self, 'tree_type') == 'multilabel_ovr':
            self.y = self.get_probabilities(self.y)
        else:
            raise AssertionError(f"Unknown tree type: {getattr(self, 'tree_type')}")

        if getattr(self, 'tree_type') in ('multilabel_ovo', 'multilabel_ovr'):
            num_classes = getattr(self, 'num_classes')
        else:
            num_classes = 1

        self.tree_root = []

        for i in range(num_classes):
            data = np.hstack((self.x, self.y[:, i][:, np.newaxis]))
            self.tree_root.append(self.build_tree(0, data))

    def build_tree(self, level, data):
        leap_size = getattr(self, 'leap_size')

        my_question = self.create_node(level, leap_size)

        if my_question.my_gini(data) == 0:
            return Leaf(level, data)
        if level == getattr(self, 'max_depth', 10):
            return Leaf(level, data)

        left, right = my_question.split_by_question(data)

        left_branch = self.build_tree(level+1, left)
        right_branch = self.build_tree(level+1, right)

        return my_question.hold_branches(left_branch, right_branch)

    def predict(self, input_data):
        num_classes = getattr(self, 'num_classes')
        out = np.zeros((input_data.shape[0], num_classes))
        for i, tree in enumerate(self.tree_root):
            out[:, i] = np.array(list(map(tree, input_data)))

        if getattr(self, 'tree_type') in ('binary', 'multilabel'):
            return out
        elif getattr(self, 'tree_type') == 'multilabel_ovo':
            return self._ovo_choose(out)
        elif getattr(self, 'tree_type') == 'multilabel_ovr':
            return self._ovr_choose(out)

    def _ovr_choose(self, x):
        """
        Возвращаем позицию наибольшего значения

        input:
            x - input array, 2 dims

        output - np.ndarray, 1 dim
        """
        return np.argmax(x, axis=1) + 1

    def _ovo_split(self, x):
        num_count = getattr(self, 'num_classes')
        pairs = combinations(range(num_count), 2)
        out = np.zeros((x.shape[0], len(list(pairs))))
        for i, vals in enumerate(pairs):
            val1, _ = vals
            out[:, i] = x[:, val1]

        return out

    def _ovo_choose(self, x):
        num_count = getattr(self, 'num_classes')
        out = np.zeros((x.shape[0], num_count))
        for i, vals in enumerate(combinations(range(num_count), 2)):
            val1, val2 = vals
            out[:, val1] += x[:, i]
            out[:, val2] -= x[:, i]

        return self._ovr_choose(x)

    def create_node(self, level, leap_size=1):
        feature_len = self.x.shape[1]

        min_val = getattr(self, 'min_val')
        max_val = getattr(self, 'max_val')

        decrease_accuracy = getattr(self, 'decrease_accuracy', 8)
        discrete_vals = list(range(min_val, max_val, decrease_accuracy))

        discrete_vals = np.array(discrete_vals)
        possibles_questions = np.ones(
            (feature_len, len(discrete_vals))) * discrete_vals
        curr_x = feature_len//2
        curr_y = max_val//decrease_accuracy//2

        for _ in range(25):
            low_x = (curr_x-leap_size) % feature_len if leap_size != -1 else 0
            high_x = (curr_x+leap_size) % feature_len if leap_size != - \
                1 else feature_len-1

            low_y = (curr_y-leap_size) % (max_val //
                                          decrease_accuracy) if leap_size != -1 else 0
            high_y = (curr_y+leap_size) % (max_val//decrease_accuracy) if leap_size != - \
                1 else (max_val//decrease_accuracy) - 1

            local_gini = np.vectorize(self.calculate_gini)(
                possibles_questions[low_x:high_x, low_y:high_y])

            max_ind = np.unravel_index(
                np.argmax(local_gini, axis=None), local_gini.shape)
            if leap_size == -1:
                break

            max_ind = max_ind[0] + curr_x + \
                leap_size, max_ind[1] + curr_y + leap_size
            if max_ind == (curr_x, curr_y):
                break
            else:
                curr_x, curr_y = max_ind

        return Question(level, curr_x, discrete_vals[curr_y])

    def calculate_gini(self, question_val):
        my_question = Question(None, None, question_val)

        left, right = my_question.split_by_question(self.x)
        return self.get_gini(left, right, getattr(self, 'min_samples', 0))

    @staticmethod
    def get_gini(left, right, min_samples=0):
        l_counts = np.unique(left[:, -1], return_counts=True)[1]
        r_counts = np.unique(right[:, -1], return_counts=True)[1]
        len_L = l_counts.sum()
        len_R = r_counts.sum()

        if len_L <= min_samples or len_R <= min_samples:
            return 0
        return np.sum(np.power(l_counts, 2))/len_L + np.sum(np.power(r_counts, 2))/len_R

    def print_tree(self, node, spacing=""):
        """World's most elegant tree printing function."""

        # Print the question at this node
        print (spacing + str(node))

        # Call this function recursively on the true branch
        print (spacing + '--> True:')
        self.print_tree(node.true_branch, spacing + "  ")

        # Call this function recursively on the false branch
        print (spacing + '--> False:')
        self.print_tree(node.false_branch, spacing + "  ")


if __name__ == "__main__":
    training_data = np.genfromtxt("datasets/light-train.csv",
                                delimiter=",", filling_values=0)
    evaluation_data = np.genfromtxt(
        "datasets/light-test.csv", delimiter=",", filling_values=0)

    evaluation_input = evaluation_data[:, 1:]
    evaluation_answers = evaluation_data[:, 0]

    my_metrics = [return_accuracy, return_recall, return_precision, return_f1]
    PCA_severe = PCA_transform(49).fit(training_data, answer_column=True) # x16
    hp = {
        'data_converter': PCA_severe,
        'num_classes': 26,
        'decrease_accuracy': 8,
        'min_val': 0,
        'max_val': 256,
        'leap_size': 2,
        'min_samples': 4,
        'max_depth': 10,
        'tree_type': 'multilabel_ovr',

    }


    TreeRunner= ModelRunner(DecisionTreeClassifier, defaults=hp, metrics=my_metrics)
    params_to_change = {
            'data_converter': [PCA_severe]
            }
    TreeRunner.run(training_data, evaluation_input, evaluation_answers, params_to_change, one_vs_one=False)
