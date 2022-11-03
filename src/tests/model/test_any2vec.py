import unittest

import numpy as np

from model.any2vec import softmax, Any2Vec


class Any2VecTestCase(unittest.TestCase):
    def test_given_array_of_numbers_when_computing_softmax_function_then_it_sums_1(self):
        x = np.array([1, 2, 3])

        actual_distribution_value = softmax(x).sum()

        expected_distribution_value = 1
        self.assertAlmostEqual(expected_distribution_value, actual_distribution_value)

    def test_given_y_predicted_and_expected_when_computing_the_error_then_is_computed_correctly(self):
        y_predicted = np.array([9, 6, 5, 4, 2])
        y_expected = np.array([1, 0, 0, 1, 0])

        actual_error = Any2Vec.error(y_predicted, y_expected)

        expected_error = np.array([17, 12, 10, 7, 4])
        self.assertTrue((expected_error == actual_error).all())
