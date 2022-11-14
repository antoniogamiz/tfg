import unittest

from any2vec.learning_rate import LearningRate


class MyTestCase(unittest.TestCase):
    def test_given_learning_rate_greater_than_1_when_building_it_error_is_thrown(self):
        invalid_learning_rate_value = 1.1

        with self.assertRaises(ValueError):
            LearningRate(invalid_learning_rate_value)

    def test_given_learning_rate_lower_than_1_when_building_it_error_is_thrown(self):
        invalid_learning_rate_value = -1.1

        with self.assertRaises(ValueError):
            LearningRate(invalid_learning_rate_value)


if __name__ == '__main__':
    unittest.main()
