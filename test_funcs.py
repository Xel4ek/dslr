import random
from unittest import TestCase
import funcs
import numpy as np


class Test(TestCase):

    def setUp(self) -> None:
        self.data = [random.random() * 100 for i in range(100)]
        self.npArray = np.array(self.data)

    def test__count(self):
        self.assertEqual(funcs._count(self.data), len(self.data))

    def test__std(self):
        self.assertEqual(funcs._std(self.data), self.npArray.std())

    def test__mean(self):
        self.assertEqual(funcs._mean(self.data), self.npArray.mean())

    def test__min(self):
        self.assertEqual(funcs._min(self.data), min(self.data))

    def test__max(self):
        self.assertEqual(funcs._max(self.data), max(self.data))

    def test__percentile(self):
        for i in [25, 50, 75]:
            print(i, funcs._percentile(self.data, i))
            self.assertEqual(funcs._percentile(self.data, i), np.percentile(self.data, i))
