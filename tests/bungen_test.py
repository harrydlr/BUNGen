# Run this test using:
# python -m unittest -v bungen_test.py
#
# Also you may test the coverage with:
# coverage run -m unittest bungen_test.py
# and then
# coverage report -m

import sys
import random
import numpy as np

import unittest

# sys.path.insert(1, './BUNGen-refactoring/')
from netgen import NetworkGenerator


class TestBungen(unittest.TestCase):

    # @unittest.skip("Skipping normal cases...")
    def test_normal_cases(self):
        """
        Test correct network generation.
        """
        for i in range(0, 50):
            with self.subTest(i=i):
                r, c = random.randint(20, 50), random.randint(20, 50)
                b = random.randint(1, 4)
                pe, mu = random.uniform(0, 1), random.uniform(0, 1)
                dmax = 1.0 / b
                d = random.uniform(0.05, dmax)

                gen = NetworkGenerator(
                    rows=r,
                    columns=c,
                    block_number=b,
                    p=pe,
                    mu=mu,
                    y_block_nodes_vec=[],
                    x_block_nodes_vec=[],
                    bipartite=True,
                    fixedConn=True,
                    link_density=d
                )

                M, Pij, crows, ccols = gen()

                # print(f"r = {r}\nc = {c}\nb = {b}\np = {pe}\nmu = {mu}\nd = {d}")
                # print(crows)
                # print(ccols)

                self.assertEqual(M.shape, (r, c))
                self.assertEqual(Pij.shape, (r, c))
                self.assertEqual(max(crows), b - 1)
                self.assertEqual(max(ccols), b - 1)
                self.assertEqual(len(crows), r)
                self.assertEqual(len(ccols), c)
                E = np.sum(M)
                dreal = E / (r * c)
                self.assertLessEqual(abs(dreal - d), 0.05, f"dexp={d:.4f}, d={dreal:.4f}, err={abs(dreal - d):.4f}")

    # @unittest.skip("Skipping corner cases...")
    def test_corner_cases(self):
        """
        Test what happens at boundary cases (i.e. strong eccentricity).
        """
        for i in range(0, 50):
            with self.subTest(i=i):
                r, c = random.randint(5, 20), random.randint(5, 20)
                b = random.randint(1, 5)
                pe, mu = random.uniform(0, 1), random.uniform(0, 1)
                dmax = 0.1
                d = random.uniform(0.01, dmax)

                gen = NetworkGenerator(
                    rows=r,
                    columns=c,
                    block_number=b,
                    p=pe,
                    mu=mu,
                    y_block_nodes_vec=[],
                    x_block_nodes_vec=[],
                    bipartite=True,
                    fixedConn=True,
                    link_density=d
                )

                M, Pij, crows, ccols = gen()

                # print(f"r = {r}\nc = {c}\nb = {b}\np = {pe}\nmu = {mu}\nd = {d}")
                # print(crows)
                # print(ccols)

                self.assertEqual(M.shape, (r, c))
                self.assertEqual(Pij.shape, (r, c))
                self.assertEqual(max(crows), b - 1)
                self.assertEqual(max(ccols), b - 1)
                self.assertEqual(len(crows), r)
                self.assertEqual(len(ccols), c)
                E = np.sum(M)
                dreal = E / (r * c)
                self.assertLessEqual(abs(dreal - d), 0.05, f"dexp={d:.4f}, d={dreal:.4f}, err={abs(dreal - d):.4f}")

    # @unittest.skip("Skipping negative cases...")
    def test_negative_cases(self):
        """
        Test incorrect network generation.
        """
        for i in range(0, 50):
            with self.subTest(i=i):
                r, c = random.randint(20, 50), random.randint(20, 50)
                b = random.randint(0, 1)
                # allow impossible p and/or mu values
                pe, mu = random.uniform(0.9, 1.1), random.uniform(-0.1, 0.1)
                dmax = 1.1
                d = random.uniform(-0.1, dmax)  # allow impossible density values

                gen = NetworkGenerator(
                    rows=r,
                    columns=c,
                    block_number=b,
                    p=pe,
                    mu=mu,
                    y_block_nodes_vec=[],
                    x_block_nodes_vec=[],
                    bipartite=True,
                    fixedConn=True,
                    link_density=d
                )

                M, Pij, crows, ccols = gen()

                # print(f"r = {r}\nc = {c}\nb = {b}\np = {pe}\nmu = {mu}\nd = {d}")
                # print(crows)
                # print(ccols)
                with self.assertRaises(ValueError):
                    pass
                with self.assertRaises(Exception):
                    pass
                # self.assertEqual(M.shape, (r,c))
                # self.assertEqual(Pij.shape, (r,c))
                # self.assertEqual(max(crows), b-1)
                # self.assertEqual(max(ccols), b-1)
                # self.assertEqual(len(crows), r)
                # self.assertEqual(len(ccols), c)
                # E = np.sum(M)
                # dreal = E/(r*c)
                # self.assertLessEqual(abs(dreal-d), 0.05, f"dexp={d:.4f}, d={dreal:.4f}, err={abs(dreal-d):.4f}")


if __name__ == '__main__':
    # Create a test suite and add test methods in the desired order
    suite = unittest.TestSuite()
    suite.addTest(TestExample('test_normal_cases'))
    suite.addTest(TestExample('test_corner_cases'))
    suite.addTest(TestExample('test_negative_cases'))

    # Run the test suite
    unittest.TextTestRunner().run(suite)
