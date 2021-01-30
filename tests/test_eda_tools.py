import unittest
from eda_and_beyond import eda_tools

import pandas as pd

class TestEdaTools(unittest.TestCase):

    def test_intitial_eda_checks(self):
        d = {'col1': [1, 2, 2, 2], 'col2': [3, 4, float('NaN'), 5]}
        df = pd.DataFrame(data=d)
        results = eda_tools.intitial_eda_checks(df)
        expected = {'dup_found': 1,
                    'pct_nan_found' : 0.333333}
        self.assertEqual(results, expected)

    # def test_isupper(self):
    #     self.assertTrue('FOO'.isupper())
    #     self.assertFalse('Foo'.isupper())

    # def test_split(self):
    #     s = 'hello world'
    #     self.assertEqual(s.split(), ['hello', 'world'])
    #     # check that s.split fails when the separator is not a string
    #     with self.assertRaises(TypeError):
    #         s.split(2)

if __name__ == '__main__':
    unittest.main()