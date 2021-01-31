import unittest
from eda_and_beyond import eda_tools

import pandas as pd
from pandas._testing import assert_frame_equal



class TestEdaTools(unittest.TestCase):

    def test_inspect_dupes(self):
        d_1 = {'col1': [1, 2, 3, 3], 'col2': [3, float('NaN'), 3, 3], 'col3': ['foo', 'bar', 'baz', 'baz']}
        df_1 = pd.DataFrame(data=d_1)
        d_2 = {'col1': [1, 2, 3], 'col2': [3, float('NaN'), 3], 'col3': ['foo', 'bar', 'baz']}
        df_2 = pd.DataFrame(data=d_2)

        results = eda_tools.inspect_dupes(df_1, True)
        expected = df_2
        assert_frame_equal(results, expected)
        

    def test_inpect_nans(self): 
        d_1 = {'col1': [1, 2, 3, 3], 'col2': [3, float('NaN'), 3, 3], 'col3': ['foo', 'bar', 'baz', 'baz']}
        df_1 = pd.DataFrame(data=d_1)

        results = eda_tools.inpect_nans(df_1)
        expected = 1
        self.assertEqual(results, expected)



if __name__ == '__main__':
    unittest.main()