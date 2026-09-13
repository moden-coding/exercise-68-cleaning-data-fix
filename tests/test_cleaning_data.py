#!/usr/bin/env python3

import unittest
import numpy as np
import pandas as pd

from src.cleaning_data import cleaning_data


class CleaningData(unittest.TestCase):

    def test_shape(self):
        df = cleaning_data()
        self.assertEqual(df.shape, (4,5), "Incorrect shape!")

    def test_columns(self):
        df = cleaning_data()
        np.testing.assert_array_equal(df.columns, ["President", "Start", "Last",
                                                   "Seasons", "Vice-president"],
                                      err_msg="Incorrect column names!")

    def test_dtypes(self):
        df = cleaning_data()
        np.testing.assert_array_equal(df.dtypes, [object, int,  float,  int, object],
                                      err_msg="Incorrect column types!")

    def test_start(self):
        df = cleaning_data()
        np.testing.assert_array_equal(df["Start"], [2017, 2009, 2001, 1993],
                                      err_msg="Incorrect values in Start column!")

    def test_last(self):
        df = cleaning_data()
        np.testing.assert_array_equal(df["Last"], [np.nan, 2017, 2009, 2001],
                                      err_msg="Incorrect values in Last column!")

    def test_seasons(self):
        df = cleaning_data()
        np.testing.assert_array_equal(df["Seasons"], [1, 2, 2, 2],
                                      err_msg="Incorrect values in Seasons column!")

    def test_president(self):
        df = cleaning_data()
        np.testing.assert_array_equal(df["President"],
                                      ["Donald Trump", "Barack Obama", "George Bush", "Bill Clinton"],
                                      err_msg="Incorrect values in President column!")

    def test_vice_president(self):
        df = cleaning_data()
        np.testing.assert_array_equal(df["Vice-president"],
                                      ["Mike Pence", "Joe Biden", "Dick Cheney", "Al Gore"],
                                      err_msg="Incorrect values in Vice-president column!")

if __name__ == '__main__':
    unittest.main()
