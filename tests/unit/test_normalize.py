import unittest
import pandas as pd

from data.normalize import NUMERIC, local_stats


class TestNormalize(unittest.TestCase):

    def test_local_stats_computes_counts_sums_and_squares(self):
        values = [1.0, 2.0, 3.0]
        data = {col: values for col in NUMERIC}
        df = pd.DataFrame(data)

        stats = local_stats(df)

        self.assertEqual(stats[NUMERIC[0]]["n"], 3)
        self.assertAlmostEqual(stats[NUMERIC[0]]["sum"], 6.0)
        self.assertAlmostEqual(stats[NUMERIC[0]]["sum_sq"], 14.0)

        for col in NUMERIC:
            self.assertEqual(stats[col]["n"], 3)
            self.assertAlmostEqual(stats[col]["sum"], 6.0)
            self.assertAlmostEqual(stats[col]["sum_sq"], 14.0)


if __name__ == "__main__":
    unittest.main()
