"""Regression checks for complete and partially appended evaluation CSV rows."""
from pathlib import Path
import tempfile
import unittest

from tools.swanlab_sync import read_rows


class ReadRowsTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.csv = Path(self.directory.name) / "eval.csv"
        self.header = "iteration,norm_return,n_failed,failed\n"

    def test_successful_evaluation_accepts_empty_failed_field(self):
        self.csv.write_text(self.header + "18000,0.9,0,\n")
        rows, cursor = read_rows(str(self.csv), 0)
        self.assertEqual(cursor, 1)
        self.assertEqual(rows[0]["failed"], "")

    def test_partial_append_is_reported_once_after_completion(self):
        self.csv.write_text(self.header + "16000,0.8,1,walk\n18000,0.9,0,")
        rows, cursor = read_rows(str(self.csv), 0)
        self.assertEqual([row["iteration"] for row in rows], ["16000"])
        with self.csv.open("a") as file:
            file.write("\n")
        rows, cursor = read_rows(str(self.csv), cursor)
        self.assertEqual([row["iteration"] for row in rows], ["18000"])
        self.assertEqual(read_rows(str(self.csv), cursor), ([], cursor))

    def test_empty_required_value_is_not_accepted(self):
        self.csv.write_text(self.header + "18000,,0,\n")
        self.assertEqual(read_rows(str(self.csv), 0), ([], 0))


if __name__ == "__main__":
    unittest.main()
