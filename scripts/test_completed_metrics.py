"""Numerical and malformed-evidence checks; standard library only."""
import csv
from pathlib import Path
import tempfile
import unittest
from evaluate_saved_predictions import binary_auc_ap, evaluate


class MetricsTests(unittest.TestCase):
    def test_rankings_and_ties(self):
        self.assertEqual(binary_auc_ap([0, 1], [0.1, 0.9]), (1.0, 1.0))
        self.assertEqual(binary_auc_ap([0, 1], [0.9, 0.1]), (0.0, 0.5))
        self.assertEqual(binary_auc_ap([0, 1], [0.5, 0.5]), (0.5, 0.5))
        self.assertAlmostEqual(binary_auc_ap([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8])[0], 0.75)

    def test_absent_class_is_not_zero_auc(self):
        self.assertEqual(binary_auc_ap([0, 0], [0.1, 0.9]), (None, None))
        self.assertEqual(binary_auc_ap([1, 1], [0.1, 0.9]), (None, 1.0))

    def test_three_class_file_and_invalid_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'predictions.csv'
            header = ['node_id', 'y_true', 'y_pred', 'prob_normal', 'prob_phishing', 'prob_scamming']
            rows = [[1, 0, 0, .8, .1, .1], [2, 1, 1, .1, .8, .1], [3, 2, 2, .1, .1, .8]]
            def write(values):
                with path.open('w', newline='') as f:
                    w = csv.writer(f); w.writerow(header); w.writerows(values)
            write(rows)
            result = evaluate(path)
            for metric in ('accuracy', 'macro_precision', 'macro_recall', 'macro_f1', 'macro_roc_auc_ovr', 'mcc'):
                self.assertEqual(result[metric], 1.0)
            write(rows + [rows[0]])
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                evaluate(path)
            write([[1, 0, 0, .8, .8, .1]])
            with self.assertRaisesRegex(ValueError, 'sum'):
                evaluate(path)
            write([[1, 0, 2, .8, .1, .1]])
            with self.assertRaisesRegex(ValueError, 'argmax'):
                evaluate(path)
            write([[1, 0, 0, 'nan', .1, .1]])
            with self.assertRaisesRegex(ValueError, 'Invalid'):
                evaluate(path)


if __name__ == '__main__':
    unittest.main()
