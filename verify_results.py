"""Independent automated checks for preprocessing, all real runs and summary tables."""
import common as c
import csv
import io
import json
import unittest
import numpy as np
from scipy.special import softmax, logsumexp
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, log_loss, confusion_matrix
from sklearn.preprocessing import StandardScaler
from prepare_data import read_source, stratified_group_indices, CLASS_COUNTS


class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = c.load_data()
        cls.x, cls.labels, _, _ = read_source(c.ROOT / 'Dry_Bean_Dataset.xlsx')

    def test_schema_and_real_class_counts(self):
        self.assertEqual(self.x.shape, (13611, 16))
        self.assertEqual(dict(zip(*np.unique(self.labels, return_counts=True))), CLASS_COUNTS)

    def test_complete_disjoint_indices(self):
        ids = [self.data['indices_'+p] for p in ('train', 'val', 'test')]
        np.testing.assert_array_equal(np.sort(np.concatenate(ids)), np.arange(len(self.x)))
        self.assertEqual([len(i) for i in ids], [9527, 2042, 2042])

    def test_duplicate_features_never_cross_splits(self):
        # Recompute groups directly from the original XLSX, independently of saved group IDs.
        _, groups = np.unique(self.x, axis=0, return_inverse=True)
        self.assertEqual(len(self.x)-len(np.unique(groups)), 68)
        sets = [set(groups[self.data['indices_'+p]]) for p in ('train', 'val', 'test')]
        for i in range(3):
            for j in range(i+1, 3):
                self.assertTrue(sets[i].isdisjoint(sets[j]))

    def test_scaler_fit_on_training_only(self):
        scaler = StandardScaler().fit(self.x[self.data['indices_train']])
        np.testing.assert_allclose(scaler.mean_, self.data['feature_mean'], atol=1e-12)
        np.testing.assert_allclose(scaler.scale_, self.data['feature_scale'], atol=1e-12)
        for part in ('train', 'val', 'test'):
            expected = scaler.transform(self.x[self.data['indices_'+part]]).astype(np.float32)
            np.testing.assert_array_equal(expected, self.data['x_'+part])

    def test_stratification_and_label_mapping(self):
        names = self.data['class_names']
        for part in ('train', 'val', 'test'):
            ids = self.data['indices_'+part]
            np.testing.assert_array_equal(names[self.data['y_'+part]], self.labels[ids])
            fractions = np.bincount(self.data['y_'+part], minlength=7) / len(ids)
            expected = np.array([CLASS_COUNTS[n] for n in names]) / 13611
            self.assertLess(np.max(np.abs(fractions-expected)), 1/len(ids))

    def test_split_reproducibility(self):
        y = np.searchsorted(self.data['class_names'], self.labels)
        indices, _, _ = stratified_group_indices(self.x, y)
        for part, ids in zip(('train', 'val', 'test'), indices):
            np.testing.assert_array_equal(ids, self.data['indices_'+part])

    def test_initial_weights_reproducible_and_seed_sensitive(self):
        hashes = [c.weights_hash(c.initial_weights(seed)) for seed in c.SEEDS]
        self.assertEqual(len(set(hashes)), 5)
        for seed, digest in zip(c.SEEDS, hashes):
            self.assertEqual(digest, c.weights_hash(c.initial_weights(seed)))
        self.assertEqual(sum(w.size+b.size for w, b in c.initial_weights(42)), 3383)

    def test_batch_order_complete_reproducible(self):
        for seed in c.SEEDS:
            arrays = list(c.batch_indices(9527, 0, seed))
            self.assertEqual(len(arrays), 38)
            self.assertEqual(len(arrays[-1]), 55)
            np.testing.assert_array_equal(np.sort(np.concatenate(arrays)), np.arange(9527))
            np.testing.assert_array_equal(np.concatenate(arrays), np.concatenate(list(c.batch_indices(9527, 0, seed))))
        self.assertEqual(len({c.batch_order_hash(9527, seed) for seed in c.SEEDS}), 5)

    def test_metrics_against_sklearn_including_absent_predictions(self):
        labels = np.array([0, 1, 2, 3, 4, 5, 6, 6], dtype=np.int64)
        logits = np.zeros((len(labels), 7), dtype=float)
        logits[:, 0] = 2.
        metrics, matrix = c.evaluate(logits, labels)
        predictions = logits.argmax(axis=1)
        p, r, f, _ = precision_recall_fscore_support(labels, predictions, labels=np.arange(7), average='macro', zero_division=0)
        for key, value in (('accuracy', accuracy_score(labels, predictions)), ('precision_macro', p),
                           ('recall_macro', r), ('f1_macro', f)):
            self.assertAlmostEqual(metrics[key], value, places=14)
        np.testing.assert_array_equal(matrix, confusion_matrix(labels, predictions, labels=np.arange(7)))

    def test_stable_cross_entropy_and_reject_nan(self):
        logits = np.array([[10000., 9999., -10000., 0., 0., 0., 0.]])
        expected = float(logsumexp(logits, axis=1)[0]-logits[0, 1])
        self.assertAlmostEqual(c.ce(logits, np.array([1])), expected, places=10)
        with self.assertRaises(FloatingPointError):
            c.ce(np.full((1, 7), np.nan), np.array([0]))
        with self.assertRaises(ValueError):
            c.ce(np.zeros((1, 7)), np.array([7]))


class CompletedRunTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = c.load_data()
        cls.rows = [c.validate_run(name, seed, cls.data) for name in c.FRAMEWORKS for seed in c.SEEDS]

    def test_twenty_completed_runs_and_thread_settings(self):
        self.assertEqual(len(self.rows), 20)
        for row in self.rows:
            self.assertEqual(row['runtime']['intra_op_threads'], 4)
            if 'inter_op_threads' in row['runtime']:
                self.assertEqual(row['runtime']['inter_op_threads'], 1)
            self.assertLess(row['initial_forward_max_abs_error'], 2e-5)

    def test_all_metrics_independently_recomputed(self):
        for row in self.rows:
            with self.subTest(framework=row['framework'], seed=row['seed']):
                with np.load(c.run_path(row['framework'], row['seed']) / 'predictions.npz') as f:
                    y, pred, logits = f['labels'], f['predicted'], f['logits']
                p, r, f1, _ = precision_recall_fscore_support(y, pred, labels=np.arange(7), average='macro', zero_division=0)
                for key, value in (('accuracy', accuracy_score(y, pred)), ('precision_macro', p),
                                   ('recall_macro', r), ('f1_macro', f1),
                                   ('test_cross_entropy', log_loss(y, softmax(logits.astype(float), axis=1), labels=np.arange(7)))):
                    self.assertAlmostEqual(row[key], value, places=9)
                cm = confusion_matrix(y, pred, labels=np.arange(7))
                np.testing.assert_array_equal(cm, row['confusion_matrix'])
                np.testing.assert_array_equal(cm.sum(axis=1), np.bincount(self.data['y_test'], minlength=7))

    def test_shared_initial_weights_and_batches_per_seed(self):
        for seed in c.SEEDS:
            rows = [r for r in self.rows if r['seed'] == seed]
            for key in ('initial_weights_sha256', 'batch_order_sha256', 'dataset_sha256'):
                self.assertEqual(len({r[key] for r in rows}), 1)

    def test_summary_means_and_sample_sd(self):
        with (c.OUTPUT_DIR / 'comparison_mean_sd.csv').open(encoding='utf-8-sig', newline='') as f:
            summary = list(csv.DictReader(f))
        self.assertEqual(len(summary), 4)
        for s in summary:
            rows = [r for r in self.rows if r['framework'] == s['framework']]
            self.assertEqual(int(s['runs']), 5)
            for field in c.METRIC_FIELDS:
                values = np.array([r[field] for r in rows])
                self.assertAlmostEqual(float(s[field+'_mean']), values.mean(), places=12)
                self.assertAlmostEqual(float(s[field+'_sd']), values.std(ddof=1), places=12)
        with (c.OUTPUT_DIR / 'all_runs.csv').open(encoding='utf-8-sig', newline='') as f:
            all_rows = list(csv.DictReader(f))
        self.assertEqual(len(all_rows), 20)
        for actual, row in zip(all_rows, self.rows):
            for field in c.METRIC_FIELDS:
                self.assertEqual(float(actual[field]), row[field])

    def test_required_graphs_exist(self):
        filenames = ['class_distribution.png', 'pca_classes.png', 'comparison_accuracy.png',
                     'comparison_f1_macro.png', 'comparison_train_seconds.png']
        for name in c.FRAMEWORKS:
            filenames += [f'{name}_learning_curves.png', f'{name}_confusion_matrix.png']
        from PIL import Image
        for name in filenames:
            with Image.open(c.ROOT / 'figures' / name) as im:
                self.assertGreaterEqual(im.width, 2000)
                im.verify()


def main():
    import sys
    suite = unittest.TestSuite()
    suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(CoreTests))
    if '--unit-only' not in sys.argv:
        suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(CompletedRunTests))
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    report = stream.getvalue()
    print(report)
    (c.ROOT / 'audit').mkdir(exist_ok=True)
    (c.ROOT / 'audit' / 'verification_tests.txt').write_text(report, encoding='utf-8')
    c.write_json(c.ROOT / 'audit' / 'verification_status.json',
                 dict(success=result.wasSuccessful(), tests_run=result.testsRun,
                      failures=len(result.failures), errors=len(result.errors), skipped=len(result.skipped),
                      scope='core only' if '--unit-only' in sys.argv else 'all 20 completed runs and aggregates'))
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == '__main__':
    main()
