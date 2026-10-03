import gc
import json
import os
import sys
import sysconfig
import tempfile
import threading
import unittest

from rocksdict import Options, Rdict, SliceTransform, SstFileWriter, WriteBatch

N_THREADS = 8
N_OPS = 2000


def run_threads(target, n=N_THREADS):
    """Run target(i) in n threads started together; re-raise the first error."""
    barrier = threading.Barrier(n)
    errors = []

    def run(i):
        barrier.wait()
        try:
            target(i)
        except BaseException as e:
            errors.append(e)

    threads = [threading.Thread(target=run, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        raise errors[0]


class TestGil(unittest.TestCase):
    @unittest.skipUnless(
        hasattr(sys, "_is_gil_enabled") and sysconfig.get_config_var("Py_GIL_DISABLED"),
        reason="not a free-threaded build",
    )
    def testImportKeepsGilDisabled(self):
        self.assertFalse(sys._is_gil_enabled())


class TestConcurrentRdict(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "db")
        self.db = Rdict(self.path, Options())

    def tearDown(self):
        # PyPy has no refcounting: collect iterators still holding the DB open
        gc.collect()
        self.db.close()
        Rdict.destroy(self.path, Options())
        self.tmp.cleanup()

    def testPutGetDelete(self):
        db = self.db

        def work(i):
            for j in range(N_OPS):
                db[f"own-{i}-{j}"] = j
                self.assertEqual(db[f"own-{i}-{j}"], j)
                if j % 2:
                    del db[f"own-{i}-{j}"]
                shared = f"shared-{j % 10}"
                db[shared] = (i, j)
                value = db.get(shared)
                self.assertTrue(value is None or len(value) == 2)
                if j % 3 == 0:
                    del db[shared]

        run_threads(work)
        for i in range(N_THREADS):
            for j in range(N_OPS):
                self.assertEqual(db.get(f"own-{i}-{j}"), None if j % 2 else j)

    def testIterateWhileWriting(self):
        db = self.db
        initial = {f"init-{j:05d}" for j in range(500)}
        for key in initial:
            db[key] = key

        def work(i):
            if i % 2:
                for j in range(N_OPS):
                    db[f"new-{i}-{j:05d}"] = j
            else:
                for _ in range(20):
                    keys = list(db.keys())
                    self.assertEqual(keys, sorted(keys))
                    self.assertTrue(initial <= set(keys))

        run_threads(work)

    def testSeparateWriteBatches(self):
        db = self.db

        def work(i):
            wb = WriteBatch()
            for j in range(N_OPS):
                wb[f"wb-{i}-{j}"] = j
            db.write(wb)

        run_threads(work)
        for i in range(N_THREADS):
            for j in range(N_OPS):
                self.assertEqual(db[f"wb-{i}-{j}"], j)

    def testCreateColumnFamiliesSavesConfig(self):
        db = self.db

        def work(i):
            options = Options()
            options.set_prefix_extractor(SliceTransform.create_max_len_prefix(i + 1))
            db.create_column_family(f"cf{i}", options).close()

        run_threads(work)
        with open(os.path.join(self.path, "rocksdict-config.json")) as f:
            config = json.load(f)
        self.assertEqual(len(config["prefix_extractors"]), N_THREADS)


class TestSharedSstFileWriter(unittest.TestCase):
    # A race between two `open` calls has no hook to block in, so it is not tested here.
    def testConcurrentUseRaisesNotCrashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            writer = SstFileWriter()
            writer.open(os.path.join(tmp, "a.sst"))
            entered = threading.Event()
            release = threading.Event()

            def blocking_dumps(value):
                entered.set()
                release.wait()
                return b"value"

            # the setter thread holds the writer while blocked in dumps
            writer.set_dumps(blocking_dumps)
            setter = threading.Thread(
                target=writer.__setitem__, args=("key", ["value"])
            )
            setter.start()
            entered.wait()
            try:
                with self.assertRaises(RuntimeError):
                    writer.open(os.path.join(tmp, "b.sst"))
                with self.assertRaises(RuntimeError):
                    writer.file_size()
            finally:
                release.set()
                setter.join()
            writer.finish()


if __name__ == "__main__":
    unittest.main()
