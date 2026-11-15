import gc
import json
import os
import sys
import sysconfig
import tempfile
import threading
import time
import unittest

from rocksdict import (
    DbClosedError,
    Options,
    Rdict,
    SliceTransform,
    SstFileWriter,
    WriteBatch,
)

N_THREADS = 8
N_OPS = 2000
FREE_THREADED = bool(sysconfig.get_config_var("Py_GIL_DISABLED"))


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

    def read_prefix_extractors(self):
        with open(os.path.join(self.path, "rocksdict-config.json")) as f:
            return json.load(f)["prefix_extractors"]

    def testFailedCreateColumnFamilyKeepsConfig(self):
        options = Options()
        options.set_prefix_extractor(SliceTransform.create_fixed_prefix(1))
        self.db.create_column_family("cf", options).close()
        options = Options()
        options.set_prefix_extractor(SliceTransform.create_fixed_prefix(2))
        with self.assertRaises(Exception):
            self.db.create_column_family("cf", options)
        self.assertEqual(self.read_prefix_extractors()["cf"], {"Fixed": 1})

    def testCreateSameColumnFamilySavesWinner(self):
        db = self.db
        for r in range(20):
            created = []

            def work(i, name=f"cf{r}"):
                options = Options()
                options.set_prefix_extractor(SliceTransform.create_fixed_prefix(i + 1))
                try:
                    db.create_column_family(name, options).close()
                except Exception:
                    return
                created.append(i + 1)

            run_threads(work)
            self.assertEqual(len(created), 1)
            self.assertEqual(self.read_prefix_extractors()[f"cf{r}"], {"Fixed": created[0]})

    def testSharedIterator(self):
        db = self.db
        for j in range(N_OPS):
            db[j] = j
        it = db.items()
        seen = [[] for _ in range(N_THREADS)]

        def work(i):
            while True:
                try:
                    key, value = next(it)
                except StopIteration:
                    return
                except RuntimeError:  # another thread holds the iterator
                    continue
                self.assertEqual(key, value)
                seen[i].append(key)

        run_threads(work)
        keys = [key for s in seen for key in s]
        self.assertEqual(sorted(keys), list(range(N_OPS)))

    def testSharedWriteBatch(self):
        wb = WriteBatch()

        def work(i):
            for j in range(200):
                while True:
                    try:
                        wb[f"wb-{i}-{j}"] = j
                        break
                    except RuntimeError:  # another thread holds the batch
                        pass

        run_threads(work)
        self.db.write(wb)
        for i in range(N_THREADS):
            for j in range(200):
                self.assertEqual(self.db[f"wb-{i}-{j}"], j)

    def testSharedSnapshot(self):
        db = self.db
        for j in range(500):
            db[j] = j
        snapshot = db.snapshot()

        def work(i):
            for j in range(500):
                if i == 0:
                    db[j] = -j
                else:
                    self.assertEqual(snapshot[j], j)

        run_threads(work)
        self.assertEqual(db[499], -499)

    def testCloseWhileInUse(self):
        path = os.path.join(self.tmp.name, "closed")
        db = Rdict(path, Options())
        for j in range(100):
            db[j] = j
        started = threading.Event()

        def work(i):
            if i == 0:
                if not started.wait(60):
                    raise AssertionError("no worker started")
                while True:
                    try:
                        db.close()
                        return
                    except RuntimeError:  # another thread is using the Rdict
                        pass
            for j in range(N_OPS):
                try:
                    db[j % 100] = j
                    db.get(j % 100)
                    for _ in zip(range(5), db.keys()):
                        pass
                except (DbClosedError, RuntimeError):
                    return
                except Exception as e:
                    if "closed" not in str(e):
                        raise
                    return
                started.set()

        run_threads(work)
        with self.assertRaises(DbClosedError):
            db[0] = 0
        gc.collect()
        Rdict.destroy(path, Options())

    def testDropColumnFamilyWhileReading(self):
        db = self.db
        cf = db.create_column_family("cf", Options())
        for j in range(100):
            cf[j] = j
        started = threading.Event()

        def work(i):
            if i == 0:
                if not started.wait(60):
                    raise AssertionError("no worker started")
                db.drop_column_family("cf")
                return
            # a dropped column family stays readable through its open handle
            for _ in range(50):
                self.assertEqual(cf[i], i)
                self.assertEqual(len(list(cf.keys())), 100)
                started.set()

        run_threads(work)


class TestSharedSstFileWriter(unittest.TestCase):
    # open() on a FIFO blocks, holding the writer, until the FIFO has a reader.
    # Under the GIL that thread would keep the GIL too, so the test needs a free-threaded build.
    @unittest.skipUnless(FREE_THREADED and hasattr(os, "mkfifo"), "needs free threading and mkfifo")
    def testConcurrentUseRaisesNotCrashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            fifo = os.path.join(tmp, "fifo.sst")
            os.mkfifo(fifo)
            writer = SstFileWriter()

            def open_fifo():
                while True:
                    try:
                        writer.open(fifo)
                        return
                    except RuntimeError:  # the main thread is in file_size()
                        pass
                    except Exception:
                        return

            # a stop-the-world collection would wait for the thread blocked in open()
            gc.disable()
            opener = threading.Thread(target=open_fifo)
            opener.start()
            try:
                deadline = time.monotonic() + 10
                while time.monotonic() < deadline:
                    try:
                        writer.file_size()
                    except RuntimeError:
                        break
                else:
                    self.fail("file_size() never raised while open() was blocked")
                with self.assertRaises(RuntimeError):
                    writer.open(os.path.join(tmp, "b.sst"))
            finally:
                # the reader stays open until open() has returned
                reader = os.open(fifo, os.O_RDONLY | os.O_NONBLOCK)
                opener.join()
                os.close(reader)
                gc.enable()


if __name__ == "__main__":
    unittest.main()
