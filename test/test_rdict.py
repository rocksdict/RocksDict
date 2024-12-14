import unittest
from rocksdict import (
    AccessType,
    Rdict,
    Options,
    PlainTableFactoryOptions,
    SliceTransform,
    CuckooTableOptions,
    DbClosedError,
    WriteBatch,
    Checkpoint
)
from random import randint, random, getrandbits
import os
import gc
import sys
import platform
from json import loads, dumps


TEST_INT_RANGE_UPPER = 999999


def randbytes(n):
    """Generate n random bytes."""
    return getrandbits(n * 8).to_bytes(n, "little")


def compare_dicts(test_case: unittest.TestCase, ref_dict: dict, test_dict: Rdict):
    # assert that the values are the same
    test_case.assertEqual({k: v for k, v in test_dict.items()}, ref_dict)


class TestGetDel(unittest.TestCase):
    test_dict = None
    opt = None
    path = "./test_get_pul_del"

    @classmethod
    def setUpClass(cls) -> None:
        cls.opt = Options()
        cls.test_dict = Rdict(cls.path, cls.opt)
        cls.test_dict["a"] = "a"
        cls.test_dict[123] = 123

    @unittest.skipIf(platform.python_implementation() == "PyPy", reason="sys.getrefcount() not available in PyPy")
    def testGetItem(self):
        assert self.test_dict is not None
        self.assertEqual(self.test_dict["a"], "a")
        self.assertEqual(self.test_dict[123], 123)
        self.assertIsNone(self.test_dict.get("b"))
        self.assertIsNone(self.test_dict.get(250))
        self.assertEqual(self.test_dict.get("b", "b"), "b")
        self.assertEqual(self.test_dict.get(250, 1324123), 1324123)
        self.assertRaises(KeyError, lambda: self.test_dict["b"] if self.test_dict is not None else None)
        self.assertRaises(KeyError, lambda: self.test_dict[250] if self.test_dict is not None else None)

    def testDelItem(self):
        assert self.test_dict is not None
        # no exception raise when deleting non-existing key
        self.test_dict.__delitem__("b")
        self.test_dict.__delitem__(250)

    @classmethod
    def tearDownClass(cls):
        assert cls.test_dict is not None
        cls.test_dict.close()
        gc.collect()
        Rdict.destroy(cls.path, Options())


class TestGetDelCustomDumpsLoads(unittest.TestCase):
    test_dict = None
    opt = None
    path = "./test_get_pul_del_loads_dumps"

    @classmethod
    def setUpClass(cls) -> None:
        cls.opt = Options()
        cls.test_dict = Rdict(cls.path, cls.opt)
        cls.test_dict.set_loads(lambda x: loads(x.decode("utf-8")))
        cls.test_dict.set_dumps(lambda x: bytes(dumps(x), "utf-8"))
        cls.test_dict["a"] = "a"
        cls.test_dict[123] = 123
        cls.test_dict["ok"] = ["o", "k"]

    def testGetItem(self):
        assert self.test_dict is not None
        self.assertEqual(self.test_dict["a"], "a")
        self.assertEqual(self.test_dict[123], 123)
        self.assertEqual(self.test_dict["ok"], ["o", "k"])
        self.assertIsNone(self.test_dict.get("b"))
        self.assertIsNone(self.test_dict.get(250))
        self.assertEqual(self.test_dict.get("b", "b"), "b")
        self.assertEqual(self.test_dict.get(250, 1324123), 1324123)
        self.assertEqual(self.test_dict["ok"], ["o", "k"])
        self.assertRaises(KeyError, lambda: self.test_dict["b"] if self.test_dict is not None else None)
        self.assertRaises(KeyError, lambda: self.test_dict[250] if self.test_dict is not None else None)

    def testDelItem(self):
        assert self.test_dict is not None
        # no exception raise when deleting non-existing key
        self.test_dict.__delitem__("b")
        self.test_dict.__delitem__(250)

    @classmethod
    def tearDownClass(cls):
        assert cls.test_dict is not None
        cls.test_dict.close()
        gc.collect()
        Rdict.destroy(cls.path, Options())


class TestIterBytes(unittest.TestCase):
    test_dict = None
    ref_dict = None
    opt = None
    path = "./temp_iter_bytes"

    @classmethod
    def setUpClass(cls) -> None:
        cls.opt = Options()
        cpu_count = os.cpu_count()
        assert cpu_count is not None
        cls.opt.increase_parallelism(cpu_count)
        cls.test_dict = Rdict(cls.path, cls.opt)
        cls.ref_dict = dict()
        for i in range(100000):
            key = randbytes(10)
            value = randbytes(20)
            cls.test_dict[key] = value
            cls.ref_dict[key] = value
        keys_to_remove = list(
            set(randint(0, len(cls.ref_dict) - 1) for _ in range(50000))
        )
        keys = [k for k in cls.ref_dict.keys()]
        keys_to_remove = [keys[i] for i in keys_to_remove]
        for key in keys_to_remove:
            del cls.test_dict[key]
            del cls.ref_dict[key]

    def test_seek_forward_key(self):
        assert self.ref_dict is not None
        assert self.test_dict is not None
        key = randbytes(10)
        ref_list = [k for k in self.ref_dict.keys() if k >= key]
        ref_list.sort()
        self.assertEqual([k for k in self.test_dict.keys(from_key=key)], ref_list)

    def test_seek_backward_key(self):
        assert self.ref_dict is not None
        assert self.test_dict is not None
        key = randbytes(20)
        ref_list = [k for k in self.ref_dict.keys() if k <= key]
        ref_list.sort(reverse=True)
        self.assertEqual(
            [k for k in self.test_dict.keys(from_key=key, backwards=True)], ref_list
        )

    def test_may_exists(self):
        assert self.ref_dict is not None
        assert self.test_dict is not None
        for k, v in self.ref_dict.items():
            fetched = self.test_dict.key_may_exist(k, fetch=True)
            assert isinstance(fetched, tuple)
            may_exists, value = fetched
            self.assertTrue(may_exists)
            if value is not None:
                self.assertEqual(v, value)

    def test_seek_forward(self):
        assert self.ref_dict is not None
        assert self.test_dict is not None
        key = randbytes(20)
        self.assertEqual(
            {k: v for k, v in self.test_dict.items(from_key=key)},
            {k: v for k, v in self.ref_dict.items() if k >= key},
        )

    def test_seek_backward(self):
        key = randbytes(20)
        assert self.ref_dict is not None
        assert self.test_dict is not None
        self.assertEqual(
            {k: v for k, v in self.test_dict.items(from_key=key, backwards=True)},
            {k: v for k, v in self.ref_dict.items() if k <= key},
        )

    @classmethod
    def tearDownClass(cls):
        assert cls.test_dict is not None
        cls.test_dict.close()
        gc.collect()
        Rdict.destroy(cls.path, Options())


class TestIterInt(unittest.TestCase):
    test_dict = None
    ref_dict = None
    opt = None
    path = "./temp_iter_int"

    @classmethod
    def setUpClass(cls) -> None:
        cls.test_dict = Rdict(cls.path)
        cls.ref_dict = dict()
        for i in range(10000):
            key = randint(0, TEST_INT_RANGE_UPPER - 1)
            value = randint(0, TEST_INT_RANGE_UPPER - 1)
            cls.ref_dict[key] = value
            cls.test_dict[key] = value
        for i in range(5000):
            key = randint(0, TEST_INT_RANGE_UPPER - 1)
            if key in cls.ref_dict:
                del cls.ref_dict[key]
                del cls.test_dict[key]

    def test_seek_forward(self):
        assert self.ref_dict is not None
        assert self.test_dict is not None
        self.assertEqual(
            {k: v for k, v in self.test_dict.items()},
            {k: v for k, v in self.ref_dict.items()},
        )

    def test_seek_backward(self):
        assert self.ref_dict is not None
        assert self.test_dict is not None
        self.assertEqual(
            {k: v for k, v in self.test_dict.items(backwards=True)},
            {k: v for k, v in self.ref_dict.items()},
        )

    def test_seek_forward_key(self):
        assert self.ref_dict is not None
        assert self.test_dict is not None
        key = randint(0, TEST_INT_RANGE_UPPER - 1)
        ref_list = [k for k in self.ref_dict.keys() if k >= key]
        ref_list.sort()
        self.assertEqual([k for k in self.test_dict.keys(from_key=key)], ref_list)

    def test_seek_backward_key(self):
        assert self.ref_dict is not None
        assert self.test_dict is not None
        key = randint(0, TEST_INT_RANGE_UPPER - 1)
        ref_list = [k for k in self.ref_dict.keys() if k <= key]
        ref_list.sort(reverse=True)
        self.assertEqual(
            [k for k in self.test_dict.keys(from_key=key, backwards=True)], ref_list
        )

    @classmethod
    def tearDownClass(cls):
        assert cls.test_dict is not None
        cls.test_dict.close()
        gc.collect()
        Rdict.destroy(cls.path, Options())


class TestInt(unittest.TestCase):
    test_dict = None
    ref_dict = None
    opt = None
    path = "./temp_int"

    @classmethod
    def setUpClass(cls) -> None:
        cls.opt = Options()
        cls.opt.create_if_missing(True)
        cls.test_dict = Rdict(cls.path, cls.opt)
        cls.ref_dict = dict()

    def test_add_integer(self):
        assert self.ref_dict is not None
        assert self.test_dict is not None
        for i in range(10000):
            key = randint(0, TEST_INT_RANGE_UPPER - 1)
            value = randint(0, TEST_INT_RANGE_UPPER - 1)
            self.ref_dict[key] = value
            self.test_dict[key] = value

        compare_dicts(self, self.ref_dict, self.test_dict)

    def test_delete_integer(self):
        assert self.ref_dict is not None
        assert self.test_dict is not None
        for i in range(5000):
            key = randint(0, TEST_INT_RANGE_UPPER - 1)
            if key in self.ref_dict:
                del self.ref_dict[key]
                del self.test_dict[key]

        compare_dicts(self, self.ref_dict, self.test_dict)

    def test_delete_range(self):
        assert self.ref_dict is not None
        assert self.test_dict is not None
        to_delete = []
        for key in self.ref_dict:
            if key >= 99999:
                to_delete.append(key)
        for key in to_delete:
            del self.ref_dict[key]
        self.test_dict.delete_range(99999, 10000000)
        compare_dicts(self, self.ref_dict, self.test_dict)

    def test_reopen(self):
        assert self.ref_dict is not None
        assert self.test_dict is not None
        self.test_dict.close()

        self.assertRaises(DbClosedError, lambda: self.test_dict.get(1) if self.test_dict is not None else None)

        gc.collect()
        test_dict = Rdict(self.path, self.opt)
        compare_dicts(self, self.ref_dict, test_dict)

    def test_get_batch(self):
        assert self.ref_dict is not None
        assert self.test_dict is not None
        keys = list(self.ref_dict.keys())[:100]
        self.assertEqual(
            self.test_dict[keys + ["no such key"] * 3],
            [self.ref_dict[k] for k in keys] + [None] * 3,
        )

    @classmethod
    def tearDownClass(cls):
        assert cls.opt is not None
        gc.collect()
        Rdict.destroy(cls.path, cls.opt)


class TestBigInt(unittest.TestCase):
    test_dict = None
    opt = None
    path = "./temp_big_int"

    @classmethod
    def setUpClass(cls) -> None:
        cls.opt = Options()
        cls.opt.create_if_missing(True)
        cls.opt.set_plain_table_factory(PlainTableFactoryOptions())
        cls.opt.set_prefix_extractor(SliceTransform.create_max_len_prefix(8))
        cls.test_dict = Rdict(cls.path, cls.opt)

    def test_big_int(self):
        assert self.test_dict is not None
        key = 13456436145354564353464754615223435465543
        value = 3456321456543245657643253647543212425364342343564
        self.test_dict[key] = value
        self.assertTrue(key in self.test_dict)
        self.assertEqual(self.test_dict[key], value)
        self.test_dict[key] = True
        self.assertTrue(self.test_dict[key])
        self.test_dict[key] = False
        self.assertFalse(self.test_dict[key])
        del self.test_dict[key]
        self.assertFalse(key in self.test_dict)

    @classmethod
    def tearDownClass(cls):
        assert cls.test_dict is not None
        cls.test_dict.close()
        assert cls.opt is not None
        gc.collect()
        Rdict.destroy(cls.path, cls.opt)


class TestFloat(unittest.TestCase):
    test_dict = None
    ref_dict = None
    opt = None
    path = "./temp_float"

    @classmethod
    def setUpClass(cls) -> None:
        cls.opt = Options()
        cls.opt.create_if_missing(True)
        cls.test_dict = Rdict(cls.path, cls.opt)
        cls.ref_dict = dict()

    def test_add_float(self):
        assert self.test_dict is not None
        assert self.ref_dict is not None
        for i in range(10000):
            key = random()
            value = random()
            self.ref_dict[key] = value
            self.test_dict[key] = value

        compare_dicts(self, self.ref_dict, self.test_dict)

    def test_delete_float(self):
        assert self.test_dict is not None
        assert self.ref_dict is not None
        for i in range(5000):
            keys = [k for k in self.ref_dict.keys()]
            key = keys[randint(0, len(self.ref_dict) - 1)]
            del self.ref_dict[key]
            del self.test_dict[key]

        compare_dicts(self, self.ref_dict, self.test_dict)

    def test_reopen(self):
        assert self.test_dict is not None
        assert self.ref_dict is not None
        self.test_dict.close()
        gc.collect()
        test_dict = Rdict(self.path, self.opt)
        compare_dicts(self, self.ref_dict, test_dict)

    def test_get_batch(self):
        assert self.test_dict is not None
        assert self.ref_dict is not None
        keys = list(self.ref_dict.keys())[:100]
        self.assertEqual(
            self.test_dict[keys + ["no such key"] * 3],
            [self.ref_dict[k] for k in keys] + [None] * 3,
        )

    @classmethod
    def tearDownClass(cls):
        assert cls.opt is not None
        gc.collect()
        Rdict.destroy(cls.path, cls.opt)


class TestBytes(unittest.TestCase):
    test_dict = None
    ref_dict = None
    opt = None
    path = "./temp_bytes"

    @classmethod
    def setUpClass(cls) -> None:
        cls.opt = Options()
        cls.opt.create_if_missing(True)
        # for the moment do not use CuckooTable on windows
        if not sys.platform.startswith("win"):
            cls.opt.set_cuckoo_table_factory(CuckooTableOptions())
            cls.opt.set_allow_mmap_reads(True)
            cls.opt.set_allow_mmap_writes(True)
        cls.test_dict = Rdict(cls.path, cls.opt)
        cls.ref_dict = dict()

    @unittest.skipIf(platform.python_implementation() == "PyPy", reason="sys.getrefcount() not available in PyPy")
    def test_add_bytes(self):
        assert self.test_dict is not None
        assert self.ref_dict is not None
        for i in range(10000):
            key = randbytes(10)
            value = randbytes(20)
            self.assertEqual(sys.getrefcount(key), 2)
            self.assertEqual(sys.getrefcount(value), 2)
            self.test_dict[key] = value
            # rdict does not increase ref_count
            self.assertEqual(sys.getrefcount(key), 2)
            self.assertEqual(sys.getrefcount(value), 2)
            self.ref_dict[key] = value
            self.assertEqual(sys.getrefcount(key), 3)
            self.assertEqual(sys.getrefcount(value), 3)

        compare_dicts(self, self.ref_dict, self.test_dict)

    @unittest.skipIf(platform.python_implementation() == "PyPy", reason="sys.getrefcount() not available in PyPy")
    def test_delete_bytes(self):
        assert self.test_dict is not None
        assert self.ref_dict is not None
        for i in range(5000):
            keys = [k for k in self.ref_dict.keys()]
            key = keys[randint(0, len(self.ref_dict) - 1)]
            # key + ref_dict + keys + sys.getrefcount -> 4
            self.assertEqual(sys.getrefcount(key), 4)
            del self.test_dict[key]
            self.assertEqual(sys.getrefcount(key), 4)
            del self.ref_dict[key]
            self.assertEqual(sys.getrefcount(key), 3)

        compare_dicts(self, self.ref_dict, self.test_dict)

    def test_reopen(self):
        assert self.test_dict is not None
        assert self.ref_dict is not None
        self.test_dict.close()
        test_dict = Rdict(self.path, self.opt)
        compare_dicts(self, self.ref_dict, test_dict)

    def test_get_batch(self):
        assert self.test_dict is not None
        assert self.ref_dict is not None
        keys = list(self.ref_dict.keys())[:100]
        self.assertEqual(
            self.test_dict[keys + ["no such key"] * 3],
            [self.ref_dict[k] for k in keys] + [None] * 3,
        )

    @classmethod
    def tearDownClass(cls):
        assert cls.opt is not None
        gc.collect()
        Rdict.destroy(cls.path, cls.opt)


class TestString(unittest.TestCase):
    test_dict = None
    opt = None
    path = "./temp_string"

    @classmethod
    def setUpClass(cls) -> None:
        cls.opt = Options()
        cls.opt.create_if_missing(True)
        cls.test_dict = Rdict(cls.path, cls.opt)

    def test_string(self):
        assert self.test_dict is not None
        self.test_dict["Guangdong"] = "Shenzhen"
        self.test_dict["Sichuan"] = "Changsha"
        # overwrite
        self.test_dict["Sichuan"] = "Chengdu"
        self.test_dict["Beijing"] = "Beijing"
        del self.test_dict["Beijing"]

        # assertions
        self.assertFalse("Beijing" in self.test_dict)
        self.assertEqual(self.test_dict["Sichuan"], "Chengdu")
        self.assertEqual(self.test_dict["Guangdong"], "Shenzhen")

    @classmethod
    def tearDownClass(cls):
        assert cls.test_dict is not None
        cls.test_dict.close()
        assert cls.opt is not None
        gc.collect()
        Rdict.destroy(cls.path, cls.opt)


class TestWideColumnsRaw(unittest.TestCase):
    test_dict = None
    opt = None
    path = "./temp_wide_columns_raw"

    @classmethod
    def setUpClass(cls) -> None:
        cls.opt = Options(True)
        cls.opt.create_if_missing(True)
        cls.test_dict = Rdict(cls.path, cls.opt)

    def test_put_wide_columns(self):
        assert self.test_dict is not None
        self.test_dict.put_entity(key=b"Guangdong", names=[b"language", b"city"], values=[b"Cantonese", b"Shenzhen"])
        self.test_dict.put_entity(key=b"Sichuan", names=[b"language", b"city"], values=[b"Mandarin", b"Chengdu"])
        self.assertEqual(self.test_dict[b"Guangdong"], b"")
        self.assertEqual(self.test_dict.get_entity(b"Guangdong"), [(b"city", b"Shenzhen"), (b"language", b"Cantonese")])
        self.assertEqual(self.test_dict[b"Sichuan"], b"")
        self.assertEqual(self.test_dict.get_entity(b"Sichuan"), [(b"city", b"Chengdu"), (b"language", b"Mandarin")])
        # overwrite
        self.test_dict.put_entity(key=b"Sichuan", names=[b"language", b"city"], values=[b"Sichuanhua", b"Chengdu"])
        self.test_dict[b"Beijing"] = b"Beijing"

        # assertions
        self.assertEqual(self.test_dict[b"Beijing"], b"Beijing")
        self.assertEqual(self.test_dict.get_entity(b"Beijing"), [(b"", b"Beijing")])
        self.assertEqual(self.test_dict[b"Guangdong"], b"")
        self.assertEqual(self.test_dict.get_entity(b"Guangdong"), [(b"city", b"Shenzhen"), (b"language", b"Cantonese")])
        self.assertEqual(self.test_dict[b"Sichuan"], b"")
        self.assertEqual(self.test_dict.get_entity(b"Sichuan"), [(b"city", b"Chengdu"), (b"language", b"Sichuanhua")])

        # iterator
        it = self.test_dict.iter()
        it.seek_to_first()
        self.assertTrue(it.valid())
        self.assertEqual(it.key(), b"Beijing")
        self.assertEqual(it.value(), b"Beijing")
        self.assertEqual(it.columns(), [(b"", b"Beijing")])
        it.next()
        self.assertTrue(it.valid())
        self.assertEqual(it.key(), b"Guangdong")
        self.assertEqual(it.value(), b"")
        self.assertEqual(it.columns(), [(b"city", b"Shenzhen"), (b"language", b"Cantonese")])
        it.next()
        self.assertTrue(it.valid())
        self.assertEqual(it.key(), b"Sichuan")
        self.assertEqual(it.value(), b"")
        self.assertEqual(it.columns(), [(b"city", b"Chengdu"), (b"language", b"Sichuanhua")])

        # iterators
        expected = [
            (b"Beijing", [(b"", b"Beijing")]),
            (b"Guangdong", [(b"city", b"Shenzhen"), (b"language", b"Cantonese")]),
            (b"Sichuan", [(b"city", b"Chengdu"), (b"language", b"Sichuanhua")]),
        ]
        for i, (key, entity) in enumerate(self.test_dict.entities()):
            self.assertEqual(key, expected[i][0])
            self.assertEqual(entity, expected[i][1])

        self.assertEqual(
            [c for c in self.test_dict.columns()],
            [
                [(b"", b"Beijing")],
                [(b"city", b"Shenzhen"), (b"language", b"Cantonese")],
                [(b"city", b"Chengdu"), (b"language", b"Sichuanhua")],
            ]
        )

    @classmethod
    def tearDownClass(cls):
        assert cls.test_dict is not None
        cls.test_dict.close()
        assert cls.opt is not None
        gc.collect()
        Rdict.destroy(cls.path, cls.opt)


class TestWriteBatchWideColumnsRaw(unittest.TestCase):
    test_dict = None
    opt = None
    path = "./temp_write_batch_wide_columns_raw"

    @classmethod
    def setUpClass(cls) -> None:
        cls.opt = Options(True)
        cls.opt.create_if_missing(True)
        cls.test_dict = Rdict(cls.path, cls.opt)

    def test_put_wide_columns(self):
        assert self.test_dict is not None
        write_batch = WriteBatch(raw_mode=True)
        default_cf_handle = self.test_dict.get_column_family_handle("default")
        write_batch.set_default_column_family(default_cf_handle)
        write_batch.put_entity(key=b"Guangdong", names=[b"language", b"city"], values=[b"Cantonese", b"Shenzhen"])
        write_batch.put_entity(key=b"Sichuan", names=[b"language", b"city"], values=[b"Mandarin", b"Chengdu"])
        # overwrite
        write_batch.put_entity(key=b"Sichuan", names=[b"language", b"city"], values=[b"Sichuanhua", b"Chengdu"])
        write_batch[b"Beijing"] = b"Beijing"

        self.test_dict.write(write_batch)

        # assertions
        self.assertEqual(self.test_dict[b"Beijing"], b"Beijing")
        self.assertEqual(self.test_dict.get_entity(b"Beijing"), [(b"", b"Beijing")])
        self.assertEqual(self.test_dict[b"Guangdong"], b"")
        self.assertEqual(self.test_dict.get_entity(b"Guangdong"), [(b"city", b"Shenzhen"), (b"language", b"Cantonese")])
        self.assertEqual(self.test_dict[b"Sichuan"], b"")
        self.assertEqual(self.test_dict.get_entity(b"Sichuan"), [(b"city", b"Chengdu"), (b"language", b"Sichuanhua")])

        # iterator
        it = self.test_dict.iter()
        it.seek_to_first()
        self.assertTrue(it.valid())
        self.assertEqual(it.key(), b"Beijing")
        self.assertEqual(it.value(), b"Beijing")
        self.assertEqual(it.columns(), [(b"", b"Beijing")])
        it.next()
        self.assertTrue(it.valid())
        self.assertEqual(it.key(), b"Guangdong")
        self.assertEqual(it.value(), b"")
        self.assertEqual(it.columns(), [(b"city", b"Shenzhen"), (b"language", b"Cantonese")])
        it.next()
        self.assertTrue(it.valid())
        self.assertEqual(it.key(), b"Sichuan")
        self.assertEqual(it.value(), b"")
        self.assertEqual(it.columns(), [(b"city", b"Chengdu"), (b"language", b"Sichuanhua")])

        # iterators
        expected = [
            (b"Beijing", [(b"", b"Beijing")]),
            (b"Guangdong", [(b"city", b"Shenzhen"), (b"language", b"Cantonese")]),
            (b"Sichuan", [(b"city", b"Chengdu"), (b"language", b"Sichuanhua")]),
        ]
        for i, (key, entity) in enumerate(self.test_dict.entities()):
            self.assertEqual(key, expected[i][0])
            self.assertEqual(entity, expected[i][1])

        self.assertEqual(
            [c for c in self.test_dict.columns()],
            [
                [(b"", b"Beijing")],
                [(b"city", b"Shenzhen"), (b"language", b"Cantonese")],
                [(b"city", b"Chengdu"), (b"language", b"Sichuanhua")],
            ]
        )

        del write_batch

    @classmethod
    def tearDownClass(cls):
        assert cls.test_dict is not None
        cls.test_dict.close()
        assert cls.opt is not None
        gc.collect()
        Rdict.destroy(cls.path, cls.opt)


class TestWideColumns(unittest.TestCase):
    test_dict = None
    opt = None
    path = "./temp_wide_columns"

    @classmethod
    def setUpClass(cls) -> None:
        cls.opt = Options()
        cls.opt.create_if_missing(True)
        cls.test_dict = Rdict(cls.path, cls.opt)

    def test_put_wide_columns(self):
        assert self.test_dict is not None
        self.test_dict.put_entity(key="Guangdong", names=["language", "city", "population"], values=["Cantonese", "Shenzhen", 1.27])
        self.test_dict.put_entity(key="Sichuan", names=["language", "city"], values=["Mandarin", "Chengdu"])
        self.assertEqual(self.test_dict["Guangdong"], "")
        self.assertEqual(self.test_dict.get_entity("Guangdong"), [("city", "Shenzhen"), ("language", "Cantonese"), ("population", 1.27)])
        self.assertEqual(self.test_dict["Sichuan"], "")
        self.assertEqual(self.test_dict.get_entity("Sichuan"), [("city", "Chengdu"), ("language", "Mandarin")])
        # overwrite
        self.test_dict.put_entity(key="Sichuan", names=["language", "city"], values=["Sichuanhua", "Chengdu"])
        self.test_dict["Beijing"] = "Beijing"

        # assertions
        self.assertEqual(self.test_dict["Beijing"], "Beijing")
        self.assertEqual(self.test_dict.get_entity("Beijing"), [("", "Beijing")])
        self.assertEqual(self.test_dict["Guangdong"], "")
        self.assertEqual(self.test_dict.get_entity("Guangdong"), [("city", "Shenzhen"), ("language", "Cantonese"), ("population", 1.27)])
        self.assertEqual(self.test_dict["Sichuan"], "")
        self.assertEqual(self.test_dict.get_entity("Sichuan"), [("city", "Chengdu"), ("language", "Sichuanhua")])

        it = self.test_dict.iter()
        it.seek_to_first()
        self.assertTrue(it.valid())
        self.assertEqual(it.key(), "Beijing")
        self.assertEqual(it.value(), "Beijing")
        self.assertEqual(it.columns(), [("", "Beijing")])
        it.next()
        self.assertTrue(it.valid())
        self.assertEqual(it.key(), "Guangdong")
        self.assertEqual(it.value(), "")
        self.assertEqual(it.columns(), [("city", "Shenzhen"), ("language", "Cantonese"), ("population", 1.27)])
        it.next()
        self.assertTrue(it.valid())
        self.assertEqual(it.key(), "Sichuan")
        self.assertEqual(it.value(), "")
        self.assertEqual(it.columns(), [("city", "Chengdu"), ("language", "Sichuanhua")])

        # iterators
        expected = [
            ("Beijing", [("", "Beijing")]),
            ("Guangdong", [("city", "Shenzhen"), ("language", "Cantonese"), ("population", 1.27)]),
            ("Sichuan", [("city", "Chengdu"), ("language", "Sichuanhua")]),
        ]
        for i, (key, entity) in enumerate(self.test_dict.entities()):
            self.assertEqual(key, expected[i][0])
            self.assertEqual(entity, expected[i][1])

        self.assertEqual(
            [c for c in self.test_dict.columns()],
            [
                [("", "Beijing")],
                [("city", "Shenzhen"), ("language", "Cantonese"), ("population", 1.27)],
                [("city", "Chengdu"), ("language", "Sichuanhua")],
            ]
        )

    @classmethod
    def tearDownClass(cls):
        assert cls.test_dict is not None
        cls.test_dict.close()
        assert cls.opt is not None
        gc.collect()
        Rdict.destroy(cls.path, cls.opt)


class TestColumnFamiliesDefaultOpts(unittest.TestCase):
    test_dict = None
    path = "./column_families"

    @classmethod
    def setUpClass(cls) -> None:
        cls.test_dict = Rdict(cls.path)

    def test_column_families(self):
        assert self.test_dict is not None
        ds = self.test_dict.create_column_family(name="string")
        di = self.test_dict.create_column_family(name="integer")

        for i in range(1000):
            di[i] = i * i
            ds[str(i)] = str(i * i)

        self.test_dict["ok"] = True

        ds.close()
        di.close()
        self.test_dict.close()

        # reopen
        self.test_dict = Rdict(self.path)
        ds = self.test_dict.get_column_family("string")
        di = self.test_dict.get_column_family("integer")
        assert self.test_dict["ok"]
        compare_dicts(self, {i: i**2 for i in range(1000)}, di)
        compare_dicts(self, {str(i): str(i**2) for i in range(1000)}, ds)
        ds.close()
        di.close()
        self.test_dict.close()

    @classmethod
    def tearDownClass(cls):
        gc.collect()
        Rdict.destroy(cls.path)


class TestColumnFamiliesDefaultOptsCreate(unittest.TestCase):
    cfs = None
    test_dict = None
    path = "./column_families_create"

    @classmethod
    def setUpClass(cls) -> None:
        cls.cfs = {"string": Options(), "integer": Options()}
        opt = Options()
        opt.create_missing_column_families(True)
        cls.test_dict = Rdict(cls.path, options=opt, column_families=cls.cfs)

    def test_column_families_create(self):
        assert self.test_dict is not None
        ds = self.test_dict.get_column_family(name="string")
        di = self.test_dict.get_column_family(name="integer")

        for i in range(1000):
            di[i] = i * i
            ds[str(i)] = str(i * i)

        self.test_dict["ok"] = True

        ds.close()
        di.close()
        self.test_dict.close()

        # reopen
        self.test_dict = Rdict(self.path)
        ds = self.test_dict.get_column_family("string")
        di = self.test_dict.get_column_family("integer")
        assert self.test_dict["ok"]
        compare_dicts(self, {i: i**2 for i in range(1000)}, di)
        compare_dicts(self, {str(i): str(i**2) for i in range(1000)}, ds)
        ds.close()
        di.close()
        self.test_dict.close()

    @classmethod
    def tearDownClass(cls):
        gc.collect()
        Rdict.destroy(cls.path)


class TestColumnFamiliesCustomOpts(unittest.TestCase):
    cfs = None
    test_dict = None
    path = "./column_families_custom_options"

    @classmethod
    def setUpClass(cls) -> None:
        plain_opts = Options()
        plain_opts.create_missing_column_families(True)
        plain_opts.set_prefix_extractor(SliceTransform.create_max_len_prefix(8))
        plain_opts.set_plain_table_factory(PlainTableFactoryOptions())
        cls.cfs = {"string": Options(), "integer": plain_opts}
        cls.test_dict = Rdict(cls.path, options=plain_opts, column_families=cls.cfs)

    def test_column_families_custom_options_auto_reopen(self):
        assert self.test_dict is not None
        ds = self.test_dict.get_column_family(name="string")
        di = self.test_dict.get_column_family(name="integer")

        for i in range(1000):
            di[i] = i * i
            ds[str(i)] = str(i * i)

        self.test_dict["ok"] = True

        ds.close()
        di.close()
        self.test_dict.close()

        # reopen
        self.test_dict = Rdict(self.path)
        ds = self.test_dict.get_column_family("string")
        di = self.test_dict.get_column_family("integer")
        assert self.test_dict["ok"]
        compare_dicts(self, {i: i**2 for i in range(1000)}, di)
        compare_dicts(self, {str(i): str(i**2) for i in range(1000)}, ds)
        ds.close()
        di.close()
        self.test_dict.close()

    @classmethod
    def tearDownClass(cls):
        gc.collect()
        Rdict.destroy(cls.path)


class TestColumnFamiliesCustomOptionsCreate(unittest.TestCase):
    cfs = None
    test_dict = None
    plain_opts = None
    path = "./column_families_custom_options_create"

    @classmethod
    def setUpClass(cls) -> None:
        cls.plain_opts = Options()
        cls.plain_opts.create_missing_column_families(True)
        cls.plain_opts.set_prefix_extractor(SliceTransform.create_max_len_prefix(8))
        cls.plain_opts.set_plain_table_factory(PlainTableFactoryOptions())
        cls.test_dict = Rdict(cls.path, options=cls.plain_opts, column_families=cls.cfs)

    def test_column_families_custom_options_auto_reopen(self):
        assert self.test_dict is not None
        ds = self.test_dict.create_column_family(name="string")
        assert self.plain_opts is not None
        di = self.test_dict.create_column_family(
            name="integer", options=self.plain_opts
        )

        for i in range(1000):
            di[i] = i * i
            ds[str(i)] = str(i * i)

        self.test_dict["ok"] = True

        ds.close()
        di.close()
        self.test_dict.close()

        # reopen
        self.test_dict = Rdict(self.path)
        ds = self.test_dict.get_column_family("string")
        di = self.test_dict.get_column_family("integer")
        assert self.test_dict["ok"]
        compare_dicts(self, {i: i**2 for i in range(1000)}, di)
        compare_dicts(self, {str(i): str(i**2) for i in range(1000)}, ds)
        ds.close()
        di.close()
        self.test_dict.close()

    @classmethod
    def tearDownClass(cls):
        gc.collect()
        Rdict.destroy(cls.path)


class TestColumnFamiliesCustomOptionsCreateReopenOverride(unittest.TestCase):
    cfs = None
    test_dict = None
    plain_opts = None
    path = "./column_families_custom_options_create_reopen_override"

    @classmethod
    def setUpClass(cls) -> None:
        cls.plain_opts = Options()
        cls.plain_opts.create_missing_column_families(True)
        cls.plain_opts.set_prefix_extractor(SliceTransform.create_max_len_prefix(8))
        cls.plain_opts.set_plain_table_factory(PlainTableFactoryOptions())
        cls.test_dict = Rdict(cls.path, options=cls.plain_opts)

    def test_column_families_custom_options_auto_reopen_override(self):
        assert self.test_dict is not None
        ds = self.test_dict.create_column_family(name="string")
        assert self.plain_opts is not None
        di = self.test_dict.create_column_family(
            name="integer", options=self.plain_opts
        )

        for i in range(1000):
            di[i] = i * i
            ds[str(i)] = str(i * i)

        self.test_dict["ok"] = True

        ds.close()
        di.close()
        self.test_dict.close()

        # reopen
        gc.collect()
        old_opts, old_cols = Options.load_latest(self.path)
        old_opts.create_missing_column_families(True)
        old_cols["bytes"] = self.plain_opts
        self.test_dict = Rdict(self.path, options=old_opts, column_families=old_cols)
        ds = self.test_dict.get_column_family("string")
        di = self.test_dict.get_column_family("integer")
        db = self.test_dict.get_column_family("bytes")
        db[b"great"] = b"hello world"
        assert self.test_dict["ok"]
        assert db[b"great"] == b"hello world"
        compare_dicts(self, {i: i**2 for i in range(1000)}, di)
        compare_dicts(self, {str(i): str(i**2) for i in range(1000)}, ds)
        ds.close()
        di.close()
        db.close()
        self.test_dict.close()

        # reopen again auto read config
        gc.collect()
        self.test_dict = Rdict(self.path)
        ds = self.test_dict.get_column_family("string")
        di = self.test_dict.get_column_family("integer")
        db = self.test_dict.get_column_family("bytes")
        db[b"great"] = b"hello world"
        assert self.test_dict["ok"]
        assert db[b"great"] == b"hello world"
        compare_dicts(self, {i: i**2 for i in range(1000)}, di)
        compare_dicts(self, {str(i): str(i**2) for i in range(1000)}, ds)
        ds.close()
        di.close()
        db.close()
        self.test_dict.close()

    @classmethod
    def tearDownClass(cls):
        gc.collect()
        Rdict.destroy(cls.path)


class TestIntWithSecondary(unittest.TestCase):
    test_dict = None
    ref_dict = None
    secondary_dict = None
    opt = None
    path = "./temp_int_with_secondary"
    secondary_path = "./temp_int_with_secondary.secondary"

    @classmethod
    def setUpClass(cls) -> None:
        cls.opt = Options()
        cls.opt.create_if_missing(True)
        cls.test_dict = Rdict(cls.path, cls.opt)

        cls.secondary_dict = Rdict(
            cls.path,
            options=cls.opt,
            access_type=AccessType.secondary(cls.secondary_path),
        )

        cls.ref_dict = dict()

    def test_add_integer(self):
        assert self.test_dict is not None
        assert self.secondary_dict is not None
        assert self.ref_dict is not None
        for i in range(10000):
            key = randint(0, TEST_INT_RANGE_UPPER - 1)
            value = randint(0, TEST_INT_RANGE_UPPER - 1)
            self.ref_dict[key] = value
            self.test_dict[key] = value

        self.test_dict.flush(True)
        self.secondary_dict.try_catch_up_with_primary()
        compare_dicts(self, self.ref_dict, self.secondary_dict)

    def test_delete_integer(self):
        assert self.test_dict is not None
        assert self.secondary_dict is not None
        assert self.ref_dict is not None
        for i in range(5000):
            key = randint(0, TEST_INT_RANGE_UPPER - 1)
            if key in self.ref_dict:
                del self.ref_dict[key]
                del self.test_dict[key]

        self.test_dict.flush(True)
        self.secondary_dict.try_catch_up_with_primary()
        compare_dicts(self, self.ref_dict, self.secondary_dict)

    def test_delete_range(self):
        assert self.test_dict is not None
        assert self.secondary_dict is not None
        assert self.ref_dict is not None
        to_delete = []
        for key in self.ref_dict:
            if key >= 99999:
                to_delete.append(key)
        for key in to_delete:
            del self.ref_dict[key]
        self.test_dict.delete_range(99999, 10000000)

        self.test_dict.flush(True)
        self.secondary_dict.try_catch_up_with_primary()
        compare_dicts(self, self.ref_dict, self.secondary_dict)

    def test_reopen(self):
        assert self.test_dict is not None
        assert self.secondary_dict is not None
        assert self.ref_dict is not None
        self.secondary_dict.close()

        self.assertRaises(DbClosedError, lambda: self.secondary_dict.get(1) if self.secondary_dict is not None else None)

        gc.collect()
        self.secondary_dict = Rdict(
            self.path,
            options=self.opt,
            access_type=AccessType.secondary(self.secondary_path),
        )
        compare_dicts(self, self.ref_dict, self.secondary_dict)

    @classmethod
    def tearDownClass(cls):
        assert cls.test_dict is not None
        assert cls.secondary_dict is not None
        cls.test_dict.close()
        cls.secondary_dict.close()
        assert cls.opt is not None
        gc.collect()
        Rdict.destroy(cls.path, cls.opt)
        Rdict.destroy(cls.secondary_path, cls.opt)


class TestCheckpoint(unittest.TestCase):
    test_dict = None
    checkpoint_path = "./temp_checkpoint"
    path = "./temp_checkpoint_db"
    opt = None

    @classmethod
    def setUpClass(cls) -> None:
        cls.opt = Options()
        cls.opt.create_if_missing(True)
        cls.test_dict = Rdict(cls.path, cls.opt)

    def test_create_checkpoint(self):
        assert self.test_dict is not None
        # Populate the database
        for i in range(1000):
            self.test_dict[i] = i * i

        # Create a checkpoint
        checkpoint = Checkpoint(self.test_dict)
        checkpoint.create_checkpoint(self.checkpoint_path)
        del checkpoint

        # Open the checkpoint as a new Rdict instance
        checkpoint_dict = Rdict(self.checkpoint_path)

        # Verify the checkpoint data
        for i in range(1000):
            self.assertTrue(i in checkpoint_dict)
            self.assertEqual(checkpoint_dict[i], i * i)

        checkpoint_dict.close()

    @classmethod
    def tearDownClass(cls):
        assert cls.test_dict is not None
        assert cls.opt is not None
        cls.test_dict.close()
        gc.collect()
        Rdict.destroy(cls.path, cls.opt)
        Rdict.destroy(cls.checkpoint_path, cls.opt)


class TestCheckpointRaw(unittest.TestCase):
    test_dict = None
    checkpoint_path = "./temp_checkpoint_raw"
    path = "./temp_checkpoint_raw_db"
    opt = None

    @classmethod
    def setUpClass(cls) -> None:
        cls.opt = Options(True)  # Enable raw mode by passing True
        cls.opt.create_if_missing(True)
        cls.test_dict = Rdict(cls.path, cls.opt)

    def test_create_checkpoint(self):
        assert self.test_dict is not None
        # Populate the database
        for i in range(1000):
            self.test_dict.put_entity(bytes(i), names=[b"value"], values=[bytes(i * i)])

        # Create a checkpoint
        checkpoint = Checkpoint(self.test_dict)
        checkpoint.create_checkpoint(self.checkpoint_path)
        del checkpoint

        # Open the checkpoint as a new Rdict instance
        checkpoint_dict = Rdict(self.checkpoint_path)

        # Verify the checkpoint data
        for i in range(1000):
            self.assertTrue(bytes(i) in checkpoint_dict)
            entity = checkpoint_dict.get_entity(bytes(i))
            self.assertEqual(entity, [(b"value", bytes(i * i))])

        checkpoint_dict.close()

    @classmethod
    def tearDownClass(cls):
        assert cls.test_dict is not None
        cls.test_dict.close()
        assert cls.opt is not None
        gc.collect()
        Rdict.destroy(cls.path, cls.opt)
        Rdict.destroy(cls.checkpoint_path, cls.opt)


if __name__ == "__main__":
    unittest.main()
