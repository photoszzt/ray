# coding: utf-8
import sys

import numpy as np
import pytest
import ray
import ray.experimental.array.distributed as rda


def test_arr_3_chunk_1(ray_start_regular_shared):
    npa = np.array([[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]])
    npb = np.array([[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]])
    A = rda.numpy_to_dist.remote(npa, chunks=(1, 1))
    B = rda.numpy_to_dist.remote(npb, chunks=(1, 1))
    C = rda.dot.remote(A, B)
    c_val = ray.get(C)
    c_val = c_val.assemble()
    npc = np.matmul(npa, npb)
    assert np.array_equal(c_val, npc)


def test_arr_3_chunk_2(ray_start_regular_shared):
    npa = np.array([[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]])
    npb = np.array([[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]])
    A = rda.numpy_to_dist.remote(npa, chunks=(2, 2))
    B = rda.numpy_to_dist.remote(npb, chunks=(2, 2))
    C = rda.dot.remote(A, B)
    c_val = ray.get(C)
    c_val = c_val.assemble()
    npc = np.matmul(npa, npb)
    assert np.array_equal(c_val, npc)


def test_arr_4_chunk_2(ray_start_regular_shared):
    npa = np.array([[1, 2, 3, 4],
                    [5, 6, 7, 8],
                    [9, 10, 11, 12],
                    [13, 14, 15, 16],
                    ])
    npb = np.array([[1, 2, 3, 4],
                    [5, 6, 7, 8],
                    [9, 10, 11, 12],
                    [13, 14, 15, 16],
                    ])
    A = rda.numpy_to_dist.remote(npa, chunks=(2, 2))
    B = rda.numpy_to_dist.remote(npb, chunks=(2, 2))
    C = rda.dot.remote(A, B)
    c_val = ray.get(C)
    c_val = c_val.assemble()
    npc = np.matmul(npa, npb)
    assert np.array_equal(c_val, npc)


def test_large_arr(ray_start_regular_shared):
    chunks = (100, 100)
    rows = 500
    cols = 500
    npa = np.random.randint(0, 255, size=(rows, cols))
    npb = np.random.randint(0, 255, size=(rows, cols))
    A = rda.numpy_to_dist.remote(npa, chunks=chunks)
    B = rda.numpy_to_dist.remote(npb, chunks=chunks)

    C = rda.dot.remote(A, B)
    c_val = ray.get(C)
    c_val = c_val.assemble()
    npc = np.matmul(npa, npb)
    assert np.array_equal(c_val, npc)


def test_carma_500(ray_start_regular_shared):
    npa = np.random.randint(0, 255, size=(500, 200))
    npb = np.random.randint(0, 255, size=(200, 300))
    m, k, n = rda.carma_split(500, 200, 300, 2)
    A = rda.numpy_to_dist.remote(npa, chunks=(500/m, 200/k))
    B = rda.numpy_to_dist.remote(npb, chunks=(200/k, 300/n))
    C = rda.dot.remote(A, B)
    c_val = ray.get(C)
    c_val = c_val.assemble()
    npc = np.matmul(npa, npb)
    assert np.array_equal(c_val, npc)


if __name__ == "__main__":
    sys.exit(pytest.main(["-s", "-v", __file__]))
