from numbers import Number, Integral
import numpy as np
import ray.experimental.array.remote as ra
import ray

BLOCK_SIZE = 10


def is_integer(i):
    """
    >>> is_integer(6)
    True
    >>> is_integer(42.0)
    True
    >>> is_integer('abc')
    False
    """
    return isinstance(i, Integral) or (isinstance(i, float) and i.is_integer())


class DistArray:
    __slots__ = ["shape", "ndim", "num_blocks", "chunks", "object_refs"]

    def __init__(self, shape, object_refs=None, chunks=BLOCK_SIZE):
        shape = tuple(map(int, shape))
        self.shape = shape
        self.ndim = len(shape)
        if isinstance(chunks, (Number, str)):
            chunks = (chunks,) * len(shape)
        if isinstance(chunks, dict):
            chunks = tuple(chunks.get(i, None) for i in range(len(shape)))
        if not all(map(is_integer, chunks)):
            raise ValueError("chunks can only contain integers.")
        if not all(map(is_integer, shape)):
            raise ValueError("shape can only contain integers.")
        self.num_blocks = [
            int(np.ceil(1.0 * a / chunks[i])) for i, a in enumerate(self.shape)
        ]
        chunks = tuple(map(int, chunks))
        self.chunks = chunks
        if object_refs is not None:
            self.object_refs = object_refs
        else:
            self.object_refs = np.empty(self.num_blocks, dtype=object)
        if self.num_blocks != list(self.object_refs.shape):
            raise ValueError(
                "The fields `num_blocks` and `object_refs` are "
                "inconsistent, `num_blocks` is {} and `object_refs` "
                "has shape {}".format(self.num_blocks,
                                      list(self.object_refs.shape)))

    @staticmethod
    def compute_block_lower(index, shape, chunks):
        if len(index) != len(shape):
            raise ValueError("The fields `index` and `shape` must have the "
                             "same length, but `index` is {} and `shape` is "
                             "{}.".format(index, shape))
        return [elem * chunks[i] for i, elem in enumerate(index)]

    @staticmethod
    def compute_block_upper(index, shape, chunks):
        if len(index) != len(shape):
            raise ValueError("The fields `index` and `shape` must have the "
                             "same length, but `index` is {} and `shape` is "
                             "{}.".format(index, shape))
        upper = []
        for i in range(len(shape)):
            upper.append(min((index[i] + 1) * chunks[i], shape[i]))
        return upper

    @staticmethod
    def compute_block_shape(index, shape, chunks):
        lower = DistArray.compute_block_lower(index, shape, chunks)
        upper = DistArray.compute_block_upper(index, shape, chunks)
        return [u - l for (l, u) in zip(lower, upper)]

    @staticmethod
    def compute_num_blocks(shape, chunks):
        return [int(np.ceil(1.0 * a / chunks[i])) for i, a in enumerate(shape)]

    def assemble(self):
        """Assemble an array from a distributed array of object refs."""
        first_block = ray.get(self.object_refs[(0, ) * self.ndim])
        dtype = first_block.dtype
        result = np.zeros(self.shape, dtype=dtype)
        for index in np.ndindex(*self.num_blocks):
            lower = DistArray.compute_block_lower(index, self.shape, self.chunks)
            upper = DistArray.compute_block_upper(index, self.shape, self.chunks)
            value = ray.get(self.object_refs[index])
            result[tuple(slice(l, u) for (l, u) in zip(lower, upper))] = value
        return result

    def __getitem__(self, sliced):
        # TODO(rkn): Fix this, this is just a placeholder that should work but
        # is inefficient.
        a = self.assemble()
        return a[sliced]


@ray.remote
def assemble(a):
    return a.assemble()


# TODO(rkn): What should we call this method?
@ray.remote
def numpy_to_dist(a, chunks=BLOCK_SIZE):
    result = DistArray(a.shape, chunks=chunks)
    for index in np.ndindex(*result.num_blocks):
        lower = DistArray.compute_block_lower(index, a.shape, chunks)
        upper = DistArray.compute_block_upper(index, a.shape, chunks)
        idx = tuple(slice(int(l), int(u)) for (l, u) in zip(lower, upper))
        obj_ref = ray.put(a[idx])
        result.object_refs[index] = obj_ref
    return result


@ray.remote
def zeros(shape, dtype_name="float", chunks=BLOCK_SIZE):
    result = DistArray(shape, chunks=chunks)
    for index in np.ndindex(*result.num_blocks):
        result.object_refs[index] = ra.zeros.remote(
            DistArray.compute_block_shape(index, shape), dtype_name=dtype_name)
    return result


@ray.remote
def ones(shape, dtype_name="float", chunks=BLOCK_SIZE):
    result = DistArray(shape, chunks=chunks)
    for index in np.ndindex(*result.num_blocks):
        result.object_refs[index] = ra.ones.remote(
            DistArray.compute_block_shape(index, shape, chunks), dtype_name=dtype_name)
    return result


@ray.remote
def copy(a, chunks=BLOCK_SIZE):
    result = DistArray(a.shape, chunks=chunks)
    for index in np.ndindex(*result.num_blocks):
        # We don't need to actually copy the objects because remote objects are
        # immutable.
        result.object_refs[index] = a.object_refs[index]
    return result


@ray.remote
def eye(dim1, dim2=-1, dtype_name="float", chunks=BLOCK_SIZE):
    dim2 = dim1 if dim2 == -1 else dim2
    shape = [dim1, dim2]
    result = DistArray(shape, chunks=chunks)
    for (i, j) in np.ndindex(*result.num_blocks):
        block_shape = DistArray.compute_block_shape([i, j], shape, chunks)
        if i == j:
            result.object_refs[i, j] = ra.eye.remote(
                block_shape[0], block_shape[1], dtype_name=dtype_name)
        else:
            result.object_refs[i, j] = ra.zeros.remote(
                block_shape, dtype_name=dtype_name)
    return result


@ray.remote
def triu(a, chunks=BLOCK_SIZE):
    if a.ndim != 2:
        raise ValueError("Input must have 2 dimensions, but a.ndim is "
                         "{}.".format(a.ndim))
    result = DistArray(a.shape, chunks=chunks)
    for (i, j) in np.ndindex(*result.num_blocks):
        if i < j:
            result.object_refs[i, j] = ra.copy.remote(a.object_refs[i, j])
        elif i == j:
            result.object_refs[i, j] = ra.triu.remote(a.object_refs[i, j])
        else:
            result.object_refs[i, j] = ra.zeros_like.remote(
                a.object_refs[i, j])
    return result


@ray.remote
def tril(a, chunks=BLOCK_SIZE):
    if a.ndim != 2:
        raise ValueError("Input must have 2 dimensions, but a.ndim is "
                         "{}.".format(a.ndim))
    result = DistArray(a.shape, chunks=chunks)
    for (i, j) in np.ndindex(*result.num_blocks):
        if i > j:
            result.object_refs[i, j] = ra.copy.remote(a.object_refs[i, j])
        elif i == j:
            result.object_refs[i, j] = ra.tril.remote(a.object_refs[i, j])
        else:
            result.object_refs[i, j] = ra.zeros_like.remote(
                a.object_refs[i, j])
    return result


@ray.remote
def blockwise_dot(*matrices):
    n = len(matrices)
    if n % 2 != 0:
        raise ValueError("blockwise_dot expects an even number of arguments, "
                         "but len(matrices) is {}.".format(n))
    shape = (matrices[0].shape[0], matrices[n // 2].shape[1])
    result = np.zeros(shape)
    for i in range(n // 2):
        result += np.dot(matrices[i], matrices[n // 2 + i])
    return result


@ray.remote
def dot(a, b):
    if a.ndim != 2:
        raise ValueError("dot expects its arguments to be 2-dimensional, but "
                         "a.ndim = {}.".format(a.ndim))
    if b.ndim != 2:
        raise ValueError("dot expects its arguments to be 2-dimensional, but "
                         "b.ndim = {}.".format(b.ndim))
    if a.shape[1] != b.shape[0]:
        raise ValueError("dot expects a.shape[1] to equal b.shape[0], but "
                         "a.shape = {} and b.shape = {}.".format(
                             a.shape, b.shape))
    if a.chunks[1] != b.chunks[0]:
        raise ValueError("dot expects a.chunks[1] to equal b.chunks[0], but "
                         "a.chunks = {} and b.chunks = {}.".format(
                             a.chunks, b.chunks))
    shape = [a.shape[0], b.shape[1]]
    chunks = [a.chunks[0], b.chunks[1]]
    result = DistArray(shape, chunks=chunks)
    for (i, j) in np.ndindex(*result.num_blocks):
        args = list(a.object_refs[i, :]) + list(b.object_refs[:, j])
        result.object_refs[i, j] = blockwise_dot.remote(*args)
    return result


@ray.remote
def subblocks(a, *ranges):
    """
    This function produces a distributed array from a subset of the blocks in
    the `a`. The result and `a` will have the same number of dimensions. For
    example,
        subblocks(a, [0, 1], [2, 4])
    will produce a DistArray whose object_refs are
        [[a.object_refs[0, 2], a.object_refs[0, 4]],
         [a.object_refs[1, 2], a.object_refs[1, 4]]]
    We allow the user to pass in an empty list [] to indicate the full range.
    """
    ranges = list(ranges)
    if len(ranges) != a.ndim:
        raise ValueError("sub_blocks expects to receive a number of ranges "
                         "equal to a.ndim, but it received {} ranges and "
                         "a.ndim = {}.".format(len(ranges), a.ndim))
    for i in range(len(ranges)):
        # We allow the user to pass in an empty list to indicate the full
        # range.
        if ranges[i] == []:
            ranges[i] = range(a.num_blocks[i])
        if not np.alltrue(ranges[i] == np.sort(ranges[i])):
            raise ValueError("Ranges passed to sub_blocks must be sorted, but "
                             "the {}th range is {}.".format(i, ranges[i]))
        if ranges[i][0] < 0:
            raise ValueError("Values in the ranges passed to sub_blocks must "
                             "be at least 0, but the {}th range is {}.".format(
                                 i, ranges[i]))
        if ranges[i][-1] >= a.num_blocks[i]:
            raise ValueError("Values in the ranges passed to sub_blocks must "
                             "be less than the relevant number of blocks, but "
                             "the {}th range is {}, and a.num_blocks = {}."
                             .format(i, ranges[i], a.num_blocks))
    last_index = [r[-1] for r in ranges]
    last_block_shape = DistArray.compute_block_shape(
        last_index, a.shape, a.chunks)
    shape = [(len(ranges[i]) - 1) * a.chunks[i] + last_block_shape[i]
             for i in range(a.ndim)]
    result = DistArray(shape, chunks=a.chunks)
    for index in np.ndindex(*result.num_blocks):
        result.object_refs[index] = a.object_refs[tuple(
            ranges[i][index[i]] for i in range(a.ndim))]
    return result


@ray.remote
def transpose(a):
    if a.ndim != 2:
        raise ValueError("transpose expects its argument to be 2-dimensional, "
                         "but a.ndim = {}, a.shape = {}.".format(
                             a.ndim, a.shape))
    result = DistArray([a.shape[1], a.shape[0]], chunks=a.chunks)
    for i in range(result.num_blocks[0]):
        for j in range(result.num_blocks[1]):
            result.object_refs[i, j] = ra.transpose.remote(a.object_refs[j, i])
    return result


# TODO(rkn): support broadcasting?
@ray.remote
def add(x1, x2):
    if x1.shape != x2.shape:
        raise ValueError("add expects arguments `x1` and `x2` to have the same "
                         "shape, but x1.shape = {}, and x2.shape = {}.".format(
                             x1.shape, x2.shape))
    if x1.chunks != x2.chunks:
        raise ValueError("add expects arguments `x1` and `x2` to have the same "
                         "block size, but x1.chunks = {}, and x2.chunks = {}.".format(
                             x1.chunks, x2.chunks))
    result = DistArray(x1.shape, chunks=x1.chunks)
    for index in np.ndindex(*result.num_blocks):
        result.object_refs[index] = ra.add.remote(x1.object_refs[index],
                                                  x2.object_refs[index])
    return result


# TODO(rkn): support broadcasting?
@ray.remote
def subtract(x1, x2):
    if x1.shape != x2.shape:
        raise ValueError("subtract expects arguments `x1` and `x2` to have the "
                         "same shape, but x1.shape = {}, and x2.shape = {}."
                         .format(x1.shape, x2.shape))
    if x1.chunks != x2.chunks:
        raise ValueError("add expects arguments `x1` and `x2` to have the same "
                         "block size, but x1.chunks = {}, and x2.chunks = {}.".format(
                             x1.chunks, x2.chunks))
    result = DistArray(x1.shape, chunks=x1.chunks)
    for index in np.ndindex(*result.num_blocks):
        result.object_refs[index] = ra.subtract.remote(x1.object_refs[index],
                                                       x2.object_refs[index])
    return result


def _dim_to_split(m, k, n):
    result = 0
    if n >= k and n >= m:
        result = 1
    elif m >= k and m >= n:
        result = 2
    else:
        result = 3
    return result


def carma_split(m, k, n, p):
    m_split_num = 1
    k_split_num = 1
    n_split_num = 1
    factor = 2
    while p > 1 and m > 1 and k > 1 and n > 1:
        if _dim_to_split(m, k, n) == 1:
            n_split_num *= factor
            n /= factor
            p /= factor
        elif _dim_to_split(m, k, n) == 2:
            m_split_num *= factor
            m /= factor
            p /= factor
        else:
            k_split_num *= factor
            k /= factor
            p /= factor
    return m_split_num, k_split_num, n_split_num
