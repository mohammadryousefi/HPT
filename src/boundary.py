import argparse

import numpy
import scipy.sparse

VERBOSE = False


def get_space_offsets(space: list):
    return numpy.cumprod((1, *space[::-1]))[len(space) - 1::-1]


'''Use only to get indices of exisiting values'''


def get_indices(index: int, offsets: numpy.ndarray):
    if offsets is None:
        offsets = get_space_offsets(space)
    if offsets.ndim > 1:
        offsets = offsets.flatten()
    indices = numpy.zeros(shape=offsets.shape, dtype=numpy.int)
    for i in range(offsets.shape[0]):
        indices[i] = index // offsets[i]
        index %= offsets[i]
    return indices


def get_index(indices: numpy.ndarray, offsets: numpy.ndarray, upper_bound):
    if numpy.any(indices < 0):
        return -1
    index = numpy.dot(indices, offsets)
    return index if index < upper_bound else -1


'''
Returns index of the neighbor voxels in linear space.
@:param indices Location of the voxel in N-Dimensional space.
@:param offsets Linear offsets for each dimension.
@:param upper_bound Maximum valid index in linear space.

@:return Set of linear space indices for neighboring voxels. Any index outside the bounds return a -1 value. 
'''


def get_neighbors(indices, offsets, upper_bound=None):
    if upper_bound is None:
        upper_bound = numpy.prod(offsets)
    one_hot = numpy.zeros(offsets.shape, dtype=numpy.int)
    one_hot[0] = 1
    neighbors = []
    for i in range(one_hot.shape[0]):
        neighbors.append(get_index(indices + one_hot, offsets, upper_bound))
        neighbors.append(get_index(indices - one_hot, offsets, upper_bound))
        one_hot = numpy.roll(one_hot, 1)
    return set(neighbors)


def compute_boundary_from_sparse(mtx, space, start_sample=0, features=0):
    '''The output of this matrix is True where the voxel does not fall on the boundary of the volume'''
    num_samples = mtx.shape[0] - start_sample
    num_values = mtx.shape[1] - features
    if num_samples < 1:
        raise ValueError('Bad start_sample. No samples remaining in data.')
    if num_values < 1:
        raise ValueError('Bad features. No label values remaining in data.')
    if start_sample == 0 and features == 0:
        m = mtx
    else:
        m = mtx[start_sample:, features:]
    offsets = get_space_offsets(space)
    upper_bound = numpy.prod(offsets)
    print(
        f'Using: Matrix with {mtx.shape[0]} samples of {mtx.shape[1]} values. Skipping {start_sample} samples. Skipping {features} values of each sample as input features.\nResulting data matrix has {m.shape[0]} samples and {m.shape[1]} label values.\n\nSpace parameters:\n  Dimensions: {space}\n  Offsets: {offsets.tolist()}\n  Linear: {numpy.prod(offsets)}, Matches Label Count? {"yes" if numpy.prod(offsets) == m.shape[1] else "no"}')
    del mtx
    d = numpy.zeros(m.shape, dtype=numpy.int).astype(numpy.bool)
    for sample_index in range(d.shape[0]):
        voxel_indices = set(m.indices[m.indptr[sample_index]:m.indptr[sample_index + 1]].tolist())
        for voxel_index in voxel_indices:
            not_boundary = get_neighbors(get_indices(voxel_index, offsets), offsets, upper_bound).issubset(
                voxel_indices)
            d[sample_index, voxel_index] = not not_boundary
    return d
