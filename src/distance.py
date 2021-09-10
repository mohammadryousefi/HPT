import argparse
import pathlib
import sys

import numpy
import scipy.sparse as sp
import scipy.ndimage as nd


def parse_args(args):
    ap = argparse.ArgumentParser(
        description='Computes the euclidean distance transform for the sparse boundary matrix in the given shape. The boundary matrix contains all voxels on the surface boundaries.')
    ap.add_argument(dest='in_file', help='Path to 2D sparse matrix data file with shape (Samples, Values)',
                    metavar='InFile')
    ap.add_argument('-s', dest='shape', nargs='+', type=int, help='Volume Space Dimensions', metavar='Dimension')
    ap.add_argument('-f', dest='features', type=int, help='The number of features at the beginning of each record',
                    default=0, metavar='Features')
    ap.add_argument('-o', dest='out_file',
                    help='Output file location. Output format is a .npz compressed file with arr_0 as the array name.',
                    default=None, metavar='OutFile')
    ns = ap.parse_args(args)

    ns.in_file = pathlib.Path(ns.in_file)
    if ns.out_file is None:
        ns.out_file = ns.in_file.withstem(ns.in_file.stem + '_edt')
    else:
        ns.out_file = ns.in_file.joinpath(ns.out_file)
        if ns.out_file.suffix != '.npz':
            ns.out_file.joinpath(ns.in_file.stem + '_edt.npz')

    return ns


def compute_edt(mtx, shape):
    edt_output = numpy.zeros(mtx.shape, dtype=numpy.float64)
    for sample_index in range(mtx.shape[0]):
        edt_output[sample_index] = numpy.ravel(
            nd.distance_transform_edt(numpy.reshape(numpy.logical_not(mtx[sample_index].toarray()), shape)))
    return edt_output


if __name__ == '__main__':
    params = parse_args(sys.argv[1:])
    if params.shape is None:
        raise ValueError('Shape undefined.')
    print('Loading data ...', end='')
    mtx = sp.load_npz(params.in_file)[:, params.features:]
    print(' done.\nComputing euclidean distances ...', end='')
    edt = compute_edt(mtx, params.shape)
    print(' done.\nSaving results ...', end='')
    numpy.savez_compressed(params.out_file, edt)
    print(' done.')
