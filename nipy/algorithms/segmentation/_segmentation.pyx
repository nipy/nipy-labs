# -*- Mode: Python -*-  

"""
Markov random field utils. 

Author: Alexis Roche, 2010.
"""

__version__ = '0.2'

# Includes
import numpy as np
cimport numpy as np

# Externals
cdef extern from "mrf.h":
    void mrf_import_array()
    void ve_step(np.ndarray ppm, 
                 np.ndarray ref,
                 np.ndarray XYZ, 
                 np.ndarray U,
                 int ngb_size, 
                 double beta)
    np.ndarray make_edges(np.ndarray mask,
                       int ngb_size)
    double interaction_energy(np.ndarray ppm, 
                              np.ndarray XYZ,
                              np.ndarray U,
                              int ngb_size)
cdef extern from "stdlib.h":
    void* calloc(int nmemb, int size)
    void* free(void* ptr) 
cdef extern from "math.h":
    double HUGE_VAL
cdef extern from "math.h":
    void* memcpy(void* dest, void* src, int n)
    void* memset(void* s, int c, int n)

# Initialize numpy
mrf_import_array()
np.import_array()



def _ve_step(ppm, ref, XYZ, U, int ngb_size, double beta):

    if not ppm.flags['C_CONTIGUOUS'] or not ppm.dtype=='double':
        raise ValueError('ppm array should be double C-contiguous')
    if not ref.flags['C_CONTIGUOUS'] or not ref.dtype=='double':
        raise ValueError('ref array should be double C-contiguous')
    if not XYZ.flags['C_CONTIGUOUS'] or not XYZ.dtype=='intp':
        raise ValueError('XYZ array should be intp C-contiguous')
    if not XYZ.shape[1] == 3: 
        raise ValueError('XYZ array should be 3D')
    if not U.flags['C_CONTIGUOUS'] or not U.dtype=='double':
        raise ValueError('U array should be double C-contiguous')
    if not ppm.shape[-1] == ref.shape[-1]:
        raise ValueError('Inconsistent shapes for ppm and ref arrays')

    ve_step(<np.ndarray>ppm, <np.ndarray>ref, <np.ndarray>XYZ, <np.ndarray>U, 
             ngb_size, beta)
    return ppm 


def _make_edges(mask, int ngb_size):
    
    if not mask.flags['C_CONTIGUOUS'] or not mask.dtype=='intp':
        raise ValueError('mask array should be intp and C-contiguous')

    return make_edges(mask, ngb_size)


def _interaction_energy(ppm, XYZ, U, int ngb_size):

    if not ppm.flags['C_CONTIGUOUS'] or not ppm.dtype=='double':
        raise ValueError('ppm array should be double C-contiguous')
    if not XYZ.flags['C_CONTIGUOUS'] or not XYZ.dtype=='intp':
        raise ValueError('XYZ array should be intp C-contiguous')
    if not XYZ.shape[1] == 3: 
        raise ValueError('XYZ array should be 3D')
    if not U.flags['C_CONTIGUOUS'] or not U.dtype=='double':
        raise ValueError('U array should be double C-contiguous')

    return interaction_energy(<np.ndarray>ppm, <np.ndarray>XYZ, <np.ndarray>U,
                               ngb_size)


cdef eval_simplex_fitting_(double y, double* q0, double* m, double c1, double c2,
                           np.npy_intp* idx, double* mat, np.npy_intp size_idx,
                           double* b, double* qe, double* aux):
    """
    Compute a feasible solution to the simplex fitting problem
    assuming that the positiveness constraints on the elements given
    by `idx` are inactive.
    """
    cdef np.npy_intp i, j, idx_i, K = 3
    cdef double summ, tmp, tmp2, data_attach, prior
    cdef double *buf_mat

    # Compute the sum of b over inactive components
    summ = 0.0
    for i from 0 <= i < size_idx:
        summ += b[idx[i]]

    # Correct b for the unit sum constraint
    for i from 0 <= i < size_idx:
        aux[i] = b[idx[i]] + (c2 - summ) / <double>size_idx

    # Get the root of the Lagrangian derivative
    memset(<void*>qe, 0, K * sizeof(double))
    buf_mat = mat
    for i from 0 <= i < size_idx:
        idx_i = idx[i]
        for j from 0 <= j < size_idx:
            qe[idx_i] += buf_mat[0] * aux[j]
            buf_mat += 1

    # Renormalize the solution
    summ = 0.0
    for i from 0 <= i < K:
        tmp = qe[i]
        if tmp < 0:
            qe[i] = 0.0
            tmp = 0.0
        summ += tmp
    if summ < 1e-20:
        for i from 0 <= i < K:
            qe[i] = 1.0 / <double>K
            summ = 1.0

    # Compute fitting objective value
    data_attach = 0.0
    prior = 0.0
    for i from 0 <= i < K:
        qe[i] /= summ
        tmp = qe[i]
        data_attach += tmp * m[i]
        tmp2 = tmp - q0[i]
        prior += tmp2 * tmp2
    tmp = y - data_attach
    return c1 * tmp * tmp + c2 * prior


cdef simplex_fitting_(double* q, double y, double* q0, double* m, double c1, double c2,
                      nonzero_elements, solver, double* b, double* qe, double* aux):
    cdef np.npy_intp i, K = 3, Kbytes = K * sizeof(double), size_idx
    cdef np.flatiter itNonzero_Elements, itSolver
    cdef np.npy_intp* idx
    cdef double* mat
    cdef np.PyArrayObject *idx_ptr, *mat_ptr
    cdef double obj, obj_current = HUGE_VAL

    itNonzero_Elements = nonzero_elements.flat
    itSolver = solver.flat

    # Compute b = c1 * y * m + c2 * q0
    for i from 0 <= i < K:
        b[i] = c1 * y * m[i] + c2 * q0[i]

    while np.PyArray_ITER_NOTDONE(itNonzero_Elements):
        idx_ptr = (<np.PyArrayObject**> np.PyArray_ITER_DATA(itNonzero_Elements))[0]
        idx = <np.npy_intp*>np.PyArray_DATA(<object> idx_ptr)
        mat_ptr = (<np.PyArrayObject**> np.PyArray_ITER_DATA(itSolver))[0]
        mat = <double*>np.PyArray_DATA(<object> mat_ptr)

        size_idx = np.PyArray_DIM(<object>idx_ptr, 0)

        obj = eval_simplex_fitting_(y, q0, m, c1, c2, idx, mat, size_idx, b, qe, aux)
        if obj < obj_current:
            memcpy(<void*>q, <void*>qe, Kbytes)
            obj_current = obj

        np.PyArray_ITER_NEXT(itNonzero_Elements)
        np.PyArray_ITER_NEXT(itSolver)

    return

def simplex_fitting(Y, M, Q0, C1, C2):
    cdef np.flatiter itY, itQ0
    cdef double *y, *q0
    cdef int axis = 1
    cdef double *m, *b, *qe, *aux
    cdef double c1 = <double>C1, c2 = <double>C2
    cdef K = 3, Kbytes = K * sizeof(double)

    if not Q0.flags['C_CONTIGUOUS'] or not Q0.dtype=='double':
        raise ValueError('q0 should be double C-contiguous')

    Y = np.asarray(Y, dtype='double')
    M = np.asarray(M, dtype='double', order='C')
    m = <double*>np.PyArray_DATA(M)
    itY = Y.flat
    itQ0 = np.PyArray_IterAllButAxis(Q0, &axis)

    # Pre-compute matrices needed to solve for the Lagrangian
    # derivative's root for each possible set of inactive constraints.
    # Make sure the matrices are stored in row-major order as it's
    # assumed in sub-routines.
    nonzero_elements = [0,1,2], [0,1], [0,2], [1,2], [0], [1], [2]
    nonzero_elements = np.array([np.array(idx, dtype='intp') for idx in nonzero_elements])
    A1 = c1 * np.dot(M.reshape((K, 1)), M.reshape((1, K)))
    A = A1 + c2 * np.eye(K)
    solver = []
    for idx in nonzero_elements:
        e = np.zeros(K)
        e[idx] = 1
        Ae = A - np.dot(np.ones((K, 1)), np.dot(e.reshape((1, K)), A1)) / float(len(idx))
        solver.append(np.asarray(np.linalg.inv(Ae[idx][:, idx]), order='C'))
    solver = np.array(solver)

    # Allocate auxiliary arrays
    b = <double*>calloc(K, sizeof(double))
    q = <double*>calloc(K, sizeof(double))
    qe = <double*>calloc(K, sizeof(double))
    aux = <double*>calloc(K, sizeof(double))

    while np.PyArray_ITER_NOTDONE(itY):
        y = <double*>(np.PyArray_ITER_DATA(itY))
        q0 = <double*>(np.PyArray_ITER_DATA(itQ0))
        simplex_fitting_(q, y[0], q0, m, c1, c2, nonzero_elements, solver, b, qe, aux)
        memcpy(<void*>q0, <void*>q, Kbytes)
        np.PyArray_ITER_NEXT(itY)
        np.PyArray_ITER_NEXT(itQ0)

    # Free auxiliary arrays
    free(b)
    free(q)
    free(qe)
    free(aux)
