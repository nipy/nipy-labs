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


cdef _simple_fitting_one_voxel(double y, double* q0, double* m, double* c,
                               np.npy_intp* idx, double* mat, np.npy_intp size_idx,
                               double* b, double* qe, double* aux):
    """
    Compute a feasible solution to the simplex fitting problem
    assuming that the positivity constraints on the elements given by
    `idx` are inactive.
    """
    cdef np.npy_intp i, j, idx_i, K = 3
    cdef double lda, mt_qe, summ, tmp
    cdef double *buf_mat

    # Compute qe = mat * b, aux = mat * 1, and the Lagrange multiplier
    # lda that matches the unit sum constraint
    memset(<void*>qe, 0, K * sizeof(double))
    memset(<void*>aux, 0, K * sizeof(double))
    lda = 0.0
    summ = 0.0
    buf_mat = mat
    for i from 0 <= i < size_idx:
        idx_i = idx[i]
        for j from 0 <= j < size_idx:
            tmp = buf_mat[0]
            qe[idx_i] += tmp * b[idx[j]]
            aux[idx_i] += tmp
            buf_mat += 1
        lda += qe[idx_i]
        summ += aux[idx_i]
    lda = (lda - 1)/ summ

    # Compute the candidate solution (replace negative components by
    # zero)
    summ = 0.0
    for i from 0 <= i < size_idx:
        idx_i = idx[i]
        tmp = qe[idx_i] - lda * aux[idx_i]
        if tmp < 0:
            tmp = 0
        qe[idx_i] = tmp
        summ += tmp

    # Replace qe with a uniform distribution for safety if all
    # components are small
    if summ < 1e-20:
        for i from 0 <= i < K:
            qe[i] = 1.0 / <double>K
            summ = 1.0

    # Renormalize qe and compute mt_qe = m' * qe and aux = qe - q0
    mt_qe = 0.0
    for i from 0 <= i < K:
        qe[i] /= summ
        tmp = qe[i]
        mt_qe += tmp * m[i]
        aux[i] = tmp - q0[i]

    # Compute (qe - q0)^t C (qe - q0)
    buf_mat = c
    summ = 0.0
    for i from 0 <= i < K:
        tmp = 0.0
        for j from 0 <= j < K:
            tmp += buf_mat[0] * aux[j] 
            buf_mat += 1
        summ += aux[i] * tmp 

    # Return the objective value
    tmp = y - mt_qe
    return tmp * tmp + summ


cdef simple_fitting_one_voxel(double* q, double y, double* q0, double* m, double* c,
                              inactives, solver, double* b, double* qe, double* aux):
    cdef np.npy_intp i, j, K = 3, Kbytes = K * sizeof(double), size_idx
    cdef np.flatiter itInactives, itSolver
    cdef np.npy_intp* idx
    cdef double *mat, *buf_mat
    cdef np.PyArrayObject *idx_ptr, *mat_ptr
    cdef double tmp, best = HUGE_VAL

    # Compute b = y * m + c * q0
    buf_mat = c
    for i from 0 <= i < K:
        tmp = y * m[i]
        for j from 0 <= j < K:
            tmp += buf_mat[0] * q0[j]
            buf_mat += 1
        b[i] = tmp

    # Evaluate each hypothesis regarding active inequality constraints
    itInactives = inactives.flat
    itSolver = solver.flat
    while np.PyArray_ITER_NOTDONE(itInactives):
        idx_ptr = (<np.PyArrayObject**> np.PyArray_ITER_DATA(itInactives))[0]
        idx = <np.npy_intp*>np.PyArray_DATA(<object> idx_ptr)
        mat_ptr = (<np.PyArrayObject**> np.PyArray_ITER_DATA(itSolver))[0]
        mat = <double*>np.PyArray_DATA(<object> mat_ptr)
        size_idx = np.PyArray_DIM(<object>idx_ptr, 0)
        tmp = _simple_fitting_one_voxel(y, q0, m, c, idx, mat, size_idx, b, qe, aux)
        if tmp < best:
            memcpy(<void*>q, <void*>qe, Kbytes)
            best = tmp
        np.PyArray_ITER_NEXT(itInactives)
        np.PyArray_ITER_NEXT(itSolver)


def simplex_fitting(Y, M, Q0, C):
    """
    simplex_fitting(y, m, q0, C)

    Find the vector on the standard simplex in dimension 3 that
    minimizes:
    
    (y - m^t * q)^2 + (q - q0)^t C (q - q0)
    
    where dimensions are:
      y : (N,)
      M : (3,)
      Q0: (N,3)
      C: (3,3)
    
    Parameter `q0` is modified in place.
    """
    cdef np.flatiter itY, itQ0
    cdef double *y, *q0
    cdef int axis = 1
    cdef double *m, *c, *b, *qe, *aux
    cdef K = 3, Kbytes = K * sizeof(double)

    if not Q0.flags['C_CONTIGUOUS'] or not Q0.dtype=='double':
        raise ValueError('q0 should be double C-contiguous')

    Y = np.asarray(Y, dtype='double')
    M = np.asarray(M, dtype='double', order='C')
    C = np.asarray(C, dtype='double', order='C')
    m = <double*>np.PyArray_DATA(M)
    c = <double*>np.PyArray_DATA(C)
    itY = Y.flat
    itQ0 = np.PyArray_IterAllButAxis(Q0, &axis)

    # Pre-compute matrices needed to solve for the Lagrangian
    # derivative's root for each possible set of inactive constraints.
    # Make sure the matrices are stored in row-major order as it's
    # assumed in sub-routines.
    inactives = [0,1,2], [0,1], [0,2], [1,2], [0], [1], [2]
    inactives = np.array([np.array(idx, dtype='intp') for idx in inactives])
    A = np.dot(M.reshape((K, 1)), M.reshape((1, K))) + C
    solver = []
    for idx in inactives:
        solver.append(np.asarray(np.linalg.inv(A[idx][:, idx]), order='C'))
    solver = np.array(solver)

    # Allocate auxiliary arrays
    b = <double*>calloc(K, sizeof(double))
    q = <double*>calloc(K, sizeof(double))
    qe = <double*>calloc(K, sizeof(double))
    aux = <double*>calloc(K, sizeof(double))

    while np.PyArray_ITER_NOTDONE(itY):
        y = <double*>(np.PyArray_ITER_DATA(itY))
        q0 = <double*>(np.PyArray_ITER_DATA(itQ0))
        simple_fitting_one_voxel(q, y[0], q0, m, c, inactives, solver, b, qe, aux)
        memcpy(<void*>q0, <void*>q, Kbytes)
        np.PyArray_ITER_NEXT(itY)
        np.PyArray_ITER_NEXT(itQ0)

    # Free auxiliary arrays
    free(b)
    free(q)
    free(qe)
    free(aux)
