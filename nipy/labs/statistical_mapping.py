# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
AR, Sep 30 2013:
Part of this stuff has moved to  nipy.algorithms.statistics.spatial_sta.py
"""
import numpy as np

# Use the nibabel image object
from nibabel import Nifti1Image as Image
from .glm import glm
from .group.permutation_test import \
     permutation_test_onesample, permutation_test_twosample

# FIXME: rename permutation_test_onesample class
#so that name starts with upper case


###############################################################################
# Statistical tests
###############################################################################


def prepare_arrays(data_images, vardata_images, mask_images):
    from .mask import intersect_masks
    # Compute mask intersection
    mask = intersect_masks(mask_images, threshold=1.)
     # Compute xyz coordinates from mask
    xyz = np.array(np.where(mask > 0))
    # Prepare data & vardata arrays
    data = np.array([(d.get_data()[xyz[0], xyz[1], xyz[2]]).squeeze()
                    for d in data_images]).squeeze()
    if vardata_images == None:
        vardata = None
    else:
        vardata = np.array([(d.get_data()[xyz[0], xyz[1], xyz[2]]).squeeze()
                            for d in vardata_images]).squeeze()
    return data, vardata, xyz, mask


def onesample_test(data_images, vardata_images, mask_images, stat_id,
                   permutations=0, cluster_forming_th=0.01):
    """
    Helper function for permutation-based mass univariate onesample
    group analysis.
    """
    # Prepare arrays
    data, vardata, xyz, mask = prepare_arrays(data_images, vardata_images,
                                              mask_images)

    # Create one-sample permutation test instance
    ptest = permutation_test_onesample(data, xyz, vardata=vardata,
                                       stat_id=stat_id)

    # Compute z-map image
    zmap = np.zeros(data_images[0].shape).squeeze()
    zmap[list(xyz)] = ptest.zscore()
    zimg = Image(zmap, data_images[0].get_affine())

    # Compute mask image
    maskimg = Image(mask.astype(np.int8), data_images[0].get_affine())

    # Multiple comparisons
    if permutations <= 0:
        return zimg, maskimg
    else:
        # Cluster definition: (threshold, diameter)
        cluster_def = (ptest.height_threshold(cluster_forming_th), None)

        # Calibration
        voxel_res, cluster_res, region_res = \
            ptest.calibrate(nperms=permutations, clusters=[cluster_def])
        nulls = {}
        nulls['zmax'] = ptest.zscore(voxel_res['perm_maxT_values'])
        nulls['s'] = cluster_res[0]['perm_size_values']
        nulls['smax'] = cluster_res[0]['perm_maxsize_values']

        # Return z-map image, mask image and dictionary of null distribution
        # for cluster sizes (s), max cluster size (smax) and max z-score (zmax)
        return zimg, maskimg, nulls


def twosample_test(data_images, vardata_images, mask_images, labels, stat_id,
                   permutations=0, cluster_forming_th=0.01):
    """
    Helper function for permutation-based mass univariate twosample group
    analysis. Labels is a binary vector (1-2). Regions more active for group
    1 than group 2 are inferred.
    """
    # Prepare arrays
    data, vardata, xyz, mask = prepare_arrays(data_images, vardata_images,
                                              mask_images)

    # Create two-sample permutation test instance
    if vardata_images == None:
        ptest = permutation_test_twosample(
            data[labels == 1], data[labels == 2], xyz, stat_id=stat_id)
    else:
        ptest = permutation_test_twosample(
            data[labels == 1], data[labels == 2], xyz,
            vardata1=vardata[labels == 1], vardata2=vardata[labels == 2],
            stat_id=stat_id)

    # Compute z-map image
    zmap = np.zeros(data_images[0].shape).squeeze()
    zmap[list(xyz)] = ptest.zscore()
    zimg = Image(zmap, data_images[0].get_affine())

    # Compute mask image
    maskimg = Image(mask, data_images[0].get_affine())

    # Multiple comparisons
    if permutations <= 0:
        return zimg, maskimg
    else:
        # Cluster definition: (threshold, diameter)
        cluster_def = (ptest.height_threshold(cluster_forming_th), None)

        # Calibration
        voxel_res, cluster_res, region_res = \
            ptest.calibrate(nperms=permutations, clusters=[cluster_def])
        nulls = {}
        nulls['zmax'] = ptest.zscore(voxel_res['perm_maxT_values'])
        nulls['s'] = cluster_res[0]['perm_size_values']
        nulls['smax'] = cluster_res[0]['perm_maxsize_values']

        # Return z-map image, mask image and dictionary of null
        # distribution for cluster sizes (s), max cluster size (smax)
        # and max z-score (zmax)
        return zimg, maskimg, nulls

###############################################################################
# Linear model
###############################################################################


def linear_model_fit(data_images, mask_images, design_matrix, vector):
    """
    Helper function for group data analysis using arbitrary design matrix
    """

    # Prepare arrays
    data, vardata, xyz, mask = prepare_arrays(data_images, None, mask_images)

    # Create glm instance
    G = glm(data, design_matrix)

    # Compute requested contrast
    c = G.contrast(vector)

    # Compute z-map image
    zmap = np.zeros(data_images[0].shape).squeeze()
    zmap[list(xyz)] = c.zscore()
    zimg = Image(zmap, data_images[0].get_affine())

    return zimg


class LinearModel(object):

    def_model = 'spherical'
    def_niter = 2

    def __init__(self, data, design_matrix, mask=None, formula=None,
                 model=def_model, method=None, niter=def_niter):

        # Convert input data and design into sequences
        if not hasattr(data, '__iter__'):
            data = [data]
        if not hasattr(design_matrix, '__iter__'):
            design_matrix = [design_matrix]

        # configure spatial properties
        # the 'sampling' direction is assumed to be the last
        # TODO: check that all input images have the same shape and
        # that it's consistent with the mask
        nomask = mask == None
        if nomask:
            self.xyz = None
            self.axis = len(data[0].shape) - 1
        else:
            self.xyz = np.where(mask.get_data() > 0)
            self.axis = 1

        self.spatial_shape = data[0].shape[0: -1]
        self.affine = data[0].get_affine()

        self.glm = []
        for i in range(len(data)):
            if not isinstance(design_matrix[i], np.ndarray):
                raise ValueError('Invalid design matrix')
            if nomask:
                Y = data[i].get_data()
            else:
                Y = data[i].get_data()[self.xyz]
            X = design_matrix[i]

            self.glm.append(glm(Y, X, axis=self.axis,
                                formula=formula, model=model,
                                method=method, niter=niter))

    def dump(self, filename):
        """Dump GLM fit as npz file.
        """
        models = len(self.glm)
        if models == 1:
            self.glm[0].save(filename)
        else:
            for i in range(models):
                self.glm[i].save(filename + str(i))

    def contrast(self, vector):
        """Compute images of contrast and contrast variance.
        """
        # Compute the overall contrast across models
        c = self.glm[0].contrast(vector)
        for g in self.glm[1:]:
            c += g.contrast(vector)

        def affect_inmask(dest, src, xyz):
            if xyz == None:
                dest = src
            else:
                dest[xyz] = src
            return dest

        con = np.zeros(self.spatial_shape)
        con_img = Image(affect_inmask(con, c.effect, self.xyz), self.affine)
        vcon = np.zeros(self.spatial_shape)
        vcon_img = Image(affect_inmask(vcon, c.variance, self.xyz),
                         self.affine)
        z = np.zeros(self.spatial_shape)
        z_img = Image(affect_inmask(z, c.zscore(), self.xyz), self.affine)
        dof = c.dof
        return con_img, vcon_img, z_img, dof


###############################################################################
# Hack to have nose skip onesample_test, which is not a unit test
onesample_test.__test__ = False
twosample_test.__test__ = False
