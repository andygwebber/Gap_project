# -*- coding: utf-8 -*-
"""
Created on Fri May  4 20:08:00 2018

@author: Valued Customer
"""

import numpy as np
from sklearn.decomposition import PCA
from copy import deepcopy

epsilon = 0.001

class ImageSet():
    """ This a set of images. A method can dirty them up.
        A method can recover them from a pca model"""
    
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.mean = None
        self.pca = None
        self.image_mask = []
        
#    @classmethod
    def dirty(self, zerofac):
        """ This method dirty's the images up by a factor of zerofac
            If zerofac is 1 the entire image is set to zero. If it is
            0 the image is not changed at all."""
        points = np.shape(self.X)[1]
        for iimage in range(np.shape(self.X)[0]):
            raw_mask = np.random.ranf(points) + (0.5 - zerofac)
            mask = np.rint(raw_mask)
            self.X[iimage] = self.X[iimage] * mask
            """ This last modification is a tweak to note where points were set
                to zero for future recovery"""
            self.X[iimage] = self.X[iimage] + epsilon*(mask-1.0)
            image_mask = self.X[iimage] > -epsilon/2.0
            self.image_mask.append(image_mask)
        
    def pca_calc(self, components):
        """ This method calculates the pca model using the using the
            the images in this Image_set """

        self.pca = PCA(n_components=components)
        self.pca.fit(self.X)
        
    def mean_calc(self):
        """ This method calculates the mean of the images over all images.
            It only calculates over the clean images so masks must be 
            calculated for each image."""
            
        image_count = np.zeros(np.shape(self.X)[1])
        image_sum = np.zeros(np.shape(self.X)[1])
        
        for iimage in range(np.shape(self.X)[0]):
            image = self.X[iimage]
#            image_mask_good = image > -0.00001
            image_mask = self.image_mask[iimage]
            image_count[image_mask] += 1
            image_sum[image_mask] += image[image_mask]
            
        self.mean = image_sum/image_count
        
    def recover_from_pca(self, pca):
        """ This method recovers images from a passed pca object"""
        
        for iimage in range(np.shape(self.X)[0]):
            image = self.X[iimage]
            image_prime = image - pca.mean_
#            image_mask_indicies = image < 0.0
            image_mask = np.invert(self.image_mask[iimage])
            eigen_vec = deepcopy(pca.components_)
            for i in range(pca.n_components_):
                eigen_vec[i][image_mask] = 0.0
            eigen_vec_transpose = eigen_vec.transpose()
            A = eigen_vec.dot(eigen_vec_transpose)
            b = np.zeros(pca.n_components_)
            for i in range(pca.n_components_):
                b[i] = image_prime.dot(eigen_vec[i])
            coeff = np.linalg.solve(A,b)
            image = np.zeros(np.shape(self.X)[1])
            for i in range(pca.n_components_):
                image += coeff[i] * pca.components_[i]
            image += pca.mean_
            
            self.X[iimage] = image
            
    def recover_from_pca_mean(self,pca):
        """ This method replaces missing values with values from the mean computed
           from a principle component analysis of clean images"""
        
        for iimage in range(np.shape(self.X)[0]):
            image = self.X[iimage]
#            image_mask_indicies = image < 0.0
            image_mask = np.invert(self.image_mask[iimage])
            image[image_mask] = pca.mean_[image_mask]
            
    def recover_from_self_mean(self):
        " This method replaces missing values with values from own mean"""
        
        for iimage in range(np.shape(self.X)[0]):
            image = self.X[iimage]
#            image_mask_indicies = image < 0.0
            image_mask = np.invert(self.image_mask[iimage])
            image[image_mask] = self.mean[image_mask]
            
    def recover_from_self_pca(self, components, iterations):
        """ This method recovers images from it's own image set using
            iterative pca technique described in Everson and Sirovich"""
            
        self.recover_from_self_mean()
        
        for _ in range(iterations):
            self.pca_calc(components = components)
            self.recover_from_pca(self.pca)
        