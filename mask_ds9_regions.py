# coding: utf-8

import pyregion

import numpy as np

from pathlib import Path

from astropy.visualization import ZScaleInterval
from astropy.io import fits

from matplotlib import pyplot as plt

import seaborn as sns


class DS9Reg (object) :
    '''
    Class to update the already existing mask frames with DS9 region files for each frame.
    The latter are meant to mask all the remaining artifacts of the pipeline such as
    cosmic rays not detected from the main authomatic routine, satellite tracks and ghosts
    owing to cross-talk effect when bright stars in neighboutring CCDs are present. 
    '''

    def __init__ (self, data) :

        self.data = fits.open(data)[0].data
        self.hdr =  fits.open(data)[0].header
        
        #newname = data.name[::-1][data.name[::-1].find('_')+1:][::-1]    # It does not work with filenames of convolved frames
        pos = [i for i, c in enumerate(data.name) if c=='_']
        newname = data.name[:pos[2]]

        self.reg_filename = newname + '.reg'

    
    def set_ds9_regions (self) :
        '''
        Method to open the region file related to data file if exists
        and to convert the DS9 regions therein in image coordinates.
        '''
        
        # Load DS9 region
        
        path_to_reg_filenames = Path('./ds9_regions').joinpath(self.reg_filename)
        
        if path_to_reg_filenames.exists() :
            self.regions = pyregion.open(path_to_reg_filenames).as_imagecoord(self.hdr)
        else :
        	self.regions = None
    
    
    def mask_regions (self) :
        '''
        In this method, a mask is created out of the pyregion.core.ShapeList
        '''
        
        if self.regions :
            self.mask = self.regions.get_mask(shape=self.data.shape)
        else :
            self.mask = np.full(self.data.shape, False)
    
    
    def plot_regions (self) :
        '''
        Check plot for the ds9 regions.
        '''   
        
        # Instantiate the patch
        patch_list, artist_list = self.regions.get_mpl_patches_texts()
        
        # Plot the image
        fig, ax = plt.subplots(1, 1, figsize=(7,7))
        zscale = ZScaleInterval()
        vmin, vmax = zscale.get_limits(self.data)
        ax.imshow(self.data, origin='lower', cmap='gray_r', vmin=vmin, vmax=vmax)

        # Plot the region
        for p in patch_list:
            ax.add_patch(p)

        ## Plot the masked DS9 region
        cmap = sns.light_palette("seagreen", as_cmap=True)
        cmap.set_bad(alpha=0)
        masked_data = np.ma.array(self.mask.astype('float'), mask=~self.mask)
        ax.imshow(masked_data.astype('float'), origin='lower', cmap=cmap, vmin=0, vmax=1, alpha=0.5)
        plt.show()



'''
# Open fits file
hdulist = fits.open('../cutout_images/NGC0628_Ha_stack5_cutout.fits')
data = hdulist[0].data
hdr = hdulist[0].header

# Load DS9 region
r = pyregion.open('NGC0628_Ha_5.reg').as_imagecoord(hdr)

# Instantiate the patch
patch_list, artist_list = r.get_mpl_patches_texts()

# Plot the image
fig, ax = plt.subplots(1, 1, figsize=(7,7))
zscale = ZScaleInterval()
vmin, vmax = zscale.get_limits(data)
ax.imshow(data, origin='lower', cmap='gray_r', vmin=vmin, vmax=vmax)

# Plot the region
for p in patch_list:
    ax.add_patch(p)
    
# Plot the masked DS9 region
cmap = sns.light_palette("seagreen", as_cmap=True)
cmap.set_bad(alpha=0)
mask = r.get_mask(shape=data.shape)
masked_data = np.ma.array(mask.astype('float'), mask=~mask)
ax.imshow(masked_data.astype('float'), origin='lower', cmap=cmap, vmin=0, vmax=1, alpha=0.5)

plt.show()
'''
