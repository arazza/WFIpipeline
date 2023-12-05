# coding: utf-8

'''
Background subtraction of WFI full FOV images.

The background is computed with masking the ellipse, masking the sources and
then compute a simple SExtractor-like background.
'''

import numpy as np
import timeit
import warnings
import sys
import gc
import argparse
import math

from pathlib import Path

from astropy.io import fits
#from astropy import units as u
from astropy.convolution import Gaussian2DKernel
from astropy.stats import gaussian_fwhm_to_sigma, sigma_clipped_stats
from astropy.nddata import Cutout2D
from astropy.modeling import models, fitting
from astropy.visualization import ZScaleInterval

from scipy import ndimage
from scipy.ndimage.filters import generic_filter
from scipy.interpolate import griddata
from scipy.optimize import least_squares

from photutils import Background2D, detect_threshold, detect_sources
from photutils.background import SExtractorBackground, StdBackgroundRMS, BkgZoomInterpolator

from photutils import EllipticalAperture

from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

import seaborn as sns

from astropy.modeling import models, fitting
from astropy.stats import sigma_clipped_stats

from galaxy_shape_v21 import GalaxyEllipse
from mask_ds9_regions import DS9Reg


class SkyImage (object) :

    def __init__ (self, pathtofile, pathtomask, pathtostd, ds9_mask, obj, filt) :

        self.path = pathtofile
        self.obj = obj
        self.filter = filt
        
        if str(pathtofile)[-13] == 'k' :
            self.filenum = str(pathtofile)[-12]
        else :
            self.filenum = str(pathtofile)[-13:-11]
        
        self.data = fits.open(pathtofile)[0].data
        self.bpm = np.round(fits.open(pathtomask)[0].data)!=99    # This create a boolean bad pixel mask (True for non-valid data)
        self.bpm = self.bpm + ds9_mask    # Update with DS9 masks implemented in NB-DR2 for masking image artifacts
        self.std = fits.open(pathtostd)[0].data
        self.hdr = fits.open(pathtofile)[0].header 
        self.mask = None
        self.bkg = None
        self.data_nosky = None


    def elliptical_aperture (self, r, instr) :

        # Instantiate elliptical aperture
        myellipse = GalaxyEllipse(self.obj, self.filter, self.path, self.bpm)
        myellipse.get_ellipse()
        myellipse.ellipse_to_pixel(instr)
        myellipse.mask_ellipse(r)

        return myellipse


    def compute_bkg (self, myellipse, exclude_percentile=10.0, dilate_size=11) :

        def mask_sources (data, input_mask, snr, dilate_size) :

            sigclip_sigma=3.0
            #sigclip_sigma=2.0  
            sigclip_iters=5
            mean, median, std = sigma_clipped_stats(data, mask=input_mask, sigma=sigclip_sigma, maxiters=sigclip_iters)    
            print (mean, median, std)
            
            threshold = detect_threshold(data, snr, background=median, error=std, mask=input_mask)
            print('Threshold --> ', threshold[0][0])
            
            segm = detect_sources(data, threshold, npixels=5)
            #dilate_size = 11
            selem = np.ones((dilate_size, dilate_size))
            #mask = ndimage.binary_dilation(segm.data.astype(np.bool), selem)    # numpy.bool name deprecated from numpy 1.20
            mask = ndimage.binary_dilation(segm.data.astype('bool'), selem)
            
            # Update the mask by masking the galaxy and the gaps
            return mask + input_mask

        
        print ('\nComputing the sky for %s ...\n'%self.path.stem)
    
        # Mask all the sources above 2*snr
        print ('Masking the image sources ...')
        self.mask = mask_sources (self.data, self.bpm+myellipse.mask, 2, dilate_size)
        #pathtomasked = self.path.parent.joinpath('%s_%s_masked_sources.fits'%(self.obj,self.filter))
        #fits.writeto(str(pathtomasked), self.mask.astype(float), header=self.hdr, overwrite=True)        
        
        # SExtractor-like background estimate
        print ('SExtractor-like background ...')
        interpolator = BkgZoomInterpolator()
        self.bkg = Background2D (self.data, (64, 64), mask=self.mask, exclude_percentile=exclude_percentile, filter_size=(3, 3), interpolator=interpolator)#, sigma_clip=None)  


    def plot_bkg (self, myellipse) :

        cmap = sns.cubehelix_palette(dark=.3, light=.8, as_cmap=True)
        
        cmap.set_bad(alpha=0)
        masked_data = np.ma.array(self.mask.astype('float'), mask=~self.mask)
        
        palette = sns.cubehelix_palette(dark=.3, light=.8, as_cmap=False)
        color1 = (palette[0][0], palette[0][1], palette[0][2])
        color2 = (palette[1][0], palette[1][1], palette[1][2])
        color3 = (palette[2][0], palette[2][1], palette[2][2])
        color4 = (palette[3][0], palette[3][1], palette[3][2])
        color5 = (palette[4][0], palette[4][1], palette[4][2])
        color6 = (palette[5][0], palette[5][1], palette[5][2])

        zscale = ZScaleInterval()
    

        # Plot 1
        
        fig1, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14.16, 7))
        
        # ax1 (image)
        data = self.data.copy()
        data[data==0] = np.nan
        vmin, vmax = zscale.get_limits(data)
        im1 = ax1.imshow(data, vmin=vmin, vmax=vmax, origin='lower', cmap='gray_r', interpolation='nearest')
        divider = make_axes_locatable(ax1)
        cax1 = divider.append_axes("top", size = "5%", pad = 0.05)
        cbar = plt.colorbar(im1, cax = cax1, orientation='horizontal', ticklocation = 'top')
        myellipse.aperture.plot(axes=ax1, color=color5)
        self.bkg.plot_meshes(axes=ax1, outlines=True, color='#1f77b4')
        ax1.set_xlabel('x [pixel]')
        ax1.set_ylabel('y [pixel]')
        ax1.tick_params(axis='y', pad=5, reset=True)
        ax1.yaxis.labelpad = 15
        
        # ax2 (background)
        vmin_res, vmax_res = zscale.get_limits(self.bkg.background)
        im2 = ax2.imshow(self.bkg.background, vmin=vmin_res, vmax=vmax_res, origin='lower', cmap='gray_r', interpolation='nearest')
        divider = make_axes_locatable(ax2)
        cax2 = divider.append_axes("top", size = "5%", pad = 0.05)
        cbar = plt.colorbar(im2, cax = cax2, orientation='horizontal', ticklocation = 'top')
        myellipse.aperture.plot(axes=ax2, color=color5, ls='--')
        ax2.imshow(masked_data.astype('float'), origin='lower', cmap=cmap.reversed(), vmin=0, vmax=1, alpha=0.25)
        ax2.set_xlabel('x [pixel]')
        ax2.yaxis.set_tick_params(labelleft=False)
        
        # ax3 (model 2D)
        vmin, vmax = zscale.get_limits(self.data_nosky)
        im3 = ax3.imshow(self.data_nosky, vmin=vmin, vmax=vmax, origin='lower', cmap='gray_r', interpolation='nearest')
        divider = make_axes_locatable(ax3)
        cax3 = divider.append_axes("top", size = "5%", pad = 0.05)
        cbar = plt.colorbar(im3, cax = cax3, orientation='horizontal', ticklocation = 'top')
        ax3.set_xlabel('x [pixel]')
        ax3.yaxis.set_tick_params(labelleft=False)

        plt.subplots_adjust(left=0.08, right=0.97, bottom=0.06, top=0.97)
        
        fig1.suptitle('%s %s frame %s '%(self.obj, self.filter, self.filenum))

        Path('bkg').mkdir(exist_ok=True)
        Path('bkg').joinpath(self.obj).mkdir(exist_ok=True)
    
        fig1.savefig('bkg/%s/%s_%s_%s_'%(self.obj, self.obj, self.filter, self.filenum)+'bkg_v2_fullFOV.png')
        
        print('\nClosing plots ..')
        plt.close()


if __name__ == "__main__":
    
    start_time = timeit.default_timer()

    wfi_galaxies = list(np.loadtxt('PHANGS_WFI_galaxies.txt', dtype='str', unpack=True))
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--wfi_fullFOV_path', action='store', type=str, default=None, help='path to calibrated WFI full FOV frames')
    parser.add_argument('-w', '--wfi', nargs='*', type=str, default=wfi_galaxies, 
        help='list of selected WFI galaxies (type a non-valid galaxy to avoid calibrating WFI)')
    parser.add_argument('--wfi_filters', nargs='*', type=str, default=['Rc', 'Ha'], help='list of WFI filters')

    parser.add_argument('--frame_number', type=int, default=1, 
        help='starting frame number for the convolution (in case you have some initial frames already convolved)')
    
    args = parser.parse_args()
    

    ##### Main loop #####
            
    wfi_galnames, wfi_paths = np.loadtxt('paths_to_calib_WFI.txt', dtype='str', unpack=True)

    wfi_galnames = np.atleast_1d(wfi_galnames)
    wfi_paths = np.atleast_1d(wfi_paths)
    
    for gal, p in zip(wfi_galnames, wfi_paths) :
    
        if (gal not in wfi_galaxies) or (gal not in args.wfi) :
            continue
        
        print('\nCompute the background for %s ...\n'%gal)
        
        if args.wfi_fullFOV_path :
            p = Path(args.wfi_fullFOV_path)
        else:
            p = Path(p)
    
        print('Images path :\n%s'%p)
        
        #for filt in ['Rc', 'Ha'] :
        for filt in args.wfi_filters :
        
            #if filt == 'Ha' :
    	    #    continue
            
            # setting the list of science, mask and error frames
            scienceframes = sorted(p.glob('%s_%s_stack?_calib.fits'%(gal,filt))) + sorted(p.glob('%s_%s_stack??_calib.fits'%(gal,filt)))
            maskframes = sorted(p.glob('%s_%s_stack?_mask_calib.fits'%(gal,filt))) + sorted(p.glob('%s_%s_stack??_mask_calib.fits'%(gal,filt)))
            errorframes = sorted(p.glob('%s_%s_stack?_std_calib.fits'%(gal,filt))) + sorted(p.glob('%s_%s_stack??_std_calib.fits'%(gal,filt)))
            
            scienceframes = scienceframes[args.frame_number-1:]
            maskframes = maskframes[args.frame_number-1:]
            errorframes = errorframes[args.frame_number-1:]
            
            if (not scienceframes) or (not maskframes) or (not errorframes) :
                
                print('\nThere are no frames with filter %s for %s\n'%(filt,gal))
                continue
            
            for scienceframe, maskframe, errorframe in zip(scienceframes, maskframes, errorframes) :
                
                #if scienceframe != scienceframes[0] :
                #    continue
                
                ##### Bkg subtraction #####
                                        
                '''
                Mask update with manually marked DS9 regions by loading
                the region file related to scienceframe file if exists.
                '''
                ds9_reg = DS9Reg(scienceframe)
                ds9_reg.set_ds9_regions()
                ds9_reg.mask_regions()
                
                print('\n\nBackground subtraction ...\n\n')
                
                # Instiantiate the sky object
                
                sky = SkyImage(scienceframe, maskframe, errorframe, ds9_reg.mask, gal, filt)
                
                myellipse = sky.elliptical_aperture(r=1.5, instr='WFI')
                
                sky.compute_bkg(myellipse)
                
                # Subtract the sky from the scienceframe
                 
                sky.data_nosky = fits.open(str(scienceframe))[0].data - sky.bkg.background.astype('float32') 
                
                # Plot the result
                
                sky.plot_bkg(myellipse)
                 
                # Saving the file
                 
                header = fits.open(str(scienceframe))[0].header
                
                file_nosky = p.joinpath(sky.path.name.replace('.fits','_nosky.fits'))
                
                fits.writeto(str(file_nosky), sky.data_nosky, header=header, overwrite=True)

                del sky, ds9_reg, myellipse, header
                print("\nDe-allocating memory...\n")
                gc.disable()
                gc.collect()
    
    
    elapsed = timeit.default_timer() - start_time
    minutes, seconds = divmod(elapsed, 60)
    hours, minutes = divmod(minutes, 60)
    print ('\nProcess terminated in %d:%02d:%02d'%(hours, minutes, seconds))
