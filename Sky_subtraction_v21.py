# coding: utf-8

'''
v4 log: Masking only the pixels (not the dilated sources) in Halpha science images (not background images)
'''

#import bottleneck as bn
#import ccdproc
import numpy as np
import timeit
import warnings
import sys
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

    def __init__ (self, pathtofile, pathtomask, pathtostd, ds9_mask, obj, filt, num) :

        self.path = pathtofile
        self.obj = obj
        self.filter = filt
        self.num = num
        self.data = fits.open(pathtofile)[0].data
        self.bpm = np.round(fits.open(pathtomask)[0].data)!=99    # This create a boolean bad pixel mask (True for non-valid data)
        self.bpm = self.bpm + ds9_mask    # Update with DS9 masks implemented in NB-DR2 for masking image artifacts
        self.std = fits.open(pathtostd)[0].data
        self.hdr = fits.open(pathtofile)[0].header 
        self.mask = None
        self.bkg = None
        self.fit = None


    def elliptical_aperture (self, r, instr) :

        # Instantiate elliptical aperture
        myellipse = GalaxyEllipse(self.obj, self.filter, self.path, self.bpm)
        myellipse.get_ellipse()
        myellipse.ellipse_to_pixel(instr)
        myellipse.mask_ellipse(r)

        return myellipse


    def compute_bkg (self, myellipse, doplot, exclude_percentile, dilate_size) :

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
            mask = ndimage.binary_dilation(segm.data.astype(np.bool), selem)
            
            # Update the mask by masking the galaxy and the gaps
            return mask + input_mask

        def fit_galaxy_bkg (bkg, mask, degree) :

            # Find the best-fit model
            print ('Fitting background surface ...')
            p_init = models.Polynomial2D(degree=degree)
            fit_p = fitting.LevMarLSQFitter()
            y, x = np.mgrid[:bkg.shape[0], :bkg.shape[1]]
            pfit = fit_p(p_init, x[~mask], y[~mask], bkg[~mask])
            
            return pfit(x, y)
        
        def plot_bkg_model_result (bkg, best_fit, res) :

            # Plot the data with the best-fit model
            fig = plt.figure(figsize=(14.16, 7))
            gs = gridspec.GridSpec(2,3, height_ratios=[0.05,1], width_ratios=[1,1,1])
            gs.update(left=0.07, right=0.93, bottom=0.05, top=0.75, hspace=0.01)

            ax1 = plt.subplot(gs[1,0])
            zscale = ZScaleInterval()
            vmin, vmax = zscale.get_limits(bkg[bkg!=0])
            plt1 = ax1.imshow(bkg, origin='lower', vmin=vmin, vmax=vmax)
            ax1.set_xlabel('x [pix]')
            ax1.set_ylabel('y [pix]')
            ax1.set_title("SExtractor Sky")
            cbax = plt.subplot(gs[0,0])
            cb = plt.colorbar(cax = cbax, mappable = plt1, orientation = 'horizontal', ticklocation = 'top')
            cbax.xaxis.set_major_locator(plt.MaxNLocator(5))
            
            ax2 = plt.subplot(gs[1,1])
            zscale = ZScaleInterval()
            vmin, vmax = zscale.get_limits(best_fit)
            plt2 = ax2.imshow(best_fit, origin='lower', vmin=vmin, vmax=vmax)
            ax2.set_xlabel('x [pix]')
            ax2.set_title("Model 2D fit")
            cbax = plt.subplot(gs[0,1])
            cb = plt.colorbar(cax = cbax, mappable = plt2, orientation = 'horizontal', ticklocation = 'top')
            cbax.xaxis.set_major_locator(plt.MaxNLocator(5))

            ax3 = plt.subplot(gs[1,2])
            zscale = ZScaleInterval()
            vmin, vmax = zscale.get_limits(res[res!=0])
            plt3 = ax3.imshow(res, origin='lower', vmin=vmin, vmax=vmax)
            ax3.set_xlabel('x [pix]')
            ax3.set_title("Residuals")
            cbax = plt.subplot(gs[0,2])
            cb = plt.colorbar(cax = cbax, mappable = plt3, orientation = 'horizontal', ticklocation = 'top')
            cbax.xaxis.set_major_locator(plt.MaxNLocator(5))
            
            fig.suptitle('%s %s frame %s '%(self.obj,self.filter, self.num))
            #plt.ion()
            #plt.draw()
            #plt.show()

            bkg_folder = self.path.parents[1].joinpath('bkg')
            bkg_folder.mkdir(exist_ok=True)

            fig.savefig(str(self.path.parents[1].joinpath('bkg/%s_%s_%s_'%(self.obj,self.filter, self.num)+'bkg_fit.png')))

            plt.close()
        

        
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
        #pathtobkg = self.path.parent.joinpath('%s_%s_%s_bkg.fits'%(self.obj,self.filter,self.num))
        #fits.writeto(str(pathtobkg), self.bkg.background, header=self.hdr, overwrite=True)
        #pathtobkgrms = self.path.parent.joinpath('%s_%s_%s_bkgrms.fits'%(self.obj,self.filter,self.num))
        #fits.writeto(str(pathtobkgrms), self.bkg.background_rms, header=self.hdr, overwrite=True)
        
        # Astropy fit 
        '''
        Note: fitting the masked backgorund gave lower residuals than fitting the masked image
        '''
        best_fit = fit_galaxy_bkg (self.bkg.background, self.mask, 1)

        # Computing residuals
        background = np.copy(self.bkg.background)
        fit = np.copy(best_fit)
        
        residuals = background - fit
        #image_residuals = image - image_fit
        #image_residuals[self.mask] = np.nan
        #pathtores = self.path.parent.joinpath('%s_%s_residuals.fits'%(self.obj,self.filter))
        #fits.writeto(str(pathtores), residuals, header=self.hdr, overwrite=True)

        # Plot data
        #if doplot :
        #    plot_bkg_model_result (self.bkg.background, best_fit, residuals)

        self.fit = best_fit.astype('float32')
        self.residuals = residuals.astype('float32')


    #def plot_bkg (self, myellipse) :
    #
    #    fig = plt.figure(figsize=(14.16, 7))
    #    gs = gridspec.GridSpec(1, 2, height_ratios=[1], width_ratios=[1,1])
    #    gs.update(left=0.1, right=0.95, bottom=0.17, top=0.82, wspace=0.3, hspace=0.03)
    #    
    #    zscale = ZScaleInterval()
    #    vmin, vmax = zscale.get_limits(self.data)
    #
    #    ax1 = plt.subplot(gs[0,0])
    #    ax1.imshow(self.data, vmin=vmin, vmax=vmax, origin='lower', cmap='gray_r', interpolation='nearest')
    #    myellipse.aperture.plot(axes=ax1, color='C1')#, color='#d62728')
    #    #myellipse.cutout.plot_on_original(color='C1')
    #    self.bkg.plot_meshes(outlines=True, color='#1f77b4')
    #   
    #    ax1.set_title('Meshes for background computation')
    #    #ax1.set_legend()
    #    
    #    zscale = ZScaleInterval()
    #    vmin, vmax = zscale.get_limits(self.bkg.background[self.bkg.background!=0])
    #
    #    ax2 = plt.subplot(gs[0,1])
    #    ax2.imshow(self.bkg.background, vmin=vmin, vmax=vmax, origin='lower', cmap='gray_r', interpolation='nearest')
    #
    #    ax2.set_title('2D background image')
    #    
    #    for ax in ([ax1,ax2]) :
    #        ax.set_xlabel('x [pixel]')
    #        ax.set_ylabel('y [pixel]')
    #        ax.tick_params(axis='x', pad=10, reset=True)
    #        ax.tick_params(axis='y', pad=5, reset=True)
    #        ax.xaxis.labelpad = 20
    #        ax.yaxis.labelpad = 20
    #    
    #    #plt.show()
    #
    #    fig.suptitle('%s %s frame %s '%(self.obj,self.filter, self.num))
    #
    #    fig.savefig(str(self.path.parents[1].joinpath('bkg/%s_%s_%s_'%(self.obj,self.filter, self.num)+'bkg.png')))
    #
    #    plt.close()


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
        vmin, vmax = zscale.get_limits(self.fit)
        im3 = ax3.imshow(self.fit, vmin=vmin, vmax=vmax, origin='lower', cmap='gray_r', interpolation='nearest')
        divider = make_axes_locatable(ax3)
        cax3 = divider.append_axes("top", size = "5%", pad = 0.05)
        cbar = plt.colorbar(im3, cax = cax3, orientation='horizontal', ticklocation = 'top')
        ax3.set_xlabel('x [pixel]')
        ax3.yaxis.set_tick_params(labelleft=False)

        plt.subplots_adjust(left=0.08, right=0.97, bottom=0.06, top=0.97)
        
        fig1.suptitle('%s %s frame %s '%(self.obj,self.filter, self.num))
    
        #fig1.savefig(str(self.path.parents[1].joinpath('bkg/%s_%s_%s_'%(self.obj,self.filter, self.num)+'bkg_v2_1.png')))
        fig1.savefig('bkg/%s/%s_%s_%s_bkg_v2_1.png'%(self.obj,self.obj,self.filter,self.num))

        plt.close()


        # Plot 2
        
        fig2, (ax4, ax5, ax6) = plt.subplots(1, 3, figsize=(14.16, 7))
        
        # ax4 (background subtracted)
        bkg_sub = self.data - self.fit
        bkg_sub[self.bpm] = np.nan
        vmin, vmax = zscale.get_limits(bkg_sub)
        im4 = ax4.imshow(bkg_sub, vmin=vmin, vmax=vmax, origin='lower', cmap='gray_r', interpolation='nearest')
        divider = make_axes_locatable(ax4)
        cax4 = divider.append_axes("top", size = "5%", pad = 0.05)
        cbar = plt.colorbar(im4, cax = cax4, orientation='horizontal', ticklocation = 'top')
        ax4.set_xlabel('x [pixel]')
        ax4.set_ylabel('y [pixel]')
        ax4.tick_params(axis='y', pad=5, reset=True)
        ax4.yaxis.labelpad = 15
        
        # ax5 (residuals)
        vmin, vmax = vmin_res - (vmax_res+vmin_res)/2., vmax_res - (vmax_res+vmin_res)/2.
        im5 = ax5.imshow(self.residuals, origin='lower', vmin=vmin, vmax=vmax, cmap='gray_r', interpolation='nearest')
        divider = make_axes_locatable(ax5)
        cax5 = divider.append_axes("top", size = "5%", pad = 0.05)
        cbar = plt.colorbar(im5, cax = cax5, orientation='horizontal', ticklocation = 'top')
        myellipse.aperture.plot(axes=ax5, color=color5, ls='--')
        ax5.imshow(masked_data.astype('float'), origin='lower', cmap=cmap.reversed(), vmin=0, vmax=1, alpha=0.25)
        ax5.set_xlabel('x [pixel]')
        ax5.yaxis.set_tick_params(labelleft=False)
        
        # ax6 (histogram)
        ax6.hist(self.residuals[~self.mask]/self.std[~self.mask], bins=100, color=color2, density=True)
        ax6.set_aspect(1.0/ax6.get_data_ratio(), adjustable='box')
        ax6.set_xlabel(r'$(\mathrm{bkg} - \mathrm{fit})\,/\,\sigma$')
        ax6.xaxis.labelpad = 10
        ax6.spines['right'].set_visible(False)
        ax6.yaxis.set_ticks_position('left')
        ax6.spines['top'].set_visible(False)
        ax6.xaxis.set_ticks_position('bottom')
        #ax6.yaxis.set_label_position('right')
        #ax6.yaxis.labelpad = 5
        #ax6.set_ylabel(r'density of unmasked pixels', rotation=-90)

        plt.subplots_adjust(left=0.08, right=0.97, bottom=0.06, top=0.97)
    
        fig2.suptitle('%s %s frame %s '%(self.obj,self.filter, self.num))
    
        #fig2.savefig(str(self.path.parents[1].joinpath('bkg/%s_%s_%s_'%(self.obj,self.filter, self.num)+'bkg_v2_2.png')))
        fig2.savefig('bkg/%s/%s_%s_%s_bkg_v2_2.png'%(self.obj,self.obj,self.filter,self.num))

        plt.close()

        

if __name__ == "__main__":

    start_time = timeit.default_timer()

    parser = argparse.ArgumentParser()
    
    parser.add_argument('pathtocalibgalaxy', help='path to galaxy calibrated final mosaic')
    parser.add_argument('galaxyname', help='name of selected galaxy')
    parser.add_argument('filter', help='filter name (Rc/Ha)')
    parser.add_argument('-e', '--ell_radius', action='store', default=1., help='galaxy ellipse increasing factor (recommended for Rc)')

    args = parser.parse_args()
    
    p = Path(args.pathtocalibgalaxy)    
    obj = args.galaxyname               
    filt = args.filter                  

    pathtofile = p.joinpath('%s_%s_stack1_cutout.fits'%(obj,filt)) 
    pathtomask = p.joinpath('%s_%s_stack1_mask_cutout.fits'%(obj,filt))
    pathtostd = p.joinpath('%s_%s_stack1_std_cutout.fits'%(obj,filt))
    
    ds9_reg = DS9Reg(pathtofile)
    ds9_reg.set_ds9_regions()
    ds9_reg.mask_regions()

    sky = SkyImage(pathtofile, pathtomask, pathtostd, ds9_reg.mask, obj, filt, 1)
    myellipse = sky.elliptical_aperture(args.ell_radius, 'WFI')
    sky.compute_bkg(myellipse, True, 10, 11)
    sky.plot_bkg(myellipse)
    
    elapsed = timeit.default_timer() - start_time
    m, s = divmod(elapsed, 60)
    h, m = divmod(m, 60)
    print ('Process terminated in %d:%02d:%02d'%(h, m, s))

