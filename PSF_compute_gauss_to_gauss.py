# coding: utf-8

'''
Broad-band images empirical PSF computation.

The code finds the brightest stars in each
photometrically calibrated frame and computes
the ePSF for each of them.

The ePSFs are assumed to be gaussian and a
gaussian kernel is found for each frame to
convolved ePSF to the worst resolution image
(also assumed to have a gaussian PSF).

It works for both WFI and DP data set.
'''

import os

import sys

import gc

import timeit

import argparse

from configparser import ConfigParser

from pathlib import Path

import numpy as np

from astropy.io import fits
from astropy.wcs import WCS
from astropy.stats import sigma_clipped_stats, gaussian_fwhm_to_sigma, SigmaClip
from astropy.coordinates import SkyCoord
from astropy.table import Table, Column
from astropy.utils.data import conf
from astropy import units as u
from astropy.visualization import ZScaleInterval, simple_norm
from astropy.nddata import NDData
from astropy.convolution import convolve, Gaussian2DKernel, Kernel

from astroquery.vizier import Vizier

from photutils import IRAFStarFinder, EPSFBuilder, extract_stars
from photutils import detect_threshold, detect_sources, SourceCatalog
from photutils import CircularAperture, CircularAnnulus, aperture_photometry
from photutils.centroids import centroid_com, centroid_2dg
from photutils.detection import find_peaks

from scipy import ndimage
from scipy.ndimage import zoom, shift

from mpdaf.obj import Image

from shutil import copy

from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import LogFormatter, MaxNLocator

from galaxy_shape_v21 import GalaxyEllipse
from mask_ds9_regions import DS9Reg
from Sky_subtraction_v21 import SkyImage


class ImageStars (object) :
    '''
    This class contains info of a set of stars for a given calibrated frame.
    '''
    def __init__ (self, path, psf_path, obj, filt, instr, filedata, filemask, filesigma, fwhm) :
        '''
        Note on the fwhm: the mean value computed from Brent's NB-DR1 WFI PSFs
        and the weighted average seeing of the du Pont observations resulted
        to be ~4.08 pixels
        '''

        self.path = path
        self.psf_path = psf_path

        self.obj = obj
        self.filename = filedata

        self.filt = filt
        self.instr = instr

        self.data = fits.open(str(filedata))[0].data
        self.mask = np.round(fits.open(str(filemask))[0].data)!=99
        self.error = fits.open(str(filesigma))[0].data

        self.hdr = fits.open(str(filedata))[0].header
        self.wcs = WCS(fits.open(str(filedata))[0].header)

        pos = [i for i, c in enumerate(filedata.name) if c=='_']

        if self.instr == 'WFI' :

            self.filenum = int(filedata.name[pos[1]+6:pos[2]])

            # DEPRECATED
            #if str(filedata)[-19] == 'k' :
            #    self.filenum = str(filedata)[-18]
            #else :
            #    self.filenum = str(filedata)[-19:-17]

        if self.instr == 'CCD' :

            if len(pos) >=3 :
                self.filenum = int(filedata.name[pos[1]+1:pos[2]])
            else :
                self.filenum = 0    # For specphot stars fields (DEPRECATED!)

        self.fwhm = fwhm


    def find_image_stars (self, r=1.2) :
        '''
        This method computes the image threshold to find stars with
        IRAFFindStars algorithm. An astropy Table is created with pixel positions
        and instrumental fluxes in microJy.
        '''

        # find stars in the input image
        sigclip_sigma=3.0
        sigclip_iters=5
        image_mean, image_median, image_std = sigma_clipped_stats(self.data, sigma=sigclip_sigma, maxiters=sigclip_iters)
        print (image_mean, image_median, image_std)
        threshold = image_median + (50 * image_std)
        print('Threshold for finding bright stars', threshold)

        iraffind = IRAFStarFinder(threshold=threshold, fwhm=self.fwhm, exclude_border=True, brightest=100)
        #dilate_size = 21
        #selem = np.ones((dilate_size, dilate_size))
        #mask = ndimage.binary_dilation(np.isnan(self.data), selem)

        # Mask an ellipse around the galaxy
        self.myellipse = GalaxyEllipse(self.obj, self.filt, self.filename, self.mask)
        self.myellipse.get_ellipse()
        self.myellipse.ellipse_to_pixel(self.instr)
        self.myellipse.mask_ellipse(r)

        self.mask = self.mask + self.myellipse.mask

        self.tbl = iraffind(self.data, mask=self.mask)
        self.tbl.keep_columns(['xcentroid', 'ycentroid', 'flux'])


    def match_with_gaia (self, gaia_stars) :
        '''
        Centroid positions are converted in sky coordinates, by means of
        the image wcs solution. Gaia DR2 stars are then retreived to be matched.
        '''
        # Convert pixel positions in coordinates
        ra, dec = self.wcs.all_pix2world(self.tbl['xcentroid'], self.tbl['ycentroid'], 0)
        self.coord = SkyCoord (ra=ra*u.deg, dec=dec*u.deg)

        # .. and update tbl
        ra = Column(self.coord.ra, name='ra')
        dec = Column(self.coord.dec, name='dec')
        self.tbl.add_column(ra)
        self.tbl.add_column(dec)

        # match with Gaia stars (closest one-to-one match)
        max_sep = 0.5 * u.arcsec
        idx, d2d, d3d = self.coord.match_to_catalog_sky(gaia_stars.coord)
        sep_constraint = d2d < max_sep
        print('Number of matching stars between the image and Gaia : ',len(self.coord[sep_constraint]))

        # update tbl
        self.tbl = self.tbl[sep_constraint]

        for col in self.tbl.colnames:
            self.tbl[col].info.format = '%.4f'


    def sigma_clipped_epsf (self, extracted_stars, oversampling_factor) :
        '''
        Compute the sigma clipped median of each pixel in the normalized star image
        to build a ePSF. The ePSF is then normalized.

        Stars are re-centered before normalizing (added in version 3.2).
        '''

        # List of stars without NaNs in their field
        clean_stars = [star for star in extracted_stars if not np.isnan(star.data).any()]

        # Oversample
        star_zoom = zoom(clean_stars[0], oversampling_factor)

        # Recentering
        #centre = (extracted_stars[0].data.shape[0] - 1) / 2
        centre_oversampled = (star_zoom.shape[0] - 1) / 2

        #xc_oversampled = extracted_stars[0].cutout_center[0] * centre_oversampled / centre
        #yc_oversampled = extracted_stars[0].cutout_center[1] * centre_oversampled / centre

        mean, median, std = sigma_clipped_stats(star_zoom, sigma=3.0)
        threshold = median + (5. * std)
        tbl = find_peaks(star_zoom, threshold, box_size=5, border_width=5, npeaks=1, centroid_func=centroid_com)
        #tbl = find_peaks(star_zoom, threshold, box_size=5, border_width=5, npeaks=1, centroid_func=centroid_2dg)

        dx = centre_oversampled - tbl['x_centroid']#xc_oversampled
        dy = centre_oversampled - tbl['y_centroid']#yc_oversampled

        star_zoom_shift = shift(star_zoom, shift=(dy,dx))

        # Normaization
        #stacked_stars = extracted_stars[0].data  / np.sum(extracted_stars[0].data)
        stacked_stars = star_zoom_shift  / np.sum(star_zoom_shift)

        for i in range(1, len(clean_stars)) :

            star_zoom = zoom(clean_stars[i], oversampling_factor)

            #xc_oversampled = extracted_stars[i].cutout_center[0] * centre_oversampled / centre
            #yc_oversampled = extracted_stars[i].cutout_center[1] * centre_oversampled / centre

            mean, median, std = sigma_clipped_stats(star_zoom, sigma=3.0)
            threshold = median + (5. * std)
            tbl = find_peaks(star_zoom, threshold, box_size=5, border_width=5, npeaks=1, centroid_func=centroid_com)
            #tbl = find_peaks(star_zoom, threshold, box_size=5, border_width=5, npeaks=1, centroid_func=centroid_2dg)

            dx = centre_oversampled - tbl['x_centroid']#xc_oversampled
            dy = centre_oversampled - tbl['y_centroid']#yc_oversampled

            star_zoom_shift = shift(star_zoom, shift=(dy,dx))

            #star_norm = extracted_stars[i].data / np.sum(extracted_stars[i].data)
            star_norm = star_zoom_shift  / np.sum(star_zoom_shift)

            stacked_stars = np.dstack((stacked_stars,star_norm))

        _, epsf, _ = sigma_clipped_stats(stacked_stars, maxiters=None, axis=2)

        return epsf / np.sum(epsf)



    def compute_psf (self, oversampling_factor) :
        '''
        Method to compute the empirical PSF (ePSF)
        and save it in a fits file.
        '''

        # Exclude stars too close to the image border
        x = self.tbl['xcentroid']
        y = self.tbl['ycentroid']
        size = 25
        hsize = (size - 1) / 2
        mask = ((x > hsize) & (x < (self.data.shape[1] -1 - hsize)) &
            (y > hsize) & (y < (self.data.shape[0] -1 - hsize)))

        self.tbl = self.tbl[mask]

        # select the 50 brightest stars in case there are more left
        if len(self.tbl) > 50 :
            n = 50
            idx = sorted(list((-self.tbl['flux'].data).argsort()[:n]))
            self.tbl = self.tbl[idx]

        with open('PSF_exclude_stars.txt', 'r') as file :
            for line in file :
                cols = line.split()
                if cols[0]==self.filename.name :
                    rows = np.array([int(num)-1 for num in cols[1:]])
                    print('\nStars number to remove %s :'%rows)
                    break
                else :
                	rows = np.array([])

        # Create a table with the centroids pixel positions
        stars_tbl = Table()
        stars_tbl['x'] = self.tbl['xcentroid']
        stars_tbl['y'] = self.tbl['ycentroid']

        # define nddata object and extract the ePSF stars
        nddata = NDData(data=self.data)
        self.extracted_stars = extract_stars(nddata, stars_tbl, size=size)

        # 2 plots of 25 stars
        nrows = 5
        ncols = 5

        for j in range(2) :

            fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(7, 7), squeeze=True)

            ax = ax.ravel()

            zscale = ZScaleInterval()

            for i in range(nrows*ncols*j, nrows*ncols*(j+1)):

                if i < 25 :
                    k = i
                else:
                    k = i - 25

                if i <= len(self.extracted_stars)-1 :
                    #vmin, vmax = zscale.get_limits(self.extracted_stars[i])
                    #ax[i].imshow(self.extracted_stars[i], origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
                    norm = simple_norm(self.extracted_stars[i], 'log', percent=99.)
                    ax[k].imshow(self.extracted_stars[i], origin='lower', cmap='viridis', norm=norm)
                    ax[k].set_title('star %s'%(i+1), fontsize=12)
                    ax[k].axes.get_xaxis().set_visible(False)
                    ax[k].axes.get_yaxis().set_visible(False)
                    if (rows.size > 0) * (i in rows) :
                        x = np.linspace(0, size-1, 100)
                        ax[k].plot(x, x, '-r', lw=1.5)
                        ax[k].plot(x, x[::-1], '-r', lw=1.5)
                else :
                    ax[k].axis('off')

            fig.suptitle('%s %s %s'%(self.obj,self.filt,self.filenum))

            plt.subplots_adjust (left=0.05, right=0.95, bottom=0.05, top=0.90, wspace=0.3, hspace=0.3)

            # plt.show() # it was creating issues with running on astronode for a particular galaxy, not sure why

            self.psf_path.joinpath('figures').mkdir(exist_ok=True)
            fig.savefig(self.psf_path.joinpath('figures','%s_stars_%s.png'%(self.filename.stem,j+1)))

            plt.close()

        # remove rejected stars
        if rows.size > 0 :
            self.tbl.remove_rows(rows)
            stars_tbl = Table()
            stars_tbl['x'] = self.tbl['xcentroid']
            stars_tbl['y'] = self.tbl['ycentroid']
            self.extracted_stars = extract_stars(nddata, stars_tbl, size=size)

        # Build the ePSF
        self.epsf = self.sigma_clipped_epsf(self.extracted_stars, oversampling_factor)

        #if self.epsf != None :    # Catch the error if the epsf fails (zero valid stars)
        if self.extracted_stars :    # Catch the error if the epsf fails (zero valid stars)

            # Plot the resulting ePSF
            fig, ax = plt.subplots (figsize=(6.92, 7))
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size = "5%", pad = 0.05)
            datacut = self.epsf
            vmin, vmax = zscale.get_limits(datacut)
            im = ax.imshow(datacut, origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
            cbar = plt.colorbar(im, cax = cax)
            cbar.mappable.set_clim(vmin, vmax)
            ax.set_title('%s %s %s'%(self.obj,self.filt,self.filenum))
            plt.subplots_adjust (left=0.10, right=0.75, bottom=0.10, top=0.90)
            # plt.show() # it was creating issues with running on astronode for a particular galaxy, not sure why
            fig.savefig(self.psf_path.joinpath('figures','%s_psf.png'%self.filename.stem))
            plt.close()
            self.psf_path.joinpath('psf_fits').mkdir(exist_ok=True)
            self.psf_model_filename = str(self.psf_path.joinpath('psf_fits','%s_psf_%sx%s.fits'%(self.filename.stem,oversampling_factor,oversampling_factor)))
            fits.writeto(self.psf_model_filename, datacut, overwrite=True)

        else :

            print('\n\n########################################')
            print('PSF not computed for %s %s %s'%(self.obj,self.filt,self.filenum))
            print('########################################\n\n')


    def compute_convolved_psf (self, oversampling_factor) :
        '''
        Method to compute the empirical PSF (ePSF)
        and save it in a fits file.
        '''

        # Exclude stars too close to the image border
        x = self.tbl['xcentroid']
        y = self.tbl['ycentroid']
        size = 25
        hsize = (size - 1) / 2
        mask = ((x > hsize) & (x < (self.data.shape[1] -1 - hsize)) &
            (y > hsize) & (y < (self.data.shape[0] -1 - hsize)))

        self.tbl = self.tbl[mask]

        # select the 50 brightest stars in case there are more left
        if len(self.tbl) > 50 :
            n = 50
            idx = sorted(list((-self.tbl['flux'].data).argsort()[:n]))
            self.tbl = self.tbl[idx]

        # Create a table with the centroids pixel positions
        stars_tbl = Table()
        stars_tbl['x'] = self.tbl['xcentroid']
        stars_tbl['y'] = self.tbl['ycentroid']

        # define nddata object and extract the ePSF stars
        nddata = NDData(data=self.data)
        self.extracted_stars = extract_stars(nddata, stars_tbl, size=size)

        # Build the ePSF
        self.epsf = self.sigma_clipped_epsf(self.extracted_stars, oversampling_factor)
        if self.instr == 'WFI' :
            epsf_filename = Path('psf_match/psf_fits/').joinpath('%s_%s_stack%s_calib_nosky_psf_%sx%s_conv.fits'%(self.obj, self.filt, self.filenum,oversampling_factor,oversampling_factor))
        if self.instr == 'CCD' :
            epsf_filename = Path('psf_match/psf_fits/').joinpath('%s_%s_%s_wcs_reproj_calib_psf_%sx%s_conv.fits'%(self.obj, self.filt, self.filenum,oversampling_factor,oversampling_factor))
        fits.writeto(epsf_filename, self.epsf, overwrite=True)

        # Open the ePSF before convolution
        if self.instr == 'WFI' :
            psf_fits = Path('psf_match/psf_fits/').joinpath('%s_%s_stack%s_calib_nosky_psf_1x1.fits'%(self.obj, self.filt, self.filenum))
        if self.instr == 'CCD' :
            psf_fits = Path('psf_match/psf_fits/').joinpath('%s_%s_%s_wcs_reproj_calib_psf_1x1.fits'%(self.obj, self.filt, self.filenum))
        epsf_noconv = fits.open(psf_fits)[0].data

        # Open the Gaussian model
        gauss_kernel_fits = Path('psf_match/psf_fits/').joinpath('%s_%s_gauss2Dmodel.fits'%(self.obj,self.instr))
        gauss_kernel_data = fits.open(gauss_kernel_fits)[0].data

        # Open the kernel
        kernels = sorted(Path('psf_match/psf_fits').glob('%s_%s_kernel_frame%s_to_???.fits'%(self.obj, self.filt, self.filenum)))
        if kernels :
            path_to_kernel = kernels[0]
            kernel = fits.open(path_to_kernel)[0].data
        else :
            sys.exit('\nKernel not found!\n')

        # Compute the residual image
        epsf_image = Image(str(epsf_filename))
        rebinned = epsf_image.rebin(oversampling_factor)
        epsf_image_norm = rebinned.data.data / np.sum(rebinned.data.data)
        residual = (epsf_image_norm - gauss_kernel_data)
        norm_residual = (epsf_image_norm - gauss_kernel_data)/(gauss_kernel_data+0.0001)

        # Open

        if self.extracted_stars :    # Catch the error if the epsf fails (zero valid stars)

            # Plot 1 (PSF plots)

            fig, ax = plt.subplots(2, 3, figsize=(14, 7))

            ax1, ax2, ax3, ax4, ax5, ax6 = ax.flatten()

            zscale = ZScaleInterval()

            # ePSF before convolution

            divider = make_axes_locatable(ax1)
            cax = divider.append_axes("right", size = "5%", pad = 0.05)
            norm = simple_norm(epsf_noconv, 'log', percent=99.)
            im = ax1.imshow(epsf_noconv, origin='lower', cmap='viridis', norm=norm)
            cbar = plt.colorbar(im, cax = cax)
            cbar.ax.tick_params(labelsize=10)
            labels = cbar.ax.get_yticklabels()
            for i in range(len(labels)) :
                if (i%2 == 0) * (i !=0) :
                    labels[-i].set_text('')
            cbar.ax.set_yticklabels(labels)

            # Target PSF (Model)

            divider = make_axes_locatable(ax2)
            cax = divider.append_axes("right", size = "5%", pad = 0.05)
            norm = simple_norm(gauss_kernel_data, 'log', percent=99.)
            im = ax2.imshow(gauss_kernel_data, origin='lower', cmap='viridis', norm=norm)
            cbar = plt.colorbar(im, cax = cax)
            cbar.ax.tick_params(labelsize=10)
            labels = cbar.ax.get_yticklabels()
            for i in range(len(labels)) :
                if (i%2 == 0) * (i !=0) :
                    labels[-i].set_text('')
            cbar.ax.set_yticklabels(labels)

            # Kernel to convolve the source PSF (ePSF before convolution) to the target PSF (model)

            divider = make_axes_locatable(ax3)
            cax = divider.append_axes("right", size = "5%", pad = 0.05)
            norm = simple_norm(kernel, 'log', percent=99.)
            im = ax3.imshow(kernel, origin='lower', cmap='viridis', norm=norm)
            cbar = plt.colorbar(im, cax = cax)
            cbar.ax.tick_params(labelsize=10)
            labels = cbar.ax.get_yticklabels()
            for i in range(len(labels)) :
                if (i%2 == 0) * (i !=0) :
                    labels[-i].set_text('')
            cbar.ax.set_yticklabels(labels)

            # ePSF after convolution

            divider = make_axes_locatable(ax4)
            cax = divider.append_axes("right", size = "5%", pad = 0.05)
            norm = simple_norm(epsf_image_norm, 'log', percent=99.)
            im = ax4.imshow(epsf_image_norm, origin='lower', cmap='viridis', norm=norm)
            cbar = plt.colorbar(im, cax = cax)
            cbar.ax.tick_params(labelsize=10)
            labels = cbar.ax.get_yticklabels()
            for i in range(len(labels)) :
                if (i%2 == 0) * (i !=0) :
                    labels[-i].set_text('')
            cbar.ax.set_yticklabels(labels)

            # Residuals

            divider = make_axes_locatable(ax5)
            cax = divider.append_axes("right", size = "5%", pad = 0.05)
            #norm = simple_norm(residual, 'log', percent=99.)
            #im = ax5.imshow(residual, origin='lower', cmap='viridis', norm=norm)
            vmin, vmax = zscale.get_limits(residual)
            im = ax5.imshow(residual, origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
            cbar = plt.colorbar(im, cax = cax)
            cbar.ax.tick_params(labelsize=10)
            #labels = cbar.ax.get_yticklabels()
            #for i in range(len(labels)) :
            #    if (i%2 == 0) * (i !=0) :
            #        labels[-i].set_text('')
            #cbar.ax.set_yticklabels(labels)

            # Normalized residuals

            divider = make_axes_locatable(ax6)
            cax = divider.append_axes("right", size = "5%", pad = 0.05)
            #norm = simple_norm(norm_residual, 'log', percent=99.)
            #im = ax6.imshow(norm_residual, origin='lower', cmap='viridis', norm=norm)
            vmin, vmax = zscale.get_limits(norm_residual)
            im = ax6.imshow(norm_residual, origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
            cbar = plt.colorbar(im, cax = cax)
            cbar.ax.tick_params(labelsize=10)
            #labels = cbar.ax.get_yticklabels()
            #for i in range(len(labels)) :
            #    if (i%2 == 0) * (i !=0) :
            #        labels[-i].set_text('')
            #cbar.ax.set_yticklabels(labels)

            ax1.set_title('Source PSF', fontsize=14)
            ax1.xaxis.set_visible(False)
            ax1.set_ylabel('y [pix]')

            ax2.set_title('Model', fontsize=14)
            ax2.xaxis.set_visible(False)
            ax2.yaxis.set_visible(False)

            ax3.set_title('Kernel', fontsize=14)
            ax3.xaxis.set_visible(False)
            ax3.yaxis.set_visible(False)

            ax4.set_title('PSF', fontsize=14)
            ax4.set_xlabel('x [pix]')
            ax4.set_ylabel('y [pix]')

            ax5.set_title('PSF - Model', fontsize=14)
            ax5.set_xlabel('x [pix]')
            ax5.yaxis.set_visible(False)

            ax6.set_title('(PSF - Model) / Model', fontsize=14)
            ax6.set_xlabel('x [pix]')
            ax6.yaxis.set_visible(False)

            fig.suptitle('%s %s %s'%(self.obj,self.filt,self.filenum))

            fig.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.90, hspace=0.2, wspace=0.2)

            self.psf_path.joinpath('check_plot').mkdir(exist_ok=True)

            fig.savefig(str(self.psf_path.joinpath('check_plot','%s_psf_check_plot.png'%self.filename.stem)))

            plt.close()

            # Plot 2 (1D plot)

            center_pixels = 12    # with python indexing from 0 to 24 the central pixel is 12
            epsf_x = epsf_image_norm[:, center_pixels]
            epsf_noconv_x = epsf_noconv[:, center_pixels]
            residuals_noconv = epsf_noconv - gauss_kernel_data
            residuals_noconv_x = residuals_noconv[:, center_pixels]
            model_x = gauss_kernel_data[:, center_pixels]
            residuals_x = residual[:, center_pixels]
            pixels = np.arange(epsf_image_norm.shape[0])

            fig, ax1 = plt.subplots(1, 1, figsize=(7, 7))

            ax1.plot(pixels, epsf_x, 'o-', c='tab:red', label='Convolved PSF', alpha=0.6)
            ax1.plot(pixels, epsf_noconv_x, 'o-', c='tab:blue', label='Source PSF', alpha=0.6)
            ax1.plot(pixels, model_x, 'o-', c='k', label='Model', alpha=0.7)
            #ax1.set_yscale('log')
            ax1.legend(loc='best')
            ax1.set_ylabel('normalized flux')
            ax1.set_title('%s %s %s'%(self.obj,self.filt,self.filenum))

            ax1.xaxis.set_visible(False)

            divider = make_axes_locatable(ax1)
            ax2 = divider.append_axes("bottom", size = "30%", pad = 0.0)
            ax2.plot(pixels, residuals_x, 'o-', c='tab:red', label='residuals', alpha=0.6)
            ax2.plot(pixels, residuals_noconv_x, 'o-', c='tab:blue', label='residuals', alpha=0.6)
            ax2.axhline(y=0, color='k', ls=':', lw=0.8)
            ax2.set_xlabel('x [pix]')
            ax2.set_ylabel('residuals')

            fig.savefig(str(self.psf_path.joinpath('check_plot','%s_psf_check_plot_1D.png'%self.filename.stem)))

            plt.close()


        else :

            print('\n\n########################################')
            print('PSF not computed for %s %s %s'%(self.obj,self.filt,self.filenum))
            print('########################################\n\n')



    def save_tbl (self) :

        self.psf_path.joinpath('tables').mkdir(exist_ok=True)
        tbl_filename = str(self.psf_path.joinpath('tables','%s_tbl.dat'%self.filename.stem))
        self.tbl.write(tbl_filename, format='ascii.fast_commented_header', overwrite=True)


    def fit_psf (self, oversampling_factor) :
        '''
        The ePSF is fitted with a Gaussian and a Moffat function.
        The image ePSF is then rebinned to the original resolution
        by rebinning with the same factor as for the oversampling.
        '''

        print('\n\nFitting %s...\n\n'%self.psf_model_filename)

        zscale = ZScaleInterval()

        # MOFFAT fit

        image = Image(str(self.psf_model_filename))
        shape = np.array(image.shape)
        self.moff_fit = image.moffat_fit(center=shape/2, circular=False, fit_back=False,
                                peak=True, unit_center=None, unit_fwhm=None,
                                full_output=True)

        # plot (MOFFAT)

        residual = (image - self.moff_fit.ima)/(self.moff_fit.ima+0.0001)

        fig, ax = plt.subplots(1, 3, figsize=(14, 7))

        divider = make_axes_locatable(ax[0])
        cax = divider.append_axes("right", size = "5%", pad = 0.05)
        norm = simple_norm(image.data.data, 'log', percent=99.)
        im = ax[0].imshow(image.data.data, origin='lower', cmap='viridis', norm=norm)
        cbar = plt.colorbar(im, cax = cax)
        cbar.ax.tick_params(labelsize=10)
        labels = cbar.ax.get_yticklabels()
        for i in range(len(labels)) :
        	if (i%2 == 0) * (i !=0) :
        		labels[-i].set_text('')
        cbar.ax.set_yticklabels(labels)

        divider = make_axes_locatable(ax[1])
        cax = divider.append_axes("right", size = "5%", pad = 0.05)
        norm = simple_norm(self.moff_fit.ima.data.data, 'log', percent=99.)
        im = ax[1].imshow(self.moff_fit.ima.data.data, origin='lower', cmap='viridis', norm=norm)
        cbar = plt.colorbar(im, cax = cax)
        cbar.ax.tick_params(labelsize=10)
        labels = cbar.ax.get_yticklabels()
        for i in range(len(labels)) :
        	if (i%2 == 0) * (i !=0) :
        		labels[-i].set_text('')
        cbar.ax.set_yticklabels(labels)

        divider = make_axes_locatable(ax[2])
        cax = divider.append_axes("right", size = "5%", pad = 0.05)
        norm = simple_norm(residual.data.data, 'log', percent=99.)
        im = ax[2].imshow(residual.data.data, origin='lower', cmap='viridis', norm=norm)
        cbar = plt.colorbar(im, cax = cax)
        cbar.ax.tick_params(labelsize=10)
        labels = cbar.ax.get_yticklabels()
        for i in range(len(labels)) :
        	if (i%2 == 0) * (i !=0) :
        		labels[-i].set_text('')
        cbar.ax.set_yticklabels(labels)

        ax[0].set_title('Original')
        ax[1].set_title('Moffat Fit')
        ax[2].set_title('(orig - fit) / fit')
        fig.suptitle('%s %s %s'%(self.obj,self.filt,self.filenum))

        fig.subplots_adjust(left=0.05, right=0.95, wspace=0.3)

        fig.savefig(self.psf_path.joinpath('figures','%s_moffatfit.png'%self.filename.stem))

        plt.close()

        # Gaussian fit

        self.gauss_fit = image.gauss_fit(center=shape/2, circular=False, fit_back=False,
                               peak=True, unit_center=None, unit_fwhm=None,
                               full_output=True)

        # plot (Gaussian)

        residual = (image - self.gauss_fit.ima)/(self.gauss_fit.ima+0.0001)

        fig, ax = plt.subplots(1, 3, figsize=(14, 7))

        divider = make_axes_locatable(ax[0])
        cax = divider.append_axes("right", size = "5%", pad = 0.05)
        norm = simple_norm(image.data.data, 'log', percent=99.)
        im = ax[0].imshow(image.data.data, origin='lower', cmap='viridis', norm=norm)
        cbar = plt.colorbar(im, cax = cax)
        cbar.ax.tick_params(labelsize=10)
        labels = cbar.ax.get_yticklabels()
        for i in range(len(labels)) :
        	if (i%2 == 0) * (i !=0) :
        		labels[-i].set_text('')
        cbar.ax.set_yticklabels(labels)

        divider = make_axes_locatable(ax[1])
        cax = divider.append_axes("right", size = "5%", pad = 0.05)
        norm = simple_norm(self.gauss_fit.ima.data.data, 'log', percent=99.)
        im = ax[1].imshow(self.gauss_fit.ima.data.data, origin='lower', cmap='viridis', norm=norm)
        cbar = plt.colorbar(im, cax = cax)
        cbar.ax.tick_params(labelsize=10)
        labels = cbar.ax.get_yticklabels()
        for i in range(len(labels)) :
        	if (i%2 == 0) * (i !=0) :
        		labels[-i].set_text('')
        cbar.ax.set_yticklabels(labels)

        divider = make_axes_locatable(ax[2])
        cax = divider.append_axes("right", size = "5%", pad = 0.05)
        norm = simple_norm(residual.data.data, 'log', percent=99.)
        im = ax[2].imshow(residual.data.data, origin='lower', cmap='viridis', norm=norm)
        cbar = plt.colorbar(im, cax = cax)
        cbar.ax.tick_params(labelsize=10)
        labels = cbar.ax.get_yticklabels()
        for i in range(len(labels)) :
        	if (i%2 == 0) * (i !=0) :
        		labels[-i].set_text('')
        cbar.ax.set_yticklabels(labels)

        ax[0].set_title('Original')
        ax[1].set_title('Gaussian Fit')
        ax[2].set_title('(orig - fit) / fit')
        fig.suptitle('%s %s %s'%(self.obj,self.filt,self.filenum))

        fig.subplots_adjust(left=0.05, right=0.95, wspace=0.3)

        fig.savefig(self.psf_path.joinpath('figures','%s_gaussfit.png'%self.filename.stem))

        plt.close()

        # plot (1D comparison)

        center_moff = np.int(self.moff_fit.center[1])
        image_x = image[:, center_moff].data
        moff_x = self.moff_fit.ima[:, center_moff].data
        gauss_x = self.gauss_fit.ima[:, center_moff].data
        pix = np.arange(shape[0])

        fig, ax = plt.subplots(1, 1, figsize=(7,7))
        ax.plot(pix, image_x, c='k', label='original')
        ax.scatter(pix, moff_x, c='red', label='moff')
        ax.scatter(pix, gauss_x, c='blue', label='gauss')
        ax.set_yscale('log')
        plt.legend(loc='best')
        ax.set_title('%s %s %s'%(self.obj,self.filt,self.filenum))

        fig.savefig(self.psf_path.joinpath('figures','%s_1Dfit_comp.png'%self.filename.stem))

        plt.close()

        print('\nRebinning %s...\n\n'%self.psf_model_filename)

        rebinned = image.rebin(oversampling_factor)
        rebinned_norm = rebinned.data.data / np.sum(rebinned.data.data)
        fits.writeto(str(self.psf_model_filename).replace('%sx%s'%(oversampling_factor,oversampling_factor),'1x1'), rebinned_norm, overwrite=True)


    def set_fwhm (self, psfs_dic, oversampling_factor) :
        '''
        Set the maximum FWHM, the major and minor axis FWHM and the angular
        rotation of the PSF by using the MOFFAT and GAUSSIAN fit results.
        These results are stored in a dictionary of lists.
        '''

        # Copute FWHM in arcsec

        if self.instr == 'WFI' :
            self.pixscale = 0.238 * u.arcsec/u.pixel
        if self.instr == 'CCD' :
            self.pixscale = 0.259 * u.arcsec/u.pixel

        self.moffat_fwhm = np.round(np.max(self.moff_fit.fwhm*u.pixel) / oversampling_factor * self.pixscale, 5)
        self.a_moffat_fwhm = np.round(self.moff_fit.fwhm[0]*u.pixel / oversampling_factor * self.pixscale, 5)
        self.b_moffat_fwhm = np.round(self.moff_fit.fwhm[1]*u.pixel / oversampling_factor * self.pixscale, 5)
        self.ang_moffat_fwhm = np.round(self.moff_fit.rot*u.deg, 5)

        self.gauss_fwhm = np.round(np.max(self.gauss_fit.fwhm*u.pixel) / oversampling_factor * self.pixscale, 5)
        self.a_gauss_fwhm = np.round(self.gauss_fit.fwhm[0]*u.pixel / oversampling_factor * self.pixscale, 5)
        self.b_gauss_fwhm = np.round(self.gauss_fit.fwhm[1]*u.pixel / oversampling_factor * self.pixscale, 5)
        self.ang_gauss_fwhm = np.round(self.gauss_fit.rot*u.deg, 5)

        psfs_dic['moffat_psfs'].append(self.moffat_fwhm.value)
        psfs_dic['a_moffat_psfs'].append(self.a_moffat_fwhm.value)
        psfs_dic['b_moffat_psfs'].append(self.b_moffat_fwhm.value)
        psfs_dic['ang_moffat_psfs'].append(self.ang_moffat_fwhm.value)

        psfs_dic['gauss_psfs'].append(self.gauss_fwhm.value)
        psfs_dic['a_gauss_psfs'].append(self.a_gauss_fwhm.value)
        psfs_dic['b_gauss_psfs'].append(self.b_gauss_fwhm.value)
        psfs_dic['ang_gauss_psfs'].append(self.ang_gauss_fwhm.value)

        psfs_dic['filenums'].append(self.filenum)
        psfs_dic['filters'].append(self.filt)


class GAIAStars (object) :

    def __init__ (self, obj, instr, my_catname = 'Gaia-DR2') :

        self.obj = obj
        self.instr = instr
        self.catname = my_catname


    def gaia_query (self) :

        new_limit = 60  # sets the timeout to 60s
        with conf.set_temp('remote_timeout', new_limit):
            obj_coord = SkyCoord.from_name(self.obj)

        vizier = Vizier(columns=['*', 'E(BR/RP)'], catalog='Gaia-DR2', row_limit=-1, timeout=60)
        result = vizier.query_region(obj_coord, radius=1*u.deg)

        gaia_tbl = result[0]

        self.coord = SkyCoord(ra=gaia_tbl['RA_ICRS'], dec=gaia_tbl['DE_ICRS'])



def save_dic (gal, dic, instr, psf_path) :
    '''
    Global function to save the PSFs dictionary to text file.
    '''

    psf_path.joinpath('psf_tables').mkdir(exist_ok=True)

    txt_file = str(psf_path.joinpath('psf_tables','%s_%s_psfs_dict.txt'%(gal,instr)))

    with open(txt_file, 'w') as f :
        f.write('# Name    Filter    Frame    moff_PSF    a_moff_PSF    b_moff_PSF    ang_moff_PSF    gauss_PSF    a_gauss_PSF    b_gauss_PSF    ang_gauss_PSF\n')
        f.write('#                            [arcsec]    [arcsec]      [arcsec]       [degree]       [arcsec]    [arcsec]        [arcsec]       [degree]\n')

    for i in range(len(dic['moffat_psfs'])) :

    	with open(txt_file, 'a') as f :
    		f.write('%s    %s    %s      %s       %s       %s       %s       %s       %s       %s       %s\n'
    			%(gal, dic['filters'][i], dic['filenums'][i], dic['moffat_psfs'][i],
    				dic['a_moffat_psfs'][i], dic['b_moffat_psfs'][i], dic['ang_moffat_psfs'][i],
    				dic['gauss_psfs'][i], dic['a_gauss_psfs'][i], dic['b_gauss_psfs'][i],
                    dic['ang_gauss_psfs'][i]))


def compute_kernels (gal, instr) :
    '''
    Global function to compute the kernels to PSF match
    N images to the target PSF that is a Gaussian 2D
    kernel a pixscale^2 bigger than the largest PSF
    between the images.

    Both worst resolution kernels among a set a frames
    in all the filters and among the a set of frames in
    only broad-band filters are created.

    Note: we assume a gaussian PSF and the kernel is then
    gaussian to convolve the source (gaussian) PSF to the
    target (gaussian) PSF.
    '''

    # Compute the lowest resolution PSF

    myfilt, framenums, moffat_fwhms, gauss_fwhms = np.loadtxt('psf_match/psf_tables/%s_%s_psfs_dict.txt'
        %(gal,instr), dtype=str , usecols=(1,2,3,7), unpack=True)

    framenums = np.atleast_1d(framenums).astype('int')
    moffat_fwhms = np.atleast_1d(moffat_fwhms).astype('float')
    gauss_fwhms = np.atleast_1d(gauss_fwhms).astype('float')

    max_fwhm = np.max(gauss_fwhms)
    #max_fwhm = np.max(moffat_fwhms)
    if instr == 'WFI' :
        max_fwhm_BB = np.max(gauss_fwhms[myfilt=='Rc'])
        #max_fwhm_BB = np.max(moffat_fwhms[myfilt=='Rc'])
        pixscale = 0.238
    if instr == 'CCD' :
        max_fwhm_BB = np.max(gauss_fwhms[myfilt=='r'])
        #max_fwhm_BB = np.max(moffat_fwhms[myfilt=='r'])
        pixscale = 0.259

    # Create the Gaussian 2D kernel out of the lowest resolution PSF (target PSF)

    gaussian_fwhm = np.sqrt(max_fwhm**2. + pixscale**2.)
    gaussian_fwhm_BB = np.sqrt(max_fwhm_BB**2. + pixscale**2.)
    gauss_fwhm_string = ('%.2f'%gaussian_fwhm).replace('.','')
    gauss_fwhm_BB_string = ('%.2f'%gaussian_fwhm_BB).replace('.','')

    target_std = gaussian_fwhm/pixscale * gaussian_fwhm_to_sigma
    target_std_BB = gaussian_fwhm_BB/pixscale * gaussian_fwhm_to_sigma

    gauss_kernel = Gaussian2DKernel(x_stddev=target_std, x_size=25, y_size=25)
    gauss_kernel_BB = Gaussian2DKernel(x_stddev=target_std_BB, x_size=25, y_size=25)

    gauss_kernel.normalize()
    gauss_kernel_BB.normalize()

    gauss_kernel_fits = Path('psf_match/psf_fits/').joinpath('%s_%s_gauss2Dmodel.fits'%(gal,instr))
    gauss_kernel_fits_BB = Path('psf_match/psf_fits/').joinpath('%s_%s_gauss2Dmodel_BB.fits'%(gal,instr))

    fits.writeto(gauss_kernel_fits, data=gauss_kernel.array, overwrite=True)
    fits.writeto(gauss_kernel_fits_BB, data=gauss_kernel_BB.array, overwrite=True)

    # Main loop into the filter names

    if instr == 'WFI' :
        filters = ['Rc', 'Ha']
    if instr == 'CCD' :
        filters = ['r', 'Ha657', 'Ha663']

    for filt in filters :

        # Compute the kernels

        for num, fwhm in zip(framenums[myfilt==filt], gauss_fwhms[myfilt==filt]) :
        #for num, fwhm in zip(framenums[myfilt==filt], moffat_fwhms[myfilt==filt]) :

            new_kernel_filename = 'psf_match/psf_fits/%s_%s_kernel_frame%s_to_%s.fits'%(gal, filt, num, gauss_fwhm_string)

            x_std = fwhm/pixscale * gaussian_fwhm_to_sigma

            kernel_std = np.sqrt(target_std**2.-x_std**2.)

            new_kernel = Gaussian2DKernel(x_stddev=kernel_std, x_size=25, y_size=25)
            new_kernel.normalize()

            fits.writeto(new_kernel_filename, data=new_kernel.array, overwrite=True)


        if (filt=='Rc') + (filt=='r') :

            for num, fwhm in zip(framenums[myfilt==filt], gauss_fwhms[myfilt==filt]) :
            #for num, fwhm in zip(framenums[myfilt==filt], moffat_fwhms[myfilt==filt]) :

                # Compute the kernels for the best BB resolution

                new_kernel_filename_BB = 'psf_match/psf_fits/%s_%s_kernel_frame%s_to_BB%s.fits'%(gal, filt, num, gauss_fwhm_BB_string)

                x_std = fwhm/pixscale * gaussian_fwhm_to_sigma

                kernel_std_BB = np.sqrt(target_std_BB**2.-x_std**2.)

                new_kernel_BB = Gaussian2DKernel(x_stddev=kernel_std_BB, x_size=25, y_size=25)
                new_kernel_BB.normalize()

                fits.writeto(new_kernel_filename_BB, data=new_kernel_BB.array, overwrite=True)



def convolve_frame(scienceframe, errorframe, mask, gal, filt, instr) :
    '''
    This is the main function to perform the convolution of each frame.
    '''

    pos = [i for i, c in enumerate(scienceframe.name) if c=='_']

    if instr == 'WFI' :

        filenum = int(scienceframe.name[pos[1]+6:pos[2]])

    if instr == 'CCD' :

        filenum = int(scienceframe.name[pos[1]+1:pos[2]])

    data_sci = fits.open(scienceframe)[0].data
    data_err = fits.open(errorframe)[0].data
    data_wcs = WCS(fits.open(scienceframe)[0].header)

    # Convolve to the worst BB + NB resolution

    kernels = sorted(Path('psf_match/psf_fits').glob('%s_%s_kernel_frame%s_to_???.fits'%(gal, filt, filenum)))

    if kernels :

        path_to_kernel = kernels[0]

        print('\nKernel %s found ...\n'%path_to_kernel)

        pos = [i for i, c in enumerate(path_to_kernel.name) if c=='_']
        res = int(path_to_kernel.name[pos[-1]+1:pos[-1]+4])
        res_string = path_to_kernel.name[pos[-1]+1:pos[-1]+4]

        kernel = Kernel(fits.open(path_to_kernel)[0].data)

        conv_sci = convolve(data_sci, kernel, mask=mask, normalize_kernel=True, preserve_nan=True)
        kernel.normalize()    # normalize the kernel before squaring it
        conv_err = np.sqrt(convolve(data_err**2., kernel.array**2., mask=mask, normalize_kernel=False, preserve_nan=True))

        conv_sci_filename = Path(str(scienceframe).replace('.fits','_conv_%s.fits'%res_string))
        conv_err_filename = Path(str(errorframe).replace('.fits','_conv_%s.fits'%res_string))

        header = fits.open(scienceframe)[0].header
        fits.writeto(conv_sci_filename, data=conv_sci, header=header, overwrite=True)
        fits.writeto(conv_err_filename, data=conv_err, header=header, overwrite=True)

    else :

        sys.exit('\nKernel not found!\n')

    # Convolve to the worst BB resolution

    if ((filt=='Rc') + (filt=='r')) :

        kernels = sorted(Path('psf_match/psf_fits').glob('%s_%s_kernel_frame%s_to_BB???.fits'%(gal, filt, filenum)))

        if kernels :

            path_to_kernel = kernels[0]

            print('\nKernel %s found ...\n'%path_to_kernel)

            pos = [i for i, c in enumerate(path_to_kernel.name) if c=='_']
            res_bb = int(path_to_kernel.name[pos[-1]+3:pos[-1]+6])
            res_bb_string = path_to_kernel.name[pos[-1]+3:pos[-1]+6]

            conv_sci_filename_bb = Path(str(scienceframe).replace('.fits','_BBconv_%s.fits'%res_bb_string))
            conv_err_filename_bb = Path(str(errorframe).replace('.fits','_BBconv_%s.fits'%res_bb_string))

            if not (res == res_bb) :

                kernel = Kernel(fits.open(path_to_kernel)[0].data)

                conv_sci = convolve(data_sci, kernel, mask=mask, normalize_kernel=True, preserve_nan=True)
                kernel.normalize()    # normalize the kernel before squaring it
                conv_err = np.sqrt(convolve(data_err**2., kernel.array**2., mask=mask, normalize_kernel=False, preserve_nan=True))

                header = fits.open(scienceframe)[0].header
                fits.writeto(conv_sci_filename_bb, data=conv_sci, header=header, overwrite=True)
                fits.writeto(conv_err_filename_bb, data=conv_err, header=header, overwrite=True)

            else :

                copy(conv_sci_filename, conv_sci_filename_bb)
                copy(conv_err_filename, conv_err_filename_bb)

        else :

            sys.exit('\nKernel not found!\n')



def convolution_check_plot(p, psf_path, gal, filt, instr, scienceframe, maskframe, errorframe, fwhm, oversampling_factor) :
    '''
    The function compute the ePSF of the convolved frame and plot the residuals
    between the gaussian model and the ePSF.
    '''

    # Instantiate ImageStars class
    stars = ImageStars(p, psf_path, gal, filt, instr, scienceframe, maskframe, errorframe, fwhm)

    # Find stars in the image
    stars.find_image_stars()

    # Instantiate Gaia stars
    gs = GAIAStars(stars.obj, stars.instr)
    gs.gaia_query()

    # Match stars found in the stacked image (common projection) with Gaia stars in the image field
    stars.match_with_gaia(gs)

    # Compute and plot the ePSF
    stars.compute_convolved_psf(oversampling_factor)




if __name__ == "__main__":

    start_time = timeit.default_timer()

    wfi_galaxies = list(np.loadtxt('PHANGS_WFI_galaxies.txt', dtype='str', unpack=True))

    ccd_galaxies = ['NGC0685', 'NGC1317', 'NGC1326', 'IC1954', 'IC1993', 'NGC1511', 'NGC1546', 'NGC1559',
                      'NGC1809',
                      'NGC1792', 'NGC2090', 'NGC2283', 'NGC2566', 'NGC2775', 'NGC3059', 'NGC3507',
                        'NGC3626', 'NGC4207', 'NGC4424', 'NGC4457', 'NGC4540', 'NGC4548', 'NGC4689', 'NGC4694',
                          'NGC4781', 'NGC4941', 'NGC4951', 'NGC5042', 'NGC5134', 'NGC5248', 'NGC5530', 'NGC5643',
                            'NGC6300', 'IC5273', 'NGC7456', 'NGC7496']

    parser = argparse.ArgumentParser()

    parser.add_argument('--wfi_path', action='store', type=str, default=None, help='path to calibrated WFI frames')
    parser.add_argument('--ccd_path', action='store', type=str,
        default='/home/arazza/NB_photometry/dirCCDreduction/DuPontDATA/PROCESSED/phot_calib/', help='path to calibrated du Pont frames')
    parser.add_argument('-w', '--wfi', nargs='*', type=str, default=wfi_galaxies,
        help='list of selected WFI galaxies (type a non-valid galaxy to avoid calibrating WFI)')
    parser.add_argument('-c', '--ccd', nargs='*', type=str, default=ccd_galaxies,
        help='list of selected CCD galaxies (type a non-valid galaxy to avoid calibrating CCD)')
    parser.add_argument('-f', '--fwhm', action='store', type=float, default=4.0, help='FWHM to be used for IRAFStarFinder')
    parser.add_argument('-e', '--ell_radius', action='store', default=1.4, help='galaxy ellipse increasing factor')
    parser.add_argument('--exclude_percentile', type=float, default=10., help='Percentile of masked pixel in a mesh to be valid for background computation')
    parser.add_argument('--dilate_size', type=int, default=11, help='Dilate size around the masked sources to compute the background')

    parser.add_argument('-o', '--oversampling_factor', action='store', type=int, default=1,
        help='oversampling factor for building the ePSF')

    parser.add_argument('--psf_fitting', action='store_true', default=False,
        help='extract stars, compute and fit the image ePSF')

    parser.add_argument('--compute_kernels', action='store_true', default=False,
        help='read the PSFs dictionary and compute the kernels')

    parser.add_argument('--wfi_filters', nargs='*', type=str, default=['Rc', 'Ha'], help='list of WFI filters')
    parser.add_argument('--ccd_filters', nargs='*', type=str, default=['r', 'Ha657', 'Ha663'], help='list of selected CCD filters')

    parser.add_argument('--bkg_subtraction', action='store_true', default=False,
        help='subtract the background from the CCD frames (do not use any --convolve_XXX argument to perform only the background subtraction)')

    parser.add_argument('--frame_number', type=int, default=1,
        help='starting frame number for the convolution (in case you have some initial frames already convolved)')

    parser.add_argument('--convolve_fullFOV', action='store_true', default=False,
        help='convolve WFI full-FOV background-subtracted frames with the computed kernels')

    parser.add_argument('--convolve_cutouts', action='store_true', default=False,
        help='subtract the background from WFI cutouts frames and convolve them with the computed kernels')

    parser.add_argument('--convolve_dp', action='store_true', default=False,
        help='subtract the background from CCD frames and convolve them with the computed kernels')

    parser.add_argument('--skip_bkg', action='store_true', default=False,
        help='skip background subtraction in the WFI cutouts or CCD frames convolution')

    args = parser.parse_args()

    config = ConfigParser()
    config.read('WFIpipeline.config')

    # Instantiate the ellipse radius for the on-the-fly background subtraction

    r = args.ell_radius

    # Instantiate the parameters for the SExtractor-like background

    exclude_percentile = args.exclude_percentile
    dilate_size = args.dilate_size

    # Instantiate the lists of galaxies

    lists_of_galaxies = [args.wfi, args.ccd]

    ##### Main loop #####

    for i,list_of_galaxy in enumerate(lists_of_galaxies) :

        # Check the lists

        if (i==0) * (not (set(list_of_galaxy) <= set(wfi_galaxies))) :

            print('\nThere is a galaxy not belonging to WFI data\n... skipping PSF computation for WFI ...\n\n')

            continue

        if (i==1) * (not (set(list_of_galaxy) <= set(ccd_galaxies))) :

            print('\nThere is a galaxy not belonging to CCD data\n... skipping PSF computation for CCD ...\n\n')

            continue

        # Instantiate the paths and start the loop for WFI

        if (i==0) * (set(list_of_galaxy) <= set(wfi_galaxies)) :

            wfi_galnames, wfi_paths = np.loadtxt('paths_to_calib_WFI.txt', dtype='str', unpack=True)

            wfi_galnames = np.atleast_1d(wfi_galnames)
            wfi_paths = np.atleast_1d(wfi_paths)

            outpath = Path(config.get('Paths','data_products'))

            for gal, p in zip(wfi_galnames, wfi_paths) :

                if gal not in list_of_galaxy :
                    continue

                print('\nWorking on %s ...\n'%gal)

                if args.wfi_path :
                    p = Path(args.wfi_path)
                else:
                    p = Path(p)

                print('Images path :\n%s'%p)

                # Dictionary of lists necessary for saving the tables and comparng the FWHMs
                psfs_dic = {
                    'moffat_psfs' : [], 'a_moffat_psfs' : [], 'b_moffat_psfs' : [], 'ang_moffat_psfs' : [],
                    'gauss_psfs' : [], 'a_gauss_psfs' : [], 'b_gauss_psfs' : [], 'ang_gauss_psfs' : [],
                    'filenums' : [], 'filters' : []
                }

                # setting the path where to save psf results
                psf_path = Path('.').joinpath('psf_match')
                psf_path.mkdir(exist_ok=True)

                if args.psf_fitting :

                    for filt in ['Rc', 'Ha'] :

                        # setting the list of science, mask and error frames
                        scienceframes = sorted(p.glob('%s_%s_stack?_calib_nosky.fits'%(gal,filt))) + sorted(p.glob('%s_%s_stack??_calib_nosky.fits'%(gal,filt)))
                        maskframes = sorted(p.glob('%s_%s_stack?_mask_calib.fits'%(gal,filt))) + sorted(p.glob('%s_%s_stack??_mask_calib.fits'%(gal,filt)))
                        errorframes = sorted(p.glob('%s_%s_stack?_std_calib.fits'%(gal,filt))) + sorted(p.glob('%s_%s_stack??_std_calib.fits'%(gal,filt)))

                        if (not scienceframes) or (not maskframes) or (not errorframes) :

                            print('\nThere are no frames with filter %s for %s\n'%(filt,gal))
                            continue

                        for scienceframe, maskframe, errorframe in zip(scienceframes, maskframes, errorframes) :

                            ##### PSF computation #####

                            # Instantiate ImageStars class
                            stars = ImageStars(p, psf_path, gal, filt, 'WFI', scienceframe, maskframe, errorframe, args.fwhm)

                            # Find stars in the image
                            stars.find_image_stars()

                            # Instantiate Gaia stars
                            gs = GAIAStars(stars.obj, stars.instr)
                            gs.gaia_query()

                            # Match stars found in the stacked image (common projection) with Gaia stars in the image field
                            stars.match_with_gaia(gs)

                            # Compute the ePSF
                            stars.compute_psf(args.oversampling_factor)

                            # Save the table
                            stars.save_tbl()

                            # Fit the ePSF
                            stars.fit_psf(args.oversampling_factor)

                            # Compute the FWHM and store them in a dictionary
                            stars.set_fwhm (psfs_dic, args.oversampling_factor)

                    # Save the dictionary in a text file
                    save_dic(gal, psfs_dic, 'WFI', psf_path)

                    print("\nDe-allocating memory...\n")
                    gc.disable()
                    gc.collect()

                # Compute the kernels
                if args.compute_kernels :

                    compute_kernels (gal, 'WFI')

                    print("\nDe-allocating memory...\n")
                    gc.disable()
                    gc.collect()

                # Loop to convolve the frames

                if args.convolve_fullFOV :

                    for filt in args.wfi_filters :

                        # setting the list of science, mask and error frames
                        scienceframes = sorted(p.glob('%s_%s_stack?_calib_nosky.fits'%(gal,filt))) + sorted(p.glob('%s_%s_stack??_calib_nosky.fits'%(gal,filt)))
                        maskframes = sorted(p.glob('%s_%s_stack?_mask_calib.fits'%(gal,filt))) + sorted(p.glob('%s_%s_stack??_mask_calib.fits'%(gal,filt)))
                        errorframes = sorted(p.glob('%s_%s_stack?_std_calib.fits'%(gal,filt))) + sorted(p.glob('%s_%s_stack??_std_calib.fits'%(gal,filt)))

                        scienceframes = scienceframes[args.frame_number-1:]
                        maskframes = maskframes[args.frame_number-1:]
                        errorframes = errorframes[args.frame_number-1:]

                        if (not scienceframes) or (not maskframes) or (not errorframes) :

                            print('\nThere are no frames with filter %s for %s)\n'%(filt,gal))
                            continue

                        for scienceframe, maskframe, errorframe in zip(scienceframes, maskframes, errorframes) :

                            #if scienceframe != scienceframes[1] :
                            #    continue

                            mask = np.round(fits.open(str(maskframe))[0].data)!=99

                            '''
                            Mask update with manually marked DS9 regions by loading
                            the region file related to scienceframe file if exists.
                            '''
                            ds9_reg = DS9Reg(Path(str(scienceframe).replace('_nosky.fits','.fits')))
                            ds9_reg.set_ds9_regions()
                            ds9_reg.mask_regions()
                            #ds9_reg.plot_regions()

                            # Update the mask if DS9 regions are defined
                            mask = mask + ds9_reg.mask

                            # Convolve the frame to the gaussian PSF
                            convolve_frame(scienceframe, errorframe, mask, gal, filt, 'WFI')

                            # Convolution check_plot
                            for BB_string in ['', 'BB'] :

                                conv_scienceframe = Path(str(scienceframe).replace('.fits','_%sconv.fits'%BB_string))
                                conv_errorframe = Path(str(errorframe).replace('.fits','_%sconv.fits'%BB_string))

                                if conv_scienceframe.exists() :
                                    convolution_check_plot(p, psf_path, gal, filt, 'WFI', conv_scienceframe,
                                        maskframe, conv_errorframe, args.fwhm, args.oversampling_factor)

                    print("\nDe-allocating memory...\n")
                    gc.disable()
                    gc.collect()


                if (args.bkg_subtraction) + (args.convolve_cutouts) :

                    for filt in args.wfi_filters :

                        # setting the list of science, mask and error frames
                        scienceframes = sorted(outpath.glob('cutout_images/%s_%s_stack?_cutout.fits'%(gal,filt))) + sorted(outpath.glob('cutout_images/%s_%s_stack??_cutout.fits'%(gal,filt)))
                        maskframes = sorted(outpath.glob('cutout_images/%s_%s_stack?_mask_cutout.fits'%(gal,filt))) + sorted(outpath.glob('cutout_images/%s_%s_stack??_mask_cutout.fits'%(gal,filt)))
                        errorframes = sorted(outpath.glob('cutout_images/%s_%s_stack?_std_cutout.fits'%(gal,filt))) + sorted(outpath.glob('cutout_images/%s_%s_stack??_std_cutout.fits'%(gal,filt)))

                        scienceframes = scienceframes[args.frame_number-1:]
                        maskframes = maskframes[args.frame_number-1:]
                        errorframes = errorframes[args.frame_number-1:]

                        if (not scienceframes) or (not maskframes) or (not errorframes) :

                            print('\nThere are no frames with filter %s for %s)\n'%(filt,gal))
                            continue

                        num = 1    # To save the Sky images with incremental numbers

                        for scienceframe, maskframe, errorframe in zip(scienceframes, maskframes, errorframes) :

                            mask = np.round(fits.open(str(maskframe))[0].data)!=99

                            '''
                            Mask update with manually marked DS9 regions by loading
                            the region file related to scienceframe file if exists.
                            '''
                            ds9_reg = DS9Reg(Path(str(scienceframe).replace('_nosky.fits','.fits')))
                            ds9_reg.set_ds9_regions()
                            ds9_reg.mask_regions()
                            #ds9_reg.plot_regions()

                            # Update the mask if DS9 regions are defined
                            mask = mask + ds9_reg.mask

                            if not args.skip_bkg :

                                # Instantiate frame number
                                pos = [i for i, c in enumerate(scienceframe.name) if c=='_']
                                num = int(scienceframe.name[pos[1]+6:pos[2]])

                                # Instiantiate the sky object
                                sky = SkyImage(scienceframe, maskframe, errorframe, ds9_reg.mask, gal, filt, num)
                                myellipse = sky.elliptical_aperture(r, 'WFI')
                                myellipse.annulus_background(r, outpath.joinpath('bkg'), dilate_size, filename='ellipse_radii_%s.png'%num)
                                sky.compute_bkg(myellipse, True, exclude_percentile, dilate_size)
                                sky.plot_bkg(myellipse)

                                ### Subtract the background and save the result

                                # Subtract the sky from the scienceframe

                                print('\n\nBackground subtraction ...\n\n')

                                data = fits.open(str(scienceframe))[0].data - sky.fit
                                hdr = fits.open(str(scienceframe))[0].header

                            nosky_scienceframe = Path(str(scienceframe).replace('.fits','_nosky.fits'))

                            if not args.skip_bkg :

                                fits.writeto(str(nosky_scienceframe), data, header=hdr, overwrite=True)

                            ### Convolve the background-subtracted frame ###

                            if args.convolve_cutouts :

                                # Convolve the frame to the gaussian PSF

                                print('\n\nConvolving the background-subtracted frame ...\n\n')
                                convolve_frame(nosky_scienceframe, errorframe, mask, gal, filt, 'WFI')

                                # Convolution check_plot and frame stacking

                                for BB_string in ['', 'BB'] :

                                    conv_scienceframe = Path(str(nosky_scienceframe).replace('.fits','_%sconv.fits'%BB_string))
                                    conv_errorframe = Path(str(errorframe).replace('.fits','_%sconv.fits'%BB_string))

                                    if conv_scienceframe.exists() :
                                        convolution_check_plot(p, psf_path, gal, filt, 'WFI', conv_scienceframe,
                                            maskframe, conv_errorframe, args.fwhm, args.oversampling_factor)

                    print("\nDe-allocating memory...\n")
                    gc.disable()
                    gc.collect()


        # Loop for CCD

        else :

            for gal in list_of_galaxy :

                print('\nWorking on %s ...\n'%gal)

                p = Path(args.ccd_path)

                print('Images path :\n%s'%p)

                # Dictionary of lists necessary for saving the tables and comparng the FWHMs
                psfs_dic = {
                    'moffat_psfs' : [], 'a_moffat_psfs' : [], 'b_moffat_psfs' : [], 'ang_moffat_psfs' : [],
                    'gauss_psfs' : [], 'a_gauss_psfs' : [], 'b_gauss_psfs' : [], 'ang_gauss_psfs' : [],
                    'filenums' : [], 'filters' : []
                }

                # setting the path where to save psf results
                psf_path = Path('.').joinpath('psf_match')
                psf_path.mkdir(exist_ok=True)

                if args.psf_fitting :

                    for filt in ['r', 'Ha657', 'Ha663'] :

                        # setting the list of science, mask and error frames
                        scienceframes = sorted(p.glob('%s_%s_*_wcs_reproj_calib.fits'%(gal,filt)))
                        maskframes = sorted(p.glob('%s_%s_*_reproj_mask_calib.fits'%(gal,filt)))
                        errorframes = sorted(p.glob('%s_%s_*_reproj_std_calib.fits'%(gal,filt)))

                        if (not scienceframes) or (not maskframes) or (not errorframes) :

                            print('\nThere are no frames with filter %s for %s\n'%(filt,gal))
                            continue

                        for scienceframe, maskframe, errorframe in zip(scienceframes, maskframes, errorframes) :

                            ##### PSF computation #####

                            # Instantiate ImageStars class
                            stars = ImageStars(p, psf_path, gal, filt, 'CCD', scienceframe, maskframe, errorframe, args.fwhm)

                            # Find stars in the image
                            stars.find_image_stars()

                            # Instantiate Gaia stars
                            gs = GAIAStars(stars.obj, stars.instr)
                            gs.gaia_query()

                            # Match stars found in the stacked image (common projection) with Gaia stars in the image field
                            stars.match_with_gaia(gs)

                            # Compute the ePSF
                            stars.compute_psf(args.oversampling_factor)

                            # Save the table
                            stars.save_tbl()

                            # Fit the ePSF
                            stars.fit_psf(args.oversampling_factor)

                            # Compute the FWHM and store them in a dictionary
                            stars.set_fwhm (psfs_dic, args.oversampling_factor)

                    # Save the dictionary in a text file
                    save_dic(gal, psfs_dic, 'CCD', psf_path)

                    print("\nDe-allocating memory...\n")
                    gc.disable()
                    gc.collect()

                # Compute the kernels
                if args.compute_kernels :

                    compute_kernels (gal, 'CCD')

                    print("\nDe-allocating memory...\n")
                    gc.disable()
                    gc.collect()


                # Loop to convolve the frames

                if (args.bkg_subtraction) + (args.convolve_dp) :

                    for filt in args.ccd_filters :

                        # setting the list of science, mask and error frames
                        scienceframes = sorted(p.glob('%s_%s_*_wcs_reproj_calib.fits'%(gal,filt)))
                        maskframes = sorted(p.glob('%s_%s_*_reproj_mask_calib.fits'%(gal,filt)))
                        errorframes = sorted(p.glob('%s_%s_*_reproj_std_calib.fits'%(gal,filt)))

                        scienceframes = scienceframes[args.frame_number-1:]
                        maskframes = maskframes[args.frame_number-1:]
                        errorframes = errorframes[args.frame_number-1:]

                        if (not scienceframes) or (not maskframes) or (not errorframes) :

                            print('\nThere are no frames with filter %s for %s)\n'%(filt,gal))
                            continue

                        for scienceframe, maskframe, errorframe in zip(scienceframes, maskframes, errorframes) :

                            #if scienceframe!=scienceframes[1] :
                            #    continue

                            mask = np.round(fits.open(str(maskframe))[0].data)!=99

                            '''
                            Mask update with manually marked DS9 regions by loading
                            the region file related to scienceframe file if exists.
                            '''
                            ds9_reg = DS9Reg(Path(str(scienceframe).replace('_nosky.fits','.fits')))
                            ds9_reg.set_ds9_regions()
                            ds9_reg.mask_regions()
                            #ds9_reg.plot_regions()

                            # Update the mask if DS9 regions are defined
                            mask = mask + ds9_reg.mask

                            if not args.skip_bkg :

                                # Instantiate frame number
                                pos = [i for i, c in enumerate(scienceframe.name) if c=='_']
                                num = int(scienceframe.name[pos[1]+1:pos[2]])

                                # Instiantiate the sky object
                                sky = SkyImage(scienceframe, maskframe, errorframe, ds9_reg.mask, gal, filt, num)
                                myellipse = sky.elliptical_aperture(r, 'CCD')
                                myellipse.annulus_background(r, outpath.joinpath('bkg'), dilate_size, filename='ellipse_radii_%s.png'%num)
                                sky.compute_bkg(myellipse, True, exclude_percentile, dilate_size)
                                sky.plot_bkg(myellipse)

                                ### Subtract the background and save the result

                                # Subtract the sky from the scienceframe

                                print('\n\nBackground subtraction ...\n\n')

                                data = fits.open(str(scienceframe))[0].data - sky.fit
                                hdr = fits.open(str(scienceframe))[0].header

                            nosky_scienceframe = Path(str(scienceframe).replace('.fits','_nosky.fits'))

                            if not args.skip_bkg :

                                fits.writeto(str(nosky_scienceframe), data, header=hdr, overwrite=True)

                            ### Convolve the background-subtracted frame ###

                            if args.convolve_dp :

                                # Convolve the frame to the gaussian PSF

                                print('\n\nConvolving the background-subtracted frame ...\n\n')

                                convolve_frame(nosky_scienceframe, errorframe, mask, gal, filt, 'CCD')

                                # Convolution check_plot and frame stacking

                                for BB_string in ['', 'BB'] :

                                    conv_scienceframe = Path(str(nosky_scienceframe).replace('.fits','_%sconv.fits'%BB_string))
                                    conv_errorframe = Path(str(errorframe).replace('.fits','_%sconv.fits'%BB_string))

                                    if conv_scienceframe.exists() :
                                        convolution_check_plot(p, psf_path, gal, filt, 'CCD', conv_scienceframe,
                                            maskframe, conv_errorframe, args.fwhm, args.oversampling_factor)

                    print("\nDe-allocating memory...\n")
                    gc.disable()
                    gc.collect()


    elapsed = timeit.default_timer() - start_time
    minutes, seconds = divmod(elapsed, 60)
    hours, minutes = divmod(minutes, 60)
    print ('\nProcess terminated in %d:%02d:%02d'%(hours, minutes, seconds))
