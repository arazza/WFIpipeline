# coding: utf-8

'''
Broad-band images photometric calibration.

The code finds the zero points for each frame
by calibrating the broad-band images against
Gaia DR2 and makes the plots to show the method.

It works for both WFI and DP data set.
'''

import sys

import timeit

import argparse

from pathlib import Path

import numpy as np

from astropy.io import fits
from astropy import units as u
from astropy.visualization import ZScaleInterval
from astropy.stats import sigma_clipped_stats, mad_std, SigmaClip
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, search_around_sky
from astropy.time import Time
from astropy.utils.data import conf
from astropy.wcs import WCS
from astropy.table import Column

from astroquery.vizier import Vizier

from photutils import IRAFStarFinder
from photutils import CircularAperture, CircularAnnulus, aperture_photometry

from scipy.spatial.distance import cdist

from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import seaborn as sns

from galaxy_shape_v21 import GalaxyEllipse


class ImageStars (object) :
    '''
    This class contains info of a set of stars for a given reduced frame.
    '''
    def __init__ (self, instr, path, tbl_path, plot_path, obj, filedata, filemask, file_newmask, file_sp_mask, filesigma, fwhm, comb=False) :
        '''
        Note on the fwhm: the mean value computed from Brent's NB-DR1 WFI PSFs
        and the weighted average seeing of the du Pont observations resulted
        to be ~4.08 pixels

        The comb variable serves to discriminate the class when instantiated
        for the final combined broad-band image, to be used in the narrow-band
        photometric calibration script, or for the broad- and narrow-band
        individual frames.
        '''

        self.path = path
        self.tbl_path = tbl_path
        self.plot_path = plot_path
        self.obj = obj
        self.wcs = WCS(fits.open(str(filedata))[0].header)
        self.filename = filedata

        self.data = fits.open(str(filedata))[0].data
        self.hdr = fits.open(str(filedata))[0].header
        if comb :
        	self.mask = np.full(self.data.shape, 99.)
        	self.mask[np.isnan(self.data)] = 1.0e6
        else :
            self.mask = fits.open(str(filemask))[0].data

        self.instr = instr

        if (self.instr == 'WFI') * (not comb) :

            self.newmask = fits.open(file_newmask)[0].data
            self.sp_mask = fits.open(file_sp_mask)[0].data
            self.error = np.sqrt(fits.open(str(filesigma))[0].data)

            if str(filedata)[-7] == 'k' :
                self.filenum = str(filedata)[-6]
            else :
                self.filenum = str(filedata)[-7:-5]

        if (self.instr == 'CCD') + (comb) :

            self.newmask = file_newmask
            self.sp_mask = file_sp_mask
            self.error = fits.open(str(filesigma))[0].data

        if (self.instr == 'CCD') * (not comb) :

            pos = [i for i, c in enumerate(filedata.name) if c=='_']
            if len(pos) >=3 :
                self.filenum = int(filedata.name[pos[1]+1:pos[2]])
            else :
                self.filenum = 0    # For specphot stars fields

        if comb :

        	self.filenum = 0

        self.fwhm = fwhm
        #self.airmass = self.hdr['AIRMASS']


    def merge_masks (self) :
        '''
        The method mergees the three types of masks available for WFI in NB-DR2
        '''

        self.mask[(np.round(self.newmask)!=99) * (self.newmask!=0)] = 1e12

        self.mask[np.round(self.sp_mask)!=99] = 1e12


    def find_image_stars (self, threshold_nstd, find_stars=IRAFStarFinder, peakmax=None) :
        '''
        This method computes the image threshold to find stars with
        IRAFFindStars algorithm. An astropy Table is created with pixel positions.
        '''

        # find stars in the input image
        sigclip_sigma=3.0
        sigclip_iters=5
        image_mean, image_median, image_std = sigma_clipped_stats(self.data, sigma=sigclip_sigma, maxiters=sigclip_iters)
        print (image_mean, image_median, image_std)
        threshold = image_median + (threshold_nstd * image_std)
        print('Threshold for finding bright stars', threshold)

        iraffind = find_stars(threshold=threshold, fwhm=self.fwhm, peakmax=peakmax)
        self.tbl = iraffind(self.data)
        self.tbl.keep_columns(['xcentroid', 'ycentroid'])

        print('\nNumber of stars found in the image : %s\n' %len(self.tbl))


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
        print('Number of matching stars between the image and Gaia : ',len(self.coord[sep_constraint]), '\n')

        ## match with Gaia stars
        #idxc, idxcatalog, d2d, d3d = gaia_stars.coord.search_around_sky(self.coord, 0.5*u.arcsec)
        #print('\nNumber of matching stars between the image and Gaia : %s\n' %len(idxc))
        #
        ## match with Gaia stars
        #idxc, idxcatalog, d2d, d3d = gaia_stars.coord.search_around_sky(self.coord, 1.*u.arcsec)
        #print('\nNumber of matching stars between the image and Gaia : %s\n' %len(idxc))

        # update tbl
        self.tbl = self.tbl[sep_constraint]
        gaia_stars.coord = gaia_stars.coord[idx[sep_constraint]]
        gaia_ra = Column(gaia_stars.coord.ra, name='gaia_ra')
        gaia_dec = Column(gaia_stars.coord.dec, name='gaia_dec')
        if self.instr == 'WFI' :
            gaia_R = Column(gaia_stars.Rmag[idx[sep_constraint]], name='gaia_R_mag')
            gaia_R_err = Column(gaia_stars.e_Rmag[idx[sep_constraint]], name='gaia_R_mag_err')
        if self.instr == 'CCD' :
            gaia_r = Column(gaia_stars.rmag[idx[sep_constraint]], name='gaia_r_mag')
            gaia_r_err = Column(gaia_stars.e_rmag[idx[sep_constraint]], name='gaia_r_mag_err')
        gaia_Gmag = Column(gaia_stars.Gmag[idx[sep_constraint]], name='gaia_Gmag')
        gaia_Gmag_err = Column(gaia_stars.e_Gmag[idx[sep_constraint]], name='gaia_Gmag_err')
        gaia_BPmag = Column(gaia_stars.BPmag[idx[sep_constraint]], name='gaia_BPmag')
        gaia_RPmag = Column(gaia_stars.RPmag[idx[sep_constraint]], name='gaia_RPmag')
        gaia_BP_RPmag = Column(gaia_stars.BP_RPmag[idx[sep_constraint]], name='gaia_BP_RPmag')
        gaia_BP_RPmag_err = Column(gaia_stars.e_BP_RPmag[idx[sep_constraint]], name='gaia_BP_RPmag_err')
        self.tbl.add_column(gaia_ra)
        self.tbl.add_column(gaia_dec)
        if self.instr == 'WFI' :
            self.tbl.add_column(gaia_R)
            self.tbl.add_column(gaia_R_err)
        if self.instr == 'CCD' :
            self.tbl.add_column(gaia_r)
            self.tbl.add_column(gaia_r_err)
        self.tbl.add_column(gaia_Gmag)
        self.tbl.add_column(gaia_Gmag_err)
        self.tbl.add_column(gaia_BPmag)
        self.tbl.add_column(gaia_RPmag)
        self.tbl.add_column(gaia_BP_RPmag)
        self.tbl.add_column(gaia_BP_RPmag_err)

        for col in self.tbl.colnames:
            self.tbl[col].info.format = '%.4f'


    def star_positions (self, instr) :
        '''
        This method filter the positions of stars too close to each other or too close to masked pixels.
        It instantiates and mask an ellipse around the galaxy.
        '''

        # Initialise the x0 and y0 pixel positions of the stars
        sources_x0 = self.tbl['xcentroid']
        sources_y0 = self.tbl['ycentroid']

        # Mask an ellipse around the galaxy
        myellipse = GalaxyEllipse(self.obj, 'Rc', self.filename, self.mask)
        myellipse.get_ellipse()
        myellipse.ellipse_to_pixel(instr)
        myellipse.mask_ellipse(r=1.2)

        # Mask the stars
        d1 = 12    # equivalent to typical FWHM * 3
        d2 = 15    # equivalent to the aperture photometry radius
        d3 = 30    # equivalent to the annulus external radius

        nstars = len(sources_x0)
        mask = np.ones((nstars), bool)
        for i in range(nstars) :
            x0 = int(round(sources_x0[i]))
            y0 = int(round(sources_y0[i]))
            # exclude stars close to masked pixels
            if (np.round(self.mask[y0-d1:y0+d1,x0-d1:x0+d1])!=99).any() :    # 99 is the background value to be distinguished from the gaps with 0 counts
                mask[i]=False
            # exclude stars in the minimal bounding box of the ellipse around the galaxy
            if myellipse.mask[y0-1,x0-1]==True :
                mask[i]=False
            # exclude stars close to the edge of the ccds or to the gaps
            if (self.data[y0-d3:y0+d3,x0-d3:x0+d3]==0).any() :
                mask[i]=False

        # exclude stars closer than the aperture photometry radius to each of the other stars
        positions = [(sources_x0[i], sources_y0[i]) for i in range(nstars)]
        distances = cdist(positions,positions)
        close_points = (distances<=d3*2) & (distances!=0)
        rejected_stars = np.where(close_points)[0]
        mask[rejected_stars] = False

        # Assign the values for the aperture photometry radii
        self.r = d2
        self.r_in = d2 + 5
        self.r_out = d3

        # Update the source list
        self.tbl = self.tbl[mask]

        print('\nNumber of remaining isolated stars (after filtering bad positions) : %s\n' %len(self.tbl))



    def annulus_photometry (self, snr=20):
        '''
        Compute aperture photometry with local background subtraction
        '''

        # Initialise the positions of the stars
        #positions = (self.tbl['xcentroid'], self.tbl['ycentroid'])    # DEPRECATED in photutils >= 1.0.0
        positions = [(xcentroid, ycentroid) for xcentroid, ycentroid in zip(self.tbl['xcentroid'], self.tbl['ycentroid'])]

        # Define the mask
        mask = np.round(self.mask)!=99

        # Aperture photometry
        self.apertures = CircularAperture(positions, r=self.r)
        self.annulus_apertures = CircularAnnulus(positions, r_in=self.r_in, r_out=self.r_out)
        phot_table = aperture_photometry(self.data, self.apertures, error=self.error, mask=mask)

        # Local background subtraction

        '''
        Error estimation:
        \sigma_\mathrm{src}^2 = \sigma_\mathrm{tot}^2 + \frac{\pi}{2}\frac{N_A^2}{N_B}\sigma_\mathrm{B/pix}^2 \\
        where
        \sigma_\mathrm{tot} \equiv \Delta F \\
        \sigma_\mathrm{B/pix} = 1.4826 \cdot \mathrm{MAD}
        '''
        mymasks = self.annulus_apertures.to_mask(method='center')

        bkg_median = np.zeros(len(mymasks))
        bkg_err = np.zeros(len(mymasks))

        for k,mask in enumerate(mymasks) :
            annulus_data = mask.multiply(self.data)
            annulus_data_1d = annulus_data[mask.data > 0]
            _, median_sigclip, _ = sigma_clipped_stats(annulus_data_1d)
            bkg_median[k] = median_sigclip
            bkg_err[k] = mad_std(annulus_data_1d, ignore_nan=True)

        phot_table['annulus_median'] = bkg_median

        N_aper = self.apertures.area
        N_back = self.annulus_apertures.area

        phot_table['aper_bkg'] = bkg_median * N_aper

        phot_table['aper_bkg_err'] = bkg_err * N_aper * np.sqrt(np.pi/2) / np.sqrt(N_back)

        phot_table['aper_sum_bkgsub'] = phot_table['aperture_sum'] - phot_table['aper_bkg']

        phot_table['aper_sum_bkgsub_err'] = np.sqrt(phot_table['aperture_sum_err']**2 + phot_table['aper_bkg_err']**2)

        # Check of WFI vs DP pipeline aperture photometry error computation formula
        #for i in range(len(phot_table)) :
        #    print(np.sqrt(phot_table['aperture_sum_err'][i]**2 + (bkg_err[i])**2 * N_aper), phot_table['aper_sum_bkgsub_err'][i])

        #Update the table

        fluxes = phot_table['aper_sum_bkgsub']
        errfluxes = phot_table['aper_sum_bkgsub_err']

        f_ap = Column(fluxes, name='flux_apert')
        err_f_ap = Column(errfluxes, name='flux_apert_err')
        m_ap = Column(-2.5*np.log10(fluxes), name='mag_apert')
        m_ap_err = Column(1.086*(errfluxes/fluxes), name='mag_apert_err')

        self.tbl.add_column(f_ap)
        self.tbl.add_column(err_f_ap)
        self.tbl.add_column(m_ap)
        self.tbl.add_column(m_ap_err)

        for col in self.tbl.colnames:
            self.tbl[col].info.format = '%.4f'

        # Cut out stars with S/N lower than snr (default 20)
        stn_cut = (self.tbl['flux_apert'] / self.tbl['flux_apert_err']) > snr

        self.tbl = self.tbl[stn_cut]
        self.apertures = self.apertures[stn_cut]
        self.annulus_apertures = self.annulus_apertures[stn_cut]

        print('\nNumber of stars used for aperture photometry (after S/N cut) : %s\n' %len(self.tbl))

        # Plot 1
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(14, 7), gridspec_kw={'width_ratios': [1, 1], 'height_ratios': [1]})
        zscale = ZScaleInterval()
        vmin, vmax = zscale.get_limits(self.data)
        self.ax1.imshow(self.data, cmap='gray_r', origin='lower', vmin=vmin, vmax=vmax)
        self.apertures.plot(axes=self.ax1, color='white', lw=2)
        self.annulus_apertures.plot(axes=self.ax1, color='red', lw=2)
        self.ax1.set_xlabel('x [pix]')
        self.ax1.set_ylabel('y [pix]')


    def set_airmass (self) :
        '''
        Method to compute the target airmass for WFI
        and to read it from the header for CCD.
        '''

        if self.instr == 'WFI' :

            #Initialization
            obj_coord = SkyCoord.from_name(self.obj)
            LaSilla = EarthLocation.of_site('La Silla Observatory')
            time = Time(self.hdr['MJD-OBS'], format='mjd')
            # Conversion to Airmass
            obj_altaz = obj_coord.transform_to (AltAz (obstime=time, location=LaSilla))
            self.airmass = obj_altaz.secz.value

        if self.instr == 'CCD' :

            self.airmass = self.hdr['AIRMASS']


    def set_filter (self, filt) :

        self.filt = filt



class GAIAStars (object) :

    def __init__ (self, obj, instr, my_catname = 'Gaia-DR2') :

        self.obj = obj
        self.instr = instr
        self.catname = my_catname


    def gaia_query (self, g_corr=False, lmag=12, hmag=20) :


        new_limit = 60  # sets the timeout to 60s
        with conf.set_temp('remote_timeout', new_limit):
            obj_coord = SkyCoord.from_name(self.obj)

        vizier = Vizier(columns=['*', 'E(BR/RP)'], catalog='Gaia-DR2', row_limit=-1, timeout=60)
        result = vizier.query_region(obj_coord, radius=1*u.deg)

        # cuts on colours as in https://gea.esac.esa.int/archive/documentation/GDR2/Data_processing/chap_cu5pho/sec_cu5pho_calibr/ssec_cu5pho_PhotTransf.html
        # + new color cut based on the Gaia bands trasformation
        mask1 = (result[0]['BPmag']-result[0]['RPmag']>0.2)&(result[0]['BPmag']-result[0]['RPmag']<1.8)
        # cuts on photometric errors as in https://gea.esac.esa.int/archive/documentation/GDR2/Data_processing/chap_cu5pho/sec_cu5pho_calibr/ssec_cu5pho_PhotTransf.html
        if self.instr == 'WFI' :
            mask2 = (result[0]['e_Gmag']<0.01)&(result[0]['e_BPmag']<0.01)&(result[0]['e_RPmag']<0.01)
        if self.instr == 'CCD' :
            mask2 = (result[0]['e_Gmag']<0.003)&(result[0]['e_BPmag']<0.01)&(result[0]['e_RPmag']<0.01)
        # check of consistency for the excess factor as in https://www.aanda.org/articles/aa/full_html/2018/08/aa32756-18/F17.html
        if self.instr == 'WFI' :
            mask3 = (result[0]['E_BR_RP_']<1.5+0.03*(result[0]['BPmag']-result[0]['RPmag'])**2.)
        if self.instr == 'CCD' :
            mask3 = (result[0]['E_BR_RP_']<1.3+0.06*(result[0]['BPmag']-result[0]['RPmag'])**2.)

        #gaia_tbl = result[0][mask1&mask2]
        #gaia_tbl = result[0][mask1&mask2&mask3]
        gaia_tbl = result[0][mask1&mask3]

        self.Gmag = gaia_tbl['Gmag']
        self.e_Gmag = gaia_tbl['e_Gmag']
        self.BPmag = gaia_tbl['BPmag']
        self.RPmag = gaia_tbl['RPmag']
        self.BP_RPmag = gaia_tbl['BPmag'] - gaia_tbl['RPmag']
        self.e_BP_RPmag = np.sqrt(gaia_tbl['e_BPmag']**2.+gaia_tbl['e_RPmag']**2)

        # G band correction as in Maiz-Apelliz & Weiler (2018)
        if g_corr :
            G_corr = self.Gmag.copy()
            group1 = self.Gmag<16
            group2 = self.Gmag>=16
            G_corr[group1] = self.Gmag[group1] - 0.0032 * (self.Gmag[group1]-6)
            G_corr[group2] = self.Gmag[group2] - 0.032    # for G > 16 it is recommended to use the correction value at G=16
            self.Gmag = G_corr.copy()

        self.coord = SkyCoord(ra=gaia_tbl['RA_ICRS'], dec=gaia_tbl['DE_ICRS'])

        if self.instr == 'WFI' :
            '''
            Transforamtion as in https://gea.esac.esa.int/archive/documentation/GDR2/Data_processing/chap_cu5pho/sec_cu5pho_calibr/ssec_cu5pho_PhotTransf.html#Ch5.T8
            G - R = -0.003226 + 0.3833*(BP-RP) - 0.1345*(BP-RP)^2
            '''
            self.Rmag = self.Gmag + 0.003226 - 0.3833*(self.BP_RPmag) + 0.1345*(self.BP_RPmag)**2.
            self.e_Rmag = np.sqrt(self.e_Gmag**2.+0.04840**2.)

            # cuts on bright and faint star magnitudes
            mask = (self.Rmag>lmag)&(self.Rmag<hmag)

            # Filter for the Gaia trasformed magnitudes
            self.Rmag = self.Rmag[mask]
            self.e_Rmag = self.e_Rmag[mask]

        if self.instr == 'CCD' :
            '''
            Transforamtion as in https://gea.esac.esa.int/archive/documentation/GDR2/Data_processing/chap_cu5pho/sec_cu5pho_calibr/ssec_cu5pho_PhotTransf.html#Ch5.T8
            G - r = - 0.12879 + 0.24662*(BP-RP) - 0.027464*(BP-RP)^2 - 0.049465*(BP-RP)^3
            '''
            self.rmag = self.Gmag + 0.12879 - 0.24662*(self.BP_RPmag) + 0.027464*(self.BP_RPmag)**2. + 0.049465*(self.BP_RPmag)**3.
            self.e_rmag = np.sqrt(self.e_Gmag**2. + 0.066739**2.)

            # cuts on bright and faint star magnitudes
            mask = (self.rmag>lmag)&(self.rmag<hmag)

            # Filter for the Gaia trasformed magnitudes
            self.rmag = self.rmag[mask]
            self.e_rmag = self.e_rmag[mask]

        # Filter for the Gaia trasformed magnitudes
        self.Gmag = self.Gmag[mask]
        self.e_Gmag = self.e_Gmag[mask]
        self.BP_RPmag = self.BP_RPmag[mask]
        self.e_BP_RPmag = self.e_BP_RPmag[mask]
        self.coord = self.coord[mask]


class ZPParams (object) :

    def __init__ (self, stars) :

        self.ZP = None
        self.sigma_ZP = None
        if stars.instr == 'WFI' :
            self.exct = 0.07    # http://www.eso.org/sci/facilities/lasilla/instruments/wfi/inst/zeropoints.html
            self.sigma_exct = 0.01
        if stars.instr == 'CCD' :
            self.exct = 0.103       # Krisciunas et al. (2017). Note: color term coeff --> -0.016
            self.sigma_exct = 0.019 # Krisciunas et al. (2017). Note: color term coeff error --> 0.015



    def find_zp (self, stars) :

        k = self.exct
        sigma_k = self.sigma_exct

        fluxes = stars.tbl['flux_apert']
        err_fluxes = stars.tbl['flux_apert_err']

        if stars.instr == 'WFI' :
            gaia_r = stars.tbl['gaia_R_mag']
            err_gaia_r = stars.tbl['gaia_R_mag_err']
        if stars.instr == 'CCD' :
            gaia_r = stars.tbl['gaia_r_mag']
            err_gaia_r = stars.tbl['gaia_r_mag_err']

        gaia_color = stars.tbl['gaia_BP_RPmag']

        ### Zero point computation ###

        # VEGA-to-AB offset from Bessel & Murphy (2012)
        if stars.instr == 'WFI' :
            vega_to_ab = 0.163
        if stars.instr == 'CCD' :
            vega_to_ab = 0.
        print('\nApplying VEGA-to-AB offset of %s\n'%vega_to_ab)

        # Before clipping
        iZP = np.array([r + vega_to_ab + 2.5*np.log10(f) + k*stars.airmass for r,f in zip(gaia_r,fluxes)])
        sigma_iZP = np.array([np.sqrt (e_rmag**2 + 1.179*(sigma_f/f)**2 + (stars.airmass*sigma_k)**2) for e_rmag, f, sigma_f in zip(err_gaia_r, fluxes, err_fluxes)])
        err_weight = np.sum(1/(sigma_iZP**2))
        ZP = np.sum(1/(sigma_iZP**2)*iZP)/err_weight
        sigma_ZP = np.sqrt(1/err_weight)
        print('Before clipping ZP : ', ZP, sigma_ZP)

        zp = Column(iZP, name='zero_point')
        err_zp = Column(sigma_iZP, name='zero_point_err')

        stars.tbl.add_column(zp)
        stars.tbl.add_column(err_zp)

        stars.tbl['zero_point'].info.format = '%.4f'
        stars.tbl['zero_point_err'].info.format = '%.4f'

        #tbl_filename = stars.filename.name.replace('wcs_reproj.fits','phot_table.dat')
        tbl_filename = '%s_%s_%s_phot_table.dat'%(stars.obj,stars.filt,stars.filenum)
        stars.tbl.write(stars.tbl_path.joinpath(tbl_filename), format='ascii.commented_header',
                        overwrite=True)

        # Plot 1
        stars.ax2.errorbar (gaia_r, iZP, xerr=err_gaia_r, yerr=sigma_iZP, fmt='none', ms=4, alpha=0.5, ecolor='lightgray', zorder=1)
        im = stars.ax2.scatter (gaia_r, iZP, c=gaia_color, marker='o', s=30, alpha=0.5, zorder=2)
        divider = make_axes_locatable(stars.ax2)
        cax = divider.append_axes("right", size = "5%", pad = 0.05)
        cb = plt.colorbar(im, cax = cax)
        cb.mappable.set_clim(0.2, 1.8)
        cb.set_label(r'$G_\mathrm{BP}-G_\mathrm{RP}$ [mag]', labelpad=10)
        if stars.instr == 'WFI' :
            stars.ax2.set_xlabel(r'GAIA-DR2 $R_\mathrm{c}$ [mag]')
        if stars.instr == 'CCD' :
            stars.ax2.set_xlabel(r'GAIA-DR2 $r$ [mag]')
        stars.ax2.set_ylabel('ZP [mag]')

        # Sigma clipping of ZP
        sigclip = SigmaClip(sigma=3, maxiters=5)
        #sigclip = SigmaClip(sigma=2.5, maxiters=None)
        iZP_filt = sigclip(iZP)
        mask = ~iZP_filt.mask
        # Number of remaing stars
        print('Number of stars for ZP after clipping: ',len(iZP[mask]))
        # After clipping
        iZP_clipped = iZP[mask]
        sigma_iZP_clipped = sigma_iZP[mask]
        err_weight = np.sum(1/(sigma_iZP_clipped**2))
        ZP = np.sum(1/(sigma_iZP_clipped**2)*iZP_clipped)/err_weight
        sigma_ZP = np.sqrt(1/err_weight)
        print('After clipping ZP : ', ZP, sigma_ZP)

        # Plot 1
        stars.ax2.scatter(gaia_r[iZP_filt.mask], iZP[iZP_filt.mask], s=50, marker='x', c='C3')
        # Plot again to resize and get the limits
        stars.ax2.scatter (gaia_r, iZP, c=gaia_color, marker='o', s=30, alpha=0.5, zorder=2)
        xmin, xmax = stars.ax2.dataLim.xmin, stars.ax2.dataLim.xmax
        ymin, ymax = stars.ax2.dataLim.ymin, stars.ax2.dataLim.ymax
        # computed zero point
        stars.ax2.axhline(y=ZP+sigma_ZP, c='r', ls=':', zorder=2)
        stars.ax2.axhline(y=ZP, c='r', ls='--', zorder=2)
        stars.ax2.axhline(y=ZP-sigma_ZP, c='r', ls=':', zorder=2)
        stars.ax2.text(0.02, ZP+sigma_ZP+0.04, r'ZP$=$%.2f$\pm$%.2f'%(ZP,sigma_ZP),
            color='r', ha='left', fontsize=10, transform=stars.ax2.get_yaxis_transform())

        stars.ax2.set_ylim(ZP-0.2, ZP+0.2)

        title_txt = '%s %s frame%s'%(stars.obj,stars.filt,stars.filenum)
        plt.suptitle(title_txt)

        stars.fig.subplots_adjust(left=0.08, right=0.92, top=0.85, bottom=0.2, wspace=0.3)

        stars.fig.savefig(stars.plot_path.joinpath(Path(stars.filename).name.replace('.fits','_zp.png')))

        print('\nClosing plots ..')
        plt.close()

        self.ZP = ZP
        self.sigma_ZP = sigma_ZP


    def calib_images (self, stars) :
        '''
        Apply ZP to calibrate science and error frames
        '''

        print('\n\nPhotometric calibration ...\n')
        '''
        f_r &= f * 10^{-0.4\cdot(\rm{ZP}-\kappa\chi-\rm{ZP}_0)}\\
        \sigma_{f_r} &= 10^{-0.4\cdot(\rm{ZP}-\kappa\chi-\rm{ZP}_0)} \,\cdot\, [\sigma_f^2+0.848\cdot f^2\cdot(\sigma_{\rm{ZP}}^2+\chi^2\sigma_\kappa^2)]^{1/2}
        '''
        ZP_0 = 23.9
        A = 10**(-0.4*(self.ZP-self.exct*stars.airmass-ZP_0))
        self.flux_calib = stars.data * A
        self.unc_flux_calib = A * np.sqrt(stars.error**2+0.848*stars.data**2*(self.sigma_ZP**2+stars.airmass**2*self.sigma_exct**2))


    def write_calib_to (self, stars) :
        '''
        Write to fits file the recalibrated science frame and variance frame
        '''

        cal_filename = stars.path.joinpath(stars.filename.name.replace('.fits','_calib.fits'))
        cal_std_filename = stars.path.joinpath(stars.filename.name.replace('.fits','_std_calib.fits'))
        cal_mask_filename = stars.path.joinpath(stars.filename.name.replace('.fits','_mask_calib.fits'))

        # Writing the unit in the FITS file header
        hdr = stars.hdr
        hdr['BUNIT'] = ('uJy/pixel', 'flux unit')
        #hdr['ZPOINT'] = ('%.3f +/- %.3f'%(self.ZP, self.sigma_ZP), 'zero point from el/sec to r mag (AB)')

        # Saving the calibrated file and its variance frame
        fits.writeto(str(cal_filename), self.flux_calib, header=hdr, overwrite=True)
        fits.writeto(str(cal_std_filename), self.unc_flux_calib, header=hdr, overwrite=True)
        fits.writeto(str(cal_mask_filename), stars.mask, header=hdr, overwrite=True)



if __name__ == "__main__":

    start_time = timeit.default_timer()

    wfi_galaxies = ['NGC1097', 'NGC1087', 'NGC1672', 'NGC2835', 'NGC3627', 'NGC4535', 'IC5332', 'NGC4254', 'NGC5068',
                      'NGC0628', 'NGC1512', 'NGC1300', 'NGC1365', 'NGC1385', 'NGC1433', 'NGC1566', 'NGC4303',
                        'NGC4321', 'NGC6744', 'NGC3511' , 'NGC3596', 'NGC4293', 'NGC4298', 'NGC4496A', 'NGC4569',
                          'NGC4571', 'NGC4654', 'NGC4731', 'NGC2997', 'NGC3351', 'NGC4540', 'NGC4548', 'NGC4689']


    ccd_galaxies = ['NGC0685', 'NGC1317', 'NGC1326', 'IC1954', 'IC1993', 'NGC1511', 'NGC1546', 'NGC1559',
                      'NGC1809',
                      'NGC1792', 'NGC2090', 'NGC2283', 'NGC2566', 'NGC2775', 'NGC3059', 'NGC3507',
                        'NGC3626', 'NGC4207', 'NGC4424', 'NGC4457', 'NGC4540', 'NGC4548', 'NGC4689', 'NGC4694',
                          'NGC4781', 'NGC4941', 'NGC4951', 'NGC5042', 'NGC5134', 'NGC5248', 'NGC5530', 'NGC5643',
                            'NGC6300', 'IC5273', 'NGC7456', 'NGC7496']

    parser = argparse.ArgumentParser()

    parser.add_argument('--wfi_path', action='store', type=str, default=None, help='path to uncalibrated WFI frames')
    parser.add_argument('--ccd_path', action='store', type=str,
        default='/home/arazza/NB_photometry/dirCCDreduction/DuPontDATA/PROCESSED/reproj/', help='path to uncalibrated du Pont frames')
    parser.add_argument('-w', '--wfi', nargs='*', type=str, default=wfi_galaxies,
        help='list of selected WFI galaxies (type a non-valid galaxy to avoid calibrating WFI)')
    parser.add_argument('-c', '--ccd', nargs='*', type=str, default=ccd_galaxies,
        help='list of selected CCD galaxies (type a non-valid galaxy to avoid calibrating CCD)')
    parser.add_argument('-f', '--fwhm', action='store', type=float, default=4.0, help='FWHM to be used for IRAFStarFinder')
    parser.add_argument('--snr', action='store', type=float, default=20., help='threshold for rejecting low SNR as computed with the aperture photometry')

    parser.add_argument('--g_corr', action='store_true', default=False, help='apply Gaia DR2 G-band correction in Casagrande (2018)')

    args = parser.parse_args()

    # Instantiate the lists of galaxies

    lists_of_galaxies = [args.wfi, args.ccd]

    ##### Main loop #####

    for i,list_of_galaxy in enumerate(lists_of_galaxies) :

        # Check the lists

        if (i==0) * (not (set(list_of_galaxy) <= set(wfi_galaxies))) :

            print('\nThere is a galaxy not belonging to WFI data\n... skipping WFI calibration ...\n\n')

            continue

        if (i==1) * (not (set(list_of_galaxy) <= set(ccd_galaxies))) :

            print('\nThere is a galaxy not belonging to CCD data\n... skipping CCD calibration ...\n\n')

            continue

        # Instantiate the paths and start the loop for WFI

        if (i==0) * (set(list_of_galaxy) <= set(wfi_galaxies)) :

            wfi_galnames, wfi_paths = np.loadtxt('paths_to_uncalib_WFI.txt', dtype='str', unpack=True)

            wfi_galnames = np.atleast_1d(wfi_galnames)
            wfi_paths = np.atleast_1d(wfi_paths)

            filt = 'Rc'

            for gal, p in zip(wfi_galnames, wfi_paths) :

                if gal not in list_of_galaxy :
                    continue

                print('\nCalibrating %s ...\n'%gal)

                if args.wfi_path :
                    p = Path(args.wfi_path)
                else:
                    p = Path(p)

                print('Images path :\n%s'%p)

                # setting the path to calibrated file
                cal_path = p.joinpath('phot_calib')
                cal_path.mkdir(exist_ok=True)

                # setting the path to save the tabels with results from photometry for each frame
                tbl_path = p.joinpath('phot_tables')
                tbl_path.mkdir(exist_ok=True)

                # setting the path to save the plots
                plot_path = p.joinpath('check_plots')
                plot_path.mkdir(exist_ok=True)

                # setting the list of science, mask and error frames
                scienceframes = sorted(p.glob('%s_Rc_stack?.fits'%gal)) + sorted(p.glob('%s_Rc_stack??.fits'%gal))
                maskframes = sorted(p.glob('%s_Rc_stack?_mask.fits'%gal)) + sorted(p.glob('%s_Rc_stack??_mask.fits'%gal))
                errorframes = sorted(p.glob('%s_Rc_stack?_var.fits'%gal)) + sorted(p.glob('%s_Rc_stack??_var.fits'%gal))

                # setting the list of bad column and saturated pixels mask frames (new in NB-DR2 for WFI)
                newmaskframes = sorted(p.glob('%s_Rc_stack?_newmask.fits'%gal)) + sorted(p.glob('%s_Rc_stack??_newmask.fits'%gal))
                sp_maskframes = sorted(p.glob('%s_Rc_stack?_sp_mask.fits'%gal)) + sorted(p.glob('%s_Rc_stack??_sp_mask.fits'%gal))

                if (not scienceframes) or (not maskframes) or (not errorframes) :

                    print('\nThere are no frames with filter Rc for %s\n'%gal)
                    continue

                for scienceframe, maskframe, newmaskframe, sp_maskframe, errorframe in zip(scienceframes, maskframes, newmaskframes, sp_maskframes, errorframes) :

                    #if scienceframe != scienceframes[3] :
                    #    continue

                    ##### Aperture photometry with ImageStars object #####

                    # Instantiate ImageStars class
                    stars = ImageStars('WFI', cal_path, tbl_path, plot_path, gal, scienceframe, maskframe, newmaskframe, sp_maskframe, errorframe, args.fwhm)

                    # Merge the masks created for NB-DR2
                    stars.merge_masks()

                    # Find stars in the image
                    stars.find_image_stars(threshold_nstd=5.)

                    # Instantiate Gaia stars
                    gs = GAIAStars(stars.obj, stars.instr)
                    gs.gaia_query(args.g_corr)

                    # Match stars found in the stacked image (common projection) with Gaia stars in the image field
                    stars.match_with_gaia(gs)

                    # Create tuples of star positions and filter those too close to each other or to masked pixels
                    stars.star_positions('WFI')

                    # Aperture photometry
                    stars.annulus_photometry(args.snr)

                    # Setting the airmass
                    stars.set_airmass()

                    # Setting the filter name
                    stars.set_filter(filt)

                    # Instantiate the ZPParams class
                    myZP = ZPParams(stars)

                    # Zero point of the given frame
                    myZP.find_zp(stars)
                    myZP.calib_images(stars)
                    myZP.write_calib_to(stars)


        # Loop for CCD

        else :

            filt = 'r'

            for gal in list_of_galaxy :

                print('\nCalibrating %s ...\n'%gal)

                p = Path(args.ccd_path)

                print('Images path :\n%s'%p)

                # setting the path to calibrated file
                cal_path = p.parent.joinpath('phot_calib')
                cal_path.mkdir(exist_ok=True)

                # setting the path to save the tabels with results from photometry for each frame
                tbl_path = p.parent.joinpath('phot_tables')
                tbl_path.mkdir(exist_ok=True)

                # setting the path to save the plots
                plot_path = p.parent.joinpath('check_plots')
                plot_path.mkdir(exist_ok=True)

                # setting the list of science, mask and error frames
                scienceframes = sorted(p.glob('%s_r*_wcs_reproj.fits'%gal))
                maskframes = sorted(p.glob('%s_r*_mask_reproj.fits'%gal))
                errorframes = sorted(p.glob('%s_r*_unc_reproj.fits'%gal))

                # setting the list of bad column and saturated pixels mask frames (new in NB-DR2 for WFI)
                newmaskframes = []
                sp_maskframes = []

                if (not scienceframes) or (not maskframes) or (not errorframes) :

                    print('\nThere are no frames with filter r for %s\n'%gal)
                    continue

                for scienceframe, maskframe, errorframe in zip(scienceframes, maskframes, errorframes) :

                    #if scienceframe != scienceframes[2] :
                    #    continue

                    ##### Aperture photometry with ImageStars object #####

                    # Instantiate ImageStars class
                    stars = ImageStars('CCD', cal_path, tbl_path, plot_path, gal, scienceframe, maskframe, newmaskframes, sp_maskframes, errorframe, args.fwhm)

                    # Find stars in the image
                    stars.find_image_stars(threshold_nstd=5.)

                    # Instantiate Gaia stars
                    gs = GAIAStars(stars.obj, stars.instr)
                    gs.gaia_query(args.g_corr)

                    # Match stars found in the stacked image (common projection) with Gaia stars in the image field
                    stars.match_with_gaia(gs)

                    # Create tuples of star positions and filter those too close to each other or to masked pixels
                    stars.star_positions('CCD')

                    # Aperture photometry
                    stars.annulus_photometry(args.snr)

                    # Setting the airmass
                    stars.set_airmass()

                    # Setting the filter name
                    stars.set_filter(filt)

                    # Instantiate the ZPParams class
                    myZP = ZPParams(stars)

                    # Zero point of the given frame
                    myZP.find_zp(stars)
                    myZP.calib_images(stars)
                    myZP.write_calib_to(stars)



    elapsed = timeit.default_timer() - start_time
    minutes, seconds = divmod(elapsed, 60)
    hours, minutes = divmod(minutes, 60)
    print ('\nProcess terminated in %d:%02d:%02d'%(hours, minutes, seconds))
