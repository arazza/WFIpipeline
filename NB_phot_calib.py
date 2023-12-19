# coding: utf-8

'''
Narrow-band images photometric calibration.

The code finds the zero points for each frame
by calibrating the narrow-band (NB) images against
the broad-band (BB) ones via a BB-NB vs Gaia colors
relation and makes the plots to show the method.

It works for both WFI and DP data set.
'''

import sys

import timeit

import argparse

from pathlib import Path

import numpy as np

import pandas as pd

import math as m

from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.table import Column
from astropy.stats import SigmaClip

from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from BB_phot_calib import ImageStars, GAIAStars, ZPParams


class ZPHalphaParams (ZPParams) :
    '''
    Class inherited from r_phot_calib
    '''

    # Zero poin computation and plots
    def find_ha_zp (self, stars) :

        # fluxes of stars in Halpha frame
        ha_fluxes = stars.tbl['flux_apert']
        ha_err_fluxes = stars.tbl['flux_apert_err']

        # Halpha--r offset computation
        '''
        H\alpha = r - \alpha - \beta\,(G_{\mathrm{BP}}-G_{\mathrm{RP}})\\
        \sigma_{H\alpha} = \sqrt{\sigma_{r}^2 + \sigma_\alpha^2 + (G_{\mathrm{BP}}-G_{\mathrm{RP}})^2\,\sigma_\beta^2 + \beta^2\,\sigma_{G_{\mathrm{BP}}-G_{\mathrm{RP}}}^2}
        '''

        #intercept, sigma_intercept, slope, sigma_slope = np.load('StdStar_phot_calib_%s.npy'%stars.filt)
        df = pd.read_csv(Path('CC_relation').joinpath('CC_relation.csv'))
        a0, a1, a2, a3, a4, y_std = df.filter(regex='%s$'%stars.filt, axis=1).values[:,0]

        ### Old CC relation (DEPRECATED) ###
        #intercept_arr = np.full(bb_fluxes.shape, intercept)
        #sigma_intercept_arr = np.full(bb_fluxes.shape, sigma_intercept)
        #NB_mag =  bb - intercept - slope*(gaia_color)
        #sigma_NB_mag = np.sqrt(bb_err**2+sigma_intercept**2+(gaia_color)**2*sigma_slope**2+slope**2*gaia_color_err**2)
        ####################################

        # broad-band fluxes and colors of stars matching those in the Halpha frame
        if stars.instr == 'WFI' :
            gaia_bb = stars.tbl['gaia_R_mag']
            gaia_bb_err = stars.tbl['gaia_R_mag_err']
        if stars.instr == 'CCD' :
            gaia_bb = stars.tbl['gaia_r_mag']
            gaia_bb_err = stars.tbl['gaia_r_mag_err']
        gaia_color = stars.tbl['gaia_BP_RPmag']
        gaia_color_err = stars.tbl['gaia_BP_RPmag_err']

        # VEGA-to-AB offset from Bessel & Murphy (2012)
        if stars.instr == 'WFI' :
            vega_to_ab = 0.163
        if stars.instr == 'CCD' :
            vega_to_ab = 0.
        print('\nApplying VEGA-to-AB offset of %s\n'%vega_to_ab)

        NB_mag = gaia_bb + vega_to_ab - a0 - a1 * gaia_color - a2 * gaia_color**2. - a3 * gaia_color**3. - a4 * gaia_color**4.
        sigma_NB_mag = np.sqrt(gaia_bb_err**2. + y_std**2.)

        # atmosferic extinction coefficient
        k = self.exct
        sigma_k = self.sigma_exct

        # Before clipping
        iZP = np.array([NB + 2.5*m.log10(f) + k*stars.airmass for NB,f in zip(NB_mag,ha_fluxes)])
        sigma_iZP = np.array([np.sqrt (sigma_NB**2 + 1.179*(sigma_f/f)**2 + (stars.airmass*sigma_k)**2) for sigma_NB, f, sigma_f in zip(sigma_NB_mag, ha_fluxes, ha_err_fluxes)])
        err_weight = np.sum(1/(sigma_iZP**2))
        ZP = np.sum(1/(sigma_iZP**2)*iZP)/err_weight
        sigma_ZP = np.sqrt(1/err_weight)
        print('Before clipping ZP : ', ZP, sigma_ZP)

        zp = Column(iZP, name='zero_point')
        err_zp = Column(sigma_iZP, name='zero_point_err')

        stars.tbl.add_column(zp)
        stars.tbl.add_column(err_zp)

        stars.tbl['zero_point'].info.format = '%.8f'
        stars.tbl['zero_point_err'].info.format = '%.8f'

        tbl_filename = '%s_%s_%s_phot_table.dat'%(stars.obj,stars.filt,stars.filenum)

        stars.tbl.write(stars.tbl_path.joinpath(tbl_filename), format='ascii.commented_header',overwrite=True)

        # Plot 1
        stars.ax2.errorbar (NB_mag, iZP, xerr=sigma_NB_mag, yerr=sigma_iZP, fmt='none', ms=4, alpha=0.5, ecolor='lightgray', zorder=1)
        im = stars.ax2.scatter (NB_mag, iZP, c=gaia_color, marker='o', s=30, alpha=0.5, zorder=2)
        divider = make_axes_locatable(stars.ax2)
        cax = divider.append_axes("right", size = "5%", pad = 0.05)
        cb = plt.colorbar(im, cax = cax)
        cb.mappable.set_clim(0.2, 1.8)
        cb.set_label(r'$G_\mathrm{BP}-G_\mathrm{RP}$ [mag]', labelpad=10)
        stars.ax2.set_xlabel(r'NB$_{H_\alpha}$ [mag]')
        stars.ax2.set_ylabel(r'ZP$_{H_\alpha}$ [mag]')

        # Sigma clipping of ZP
        sigclip = SigmaClip(sigma=3, maxiters=5)
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
        stars.ax2.scatter(NB_mag[iZP_filt.mask], iZP[iZP_filt.mask], s=50, marker='x', c='C3')
        # Plot again to resize and get the limits
        stars.ax2.scatter (NB_mag, iZP, c=gaia_color, marker='o', s=30, alpha=0.5, zorder=2)
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

    parser.add_argument('--wfi_path', action='store', type=str, default=None, help='path to uncalibrated WFI frames')
    parser.add_argument('--ccd_path', action='store', type=str,
        default='/home/arazza/NB_photometry/dirCCDreduction/DuPontDATA/PROCESSED/reproj/', help='path to uncalibrated du Pont frames')
    parser.add_argument('-w', '--wfi', nargs='*', type=str, default=wfi_galaxies,
        help='list of selected WFI galaxies (type a non-valid galaxy to avoid calibrating WFI)')
    parser.add_argument('-c', '--ccd', nargs='*', type=str, default=ccd_galaxies,
        help='list of selected CCD galaxies (type a non-valid galaxy to avoid calibrating CCD)')
    parser.add_argument('-f', '--fwhm', action='store', type=float, default=4.0, help='FWHM to be used for IRAFStarFinder')
    parser.add_argument('--snr', action='store', type=float, default=20., help='threshold for rejecting low SNR as computed with the aperture photometry')

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
                scienceframes = sorted(p.glob('%s_Ha_stack?.fits'%gal)) + sorted(p.glob('%s_Ha_stack??.fits'%gal))
                maskframes = sorted(p.glob('%s_Ha_stack?_mask.fits'%gal)) + sorted(p.glob('%s_Ha_stack??_mask.fits'%gal))
                errorframes = sorted(p.glob('%s_Ha_stack?_var.fits'%gal)) + sorted(p.glob('%s_Ha_stack??_var.fits'%gal))

                # setting the list of bad column and saturated pixels mask frames (new in NB-DR2 for WFI)
                newmaskframes = sorted(p.glob('%s_Ha_stack?_newmask.fits'%gal)) + sorted(p.glob('%s_Ha_stack??_newmask.fits'%gal))
                sp_maskframes = sorted(p.glob('%s_Ha_stack?_sp_mask.fits'%gal)) + sorted(p.glob('%s_Ha_stack??_sp_mask.fits'%gal))

                if (not scienceframes) or (not maskframes) or (not errorframes) :

                    print('\nThere are no frames with filter Ha for %s\n'%gal)
                    continue

                for scienceframe, maskframe, newmaskframe, sp_maskframe, errorframe in zip(scienceframes, maskframes, newmaskframes, sp_maskframes, errorframes) :

                    #if scienceframe != scienceframes[0] :
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
                    gs.gaia_query()

                    # Match stars found in the stacked image (common projection) with Gaia stars in the image field
                    stars.match_with_gaia(gs)

                    # Create tuples of star positions and filter those too close to each other or to masked pixels
                    stars.star_positions('WFI')

                    # Aperture photometry
                    stars.annulus_photometry(args.snr)

                    # Setting the airmass
                    stars.set_airmass()

                    # Setting the filter name
                    stars.set_filter('Ha')

                    # Instantiate the ZPParams class
                    myZP = ZPHalphaParams(stars)

                    # Zero point of the given frame
                    myZP.find_ha_zp(stars)
                    myZP.calib_images (stars)
                    myZP.write_calib_to(stars)


        # Loop for CCD

        else :

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
                scienceframes = sorted(p.glob('%s_Ha*_wcs_reproj.fits'%gal))
                maskframes = sorted(p.glob('%s_Ha*_mask_reproj.fits'%gal))
                errorframes = sorted(p.glob('%s_Ha*_unc_reproj.fits'%gal))

                # filter name determination
                if scienceframes :
                    name = scienceframes[0].name
                    pos1 = name.find('_')
                    newname = name[pos1+1:]
                    pos2 = newname.find('_')
                    filt = newname[:pos2]

                # setting the list of bad column and saturated pixels mask frames (new in NB-DR2 for WFI)
                newmaskframes = []
                sp_maskframes = []

                if (not scienceframes) or (not maskframes) or (not errorframes) :

                    print('\nThere are no frames with filter Ha for %s\n'%gal)
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
                    gs.gaia_query()

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
                    myZP = ZPHalphaParams(stars)

                    # Zero point of the given frame
                    myZP.find_ha_zp(stars)
                    myZP.calib_images (stars)
                    myZP.write_calib_to(stars)



    elapsed = timeit.default_timer() - start_time
    minutes, seconds = divmod(elapsed, 60)
    hours, minutes = divmod(minutes, 60)
    print ('\nProcess terminated in %d:%02d:%02d'%(hours, minutes, seconds))
