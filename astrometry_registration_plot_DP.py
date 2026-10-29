# coding: utf-8

import timeit
import sys
import math

import argparse

import numpy as np

from pathlib import Path

from astropy.coordinates import SkyCoord
from astropy.io import ascii, fits
from astropy.table import Table
from astropy import units as u
from astropy.visualization import ZScaleInterval
from astropy.wcs import WCS
from astropy.wcs.utils import skycoord_to_pixel

from photutils.aperture import EllipticalAperture

import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

from galaxy_shape_v21 import GalaxyEllipse


class ImageAstrometry (object) :
    '''
    Class to store the astrometry table for each galaxy.
    '''
    
    def __init__ (self, gal_file, path, suffix) :
        
        self.gal_name = gal_file.name[:gal_file.name.find('_')]
        
        print('\nReading table %s ...'%gal_file.name)
        
        # Check if the table is empty
        
        mat = np.loadtxt(gal_file)
        if mat.size != 0 :
            self.tbl = ascii.read(gal_file)
        else :
            self.tbl = Table()

        self.filename = Path(path).joinpath(gal_file.name.replace(suffix, '.fits'))

    
    def set_coordinates (self) :
        
        self.gaia_coord = SkyCoord(ra=self.tbl['gaia_ra']*u.deg, dec=self.tbl['gaia_dec']*u.deg, frame='icrs')
        self.img_coord = SkyCoord(ra=self.tbl['ra']*u.deg, dec=self.tbl['dec']*u.deg, frame='icrs')


    def galaxy_ellipse (self, filt) :
        '''
        Methods to call the GalaxyEllipse class and its methods
        '''

        self.myellipse = GalaxyEllipse(self.gal_name, filt, self.filename, None)
        self.myellipse.get_ellipse()
        self.myellipse.ellipse_to_pixel('CCD')


    def ellipse_check (self, plot_path, filt, string, phot_sel, r) :
        '''
        Method to check whether a point in RA/DEC coordinates is inside or outside the galaxy ellipse
        '''
        
        def rotate(origin, point, angle):             
            """                            
            Rotate a point counterclockwise by a given angle around a given origin.
                                            
            The angle should be given in radians.  
            """                   
            ox, oy = origin
            px, py = point
            
            qx = math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
            qy = math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
            
            return qx, qy

        origin = (self.myellipse.x0, self.myellipse.y0)
        angle = math.radians(90-self.myellipse.pa.value)
        self.myellipse.a_pix = self.myellipse.a_pix * float(r)
        self.myellipse.b_pix = self.myellipse.b_pix * float(r)
        a = self.myellipse.a_pix.value
        b = self.myellipse.b_pix.value
        
        self.myellipse.aperture = EllipticalAperture(origin, a, b, theta=math.radians(self.myellipse.pa.value+90))
        
        fig, ax = plt.subplots(1, 1, figsize=(7, 7))
        
        zscale = ZScaleInterval()
        
        vmin, vmax = zscale.get_limits(self.myellipse.data)
        ax.imshow(self.myellipse.data, cmap='gray_r', origin='lower', vmin=vmin, vmax=vmax)

        ax.set_title('%s (%s)'%(self.gal_name,filt))
        ax.set_xlabel('x [pixel]')
        ax.set_ylabel('y [pixel]')

        self.myellipse.aperture.plot(axes=ax, color='C1', ls='--', lw=1.2)

        self.position = []
        
        for i in range(len(self.gaia_coord)) :
            
            #ra, dec = (self.gaia_coord[i].ra.deg, self.gaia_coord[i].dec.deg)
            
            xp, yp = skycoord_to_pixel(self.gaia_coord[i], self.myellipse.wcs)

            x, y = rotate(origin, (xp, yp), angle)
            
            if (x/a)**2+(y/b)**2<=1 :
                #print(ra, dec, 'inside')
                self.position.append('inside')
                color = 'red'
            else :
                #print(ra, dec, 'outside')
                self.position.append('outside')
                color = 'blue'

            plt.scatter(xp, yp, s=50, marker='o', facecolor = 'None', color=color)

        if phot_sel :
            string = string + '_phot_sel'

        fig.savefig(Path(plot_path).joinpath('%s_%s_%s_stars.png'%(self.gal_name,filt,string)))

        plt.close()


    def compute_radec_offset (self, include_ellipse) :
        '''
        This method makes use of the SkyCoord method to compute 
        RA and DEC spherical offset from two coordinates given 
        in the same system. In this case the offset is between 
        stars in the image and Gaia. The mean offset is
        computed.
        When include_ellipse is True, the stars within the galaxy ellipse
        are included in the astrometric precision computation. 
        '''
        
        ra_sep, dec_sep = self.gaia_coord.spherical_offsets_to(self.img_coord)
        self.ra_sep = ra_sep.to(u.arcsec)
        self.dec_sep = dec_sep.to(u.arcsec)
        
        if include_ellipse :

            self.ra_mean_sep = np.mean(self.ra_sep) 
            self.dec_mean_sep = np.mean(self.dec_sep)
            self.ra_std_sep = np.std(self.ra_sep)
            self.dec_std_sep = np.std(self.dec_sep)

        else :
            
            self.ra_mean_sep = np.mean(self.ra_sep[np.array(self.position)=='outside']) 
            self.dec_mean_sep = np.mean(self.dec_sep[np.array(self.position)=='outside'])
            self.ra_std_sep = np.std(self.ra_sep[np.array(self.position)=='outside'])
            self.dec_std_sep = np.std(self.dec_sep[np.array(self.position)=='outside'])


    def compute_ang_sep (self, include_ellipse) :
        '''
        The method computes the mean angular separation between 
        stars in the image and Gaia.
        When include_ellipse is True, the stars within the galaxy ellipse
        are included from the astrometric precision computation. 
        '''

        seps = self.gaia_coord.separation(self.img_coord)
        self.seps = seps.to(u.arcsec)
        
        if include_ellipse :
            
            self.mean_sep = np.mean(self.seps)
            self.std_sep = np.std(self.seps)

        else :

            self.mean_sep = np.mean(self.seps[np.array(self.position)=='outside'])
            self.std_sep = np.std(self.seps[np.array(self.position)=='outside'])


    def set_image_header (self) :
        '''
        Method to open the fits image and load the header.
        The fits filename is also saved in an attribute.
        ''' 
        
        self.hdr = fits.open(self.filename)[0].header


    def set_image_wcs (self) :

        self.wcs = WCS(self.hdr)


    def update_image_header (self, filt, catname) :
        '''
        Update the image header with astrometric precision.
        A card with the ZERO POINT from micro Jansky to 
        bandpass AB magnitude is also provided.
        '''
        
        # Save the zero point
        self.hdr.set('ZPOINT', 23.9, 'ZERO POINT from micro Jansky to bandpass AB magnitude')
        
        # Save the astrometric precision in the reprojected image header
        print('\nSaving the following estimates of the astrometric solution in the header ...')
        print('Number of stars used to estimate the astrometric solution: ',len(self.tbl))
        print('Mean separation in RA with %s [arcsec]: '%catname,self.ra_mean_sep.value)
        print('Mean separation in DEC with %s [arcsec]: '%catname,self.dec_mean_sep.value)
        print('Std of separation in RA with %s [arcsec]: '%catname,self.ra_std_sep.value)
        print('Std of separation in DEC with %s [arcsec]: '%catname,self.dec_std_sep.value)
        print('Mean angular separation with %s [arcsec]: '%catname,self.mean_sep.value)
        print('Std of the angular separation with %s [arcsec]: ',self.std_sep.value)
        self.hdr.set('NSTARS', len(self.tbl), '# of stars to estimate the astrometric solution')
        self.hdr.set('RASEP', self.ra_mean_sep.value, '[arcsec] Mean separation in RA (%s)'%catname)
        self.hdr.set('DESEP', self.dec_mean_sep.value, '[arcsec] Mean separation in DEC (%s)'%catname)
        self.hdr.set('RASEPRMS', self.ra_std_sep.value, '[arcsec] Std of separation in RA (%s)'%catname)
        self.hdr.set('DESEPRMS', self.dec_std_sep.value, '[arcsec] Std of separation in DEC (%s)'%catname)
        self.hdr.set('ANGSEP', self.mean_sep.value, '[arcsec] Mean ang separation (%s)'%catname)
        self.hdr.set('ANGRMS', self.std_sep.value, '[arcsec] Std of the ang separation (%s)'%catname)
        
        # Update the image fits file header
        
        hdulist = fits.open(str(self.filename))
        hdulist[0].header = self.hdr
        hdulist.writeto(str(self.filename), overwrite=True)

        hdulist = fits.open(str(self.filename).replace('flux','std'))
        hdulist[0].header = self.hdr
        hdulist.writeto(str(self.filename).replace('flux','std'), overwrite=True)
        
        hdulist = fits.open(str(self.filename).replace('comb','wcomb'))
        hdulist[0].header = self.hdr
        hdulist.writeto(str(self.filename).replace('comb','wcomb'), overwrite=True)
        
        hdulist = fits.open(str(self.filename).replace('flux','std').replace('comb','wcomb'))
        hdulist[0].header = self.hdr
        hdulist.writeto(str(self.filename).replace('flux','std').replace('comb','wcomb'), overwrite=True)
            

    def correct_astrometry (self) :
        '''
        Method to apply the offset found to further correct the astrometry.
        '''
    
        input_data_comb = str(self.filename)
        input_std_comb = str(self.filename).replace('flux','std')
        input_data_wcomb = str(self.filename).replace('comb','wcomb')
        input_std_wcomb = str(self.filename).replace('flux','std').replace('comb','wcomb')
        input_hdr = self.hdr
        
        ra_offset = self.ra_mean_sep
        dec_offset = self.dec_mean_sep
            
        # Apply a simple traslation to the image WCS opposite to the RA/DEC offsets
        
        if 'RAOFFSET' not in input_hdr.keys() :
            
            input_hdr['CRVAL1'] = input_hdr['CRVAL1'] - ra_offset.to('deg').value
            input_hdr['CRVAL2'] = input_hdr['CRVAL2'] - dec_offset.to('deg').value
            
            if 'Astrometry' not in input_hdr['COMMENT'] : 
                input_hdr.add_comment('', after='ZPOINT')
                input_hdr.add_comment('Astrometry', before='NSTARS')
            
            input_hdr.set('RAOFFSET', - ra_offset.value, '[arcsec] Offset in RA applied to CRVAL1', after='ANGRMS')
            input_hdr.set('DEOFFSET', - dec_offset.value, '[arcsec] Offset in DEC applied to CRVAL2', after='RAOFFSET')
            
            print('\nApplying offset to %s ...'%input_data_comb)
            fits.writeto(input_data_comb, data=fits.open(input_data_comb)[0].data, header=input_hdr, overwrite=True)
            print('Applying offset to %s ...'%input_std_comb)
            fits.writeto(input_std_comb, data=fits.open(input_std_comb)[0].data, header=input_hdr, overwrite=True)
            print('\nApplying offset to %s ...'%input_data_wcomb)
            fits.writeto(input_data_wcomb, data=fits.open(input_data_wcomb)[0].data, header=input_hdr, overwrite=True)
            print('Applying offset to %s ...'%input_std_wcomb)
            fits.writeto(input_std_wcomb, data=fits.open(input_std_wcomb)[0].data, header=input_hdr, overwrite=True)



def make_astrometry_plots (sep_dic, plot_path, filt, string, phot_sel, show_inside, gal=None, add_cross=False) :

    dp_ra_sep = np.array([item.value for sublist in sep_dic['dp_ra_sep'] for item in sublist]) 
    dp_de_sep = np.array([item.value for sublist in sep_dic['dp_de_sep'] for item in sublist]) 
    position = np.array([item for sublist in sep_dic['position'] for item in sublist]) 

    cmap = sns.cubehelix_palette(dark=.3, light=.8, as_cmap=True)
    #cmap = sns.cubehelix_palette(dark=.2, light=.8, as_cmap=True)
    palette = sns.cubehelix_palette(dark=.3, light=.8, as_cmap=False)
    color1 = (palette[0][0], palette[0][1], palette[0][2])
    color2 = (palette[1][0], palette[1][1], palette[1][2])
    color3 = (palette[2][0], palette[2][1], palette[2][2])
    color4 = (palette[3][0], palette[3][1], palette[3][2])
    color5 = (palette[4][0], palette[4][1], palette[4][2])
    color6 = (palette[5][0], palette[5][1], palette[5][2])

    fig = plt.figure(figsize=(7,7))
    
    gs = gridspec.GridSpec(2, 2, height_ratios=[0.3,1], width_ratios=[1,0.3])
    gs.update(left=0.15, right=0.95, bottom=0.15, top=0.85, wspace=0.03, hspace=0.03)
       
    # ax1
    
    ax1 = fig.add_subplot(gs[1,0])
    ax1_histx = fig.add_subplot(gs[0,0], sharex=ax1) 
    ax1_histy = fig.add_subplot(gs[1,1], sharey=ax1)
    
    ax1.scatter(dp_ra_sep[position=='outside'], dp_de_sep[position=='outside'], color=color2, alpha=0.4)
    
    if show_inside :
        ax1.scatter(dp_ra_sep[position=='inside'], dp_de_sep[position=='inside'], color='tab:orange', alpha=0.4)
    
    mean_dist_ra = np.mean(dp_ra_sep[position=='outside'])
    mean_dist_de = np.mean(dp_de_sep[position=='outside'])
            
    std_dist_ra = np.std(dp_ra_sep[position=='outside'])
    std_dist_de = np.std(dp_de_sep[position=='outside'])
    
    print('\n\nMean deltaRA : %s +/- %s'%(mean_dist_ra, std_dist_ra))
    print('Mean deltaDEC : %s +/- %s\n\n'%(mean_dist_de, std_dist_de))
    
    # Rectagle representing the mean of RA/DEC distribution and its error
    xy = (mean_dist_ra-std_dist_ra, mean_dist_de-std_dist_de)
     
    rect_ax1 = Rectangle(xy, width=2*std_dist_ra, height=2*std_dist_de, edgecolor=color5, facecolor='None', lw=2, zorder=99)
    
    # Sigma of the RA/DEC distributions
    ax1_histx.errorbar(mean_dist_ra, 0.6, xerr=std_dist_ra, 
        color=color5, fmt='none', lw=2, capsize=4, transform=ax1_histx.get_xaxis_transform(), zorder=99)
    ax1_histx.text(0.7, 0.75, r'$\sigma=%s$ mas'%int(round(std_dist_ra*1000)), transform=ax1_histx.transAxes,
        color=color5, fontsize=11, horizontalalignment='left', verticalalignment='center', zorder=99)
    ax1_histy.errorbar(0.6, mean_dist_de, yerr=std_dist_de, 
        color=color5, fmt='none', lw=2, capsize=4, transform=ax1_histy.get_yaxis_transform(), zorder=99)
    ax1_histy.text(0.75, 0.3, r'$\sigma=%s$ mas'%int(round(std_dist_de*1000)), rotation=-90, transform=ax1_histy.transAxes,
        color=color5, fontsize=11, horizontalalignment='center', verticalalignment='top', zorder=99)
    
    # Cross representing the mean of RA/DEC distribution
    
    if add_cross :
        
        mean_dist_ra_inside = np.mean(dp_ra_sep[position=='inside'])
        mean_dist_de_inside = np.mean(dp_de_sep[position=='inside'])
        ax1.scatter(mean_dist_ra_inside, mean_dist_de_inside, marker='+', 
            ec='tab:red', s=80, label='%.3f %.3f'%(mean_dist_ra_inside, mean_dist_de_inside))
        ax1.legend(loc='lower right', frameon=False, labelcolor='tab:red', handletextpad=0.2, fontsize=12)
    
    else :
    
        mean_dist_ra_outside = np.mean(dp_ra_sep[position=='outside'])
        mean_dist_de_outside = np.mean(dp_de_sep[position=='outside'])
        std_dist_ra_outside = np.std(dp_ra_sep[position=='outside'])
        std_dist_de_outside = np.std(dp_de_sep[position=='outside'])
        string1 = r'$\Delta\alpha = %.3f\pm %.3f$'%(mean_dist_ra_outside, std_dist_ra_outside)
        string2 = r'$\Delta\delta = %.3f\pm %.3f$'%(mean_dist_de_outside, std_dist_de_outside)
        label = string1 + '\n' + string2 
        ax1.scatter(mean_dist_ra_outside, mean_dist_de_outside, marker='+', ec='tab:red', s=80, label=label)
        ax1.legend(loc='lower right', frameon=False, labelcolor='tab:red', handletextpad=0.2, fontsize=12)
       
    ax1.set_xlabel(r'$\Delta\alpha$ [arcsec]')
    ax1.set_ylabel(r'$\Delta\delta$ [arcsec]')
    if gal==None :
        ax1.text(0.05, 0.9, 'CCD (%s)'%filt, transform=ax1.transAxes)
    else :
        ax1.text(0.05, 0.9, '%s (%s)'%(gal,filt), transform=ax1.transAxes)
    
    ax1.axvline(x=0, color='k', ls=':', lw=1)
    ax1.axhline(y=0, color='k', ls=':', lw=1)
    
    ax1_histx.axvline(x=0, color='k', ls=':', lw=1)
    ax1_histy.axhline(y=0, color='k', ls=':', lw=1)
            
    ax1_histx.xaxis.set_tick_params(labelbottom=False)
    ax1_histy.yaxis.set_tick_params(labelleft=False)
    
    ax1.add_patch(rect_ax1)


    # Both axes
    
    # Set x- and y-axes limit equal
    max_lim = 0.5
    ax1.set_xlim(-max_lim,max_lim)
    ax1.set_ylim(-max_lim,max_lim)
    
    bins = np.arange(-max_lim,max_lim,2*max_lim/50)  
        
    ax1_histx.hist(dp_ra_sep, bins=bins, color=color2)
    ax1_histy.hist(dp_de_sep, bins=bins, orientation='horizontal', color=color2)
    
    plt.show()

    if phot_sel :
        string = string + '_phot_sel'
                  
    if gal==None :   
        fig.savefig(Path(plot_path).joinpath('CCD_all_%s_%s.png'%(filt,string)))
    else :
        fig.savefig(Path(plot_path).joinpath('%s_%s_%s.png'%(gal,filt,string)))

    plt.close()



if __name__ == "__main__":
    
    start_time = timeit.default_timer()
     
    ccd_galaxies = ['NGC0685', 'NGC1317', 'NGC1326', 'IC1954', 'IC1993', 'NGC1511', 'NGC1546', 'NGC1559',
                      'NGC1809',
                      'NGC1792', 'NGC2090', 'NGC2283', 'NGC2566', 'NGC2775', 'NGC3059', 'NGC3507',
                        'NGC3626', 'NGC4207', 'NGC4424', 'NGC4457', 'NGC4540', 'NGC4548', 'NGC4689', 'NGC4694',
                          'NGC4781', 'NGC4941', 'NGC4951', 'NGC5042', 'NGC5134', 'NGC5248', 'NGC5530', 'NGC5643',
                            'NGC6300', 'IC5273', 'NGC7456', 'NGC7496',
                            'NGC3489', 'NGC3599', 'NGC4459', 'NGC4476', 'NGC4477', 'NGC4596' # ETGs 
                            ]

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', action='store', type=str, 
        default='/Users/arazza/DAS/Observations/NB_DR31_pipeline/final_images/CCD/fits', help='path to galaxy final images')
    parser.add_argument('--tbl_path', action='store', type=str, 
        default='/Users/arazza/DAS/Observations/NB_DR31_pipeline/CCD_tables', help='path to astrometric tables')
    parser.add_argument('--plot_path', action='store', type=str, 
        default='/Users/arazza/DAS/Observations/NB_DR31_pipeline/CCD_astrometry_plots', help='path where to save duPont astrometric plots')
    parser.add_argument('--ccd_galaxies', nargs='*', type=str, default=ccd_galaxies, help='list of selected duPont galaxies')
    parser.add_argument('--ccd_filters', nargs='*', type=str, default=['r', 'Ha657', 'Ha663'], help='list of CCD filters')
    parser.add_argument('--gaia_dr', action='store', type=str, default='Gaia-DR1', 
        help='gaia data release to use as reference (GAIA-DR1, GAIA-DR2, GAIA-EDR3, default GAIA-DR1)')
    parser.add_argument('--phot_sel', action='store_true', default=False, help='create table with Gaia photometrically selected stars')
    #parser.add_argument('--exclude_ellipse', action='store_true', default=False, 
    #    help='exclude stars in the galaxy ellipse in astrometry precision computation')
    parser.add_argument('-e', '--ell_radius', action='store', default=1.2, help='galaxy ellipse increasing factor')
    parser.add_argument('--include_ellipse', action='store_true', default=False, 
        help='include stars in the galaxy ellipse in astrometry precision computation')
    parser.add_argument('--correct_astrometry', action='store_true', default=False, help='correct the astrometry and save the statistics in the headers')
    parser.add_argument('--make_allstars_plot', action='store_true', default=False, help='make the plot with all the stars from all the galaxies')
    parser.add_argument('--show_inside', action='store_true', default=False, help='show the stars inside the galaxy ellipse in the plots')
    args = parser.parse_args()
    
    path_to_tables = Path(args.tbl_path)
    
    plot_path = Path(args.plot_path)
    plot_path.mkdir(exist_ok=True)

    for filt in args.ccd_filters :

        print('\nFilter %s ...\n'%filt)

        # Dictionary of lists necessary for all stars plots
        
        sep_dic_all = {
            'dp_object' : [], 'dp_nstars' : [], 'dp_angsep' : [], 
            'dp_angerr' : [], 'dp_ra_sep' : [], 'dp_de_sep' : [],
            'position' : []
            }
        
        for gal in args.ccd_galaxies :

            # Dictionary of lists necessary for individual galaxies plots

            sep_dic_gal = {
            'dp_object' : [], 'dp_nstars' : [], 'dp_angsep' : [], 
            'dp_angerr' : [], 'dp_ra_sep' : [], 'dp_de_sep' : [],
            'position' : []
            }

            # Instantiate the galaxy table name

            catname = args.gaia_dr

            string = 'gaia' + catname[catname.find('-')+1:]

            if (args.phot_sel) * (catname!='Gaia-DR1') :

                suffix = '_%s_tbl_phot_sel.dat'%string
            
            else :

                suffix = '_%s_tbl.dat'%string

            gal_file = Path(path_to_tables).joinpath('%s_%s_flux_comb%s'%(gal,filt,suffix))
            
            if not gal_file.exists() :
                continue

            # Instantiate ImageAstrometry object
            
            imastro = ImageAstrometry(gal_file, args.path, suffix)
            
            if len(imastro.tbl) == 0 :
                continue
            
            imastro.set_coordinates()   

            imastro.galaxy_ellipse(filt)
        
            imastro.ellipse_check(plot_path, filt, string, args.phot_sel, args.ell_radius)
             
            imastro.compute_radec_offset(args.include_ellipse)
             
            imastro.compute_ang_sep(args.include_ellipse)
            
            imastro.set_image_header()

            imastro.set_image_wcs()
                
            imastro.update_image_header(filt, catname)

            if args.correct_astrometry :
                
                imastro.correct_astrometry()
        
            sep_dic_all['dp_object'].append(gal)
            sep_dic_all['dp_nstars'].append(len(imastro.tbl))
            sep_dic_all['dp_angsep'].append(imastro.mean_sep.value)
            sep_dic_all['dp_angerr'].append(imastro.std_sep.value/len(imastro.tbl))
            sep_dic_all['dp_ra_sep'].append(imastro.ra_sep) 
            sep_dic_all['dp_de_sep'].append(imastro.dec_sep)
            sep_dic_all['position'].append(imastro.position)

            sep_dic_gal['dp_object'].append(gal)
            sep_dic_gal['dp_nstars'].append(len(imastro.tbl))
            sep_dic_gal['dp_angsep'].append(imastro.mean_sep.value)
            sep_dic_gal['dp_angerr'].append(imastro.std_sep.value/len(imastro.tbl))
            sep_dic_gal['dp_ra_sep'].append(imastro.ra_sep) 
            sep_dic_gal['dp_de_sep'].append(imastro.dec_sep)
            sep_dic_gal['position'].append(imastro.position)

            make_astrometry_plots (sep_dic_gal, plot_path, filt, string, args.phot_sel, args.show_inside, gal)
        
        if (sep_dic_all['dp_object']) * (args.make_allstars_plot) :

            make_astrometry_plots (sep_dic_all, plot_path, filt, string, args.phot_sel, args.show_inside)	   
    
    
    elapsed = timeit.default_timer() - start_time
    minutes, seconds = divmod(elapsed, 60)
    hours, minutes = divmod(minutes, 60)
    print ('Process terminated in %d:%02d:%02d'%(hours, minutes, seconds))

