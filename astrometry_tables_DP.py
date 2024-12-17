#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 2021

@author: arazza
"""

import sys
import timeit
import numpy as np
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from astroquery.vizier import Vizier
from astropy import units as u
from astropy.table import Table, Column
from astropy.io import fits
from astropy.wcs import WCS
from astropy.stats import sigma_clipped_stats, gaussian_fwhm_to_sigma
from astropy.coordinates import SkyCoord, search_around_sky
from astropy.nddata import NDData
from astropy.visualization import ZScaleInterval
from astropy.utils.data import conf

from scipy import ndimage

from photutils.detection import IRAFStarFinder

from galaxy_shape_v21 import GalaxyEllipse


class DPImage (object) :
    
    def __init__ (self, data, std, tbl_path, fwhm=5.0) :
        
        self.filename = data
        self.std_name = std
        self.data = fits.open(str(data))[0].data
        self.std = fits.open(str(std))[0].data
        self.hdr = fits.open(str(data))[0].header
        self.wcs = WCS(fits.open(str(data))[0].header)
        self.obj = data.stem[:data.stem.find('_')]    # string object name
        filter_string = data.stem[data.stem.find('_')+1:]
        self.filter = filter_string[:filter_string.find('_')]
        self.fwhm = fwhm
        self.tbl_path = tbl_path


    def mask_sources (self, snr) :

        #sigclip_sigma=3.0
        sigclip_sigma=2.0  
        sigclip_iters=5
        self.image_mean, self.image_median, self.image_std = sigma_clipped_stats(self.data, mask=np.isnan(self.data), sigma=sigclip_sigma, maxiters=sigclip_iters)    
        print (self.image_mean, self.image_median, self.image_std)

        
    def find_image_stars (self, ellipse) :
        '''
        This method computes the image threshold to find stars with 
        IRAFFindStars algorithm. An astropy Table is created with pixel positions 
        and instrumental fluxes in microJy. 
        '''
        
        # find stars in the input image
        threshold = self.image_median + (50. * self.image_std)
        print('Thereshold for finding bright stars', threshold)
        
        iraffind = IRAFStarFinder(threshold=threshold, fwhm=self.fwhm, exclude_border=True)
        dilate_size = 21
        selem = np.ones((dilate_size, dilate_size))
        mask = ndimage.binary_dilation(np.isnan(self.data), selem)

        self.myellipse = GalaxyEllipse(self.obj, self.filter, self.filename, mask)
        self.myellipse.get_ellipse()
        self.myellipse.ellipse_to_pixel('CCD')
        self.myellipse.mask_ellipse(1.2)
        
        if ellipse :
            
            self.mask = mask
            
        else :
            
            self.mask = mask + self.myellipse.mask
            
        self.tbl = iraffind(self.data, mask=self.mask)
        self.tbl.keep_columns(['xcentroid', 'ycentroid', 'flux'])
        
        
        
    def match_with_gaia (self, gaia_stars) :
        '''
        Centroid positions are converted in sky coordinates, by means of 
        the image wcs solution. Gaia stars are then retreived to be matched.
        '''
        
        # Convert pixel positions in coordinates
        ra, dec = self.wcs.all_pix2world(self.tbl['xcentroid'], self.tbl['ycentroid'], 0)
        coord = SkyCoord (ra=ra*u.deg, dec=dec*u.deg)
        
        # .. and update tbl
        ra = Column(coord.ra, name='ra')
        dec = Column(coord.dec, name='dec')
        self.tbl.add_column(ra)
        self.tbl.add_column(dec)
        
        # match with Gaia stars (closest one-to-one match)
        max_sep = 0.5 * u.arcsec
        idx, d2d, d3d = coord.match_to_catalog_3d(gaia_stars.coord)
        sep_constraint = d2d < max_sep
        print('Number of matching stars between the image and Gaia : ',len(coord[sep_constraint])) 
        
        # update tbl
        self.tbl = self.tbl[sep_constraint]
        gaia_ra = Column(gaia_stars.coord.ra[idx[sep_constraint]], name='gaia_ra')
        gaia_dec = Column(gaia_stars.coord.dec[idx[sep_constraint]], name='gaia_dec')
        
        self.tbl.add_column(gaia_ra)
        self.tbl.add_column(gaia_dec)

        self.tbl['xcentroid'].info.format = '%.4f'
        self.tbl['ycentroid'].info.format = '%.4f' 
        self.tbl['flux'].info.format = '%.4f' 
        self.tbl['ra'].info.format = '%.8f'
        self.tbl['dec'].info.format = '%.8f'
        self.tbl['gaia_ra'].info.format = '%.8f' 
        self.tbl['gaia_dec'].info.format = '%.8f'   


    def save_tbl (self, phot_sel, catname) :

        string = '_gaia' + catname[catname.find('-')+1:]

        if (phot_sel) * (catname!='Gaia-DR1') :
            
            tbl_filename = str(self.tbl_path.joinpath('%s%s_tbl_phot_sel.dat'%(self.filename.stem,string)))
        
        else :
            
            tbl_filename = str(self.tbl_path.joinpath('%s%s_tbl.dat'%(self.filename.stem,string)))

        self.tbl.write(tbl_filename, format='ascii.commented_header', overwrite=True)


class GAIAStars (object) :
    
    def __init__ (self, obj, my_catname) : 
        
        self.obj = obj
        self.catname = my_catname
        
    def gaia_query (self, phot_sel) :
        
        new_limit = 60  # sets the timeout to 60s
        with conf.set_temp('remote_timeout', new_limit):
            obj_coord = SkyCoord.from_name(self.obj)
        
        vizier = Vizier(columns=['*', 'E(BR/RP)'], catalog=self.catname, row_limit=-1, timeout=60)
        result = vizier.query_region(obj_coord, radius=1*u.deg)

        if (phot_sel) * (self.catname!='Gaia-DR1') :

            # cuts on colours as in https://gea.esac.esa.int/archive/documentation/GDR2/Data_processing/chap_cu5pho/sec_cu5pho_calibr/ssec_cu5pho_PhotTransf.html
            # + new color cut based on the Gaia bands trasformation
            mask1 = (result[0]['BPmag']-result[0]['RPmag']>0.2)&(result[0]['BPmag']-result[0]['RPmag']<1.8)
            # cuts on photometric errors as in https://gea.esac.esa.int/archive/documentation/GDR2/Data_processing/chap_cu5pho/sec_cu5pho_calibr/ssec_cu5pho_PhotTransf.html
            mask2 = (result[0]['e_Gmag']<0.01)&(result[0]['e_BPmag']<0.01)&(result[0]['e_RPmag']<0.01)
            # check of consistency for the excess factor as in https://www.aanda.org/articles/aa/full_html/2018/08/aa32756-18/F17.html
            mask3 = (result[0]['E_BR_RP_']<1.3+0.06*(result[0]['BPmag']-result[0]['RPmag'])**2.)
            
            gaia_tbl = result[0][mask1&mask3]
            
        else:
            
            gaia_tbl = result[0]
        
        self.coord = SkyCoord(ra=gaia_tbl['RA_ICRS'], dec=gaia_tbl['DE_ICRS'])
        
        
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
        default='/Users/arazza/DAS/Observations/NB_DR31_pipeline/CCD_tables', help='path where to save the astrometric tables')
    parser.add_argument('--ccd_galaxies', nargs='*', type=str, default=ccd_galaxies, help='list of selected duPont/DirectCCD galaxies')
    parser.add_argument('--ccd_filters', nargs='*', type=str, default=['r', 'Ha657', 'Ha663'], help='list of DirectCCD filters')
    parser.add_argument('--gaia_dr', action='store', type=str, default='Gaia-DR1', 
        help='gaia data release to use as reference (GAIA-DR1, GAIA-DR2, GAIA-EDR3, default GAIA-DR1)')
    parser.add_argument('--phot_sel', action='store_true', default=False, help='create table with Gaia photometrically selected stars')
    parser.add_argument('--include_ellipse', action='store_true', default=False, help='include stars in the galaxy ellipse')
    args = parser.parse_args()
    
    p = Path(args.path)
    tbl_path = Path(args.tbl_path)
    tbl_path.mkdir(exist_ok=True)
    
    for filt in args.ccd_filters :
        
        if args.ccd_galaxies == None :
            data = sorted(p.glob('*_%s_flux_comb.fits'%filt))
            std = sorted(p.glob('*_%s_std_comb.fits'%filt))
        else:
            listoffluxes = []
            listofstds = []
            for gal in args.ccd_galaxies :
                listoffluxes.append(sorted(p.glob('%s_%s_flux_comb.fits'%(gal,filt))))
                data = [item for sublist in listoffluxes for item in sublist]
                listofstds.append(sorted(p.glob('%s_%s_std_comb.fits'%(gal,filt))))
                std = [item for sublist in listofstds for item in sublist]
                
        if (not data) + (not std) :
            
            print('\nGalaxy or filter not found!\n')
            
            continue
                
        for idata, istd in zip(data, std) :
            
            dp_im = DPImage(idata, istd, tbl_path)
            
            dp_im.mask_sources(snr=3)
            
            dp_im.find_image_stars(args.include_ellipse)
            
            gs = GAIAStars(dp_im.obj, args.gaia_dr)
            print('\nQuerying %s...\n'%gs.catname)
            gs.gaia_query(args.phot_sel)
            
            dp_im.match_with_gaia(gs)
            
            dp_im.save_tbl(args.phot_sel, gs.catname)


    elapsed = timeit.default_timer() - start_time
    minutes, seconds = divmod(elapsed, 60)
    hours, minutes = divmod(minutes, 60)
    print ('\nProcess terminated in %d:%02d:%02d'%(hours, minutes, seconds))
    