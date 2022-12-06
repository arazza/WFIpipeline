#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adapted from the script created on Wed Mar 28 15:15:37 2018

@author: arazza
"""

import numpy as np
import math as m
from ccdproc import CCDData, Combiner, trim_image
from pathlib import Path
import timeit
import sys
from shutil import move
import argparse
import astropy.units as u
from astropy.io import fits
from astropy.stats import median_absolute_deviation
from astropy.visualization import ZScaleInterval
from astropy.wcs import WCS

from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

from Sky_subtraction_v21 import SkyImage
from galaxy_shape_v21 import GalaxyEllipse
from mask_ds9_regions import DS9Reg


class PHANGSImage (object) :

    def __init__ (self, path, obj, filt) :

        self.path = path
        self.obj = obj
        self.filter = filt

    
    def weight_combine (self, p, scienceframes, maskframes, errorframes, image_type, full_FOV=False) :
        '''
        Inverse-variance weighted combination. 
        The science frames are combined with a weighted average where the weights are the inverse of non-convolved variance maps.
        The final error maps are from the (square of) weighted average of the convovled variance maps and as weights the non-convovled
        variance maps are used in this case as well.
        '''
        
        data = fits.open(str(scienceframes[0]))[0].data
        size = (data.shape[0], data.shape[1])
        arr = np.ma.array(np.zeros(size, np.float32), mask=np.full(size, False))
        wgt = np.ma.array(np.zeros(size, np.float32), mask=np.full(size, False))
        prod = np.ma.prod([arr,wgt], axis=0)

        for scienceframe, maskframe, errorframe in zip(scienceframes, maskframes, errorframes) :

            #if scienceframe!=scienceframes[0] :
            #    continue
            
            self.mask = np.round(fits.open(str(maskframe))[0].data)!=99
            
            '''
            Mask update with manually marked DS9 regions by loading
            the region file related to scienceframe file if exists.
            '''
            self.ds9_reg = DS9Reg(scienceframe)
            self.ds9_reg.set_ds9_regions()
            self.ds9_reg.mask_regions()
            #self.ds9_reg.plot_regions()
            
            # Update the mask if DS9 regions are defined 
            self.mask = self.mask + self.ds9_reg.mask
                
            # Open the saved convolved frame

            print('\nOpening %s ...\n'%scienceframe.name)
            data = fits.open(str(scienceframe))[0].data
            a = np.ma.array(data, mask=self.mask)

            print('\nOpening %s ...\n'%errorframe.name)
            error = fits.open(str(errorframe))[0].data
            w = np.ma.array(1/(error)**2., mask=self.mask)
            
            # Stacking the frames together in the inverse variance combination
            
            p = np.ma.prod([a,w], axis=0)
            
            prod = np.ma.sum([prod,p], axis=0)
            
            wgt = np.ma.sum([wgt,w], axis=0)

        # Final computation of the weighted combination and its uncertainty

        avg = np.ma.prod([prod,1/wgt], axis=0)
        
        unc = np.sqrt(1/wgt)

        # Saving the final images

        header = fits.open(str(scienceframes[0]))[0].header
        
        if image_type == 'WFI_fullFOV' :
            final_imgs_path = Path('.').joinpath('final_images', 'WFI', 'full_FOV')
        if image_type == 'WFI_cutouts' :
            final_imgs_path = Path('.').joinpath('final_images', 'WFI', 'cutouts')
        if image_type == 'CCD' : 
            final_imgs_path = Path('.').joinpath('final_images', 'CCD') 

        if full_FOV :
            full_FOV_string = '_fullFOV'
        else :
            full_FOV_string = ''
        
        self.file_flux = final_imgs_path.joinpath('fits/%s_%s_flux%s_wcomb.fits'%(self.obj,self.filter,full_FOV_string))
        self.file_unc = final_imgs_path.joinpath('fits/%s_%s_std%s_wcomb.fits'%(self.obj,self.filter,full_FOV_string))
        
        # Trasform pixel with 0 values to NaN and homogenize the header with NB-DR2 (duPont data)
        avg.data[avg.data==0] = np.nan
        unc.data[unc.data==0] = np.nan
        
        if (image_type == 'WFI_fullFOV') + (image_type == 'WFI_cutouts') :
            if 'ZUNIT' in header :
                header.remove('ZUNIT')
            if 'ZPOINT' in header :
                header.remove('ZPOINT')
            header.set('BUNIT', 'uJy/pixel', 'flux unit')
            header.set('SCALE', '0.238', 'arcsec/pixel')
        
        # Save fits files
        fits.writeto(str(self.file_flux), avg.data, header=header, overwrite=True)
        fits.writeto(str(self.file_unc), unc.data, header=header, overwrite=True)
        
    
    def combine (self, p, scienceframes, maskframes, errorframes, image_type, full_FOV=False) :
        '''
        Normal combination of the frames. 
        This set of final images is meant to serve for the unbiased aperture photometry
        since the weighted combination introduces a bias due to the different weighting
        the stars profiles have in frames with varying PSFs.
        '''
        
        data = fits.open(str(scienceframes[0]))[0].data
        size = (data.shape[0], data.shape[1])
        sci = np.ma.array(np.zeros(size, np.float32), mask=np.full(size, False))
        err = np.ma.array(np.zeros(size, np.float32), mask=np.full(size, False))
        num = np.ma.array(np.zeros(size, np.float32), mask=np.full(size, False))    # number of frames where the pixels are valid

        for scienceframe, maskframe, errorframe in zip(scienceframes, maskframes, errorframes) :

            #if scienceframe!=scienceframes[0] :
            #    continue
            
            self.mask = np.round(fits.open(str(maskframe))[0].data)!=99
            
            '''
            Mask update with manually marked DS9 regions by loading
            the region file related to scienceframe file if exists.
            '''
            self.ds9_reg = DS9Reg(scienceframe)
            self.ds9_reg.set_ds9_regions()
            self.ds9_reg.mask_regions()
            #self.ds9_reg.plot_regions()
            
            # Update the mask if DS9 regions are defined 
            self.mask = self.mask + self.ds9_reg.mask
                
            # Open the saved convolved frame

            print('\nOpening %s ...\n'%scienceframe.name)
            data = fits.open(str(scienceframe))[0].data
            s = np.ma.array(data, mask=self.mask)

            print('\nOpening %s ...\n'%errorframe.name)
            error = fits.open(str(errorframe))[0].data
            e = np.ma.array(error, mask=self.mask)
            
            # Update the mask in the num array

            num = np.ma.array(num, mask=self.mask)
            
            # Stacking the frames together
            
            sci = np.ma.sum([sci,s], axis=0)
            
            err = np.ma.sum([err,e], axis=0)

            num = np.ma.sum([num,1.], axis=0)

        # Final computation of the combination and its uncertainty

        avg = np.ma.prod([sci,1/num], axis=0)
        
        unc = np.ma.prod([err,1/num**2.], axis=0)
        
        # Saving the final images

        header = fits.open(str(scienceframes[0]))[0].header
        
        if image_type == 'WFI_fullFOV' :
            final_imgs_path = Path('.').joinpath('final_images', 'WFI', 'full_FOV')
        if image_type == 'WFI_cutouts' :
            final_imgs_path = Path('.').joinpath('final_images', 'WFI', 'cutouts')
        if image_type == 'CCD' : 
            final_imgs_path = Path('.').joinpath('final_images', 'CCD') 

        if full_FOV :
            full_FOV_string = '_fullFOV'
        else :
            full_FOV_string = ''
        
        self.file_flux = final_imgs_path.joinpath('fits/%s_%s_flux%s_comb.fits'%(self.obj,self.filter,full_FOV_string))
        self.file_unc = final_imgs_path.joinpath('fits/%s_%s_std%s_comb.fits'%(self.obj,self.filter,full_FOV_string))
        
        # Trasform pixel with 0 values to NaN and homogenize the header with NB-DR2 (duPont data)
        avg.data[avg.data==0] = np.nan
        unc.data[unc.data==0] = np.nan
        
        if (image_type == 'WFI_fullFOV') + (image_type == 'WFI_cutouts') :
            if 'ZUNIT' in header :
                header.remove('ZUNIT')
            if 'ZPOINT' in header :
                header.remove('ZPOINT')
            header.set('BUNIT', 'uJy/pixel', 'flux unit')
            header.set('SCALE', '0.238', 'arcsec/pixel')
        
        # Save fits files
        fits.writeto(str(self.file_flux), avg.data, header=header, overwrite=True)
        fits.writeto(str(self.file_unc), unc.data, header=header, overwrite=True)



    def plot (self, instr, full_FOV=False) :

        fig1 = plt.figure(figsize=(6.92, 7))
        
        data = fits.open(self.file_flux)[0].data
        zscale = ZScaleInterval()
        vmin, vmax = zscale.get_limits(data)

        wcs = WCS(fits.open(self.file_flux)[0].header)

        ax1 = fig1.add_subplot(1, 1, 1, projection=wcs)
        
        ax1.imshow(data, origin='lower', cmap='gray_r', vmax=vmax, vmin=vmin)
        ax1.set_title('%s data image (%s)'%(self.obj,self.filter))
        ax1.set_xlabel('R.A. [J2000]')
        ax1.set_ylabel('Dec. [J2000]')
        
        plt.subplots_adjust(wspace=0.20, hspace=0.20, left=0.23, bottom=0.11, right=0.90, top=0.88)

        #plt.show()
        
        if instr == 'WFI' :
            if full_FOV :
                plot_path = Path('.').joinpath('final_images', 'WFI', 'full_FOV', 'png')
                #full_FOV_string = '_fullFOV'
            else :
                plot_path = Path('.').joinpath('final_images', 'WFI', 'cutouts', 'png')
                #full_FOV_string = ''
        if instr == 'CCD' :
            plot_path = Path('.').joinpath('final_images', 'CCD', 'png')
        
        fig1.savefig(plot_path.joinpath(str(self.file_flux.name).replace('.fits','.png')))#'%s.png'%full_FOV_string)))

        plt.close()

        fig2 = plt.figure(figsize=(6.92, 7))

        unc = fits.open(self.file_unc)[0].data
        vmin, vmax = zscale.get_limits(unc)
        
        wcs = WCS(fits.open(self.file_unc)[0].header)

        ax2 = fig2.add_subplot(1, 1, 1, projection=wcs)

        ax2.imshow(unc, origin='lower', cmap='gray_r', vmax=vmax, vmin=vmin)
        ax2.set_title('%s error image (%s)'%(self.obj,self.filter))
        ax2.set_xlabel('R.A. [J2000]')
        ax2.set_ylabel('Dec. [J2000]')
        
        plt.subplots_adjust(wspace=0.20, hspace=0.20, left=0.23, bottom=0.11, right=0.90, top=0.88)

        #plt.show()
        
        fig2.savefig(plot_path.joinpath(str(self.file_unc.name).replace('.fits','.png')))#'%s.png'%full_FOV_string)))

        plt.close()


# Global functions to weight combine WFI fullFOV, WFI cutouts and DP frames

def combine_fullFOV (p, obj, filt) :
    '''
    Global function to prepare the lists of WFI full FOV images to be combined
    '''

    print('\nWeight combine WFI full FOV ...\n')
    
    scienceframes = sorted(p.glob('%s_%s_stack?_calib_nosky.fits'%(obj,filt))) + sorted(p.glob('%s_%s_stack??_calib_nosky.fits'%(obj,filt)))
    maskframes = sorted(p.glob('%s_%s_stack?_mask_calib.fits'%(obj,filt))) + sorted(p.glob('%s_%s_stack??_mask_calib.fits'%(obj,filt)))
    errorframes = sorted(p.glob('%s_%s_stack?_std_calib.fits'%(obj,filt))) + sorted(p.glob('%s_%s_stack??_std_calib.fits'%(obj,filt)))
    
    if (not scienceframes) or (not maskframes) or (not errorframes) :
        
        print('\nThere are no calibrated frames with filter %s for %s\n'%(filt,obj))
         
    else :

        # Instantiate the paths to the final fullFOV fits and png files
        
        nested_paths = ['final_images', 'WFI', 'full_FOV']
        newdir = Path('.')
        for path in nested_paths :
            newdir = newdir.joinpath(path)
            newdir.mkdir(exist_ok=True)
        
        newdir_fits = newdir.joinpath('fits')
        newdir_fits.mkdir(exist_ok=True)

        newdir_png = newdir.joinpath('png')
        newdir_png.mkdir(exist_ok=True)
        
        # weight-combine the frames
        
        image = PHANGSImage(p, obj, filt)
        
        # weighted combination
        print('\nWeighted combination of the calibrated frames ...\n')
        image.weight_combine(p, scienceframes, maskframes, errorframes, 'WFI_fullFOV', True)
        image.plot('WFI', full_FOV=True)
        
        # normal combination
        print('\nCombination of the calibrated frames ...\n')
        image.combine(p, scienceframes, maskframes, errorframes, 'WFI_fullFOV', True)
        image.plot('WFI', full_FOV=True)
            
        return image


def combine_cutouts (p, obj, filt) :
    '''
    Global function to prepare the lists of WFI cutouts to be combined as in NB-DR2 (script weight_combine_v2.py)
    '''
    
    print('\nWeight combine cutouts ...\n')

    scienceframes = sorted(Path('.').glob('cutout_images/%s_%s_stack?_cutout_nosky.fits'%(obj,filt))) + sorted(Path('.').glob('cutout_images/%s_%s_stack??_cutout_nosky.fits'%(obj,filt)))
    maskframes = sorted(Path('.').glob('cutout_images/%s_%s_stack?_mask_cutout.fits'%(obj,filt))) + sorted(Path('.').glob('cutout_images/%s_%s_stack??_mask_cutout.fits'%(obj,filt)))
    errorframes = sorted(Path('.').glob('cutout_images/%s_%s_stack?_std_cutout.fits'%(obj,filt))) + sorted(Path('.').glob('cutout_images/%s_%s_stack??_std_cutout.fits'%(obj,filt)))

    if (not scienceframes) or (not maskframes) or (not errorframes) :
        
        print('\nThere are no calibrated frames with filter %s for %s\n'%(filt,obj))

    else :

        # Instantiate the paths to the final fullFOV fits and png files

        nested_paths = ['final_images', 'WFI', 'cutouts']
        newdir = Path('.')
        for path in nested_paths :
            newdir = newdir.joinpath(path)
            newdir.mkdir(exist_ok=True)
        
        newdir_fits = newdir.joinpath('fits')
        newdir_fits.mkdir(exist_ok=True)

        newdir_png = newdir.joinpath('png')
        newdir_png.mkdir(exist_ok=True)
    
        # weight-combine the frames

        image = PHANGSImage(p, obj, filt)
        
        # weighted combination
        print('\nWeighted combination of the calibrated frames ...\n')
        image.weight_combine(p, scienceframes, maskframes, errorframes, 'WFI_cutouts', False)
        image.plot('WFI')
        
        # normal combination
        print('\nCombination of the calibrated frames ...\n')
        image.combine(p, scienceframes, maskframes, errorframes, 'WFI_cutouts', False)
        image.plot('WFI')

        return image


def combine_CCD (p, obj, filt) :
    '''
    Global function to prepare the lists of Direct CCD images to be combined
    '''
    
    print('\nWeight combine CCD ...\n')

    scienceframes = sorted(p.glob('%s_%s*_wcs_reproj_calib_nosky.fits'%(obj,filt)))
    maskframes = sorted(p.glob('%s_%s*_wcs_reproj_mask_calib.fits'%(obj,filt)))
    errorframes = sorted(p.glob('%s_%s*_wcs_reproj_std_calib.fits'%(obj,filt)))

    if (not scienceframes) or (not maskframes) or (not errorframes) :
        
        print('\nThere are no calibrated frames with filter %s for %s\n'%(filt,obj))

    else :
        
        # Instantiate the paths to the final fullFOV fits and png files
        
        nested_paths = ['final_images', 'CCD']
        newdir = Path('.')
        for path in nested_paths :
            newdir = newdir.joinpath(path)
            newdir.mkdir(exist_ok=True)
        
        newdir_fits = newdir.joinpath('fits')
        newdir_fits.mkdir(exist_ok=True)
        
        newdir_png = newdir.joinpath('png')
        newdir_png.mkdir(exist_ok=True)
    
        # weight-combine the frames

        image = PHANGSImage(p, obj, filt)

        # weighted combination
        print('\nWeighted combination of the calibrated frames ...\n')
        image.weight_combine(p, scienceframes, maskframes, errorframes, 'CCD', False)
        image.plot('CCD')
        
        # normal combination
        print('\nCombination of the calibrated frames ...\n')
        image.combine(p, scienceframes, maskframes, errorframes, 'CCD', False)
        image.plot('CCD')
        
        return image


if __name__ == "__main__":
    
    start_time = timeit.default_timer()

    wfi_galaxies = ['NGC1087', 'NGC1672', 'NGC2835', 'NGC3627', 'NGC4535', 'IC5332', 'NGC4254', 'NGC5068', 
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
    
    parser.add_argument('--wfi_galaxies', nargs='*', type=str, default=wfi_galaxies, 
        help='list of selected WFI galaxies (type a non-valid galaxy to avoid calibrating WFI)')
    parser.add_argument('--ccd_galaxies', nargs='*', type=str, default=ccd_galaxies, 
        help='list of selected CCD galaxies (type a non-valid galaxy to avoid calibrating CCD)')

    parser.add_argument('--wfi_filters', nargs='*', type=str, default=['Rc', 'Ha'], help='list of WFI filters')
    parser.add_argument('--ccd_filters', nargs='*', type=str, default=['r', 'Ha657', 'Ha663'], help='list of selected CCD filters')

    parser.add_argument('--full_FOV', action='store_true', default=False, help='combine WFI full FOV images and not the cutouts')
    
    args = parser.parse_args()

    # Instantiate the lists of galaxies

    lists_of_galaxies = [args.wfi_galaxies, args.ccd_galaxies]

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

        if set(list_of_galaxy) <= set(wfi_galaxies) :
            
            wfi_galnames, wfi_paths = np.loadtxt('paths_to_uncalib_WFI.txt', dtype='str', unpack=True)
            
            for gal, p in zip(wfi_galnames, wfi_paths) :

                if gal not in list_of_galaxy :
                    continue
                
                print('\nCalibrating %s ...\n'%gal)
                
                if args.wfi_path :
                    p = Path(args.wfi_path).joinpath('phot_calib')
                else:
                    p = Path(p).joinpath('phot_calib')

                print('Images path :\n%s'%p)

                for filt in args.wfi_filters :

                    print('\n%s\n'%filt)
                    
                    # Call the weighted combination for WFI full FOV frames
                    
                    if args.full_FOV :
                        
                        image_fullFOV = combine_fullFOV (p, gal, filt)

                    # Call the weighted combination for WFI cutouts

                    else :
                        
                        image_cutout = combine_cutouts (p, gal, filt)



        # Loop for CCD
        
        else :

            for gal in list_of_galaxy :
                
                print('\nCalibrating %s ...\n'%gal)

                p = Path(args.ccd_path).parent.joinpath('phot_calib')
                
                print('Images path :\n%s'%p)

                for filt in args.ccd_filters :

                    print('\n%s\n'%filt)

                    # Call the weighted combination for DP frames
                    image_ccd = combine_CCD (p, gal, filt)
                
                
    elapsed = timeit.default_timer() - start_time
    minutes, seconds = divmod(elapsed, 60)
    hours, minutes = divmod(minutes, 60)
    print ('Process terminated in %d:%02d:%02d'%(hours, minutes, seconds))