# coding: utf-8

import timeit

import argparse

from pathlib import Path

import numpy as np

from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from astropy.visualization import ZScaleInterval

from matplotlib import pyplot as plt 

from galaxy_shape_v21 import GalaxyEllipse


class WFIMosaic (object) :

    
    def __init__ (self, path, obj, filt) :

        self.path = path
        self.obj = obj
        self.filter = filt

    
    def set_arrays (self, mosaic, maskframe, newmaskframe, sp_maskframe, errorframe) :

        self.mosaic = mosaic
        self.maskframe = maskframe
        self.newmaskframe = newmaskframe
        self.sp_maskframe = sp_maskframe
        self.errorframe = errorframe
        
        self.hdulist = fits.open(mosaic)
        self.data = self.hdulist[0].data
        self.hdr = self.hdulist[0].header
        self.wcs = WCS(self.hdr)
        
        self.mask = fits.open(maskframe)[0].data

        self.newmask = fits.open(newmaskframe)[0].data

        self.sp_mask = fits.open(sp_maskframe)[0].data
        
        self.std = fits.open(errorframe)[0].data


    def merge_masks (self) :
        
        self.mask[(np.round(self.newmask)!=99) * (self.newmask!=0)] = 1e12

        self.mask[np.round(self.sp_mask)!=99] = 1e12

    
    def trim_data (self, myell) :
        
        new_dim = (4.0 * myell.a_pix).round()

        data_cutout = Cutout2D(self.data, myell.obj_coord, (new_dim, new_dim), wcs=self.wcs)
        mask_cutout = Cutout2D(self.mask, myell.obj_coord, (new_dim, new_dim), wcs=self.wcs)
        std_cutout = Cutout2D(self.std, myell.obj_coord, (new_dim, new_dim), wcs=self.wcs)

        self.data = data_cutout.data
        self.wcs = data_cutout.wcs
        self.shape = data_cutout.shape

        self.mask = mask_cutout.data

        self.std = std_cutout.data

    
    def plot_trimmed (self) :
        
        fig = plt.figure(figsize=(14, 7))
        
        zscale = ZScaleInterval()
        
        ax1 = fig.add_subplot(1, 3, 1, projection=self.wcs)
        vmin, vmax = zscale.get_limits(self.data)
        ax1.imshow(self.data, origin='lower', cmap='gray_r', vmin=vmin, vmax=vmax)
        ax1.set_xlabel('Right Ascension (J2000)')
        ax1.set_ylabel('Declination (J2000)')
        ax1.set_title('Image')
        
        ax2 = fig.add_subplot(1, 3, 2, projection=self.wcs)
        vmin, vmax = zscale.get_limits(self.std)
        ax2.imshow(self.std, origin='lower', cmap='gray_r', vmin=vmin, vmax=vmax)
        ax2.set_xlabel('Right Ascension (J2000)')
        ax2.coords[1].set_ticklabel_visible(False)
        ax2.coords[1].set_axislabel(' ')
        ax2.set_title('Error')
        
        ax3 = fig.add_subplot(1, 3, 3, projection=self.wcs)
        vmin, vmax = zscale.get_limits(self.mask)
        ax3.imshow(self.mask, origin='lower', cmap='gray_r', vmin=vmin, vmax=vmax)
        ax3.set_xlabel('Right Ascension (J2000)')
        ax3.coords[1].set_ticklabel_visible(False)
        ax3.coords[1].set_axislabel(' ')
        ax3.set_title('Mask')
        
        pos = [i for i, c in enumerate(self.mosaic.name) if c=='_']
        self.filenum = int(self.mosaic.name[pos[1]+6:pos[2]])
        plt.suptitle('%s %s %s'%(self.obj,self.filter,self.filenum))

        plt.show()

        self.cutout_folder = Path('.').joinpath('cutout_images')
        self.cutout_folder.mkdir(exist_ok=True)

        self.cutouts_plots_folder = self.cutout_folder.joinpath('cutout_plots')
        self.cutouts_plots_folder.mkdir(exist_ok=True)

        fig.savefig(self.cutouts_plots_folder.joinpath('%s_%s_frame%s_cutout_plot.png'%(self.obj,self.filter,self.filenum)))

        plt.close()


    def save_cutout (self) :
        
        new_hdr = self.hdr.copy()
        new_hdr['NAXIS1'] = self.shape[1]    # NAXIS1 is X-axis in FITS convention that is column in python convention
        new_hdr['NAXIS2'] = self.shape[0]
        new_hdr['CRPIX1'] = self.wcs.wcs.crpix[0]
        new_hdr['CRPIX2'] = self.wcs.wcs.crpix[1]
        
        flux_filename = self.cutout_folder.joinpath(self.mosaic.name.replace('calib', 'cutout'))
        #std_filename = self.cutout_folder.joinpath(self.errorframe.name.replace('var', 'std').replace('calib', 'cutout'))    # DEPRECATED
        std_filename = self.cutout_folder.joinpath(self.errorframe.name.replace('calib', 'cutout'))
        mask_filename = self.cutout_folder.joinpath(self.maskframe.name.replace('calib', 'cutout'))
        
        fits.writeto(flux_filename, data=self.data, header=new_hdr, overwrite=True)
        fits.writeto(std_filename, data=self.std, header=new_hdr, overwrite=True)
        fits.writeto(mask_filename, data=self.mask, header=new_hdr, overwrite=True)



if __name__ == "__main__":
    
    start_time = timeit.default_timer()
    
    galnames, paths = np.loadtxt('paths_to_calib_WFI.txt', dtype='str', unpack=True)
    
    galnames = np.atleast_1d(galnames)
    paths = np.atleast_1d(paths)
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-g', '--gals', nargs='*', type=str, default=galnames,
        help='list of selected WFI galaxies')
    
    args = parser.parse_args()
    
    # Instantiate the lists of galaxies and list of paths
    
    if set(args.gals) <= set(galnames) :
        
        list_of_gals = []
        list_of_paths = []
    
        for gal in args.gals :
            
            pos = [i for i, g in enumerate(galnames) if g==gal][0]
            
            list_of_gals.append(gal)
            list_of_paths.append(paths[pos])
    
    else :
        
        list_of_gals = galnames.copy()
        list_of_paths = paths.copy()
        
    # Main loop
    
    for obj, p in zip(list_of_gals, list_of_paths) :
        
        p = Path(p)
            
        for filt in ['Rc', 'Ha'] :
            
            mosaics = sorted(p.glob('%s_%s_stack?_calib.fits'%(obj,filt))) + sorted(p.glob('%s_%s_stack??_calib.fits'%(obj,filt)))
            maskframes = sorted(p.glob('%s_%s_stack?_mask_calib.fits'%(obj,filt))) + sorted(p.glob('%s_%s_stack??_mask_calib.fits'%(obj,filt)))
            newmaskframes = sorted(p.parent.glob('%s_%s_stack?_newmask.fits'%(obj,filt))) + sorted(p.parent.glob('%s_%s_stack??_newmask.fits'%(obj,filt)))
            sp_maskframes = sorted(p.parent.glob('%s_%s_stack?_sp_mask.fits'%(obj,filt))) + sorted(p.parent.glob('%s_%s_stack??_sp_mask.fits'%(obj,filt)))
            errorframes = sorted(p.glob('%s_%s_stack?_std_calib.fits'%(obj,filt))) + sorted(p.glob('%s_%s_stack??_std_calib.fits'%(obj,filt)))
            
            if mosaics == [] :
                continue

            myell = GalaxyEllipse(obj, filt, mosaics[0], fits.open(maskframes[0])[0].data)
            myell.get_ellipse()
            myell.ellipse_to_pixel('WFI')
            myell.mask_ellipse(r=1)
            
            for mosaic, maskframe, newmaskframe, sp_maskframe, errorframe in zip(mosaics, maskframes, newmaskframes, sp_maskframes, errorframes) :
                
                wfi = WFIMosaic(p, obj, filt)
                
                wfi.set_arrays(mosaic, maskframe, newmaskframe, sp_maskframe, errorframe)
                
                #wfi.merge_masks()    # This step can be removed in pipeline v2.1
                
                wfi.trim_data(myell)
                
                wfi.plot_trimmed()
                
                wfi.save_cutout() 
                
                
    elapsed = timeit.default_timer() - start_time
    minutes, seconds = divmod(elapsed, 60)
    hours, minutes = divmod(minutes, 60)
    print ('Process terminated in %d:%02d:%02d'%(hours, minutes, seconds))
