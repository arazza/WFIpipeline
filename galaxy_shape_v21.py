#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 7 21:14 2020

@author: arazza
"""

import numpy as np
from pathlib import Path
import timeit
import sys
import math
import argparse

from astroquery.simbad import Simbad
from astroquery.vizier import Vizier

from astropy.coordinates import SkyCoord, Angle
import astropy.units as u 
from astropy.wcs import WCS
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.utils.data import conf
from astropy.table import Table
from astropy.stats import sigma_clipped_stats
from astropy.visualization import ZScaleInterval

from photutils import EllipticalAperture, EllipticalAnnulus
from photutils import detect_threshold, detect_sources

from scipy import ndimage

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from astropy.visualization import ZScaleInterval

from cubehelix_palette import PaperPalette


class GalaxyEllipse (object) :
    
    
    def __init__ (self, name, filt, filename, bpm) :
        
        self.name = name
        self.filter = filt
        self.data = fits.open(str(filename))[0].data
        self.bpm = bpm
        self.wcs = WCS(str(filename))
        
    
    def get_ellipse (self) :
            
        phangs_tbl = Table.read('phangs_sample_table_v1p6.fits')
        #phangs_tbl = Table.read('/Users/arazza/DAS/Papers_in_prep/PHANGS/Halpha_Survey/All_plots/Files/phangs_sample_table_v1p6.fits')

        pgc = [phangs_tbl['pgc'][i] for i,galname in enumerate(phangs_tbl['name']) if galname.strip()==self.name.lower()][0]
        my_gal = phangs_tbl['pgc']==pgc

        self.a = Angle(phangs_tbl['size_r25'][my_gal][0] * u.arcsec).to(u.arcmin)
        self.b = self.a * np.cos(Angle(phangs_tbl['orient_incl'][my_gal][0] * u.deg).radian)
        self.pa = phangs_tbl['orient_posang'][my_gal][0] * u.deg
        self.obj_coord = SkyCoord (ra=phangs_tbl['orient_ra'][my_gal][0], dec=phangs_tbl['orient_dec'][my_gal][0], unit=(u.deg, u.deg))
     
        print('Position of %s : '%self.name, self.obj_coord.to_string('hmsdms'))
        print('Semi-major axis : ',round(self.a.value, 3) * u.arcmin) 
        print('Semi-minor axis : ',round(self.b.value, 3) * u.arcmin)
        print('Position angle : ',self.pa.value)
        
    
    def ellipse_to_pixel (self, instr) :
        
        self.x0, self.y0 = self.obj_coord.to_pixel(self.wcs)
        
        if instr == 'WFI' :
            pixel_scale = 0.238 * u.arcsec / u.pixel
        if instr == 'CCD' :
            pixel_scale = 0.259 * u.arcsec / u.pixel
        self.a_pix = self.a.to(u.arcsec) / pixel_scale
        self.b_pix = self.b.to(u.arcsec) / pixel_scale

        print('Position of %s (pixels) : '%self.name, np.round(self.x0, 2), np.round(self.y0, 2))
        print('Semi-major axis (pixels) : ', round(self.a_pix.value, 2)) 
        print('Semi-minor axis (pixels) : ', round(self.b_pix.value, 2))
        

    
    def mask_ellipse (self, r) :

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

        origin = (self.x0, self.y0)
        angle = math.radians(90-self.pa.value)
        self.a_pix = self.a_pix * float(r)
        self.b_pix = self.b_pix * float(r)
        a = self.a_pix.value
        b = self.b_pix.value
        
        self.aperture = EllipticalAperture(origin, a, b, theta=math.radians(self.pa.value+90))    
        # Note on +90: photutils counts the angles counter-clockwise from X axis towards Y, which is West towards North in sky coordinates
        box = self.aperture.to_mask()
        self.cutout = Cutout2D(self.data, self.obj_coord, box.shape, wcs=self.wcs)
        
        xmin = self.cutout.xmin_original
        xmax = self.cutout.xmax_original
        ymin = self.cutout.ymin_original
        ymax = self.cutout.ymax_original

        self.mask = np.full(self.data.shape, False)

        for i in range(ymin,ymax) :               
            for j in range(xmin,xmax) :
                x, y = rotate(origin, (j,i), angle)                    
                if (x/a)**2+(y/b)**2<=1 :
                    self.mask[i,j]=True


    def annulus_background (self, r, path, dilate_size, filename='ellipse_radii.png') :

        def mask_sources_old (data, bpm, mask, snr, dilate_size) :

            sigclip_sigma=3.0
            #sigclip_sigma=2.0  
            sigclip_iters=5
            mean, median, std = sigma_clipped_stats(data, mask=bpm+mask, sigma=sigclip_sigma, maxiters=sigclip_iters)    
            print (mean, median, std)
            
            threshold = detect_threshold(data, snr, background=median, error=std, mask=bpm+mask, mask_value=None)
            print('Threshold --> ', threshold[0][0])
            
            segm = detect_sources(data, threshold, npixels=5,  
                                     mask=mask)    # pixels within the ellipse are excluded from any source 
            #dilate_size = 11
            selem = np.ones((dilate_size, dilate_size))
            mask = ndimage.binary_dilation(segm.data.astype(np.bool), selem)
            
            # Update the mask by masking the galaxy and the gaps
            return mask + bpm

        def mask_sources (data, snr, dilate_size) :

            sigclip_sigma=3.0
            #sigclip_sigma=2.0  
            sigclip_iters=5
            mean, median, std = sigma_clipped_stats(data, sigma=sigclip_sigma, maxiters=sigclip_iters)    
            print (mean, median, std)
            
            threshold = detect_threshold(data, snr, background=median, error=std)
            print('Threshold --> ', threshold[0][0])
            
            segm = detect_sources(data, threshold, npixels=5) 
                                     
            #dilate_size = 11
            selem = np.ones((dilate_size, dilate_size))
            mask = ndimage.binary_dilation(segm.data.astype(np.bool), selem)
            
            # Update the mask by masking the galaxy and the gaps
            return mask

        # Create path
        Path('bkg').mkdir(exist_ok=True)
        path.joinpath(self.name).mkdir(exist_ok=True)
        
        # Define the aperture without incrementing factor r
        self.aperture.a = self.a_pix.value / float(r)
        self.aperture.b = self.b_pix.value / float(r)

        radii = np.arange(0.5, 2.0, 0.1)

        apers = [EllipticalAnnulus(self.aperture.positions, self.aperture.a * rad, self.aperture.a * (rad + 0.1), 
                                       self.aperture.b * (rad + 0.1), theta=self.aperture.theta) for rad in radii]

        rad_aper = EllipticalAperture(self.aperture.positions, self.aperture.a * float(r), 
                                       self.aperture.b * float(r), self.aperture.theta)
        
        # CHECK SIZE ANNULI
        #print(self.data.shape)
        #print(self.a_pix.value, self.b_pix.value, float(r))
        #print(self.aperture.a, self.aperture.b)
        #for rad in radii :
        #    print(self.aperture.a * rad, self.aperture.b * (rad + 0.1) * self.aperture.a * rad / (self.aperture.a * (rad + 0.1)),
        #        self.aperture.a * (rad + 0.1), self.aperture.b * (rad + 0.1))

        # Restore the incremented aperture
        self.aperture.a = self.a_pix.value 
        self.aperture.b = self.b_pix.value
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

        zscale = ZScaleInterval()

        vmin, vmax = zscale.get_limits(self.data)
        ax1.imshow(self.data, cmap='gray_r', origin='lower', vmin=vmin, vmax=vmax)

        # Update the mask
        #self.new_mask = mask_sources (self.data, self.bpm, self.mask, 2, dilate_size)
        self.new_mask = mask_sources (self.data, 2, dilate_size)

        #pal = PaperPalette()
        #pal.cmap.set_bad(alpha=0)
        #masked_data = np.ma.array(self.new_mask.astype('float'), mask=~self.new_mask)
        #
        #ax1.imshow(masked_data.astype('float'), origin='lower', cmap=pal.cmap.reversed(), vmin=0, vmax=1, alpha=0.5)

        medians = []

        for aper in apers :
            
            annulus_mask = aper.to_mask(method='center')
            annulus_data = annulus_mask.multiply(self.data)
            #fig_test, (ax3, ax4) = plt.subplots(1, 2, figsize=(14,7))
            #ax3.imshow(annulus_data, origin='lower')
            annulus_data_mask = annulus_mask.multiply(self.bpm+self.new_mask)
            #ax4.imshow(annulus_data_mask, origin='lower')
            
            annulus_data_1d = annulus_data[annulus_mask.data>0]
            annulus_data_1d_mask = annulus_data_mask[annulus_mask.data>0]

            _, median_sigclip, _ = sigma_clipped_stats(annulus_data_1d, mask=annulus_data_1d_mask)
            #_, median_sigclip, _ = sigma_clipped_stats(annulus_data_1d)

            medians.append(median_sigclip)

            aper.plot(axes=ax1, color='C1', lw=0.7)
        
        rad_aper.plot(axes=ax1, color='k', lw=1.0, zorder=2)
        
        ax1.set_xlabel('x [pix]')
        ax1.set_ylabel('y [pix]')

        #mid_radii = np.arange(1.025, 1.775, 0.05)

        ax2.plot(radii, medians, '-d', color='C1')
        ax2.axvline(x=float(r), color='k', ls='--', lw=1.0)
        
        #ax.set_title('%s - filter %s'%(self.name,self.filter))
        ax2.set_xlabel(r'ellipse radius')
        ax2.set_ylabel(r'background $[\mu \rm{Jy}]$')
        
        ax2.grid(ls=':', lw=0.5)
        
        plt.suptitle('%s - filter %s'%(self.name,self.filter))

        fig.subplots_adjust(left=0.08, right=0.92, top=0.85, bottom=0.2, wspace=0.3)
        
        plt.show()
        
        fig.savefig(str(path.joinpath(self.name, '%s_%s_'%(self.name,self.filter)+filename)))
        
        plt.close()
        
        # Save results in a table
        
        table_file = str(path.joinpath(self.name, '%s_%s_'%(self.name,self.filter)+filename)).replace('png','dat')
        
        with open(table_file, 'w') as f :
            f.write('# radius, median (within radius and radius + 0.1), relative error (%)\n')
        
        last_median = median_sigclip
        
        for radius, median in zip(radii, medians) :

            if not isinstance(median, np.ma.core.MaskedConstant) :
                
                with open(table_file, 'a') as f :
                    f.write('%s %s %s\n'%(round(radius, 1), round(median, 6), round((median-last_median)/last_median*100, 2)))

        

    def plot_ellipse (self, path, filename='ellipse.png', title_suffix=' ') :

        fig = plt.figure(figsize=(6.92, 7))
        gs = gridspec.GridSpec(1, 1, height_ratios=[1], width_ratios=[1])
        gs.update(left=0.05, right=0.95, bottom=0.17, top=0.82, wspace=0.47, hspace=0.03)
        
        zscale = ZScaleInterval()
        vmin, vmax = zscale.get_limits(self.data)

        ax1 = plt.subplot(gs[0])
        ax1.imshow(self.data, vmin=vmin, vmax=vmax, origin='lower', cmap='gray_r', interpolation='nearest')
        self.aperture.plot(ax=ax1, color='C1')

        ax1.set_title('%s - %s filter %s'%(self.name,self.filter,title_suffix))
        ax1.set_xlabel('x [pixel]')
        ax1.set_ylabel('y [pixel]')
        
        ax1.tick_params(axis='x', pad=10, reset=True)
        ax1.tick_params(axis='y', pad=5, reset=True)
        ax1.xaxis.labelpad = 20
        ax1.yaxis.labelpad = 20
        
        plt.show()

        fig.savefig(str(path.joinpath('%s_%s_'%(self.name,self.filter)+filename)))

        plt.close()
        


if __name__ == "__main__":

    start_time = timeit.default_timer()

    parser = argparse.ArgumentParser()
    
    parser.add_argument('pathtogalaxy', help='path to reduced frames')
    parser.add_argument('objname', help='name of selected galaxy')
    parser.add_argument('filter', help='filter name (Rc/Ha)')
    parser.add_argument('-e', '--ell_radius', action='store', default=1., help='galaxy ellipse increasing factor (recommended for Rc)')
    
    args = parser.parse_args()

    p = Path(args.pathtogalaxy)    
    obj = args.objname             
    filt = args.filter             
    
    mosaics = sorted(p.glob('%s_%s_stack?_cutout.fits'%(obj,filt))) + sorted(p.glob('%s_%s_stack??_cutout.fits'%(obj,filt)))

    myell = GalaxyEllipse(obj, filt, mosaics[0])
    
    myell.get_ellipse()
    myell.ellipse_to_pixel()
    myell.mask_ellipse(args.ell_radius)

    myell.annulus_background(args.ell_radius, Path('ellipse_plots'))
    
    myell.plot_ellipse(Path('ellipse_plots'), 'ellipse_cutout.png')

    elapsed = timeit.default_timer() - start_time
    minutes, seconds = divmod(elapsed, 60)
    hours, minutes = divmod(minutes, 60)
    print ('Process terminated in %d:%02d:%02d'%(hours, minutes, seconds))

