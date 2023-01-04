#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 03 16:34:00 2018

@author: arazza
"""

from pathlib import Path
from ccdproc import ImageFileCollection
import argparse
import os
import sys
import astropy.io.fits as fits

from astropy.coordinates import SkyCoord
from astropy import units as u



def change_objname_keyword(p,images):
    objnames = images.summary['OBJECT'].data.data
    select = [i for i in range(len(objnames)) if objnames[i].endswith('small') or objnames[i].endswith('large')]
    for i in select:
        with fits.open(p.joinpath(images.summary['file'][i]),'update') as hdul:
            newkey = hdul[0].header['OBJECT'].split('_')[0]
            hdul[0].header['OBJECT']=newkey

                  
                  
def make_file (p, name, tbl, filt, ccd, PhangsTbl) :
    
    if 'Rc' in filt :
        myfilt = 'Rc'
    else :
        myfilt = 'Ha'
    
    obsdate = str(p)[-18:-8]
    
    obj_file = 'objects_txt/%s_%s_%s.txt'%(name,myfilt,obsdate)
    with open(obj_file, 'w') as f :
        f.write('# %s in %s \n# Observing Run %s \n# first row: CCD number where the object is centred \n# second row: RA, DEC (hours, degrees) \n# other rows: crpix1, crpix2 for each dithered frame (pixel)\nCCD %s\n'%(name,myfilt,obsdate,ccd))

    with open(obj_file, 'a') as f :
        
        try:
            with fits.open(PhangsTbl) as sample:
                ra, dec = sample[1].data['orient_ra'][sample[1].data['name']==name.lower()][0], sample[1].data['orient_dec'][sample[1].data['name']==name.lower()][0]
                mycoord = SkyCoord(ra=ra * u.degree, dec=dec * u.degree, frame='icrs')
        
        except:
            print('{} is not part of the PHANGS sample. Using CDS catalogue for RA, DEC coordinates.'.format(name))
            mycoord = SkyCoord.from_name(name)
            
        f.write(mycoord.to_string('hmsdms') + '\n')

    files_to_load = []
    
    for i in range(len(tbl['file'])) :
        files_to_load.append(str(p.joinpath(tbl['file'][i])) + '[%s]'%ccd)
    
    command = 'ds9 -zscale '
    for f in files_to_load :
        command = command + f + ' '
    
    os.system(command)

    while True :
        pixels = input()
        if pixels=='' :
            break
        else :
            with open(obj_file, 'a') as f :
                f.write(pixels + '\n')


def main(args) :
    
    # Path to rawdata
    p = Path(args.pathtogalaxy)
    
    # Load the images in the input rawdata folder
    images = ImageFileCollection(str(p), keywords=['OBJECT', 'ESO INS FILT1 NAME'])
    
    # Galaxy name
    galaxyname = args.objname 
    
    # PHANGS sample table
    PhangsTbl = Path(args.pathtosample)

    # CCD number 
    ccd = args.ccd
    
    # change keywords
    chkey = args.chkeys
    if chkey:
        change_objname_keyword(p,images)
        images = ImageFileCollection(str(p), keywords=['OBJECT', 'ESO INS FILT1 NAME'])
    
    for filt in ['BB#Rc/162_ESO844','NB#Halpha/7_ESO856'] :
        tbl = images.summary['file','OBJECT','ESO INS FILT1 NAME'][(images.summary['OBJECT']==galaxyname)&(images.summary['ESO INS FILT1 NAME']==filt)]
        if len(tbl['file'])==0 :
            print('Object name %s not found!'%galaxyname)
            tbl = images.summary['file','OBJECT','ESO INS FILT1 NAME'][images.summary['ESO INS FILT1 NAME']==filt]
            tbl.show_in_browser()
        else :
            make_file(p, galaxyname, tbl, filt, ccd, PhangsTbl)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('pathtogalaxy', help='path to raw frames')
    parser.add_argument('objname', help='name of selected galaxy')
    parser.add_argument('pathtosample', help='path to PHANGS sample table')
    
    parser.add_argument('-c', '--ccd', action="store", type=int, default=2, help='define the CCD number where the galaxy is centred (default CCD 2)')
    parser.add_argument('-k', '--chkeys', action="store_true", default=False, help='change OBJECT keywords to contain only the galaxy name')

    
    args = parser.parse_args()

    main(args)