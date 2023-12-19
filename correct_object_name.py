#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 03 16:34:00 2018

@author: arazza
"""

from pathlib import Path
from ccdproc import ImageFileCollection
import argparse
import astropy.io.fits as fits

def change_objname_keyword(p,images):
    objnames = images.summary['OBJECT'].data.data
    select = [i for i in range(len(objnames)) if objnames[i].endswith('small') or objnames[i].endswith('large')]
    for i in select:
        with fits.open(p.joinpath(images.summary['file'][i]),'update') as hdul:
            newkey = hdul[0].header['OBJECT'].split('_')[0]
            hdul[0].header['OBJECT']=newkey
    select2 = [i for i in range(len(objnames)) if objnames[i].endswith(' ')]
    for i in select2:
        with fits.open(p.joinpath(images.summary['file'][i]),'update') as hdul:
            newkey = hdul[0].header['OBJECT'].strip(' ')
            hdul[0].header['OBJECT']=newkey



def main(args) :

    # Path to rawdata
    p = Path(args.pathtogalaxy)

    # Load the images in the input rawdata folder
    images = ImageFileCollection(str(p), keywords=['OBJECT', 'ESO INS FILT1 NAME'])
    # change keywords
    change_objname_keyword(p, images)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('pathtogalaxy', help='path to raw frames')

    args = parser.parse_args()

    main(args)