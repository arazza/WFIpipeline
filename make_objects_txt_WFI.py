#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: arazza
"""

from pathlib import Path
from ccdproc import ImageFileCollection
import argparse
import os
import sys
from shutil import copy

from astropy.coordinates import SkyCoord

def make_file (p, name, tbl, filt) :
    
    obsdate = str(p.parts[-2])

    if 'Rc' in filt :
        filtname = 'Rc'
    if 'Ha' in filt :
        filtname = 'Ha'
    
    obj_file = 'objects_txt/%s_%s_%s.txt'%(name,filtname,obsdate)
    
    with open(obj_file, 'w') as f :
        f.write('# %s in %s \n# Observing Run %s \n# first row: RA, DEC (hours, degrees) \n# other rows: crpix1, crpix2 for each dithered frame (pixel)\n'%(name,filt,obsdate))

    with open(obj_file, 'a') as f :
        mycoord = SkyCoord.from_name(name)
        f.write(mycoord.to_string('hmsdms') + '\n')

    files_to_load = []
    
    for i in range(len(tbl['file'])) :
        files_to_load.append(str(p.joinpath(tbl['file'][i])))
    
    command = 'ds9 -zscale '
    for f in files_to_load :
        command = command + f + ' '

    print(command)
    
    try :
        os.system(command)
    except OSError :
        pass

    while True :
        pixels = input()
        if pixels=='' :
            break
        else :
            with open(obj_file, 'a') as f :
                f.write(pixels + '\n')    
    


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('pathtogalaxy', help='path to raw frames')
    parser.add_argument('objname', help='name of selected object')
    
    args = parser.parse_args()

    # Path to rawdata
    p = Path(args.pathtogalaxy)
    
    # Load the images in the input rawdata folder
    images = ImageFileCollection(str(p), keywords=['OBJECT', 'ESO INS FILT1 NAME'])
    
    # Galaxy name
    objname = args.objname 
     
    for filt in ['BB#Rc/162_ESO844','NB#Halpha/7_ESO856'] :

        print('\n*** Filter %s ***\n'%filt)
        
        tbl = images.summary['file','OBJECT','ESO INS FILT1 NAME'][(images.summary['OBJECT']==objname)&(images.summary['ESO INS FILT1 NAME']==filt)]
        
        if len(tbl['file'])==0 :
            
            print('Object name %s or filter %s not found!'%(objname,filt))
        
        else :
            
            make_file(p, objname, tbl, filt)
