#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 16:46:00 2019

@author: arazza
"""

from pathlib import Path
from ccdproc import ImageFileCollection
import argparse
import os
import re
import sys
from shutil import copy

from astropy.coordinates import SkyCoord

def make_file (p, name, tbl, filt) :

    def sort_nicely (l): 
        ''' 
        Sort the given list in the way that humans expect (human sorting). 
        Natural (human) sort fixes the problem of frame #10 being the last 
        created but #9 the last one in the list generated with glob method. 
        Withouth this fix #10 is overwritten N times by N new frames. 
        ''' 
        convert = lambda text: int(text) if text.isdigit() else text 
        alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)] 
        
        return sorted(l, key=alphanum_key)
    
    pos = str(p).find('ut')    
    obsdate = str(p)[pos:pos+11]
    
    Path('objects_txt').mkdir(exist_ok=True)
    obj_file = 'objects_txt/%s_%s_%s.txt'%(name,filt,obsdate)
    
    with open(obj_file, 'w') as f :
        f.write('# %s in %s \n# Observing Run %s \n# first row: RA, DEC (hours, degrees) \n# other rows: crpix1, crpix2 for each dithered frame (pixel)\n'%(name,filt,obsdate))

    with open(obj_file, 'a') as f :
        mycoord = SkyCoord.from_name(name)
        f.write(mycoord.to_string('hmsdms') + '\n')
    
    list_of_files = sort_nicely(tbl['file'])

    files_to_load = []
    
    for i in range(len(list_of_files)) :
        files_to_load.append(str(p.joinpath(list_of_files[i])))
    
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

    # Look for the folder with fits files (in case they were moved to "detrend" by dirCCDreduction.py script)
    
    list_of_fits = sorted(p.glob('*.fits'))
    
    if list_of_fits == [] :
        
        detrend_path = p.joinpath('output', 'detrend')
        
        images = ImageFileCollection(detrend_path, keywords=['OBJECT', 'FILTER'])
        
    else :
        
        images = ImageFileCollection(p, keywords=['OBJECT', 'FILTER'])
    
    
    # Make objects_txt file for each valid filter 
    for filt in ['r','Ha657','Ha663'] :
        
        print('\n*** Filter %s ***\n'%filt)
        
        tbl = images.summary['file','OBJECT','FILTER'][(images.summary['OBJECT']==args.objname)&(images.summary['FILTER']==filt)]
        
        if len(tbl['file'])==0 :
                
            print('Object name %s or filter %s not found!'%(args.objname,filt))
            
        else :
            
            make_file(images.location, args.objname, tbl, filt)
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('pathtogalaxy', help='path to raw frames')
    parser.add_argument('objname', help='name of selected galaxy')
    
    args = parser.parse_args()
    
    main(args)