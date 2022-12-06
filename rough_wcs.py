# coding: utf-8

import ccdproc
from astropy.io import fits
import astropy.units as u
from astropy.coordinates import SkyCoord
import numpy as np  
import sys
from glob import glob
import os


def load_object (txtfile) :

    x, y = np.loadtxt (txtfile, dtype='str', unpack=True)
    c = SkyCoord (ra=x[1], dec=y[1], unit=(u.hourangle, u.deg))
    #c = c.transform_to('fk5')

    return int(y[0]), c.ra.degree, c.dec.degree, x[2:].astype(float), y[2:].astype(float)

    
def first_wcs (header, X, Y, pixscale, ra, dec, dra, ddec, x, y) :    #, obj, filt, nhdu) :
    
    # increment for the distance (in degrees) between the CCDs
    totdra = (X - header['CHIPX']) * dra
    totddec = (header['CHIPY'] - Y) * ddec
    # write the coordinates of the reference point (centre of the galaxy, standard star)
    header['CRVAL1'] = ra + totdra/np.cos(dec*np.pi/180.)
    header['CRVAL2'] = dec + totddec
    '''
    apply the transformation
    x = -pixscale * (X - CRPIX1)
    y = pixscale * (Y - CRPIX2)
    '''
    header['CRPIX1']=x
    header['CRPIX2']=y
    header['CD1_2']=0
    header['CD2_1']=0
    header['CD1_1']=-pixscale
    header['CD2_2']=pixscale
    header['CTYPE1']='RA---TAN'
    header['CTYPE2']='DEC--TAN'
    header['CUNIT1']='deg'
    header['CUNIT2']='deg'
    
    return header

    


def main() :
    
    if len(sys.argv) == 1 :
        print ("Type the name of a galaxy text file in the folder, after the python code name\ne.g. >>> python rough_wcs.py NGC1365_Rc.txt")
        sys.exit()
    else:
        ra, dec, crpix1, crpix2 = load_object (sys.argv[1])
    
    objectname = sys.argv[1][0:7]
    if sys.argv[1][8:10] == 'Rc' :
        filtname = 'BB#Rc/162_ESO844'
    elif sys.argv[1][8:10] == 'Ha' :    
        filtname = 'NB#Halpha/7_ESO856'
    else :
        sys.exit('Error: filter not defined!')
 
    obsdate = '20161007'
    obsdir = os.path.join ('..', 'DATA', obsdate)
    outdir = os.path.join (obsdir, 'processed', 'output')
    wcsdir = os.path.join (obsdir, 'processed', 'wcs_corrected')
    try:
        os.makedirs(wcsdir)
    except OSError:
        pass

    keys = ['object', 'filtname', 'chipinde']
    reduced_files = sorted(glob(os.path.join(outdir,'nfbtocrg*')))
    filenames = [f[len(outdir)+1:] for f in reduced_files]
    images = ccdproc.ImageFileCollection(outdir, keywords=keys, filenames=filenames)
    match = (images.summary['object'] == objectname) & (images.summary['filtname'] == filtname) #& (images.summary['chipinde'] == nhdu)
    frames = images.summary['file'][match]

    # defining X & Y positions of the CCDs 
    nhdu = 2
    pos = lambda x: {5:[4,1],6:[3,1],7:[2,1],8:[1,1]}.get(x,[x,2])
    X = pos(nhdu)[0]
    Y = pos(nhdu)[1]
    
    # WFI camera characteristic sizes
    pixscale = 0.238/3600. # * u.degree / u.pixel
    xarcsec = 486.948 # x axis dimension in arcsec
    yarcsec = 975.324 # y axis dimension in arcsec
    xgap = 22.848 # gap between ccd in x direction
    ygap = 14.28 # gap between ccd in y direction
    dra = (xarcsec + xgap) / 3600. # deg
    ddec = (yarcsec + ygap)/ 3600. # deg 
    
    for i, f in enumerate(frames) :
        hdu = first_wcs (f, outdir, X, Y, pixscale, ra, dec, dra, ddec, crpix1[i//8], crpix2[i//8], objectname, filtname, nhdu)
        # write the file with rough wcs
        print ('Writing a rough WCS solution for %s ...' %f)
        hdu.writeto (os.path.join(wcsdir, 'rwcs%s_%s_%s_frame%s.fits' %(objectname, filtname[3:5], hdu.header['CHIPINDE'], i//8+1)), overwrite=True)
        print ('Fits file rwcs%s_%s_%s_frame%s.fits created' %(objectname, filtname[3:5], hdu.header['CHIPINDE'], i//8+1))
    

if __name__ == "__main__":
    
    main()
