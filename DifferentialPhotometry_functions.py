#Differential Photometry script written in April 2019 by SKB, MP, KWD for WIYN 0.9m HDI data

#This script calculates photometry and differential photometry for all stars in an image and takes target positions to pull out differential photometry of target stars.  Auto calculates comparison stars based on lowest percentile of variability of stars in the image.   
# Script is run through a shell jupyter notebook script. 

#Initially created by Mike Peterson as a juypter notebook 2018
#Turned into modular form by Sarah Betti April 2019
#Modified by Mike Peterson, Kim Ward-Duong, Sarah Betti April 2019


# python 2/3 compatibility
from __future__ import print_function
# numerical python
import numpy as np
# file management tools
import glob
import os
# good module for timing tests
import time
# plotting stuff
import matplotlib.pyplot as plt
import matplotlib.cm as cm
# ability to read/write fits files
from astropy.io import fits
# fancy image combination technique
from astropy.stats import sigma_clip
# median absolute deviation: for photometry
from astropy.stats import mad_std
# photometric utilities
from photutils import DAOStarFinder,aperture_photometry, CircularAperture, CircularAnnulus, Background2D, MedianBackground
# periodograms
from astropy.stats import LombScargle
from regions import read_ds9, write_ds9
from astropy.wcs import WCS
import warnings
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.visualization import ZScaleInterval
import numpy.ma as ma
warnings.filterwarnings("ignore")

np.set_printoptions(suppress=True)


def construct_astrometry(hdr_wcs):
    '''
    construct_astrometry
    
    make the pixel to RA/Dec conversion (and back) from the header of an astrometry.net return
    
    inputs
    ------------------------------
    hdr_wcs  :  header with astrometry information, typically from astrometry.net
    
    returns
    ------------------------------
    w        :  the WCS instance
    
    '''
    
    # initialize the World Coordinate System
    w = WCS(naxis=2)
    
    # specify the pixel to RA/Dec conversion
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.cd = np.array([[hdr_wcs['CD1_1'],hdr_wcs['CD1_2']],[hdr_wcs['CD2_1'],hdr_wcs['CD2_2']]])
    w.wcs.crpix = [hdr_wcs['CRPIX1'], hdr_wcs['CRPIX2']]
    w.wcs.crval = [hdr_wcs['CRVAL1'],hdr_wcs['CRVAL2']]
    w.wcs.cunit = [hdr_wcs['CUNIT1'],hdr_wcs['CUNIT2']]
    w.wcs.latpole = hdr_wcs['LATPOLE']
    #w.wcs.lonpole = hdr_wcs['LONPOLE']
    w.wcs.theta0 = hdr_wcs['LONPOLE']
    w.wcs.equinox = hdr_wcs['EQUINOX']

    # calculate the RA/Dec to pixel conversion
    w.wcs.fix()
    w.wcs.cdfix()
    w.wcs.set()
    
    # return the instance
    return w



def StarFind(imname, FWHM, nsigma):
    '''
    StarFind
    
    find all stars in a .fits image
    
    inputs
    ----------
    imname: name of .fits image to open.  
    FWHM: fwhm of stars in field
    nsigma: number of sigma above background above which to select sources.  (~3 to 4 is a good estimate)
    
    outputs
    --------
    xpos: x positions of sources
    ypos: y positions of sources
    nstars: number of stars found in image
    '''
    
    #open image
    im,hdr=fits.getdata(imname, header=True)

    im = np.array(im).astype('float')
    #determine background
    bkg_sigma = mad_std(im)
    
    print('begin: DAOStarFinder')
    
    daofind = DAOStarFinder(fwhm=FWHM, threshold=nsigma*bkg_sigma, exclude_border=True)
    sources = daofind(im)
    
    #x and y positions
    xpos = sources['xcentroid']
    ypos = sources['ycentroid']
    
    
    #number of stars found
    nstars = len(xpos)
    
    print('found ' + str(nstars) + ' stars')
    return xpos, ypos, nstars


def makeApertures(xpos, ypos, aprad,skybuff, skywidth):
    '''
    makeApertures
    
    makes a master list of apertures and the annuli
    
    inputs
    ---------
    xpos: list - x positions of stars in image
    ypos: list - y positions of stars in image 
    aprad: float - aperture radius
    skybuff: float - sky annulus inner radius
    skywidth: float - sky annulus outer radius
    
    outputs
    --------
    apertures: list - list of aperture positions and radius
    annulus_apertures: list - list of annuli positions and radius
    see: https://photutils.readthedocs.io/en/stable/api/photutils.CircularAperture.html#photutils.CircularAperture 
    for more details
    
    '''
    
    # make the master list of apertures
    apertures = CircularAperture((xpos, ypos), r=aprad)
    annulus_apertures = CircularAnnulus((xpos, ypos), r_in=aprad+skybuff, r_out=aprad+skybuff+skywidth)
    apers = [apertures, annulus_apertures]

    return apertures, annulus_apertures

def apertureArea(apertures):
    ''' returns the area of the aperture'''
    return apertures.area()  ### should be apertures

def backgroundArea(back_aperture):
    '''returns the area of the annuli'''
    return back_aperture.area() ### should be annulus_apertures


def doPhotometry(imglist, xpos, ypos, aprad, skybuff, skywidth,timekey='MJD-OBS',verbose=1): 
    '''
    doPhotomoetry*
    
    determine the flux for each star from aperture photometry
    
    inputs
    -------
    imglist: list - list of .fits images
    xpos, ypos: lists - lists of x and y positions of stars
    aprad, skybuff, skywidth: floats - aperture, sky annuli inner, sky annuli outer radii 
    
    outputs
    -------
    Times: list - time stamps of each observation from the .fits header
    Photometry: list - aperture photometry flux values found at each xpos, ypos position
    
    '''
    
    #number of images
    nimages = len(imglist)
    nstars = len(xpos)
    print('Found {} images'.format(nimages))
    
    #create lists for timestamps and flux values
    Times = np.zeros(nimages)
    Photometry = np.zeros((nimages,nstars))
    
    print('making apertures')
    #make the apertures around each star
    apertures, annulus_apertures = makeApertures(xpos, ypos, aprad, skybuff, skywidth)
    

    #plot apertures
    plt.figure(figsize=(12,12))
    interval = ZScaleInterval()
    vmin, vmax = interval.get_limits(fits.getdata(imglist[0]))
    plt.imshow(fits.getdata(imglist[0]), vmin=vmin,vmax=vmax, origin='lower')
    apertures.plot(color='white', lw=2)
    #annulus_apertures.plot(color='red', lw=2)
    plt.title('apertures')
    plt.show()
    
    #determine area of apertures
    area_of_ap = apertureArea(apertures)    

    #determine area of annuli
    area_of_background = backgroundArea(annulus_apertures)

    checknum = np.linspace(0,nimages,10).astype(int)


    #go through each image and run aperture photometry
    for ind in np.arange(nimages):

        if ((ind in checknum) & (verbose==1)):
            print('running aperture photometry on image: ', ind )

        if (verbose>1):
            print('running aperture photometry on image: ', ind )
            
        #open image
        data_image, hdr = fits.getdata(imglist[ind], header=True)
        
        #find time stamp and append to list
        Times[ind] = hdr[timekey]
        
        #do photometry
        phot_table = aperture_photometry(data_image, (apertures,annulus_apertures))
        
        #determine flux: (aperture flux) - [(area of aperture * annuli flux)/area of background ]
        flux0 = np.array(phot_table['aperture_sum_0']) - (area_of_ap/area_of_background)*np.array(phot_table['aperture_sum_1'])

        #append to list 
        Photometry[ind,:] = flux0
           
    return Times,Photometry



def doPhotometryError(imglist,xpos, ypos,aprad, skybuff, skywidth, flux0, GAIN=1.3, manual = False, **kwargs):
    '''
    doPhotometryError
    
    determine error in photometry from background noise 
    two options:
    - use sigma clipping and use whole background
    - manually input background box positions as kwargs
    
    inputs
    --------
    imglist: list - list of .fits images
    xpos, ypos: lists - lists of x and y positions of stars
    aprad, skybuff, skywidth: floats - aperture, sky annuli inner, sky annuli outer radii 
    flux0: list - aperture photometry found from doPhotometry() function
    GAIN: float - average gain 
    manual: boolean - switch between manually inputting box (True) or using sigma clipping (False)
                        if True -- must have kwargs
                        manual = False is default
    **kwargs
        kwargs[xboxcorner]: float - x edge of box in pixel coords
        kwargs[yboxcorner]: float - y edge of box in pixel coords
        kwargs[boxsize]: float - size of box in pixel coords
     

    '''
    
    # find number of images in list
    nimages = len(imglist)
    nstars = len(xpos)
    print('Found {} images'.format(nimages))
    
    #make apertures
    apertures, annulus_apertures = makeApertures(xpos, ypos, aprad, skybuff, skywidth)
    
    
    #find areas of apertures and annuli
    area_of_ap = apertureArea(apertures)    

    area_of_background = backgroundArea(annulus_apertures)

    checknum = np.linspace(0,nimages,10).astype(int)


    #find error in photometry
    ePhotometry = np.zeros((nimages,nstars))
    for ind in np.arange(nimages):
        #open images
        im = fits.getdata(imglist[ind])

        if ind in checknum:
            print('running error analysis on image ', ind)
        
        #determine variance in background 
        if manual == True: #manual method -- choose back size
            skyvar = np.std(im[kwargs['xboxcorner']:kwargs['xboxcorner']+kwargs['boxsize'],kwargs['yboxcorner']:kwargs['yboxcorner']+kwargs['boxsize']])**2.
            err1 = skyvar*(area_of_ap)**2./(kwargs['boxsize']*kwargs['boxsize'])  # uncertainty in mean sky brightness
        if manual == False: #automatic method -- use sigma clipping
            filtered_data = sigma_clip(im, sigma=3)
            skyvar = np.std(filtered_data)**2.
            err1 = skyvar*(area_of_ap)**2./(np.shape(im[0])[0]*np.shape(im[1])[0])  # uncertainty in mean sky brightness
    
        err2 = area_of_ap * skyvar  # scatter in sky values

        err3 = flux0[ind]/GAIN # Poisson error
    
        print ('Scatter in sky values: ',err2**0.5,', uncertainty in mean sky brightness: ',err1**0.5)
    
        # sum souces of error in quadrature
        errtot = (err1 + err2 + err3)**0.5
        
        #append to list
        ePhotometry[ind,:] = errtot
        
    return ePhotometry  


def mask(Photometry, ePhotometry, sn_thresh=3.):

    Photometry_mask1 = ma.masked_where(Photometry <= 0, Photometry)

    sn = Photometry_mask1 / ePhotometry
    Photometry_mask2 = ma.masked_where(sn < sn_thresh, Photometry_mask1)
    
    ePhotometry_mask1 = ma.masked_where(Photometry <= 0, ePhotometry)

    sn = Photometry_mask1 / ePhotometry
    ePhotometry_mask2 = ma.masked_where(sn < sn_thresh, ePhotometry_mask1)
    
    return Photometry_mask2, ePhotometry_mask2


# detrend all stars
def detrend(idx, Photometry_initial, ePhotometry, nstars, sn_thresh):
    '''
    detrend
    
    detrend the background for each night so we don't have to worry about changes in background noise levels
    
    inputs
    -------
    photometry: list - list of flux values from aperture photometry
    ephotometry: list - list of flux errors from aperture photometry
    nstars: float - number of stars in the field
    
    outputs
    --------
    finalPhot: list - final aperture photometry of sources with bad sources replaced with nans.  
                        << this is the list you want to use from now on. >> 
    cPhotometry: list - detrended aperture photometry
    
    '''
    sn = Photometry_initial / ePhotometry
    
    Photometry_mask1 = ma.masked_where(Photometry_initial <= 0, Photometry_initial)
    Photometry_mask2 = ma.masked_where(sn < sn_thresh, Photometry_mask1)
    #mask out target stars
    m = np.zeros_like(Photometry_mask2)
    m[:,idx] = 1
    Photometry_initial_mask3 = ma.masked_array(Photometry_mask2, m)
       
    med_val = np.median(Photometry_initial_mask3, axis=0)
    
    c = np.zeros_like(Photometry_initial_mask3)
    c[:,med_val<=0] = 1

    # get median flux value for each star (find percent change)
    cPhotometry = ma.masked_array(Photometry_initial_mask3, c)
    cPhotometry = cPhotometry / med_val

    # do a check for outlier photometry?
    for night in np.arange(len(cPhotometry)):
        # remove large-scale image-to-image variation to find best stars
        cPhotometry[night] = cPhotometry[night] / np.median(cPhotometry[night])
    # eliminate stars with outliers from consideration 
    cPhotometry_mask = ma.masked_where( ((cPhotometry < 0.5) | (cPhotometry > 1.5)), cPhotometry)
    
    return Photometry_initial_mask3, cPhotometry_mask
        

def plotPhotometry(Times,cPhotometry):
    '''plot detrended photometry'''
    plt.figure()
    for ind in np.arange(np.shape(cPhotometry)[1]):
        plt.scatter(Times-np.nanmin(Times),cPhotometry[:,ind],s=1.,color='black')


    # make the ranges a bit more general
    plt.xlim(-0.1,1.1*np.max(Times-np.nanmin(Times)))

    plt.ylim(np.nanmin(cPhotometry),np.nanmax(cPhotometry))

    plt.xlabel('Observation Time [days]')
    plt.ylabel('Detrended Flux')
    plt.show()


def CaliforniaCoast(Photometry,cPhotometry,comp_num=9,flux_bins=6):
    """
    Find the least-variable stars as a function of star brightness*

    (it's called California Coast because the plot looks like California and we are looking for edge values: the coast)
    
    
    inputs
    --------------
    Photometry  : input Photometry catalog
    cPhotometry : input detrended Photometry catalog
    flux_bins   : (default=10) maximum number of flux bins
    comp_num    : (default=5)  minimum number of comparison stars to use
    
    
    outputs
    --------------
    BinStars      : dictionary of stars in each of the flux partitions
    LeastVariable : dictionary of least variable stars in each of the flux partitions
    
    
    """
    
    tmpX = np.nanmedian(Photometry,axis=0)
    tmpY = np.nanstd(cPhotometry, axis=0)
    
    xvals = tmpX[(np.isfinite(tmpX) & np.isfinite(tmpY))]
    yvals = tmpY[(np.isfinite(tmpX) & np.isfinite(tmpY))]
    kept_vals = np.where((np.isfinite(tmpX) & np.isfinite(tmpY)))[0]
    #print('Keep',kept_vals)
    
    # make the bins in flux, equal in percentile
    flux_percents = np.linspace(0.,100.,flux_bins)
    print('Bin Percentiles to check:',flux_percents)
    
    # make the dictionary to return the best stars
    LeastVariable = {}
    BinStars = {}
    
    for bin_num in range(0,flux_percents.size-1):
        
        # get the flux boundaries for this bin
        min_flux = np.percentile(xvals,flux_percents[bin_num])
        max_flux = np.percentile(xvals,flux_percents[bin_num+1])
        #print('Min/Max',min_flux,max_flux)

        # select the stars meeting the criteria
        w = np.where( (xvals >= min_flux) & (xvals < max_flux))[0]
        
        BinStars[bin_num] = kept_vals[w]
        
        # now look at the least variable X stars
        nstars = w.size
        #print('Number of stars in bin {}:'.format(bin_num),nstars)
        
        # organize stars by flux uncertainty
        binStarsX = xvals[w]
        binStarsY = yvals[w]
                
        # mininum Y stars in the bin:
        lowestY = kept_vals[w[binStarsY.argsort()][0:comp_num]]
        
        #print('Best {} stars in bin {}:'.format(comp_num,bin_num),lowestY)
        LeastVariable[bin_num] = lowestY
        
    return BinStars,LeastVariable
    


def findComparisonStars(Photometry, cPhotometry, accuracy_threshold = 0.2, plot=True,comp_num=6): #0.025
    '''
    findComparisonStars*
    
    finds stars that are similar over the various nights to use as comparison stars
    
    inputs
    --------
    Photometry: list - photometric values taken from detrend() function. 
    cPhotometry: list - detrended photometric values from detrend() function
    accuracy_threshold: float - level of accuracy in fluxes between various nights
    plot: boolean - True/False plot various stars and highlight comparison stars

    outputs
    --------
    most_accurate: list - list of indices of the locations in Photometry which have the best stars to use as comparisons
    '''

    BinStars,LeastVariable = CaliforniaCoast(Photometry,cPhotometry,comp_num=comp_num)

    star_err = ma.std(cPhotometry, axis=0)

    if plot:
        xvals = np.log10(ma.median(Photometry,axis=0))
        yvals = np.log10(ma.std(cPhotometry, axis=0))

        plt.figure()
        plt.scatter(xvals,yvals,color='black',s=1.)
        plt.xlabel('log Median Flux per star')
        plt.ylabel('log De-trended Standard Deviation')
        plt.text(np.nanmin(np.log10(ma.median(Photometry,axis=0))),np.nanmin(np.log10(star_err[star_err>0.])),\
            'Less Variable',color='red',ha='left',va='bottom')
        plt.text(np.nanmax(np.log10(ma.median(Photometry,axis=0))),np.nanmax(np.log10(star_err[star_err>0.])),\
            'More Variable',color='red',ha='right',va='top')

        for k in LeastVariable.keys():
            plt.scatter(xvals[LeastVariable[k]],yvals[LeastVariable[k]],color='red')


    # this is the middle key for safety
    middle_key = np.array(list(LeastVariable.keys()))[len(LeastVariable.keys())//2]

    # but now let's select the brightest one
    best_key = np.array(list(LeastVariable.keys()))[-1]
    
    return LeastVariable[best_key]


def runDifferentialPhotometry(photometry, ephotometry, nstars, most_accurate, sn_thresh):
    
    

    '''
    runDifferentialPhotometry
    
    as the name says!
    
    inputs
    ----------
    Photometry: list - list of photometric values from detrend() function
    ePhotometry: list - list of photometric error values
    nstars: float - number of stars 
    most_accurate: list - list of indices of non variable comparison stars
    
    outputs
    ---------
    dPhotometry: list - differential photometry list
    edPhotometry: list - scaling factors to photometry error
    tePhotometry: list - differential photometry error
    
    '''
    
    Photometry = ma.masked_where(photometry <= 0, photometry)
    ePhotometry = ma.masked_where(photometry <= 0, ephotometry)
    
    
    #number of nights of photometry
    nimages = len(Photometry)
    
    #range of number of nights
    imgindex = np.arange(0,nimages,1)

    #create lists for diff photometry
    dPhotometry = ma.ones([nimages, len(Photometry[0])])
    edPhotometry = ma.ones([nimages, len(Photometry[0])])
    eedPhotometry = ma.ones([nimages, len(Photometry[0])])
    tePhotometry = ma.ones([nimages, len(Photometry[0])])
    
    checknum = np.linspace(0,nstars,10).astype(int)

    for star in np.arange(nstars):
        if star in checknum:
            print('running differential photometry on star: ', star+1, '/', nstars)
        starPhotometry = Photometry[:,star]
        starPhotometryerr = ePhotometry[:,star]
         #create temporary photometry list for each comparison star
        tmp_phot = ma.ones([nimages,len(most_accurate)])
        #go through comparison stars and determine differential photometry
        for ind, i in enumerate(most_accurate):
             #pull out each star's photometry + error Photometry for each night and place in list 
            compStarPhotometry = Photometry[:,i]
             #calculate differential photometry
            tmp = starPhotometry*ma.median(compStarPhotometry)/(compStarPhotometry*ma.median(starPhotometry))
            tmp_phot[:,ind] = tmp
        
        #median combine differential photometry found with each comparison star for every other star
        dPhotometry[:,star] = ma.median(tmp_phot,axis=1)

        # apply final scaling factors to the photometric error
        edPhotometry[:,star] = starPhotometryerr*(ma.median(tmp_phot,axis=1)/starPhotometry)
        
        # the differential photometry error
        eedPhotometry[:,star] = ma.std(tmp_phot,axis=1)

        # the differential photometry error
        tePhotometry[:,star] = ((starPhotometryerr*(ma.median(tmp_phot,axis=1)/starPhotometry))**2. + (ma.std(tmp_phot,axis=1))**2.)**0.5

    return dPhotometry, edPhotometry, tePhotometry


def target_list(memberlist, ra_all, dec_all):
    #checks to see if memberlist is a tuple or region file
    if isinstance(memberlist, tuple):
        ra_mem = [memberlist[0]]
        dec_mem = [memberlist[1]]
    elif isinstance(memberlist, str):
        try:
            regions = read_ds9(memberlist)
            ra_mem = [i.center.ra.deg for i in regions]
            dec_mem = [i.center.dec.deg for i in regions]
        except:
            print('memberlist must be a region file or tuple')
        
    else:
        ValueError('memberlist must be region file or tuple')
    
    print('number of targets: {}'.format(len(ra_mem)))
    
    #finds your target star index in the catalog found with DAOStarFinder
    c = SkyCoord(ra=ra_mem*u.degree, dec=dec_mem*u.degree)  
    catalog = SkyCoord(ra=ra_all*u.degree, dec=dec_all*u.degree)  
    
    max_sep = 20.0 * u.arcsec 
    ind, d2d, d3d = c.match_to_catalog_3d(catalog) 
    sep_constraint = d2d < max_sep 
    
    idx = ind[sep_constraint]
    
    return idx, ra_all[idx], dec_all[idx]


def diffPhot_IndividualStars(datadir, idx, ra, dec, xpos, ypos, dPhotometry, edPhotometry, tePhotometry, times, target, filt, fitsimage, most_accurate):
    '''
    diffPhot_IndividualStars
    
    pull out differential photometry for objects of interest from region file
    
    inputs
    --------
    memberlist: tuple OR region file 
         can either be a tuple (<RA>, <DEC>)  or a region file listing ra and dec of sources
    ra_all, dec_all, xpos, ypos: list - list of ra, dec, x pixel, y pixel positions of all stars in field
    dPhotometry, edPhotometry, eedPhotometry, tePhotometry: lists - lists of differential, photometric err, differential photometric error fluxes found with runDifferentialPhotometry() function
    times: list - list of time stamps
    target: string - name of target
    
    outputs
    --------
    npz save file: numpy save file with ra, dec, xpos, ypos, time, phase, period, flux, fluxerr, and power for each target star 
    
    can be read back in by:
    data = np.load('<name of file>.npz')
    # get column names
    print(data.files)
    # read ra positions of all stars
    print(data['ra'])
    
    '''

    
    print('number of target stars:', len(idx))
    print('index of target stars:', idx)
    
    interval = ZScaleInterval()
    vmin, vmax = interval.get_limits(fits.getdata(fitsimage))
    plt.figure(figsize=(10,10))
    plt.imshow(fits.getdata(fitsimage), vmin=vmin, vmax=vmax)
    plt.plot(xpos[idx], ypos[idx], 'ro')
    plt.plot(xpos[most_accurate], ypos[most_accurate], 'kx')
    plt.title('target stars')
    plt.show()
    

    #determine time
    time = np.array(times - np.nanmin(times))
    #pulls out differential photometry of target stars
    
    tarra = []
    tardec = []
    tarxpos = []
    tarypos = []
    flux = []
    fluxerr = []
    foldedphase = []
    totperiod = []
    totpower = []
    
    for star in idx:
        dPhot = dPhotometry[:,star]
        #pulls out differetial photometry error of target stars
        tePhot = tePhotometry[:,star]
        
        DD = ma.filled(dPhot, fill_value = np.nan)
        ED = ma.filled(tePhot, fill_value = np.nan)
        TT = time
        
        #run L-S analysis
        print('starting L-S analysis on star at: (x,y) = {}'.format(xpos[star], ypos[star]) )

        #determine where the values are good.
        w = np.where(np.isfinite(TT) & np.isfinite(DD) & np.isfinite(ED))
        print('Differential flux: ')
        print(DD[w])
        
        #run LombScargle
        ls = LombScargle(TT[w], DD[w], ED[w])

        #get power and frequency
        try:
            frequency, power = ls.autopower()
            #determine period
            per = 1./frequency

            #determine best frequency from LombScargle
            w = np.where( (per > 0.2) & (per< 4.))
            best_frequency = frequency[w][np.argmax(power[w])]

            #phase folding
            newtime = TT % (1./best_frequency)
            phase = newtime - np.round(newtime,0) + 0.5
            foldedphase.append(phase)
            totperiod.append(per)
            totpower.append(power)
        except:
            print('L-S failed.  Potentially one 1 night had good photometry.  All phase, power, and period will nans')
            foldedphase.append(np.ones_like(DD)*np.nan)
            totperiod.append(np.array([np.nan]))
            totpower.append(np.array([np.nan]))
        print('--------')    

        #append all to list
        flux.append(DD)
        fluxerr.append(ED)
        tarra.append(ra[star])
        tardec.append(dec[star])
        tarxpos.append(xpos[star])
        tarypos.append(ypos[star])
        


    savefile = datadir + 'differentialPhot_field' + target + filt

    #save as npz file.  much easier and saves lists as lists
    np.savez(savefile, ra =tarra, dec=tardec, xpos=tarxpos, ypos=tarypos, time=time,flux=flux,fluxerr= fluxerr, foldedphase =foldedphase, period=totperiod, power=totpower  )
    
    print('finished.  saving catalogue to:', savefile)

    return tarra, tardec, tarxpos, tarypos, time, flux, fluxerr, foldedphase, totperiod, totpower 

    #save as npz file.  much easier and saves lists as lists
#    np.savez(savefile, ra =ra[idx], dec=dec[idx], xpos=xpos[idx], ypos=ypos[idx], time=time,flux=flux,fluxerr= fluxerr)
#    print('finished.  saving catalogue to:', savefile)
#
#    return ra[idx], dec[idx], xpos[idx], ypos[idx], time, flux, fluxerr
#      
      

