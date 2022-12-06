# -*- coding: utf-8 -*-

'''
Spectro-photometric standard stars (SPSS) from Gaia SPSS survey
are used to compute the color-color (CC) relation, i.e. the difference
between broad- and narrow-band versus Gaia color.

Two classes are created:
- GaiaSPSS
- SpecPhotStdStar

GaiaSPSS class has the following methods:
1) update_data_frame --> update the data frame with new synthetic magnitudes
2) plot_cc --> plot the CC relation

SpecPhotStdStar class has the following methods:
1) synthetic_magnitude --> compute the synthetic magnitudes
'''

import timeit

import sys

import argparse

import numpy as np

from pathlib import Path

from astropy.io import ascii
from astropy import units as u
from astropy import constants as const

import pandas as pd

import seaborn as sns

from matplotlib import pyplot as plt

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


class GaiaSPSS (object) :
    '''
    Class to store and update the SPSS V2 data table.
    '''
    
    def __init__ (self, csv_table) :
        
        self.df = pd.read_csv(csv_table, skiprows=[1])

    
    def update_data_frame (self) :
        '''
        Mathod to reate new columns in the DataFrame to include the colors.
        '''
        
        self.df['GBP-GRP'] = self.df['G_BPsynthetic'] - self.df['G_RPsynthetic']
        self.df['GBP-GRP_ERR'] = np.sqrt(self.df['G_BPsyntERR']**2. + self.df['G_RPsyntERR']**2.)
        
        self.df['Rc-Ha'] = self.df['Rc_synthetic'] - self.df['Ha_synthetic']
        self.df['Rc-Ha_ERR'] = np.sqrt(self.df['Rc_syntERR']**2. + self.df['Ha_syntERR']**2.)
        
        self.df['r-Ha657'] = self.df['r_synthetic'] - self.df['Ha657_synthetic']
        self.df['r-Ha657_ERR'] = np.sqrt(self.df['r_syntERR']**2. + self.df['Ha657_syntERR']**2.)
        
        self.df['r-Ha663'] = self.df['r_synthetic'] - self.df['Ha663_synthetic']
        self.df['r-Ha663_ERR'] = np.sqrt(self.df['r_syntERR']**2. + self.df['Ha663_syntERR']**2.)


    def group_stellar_type (self) :
        '''
        Group together sub-type of stars to reduce the total number of types.
        '''
        
        stellar_type = gaia_spss.df['G_LSPTYP'].fillna('None').to_numpy()
        
        newtype = []
        
        for st in stellar_type :
            
            if st[0]=='O' :
                newtype.append('O')
            
            if st[0]=='B' :
                newtype.append('B')
            
            if st[0]=='A' :
                newtype.append('A')
            
            if st[0]=='F' :
                newtype.append('F')
            
            if st[0]=='G' :
                newtype.append('G')
            
            if st[0]=='K' :
                newtype.append('K')
            
            if st[0]=='M' :
                newtype.append('M')
            
            if (st[0]=='D') :
                if (st[1]=='A') :
                    newtype.append('DA')
                else :
                    newtype.append('D')
             
            if (st=='None') :
                newtype.append('None')
        
        stellar_type = pd.Series(data=newtype, index=self.df.index, name='Stellar Type')
            
        self.df = pd.concat([self.df, stellar_type], axis=1)


    def sort_by_stellar_type (self) :

        # Define the sorter
        sorter = ['O', 'B', 'A', 'F', 'G', 'K', 'M', 'DA', 'D', 'None']

        # Create the dictionary that defines the order for sorting
        sorterIndex = dict(zip(sorter, range(len(sorter))))

        # Generate a rank column that will be used to sort
        # the dataframe numerically
        self.df['ST_Rank'] = self.df['Stellar Type'].map(sorterIndex)
        
        # Here is the result asked with the lexicographic sort
        self.df.sort_values('ST_Rank', inplace=True)
        self.df.drop('ST_Rank', 1, inplace = True)



    def plot_cc (self, draw_fit) :
        
        # update column name
        #self.df['Stellar Type'] = self.df['G_LSPTYP']
        
        # Instantiate the vectors
        BP_RPmag = self.df['GBP-GRP'].to_numpy()
        Rc_Ha = self.df['Rc-Ha'].to_numpy()
        r_Ha657 = self.df['r-Ha657'].to_numpy() 
        r_Ha663 = self.df['r-Ha663'].to_numpy()
        BP_RPmag_err = self.df['GBP-GRP_ERR'].to_numpy()
        Rc_Ha_err = self.df['Rc-Ha_ERR'].to_numpy()
        r_Ha657_err = self.df['r-Ha657_ERR'].to_numpy() 
        r_Ha663_err = self.df['r-Ha663_ERR'].to_numpy()
        stellar_type = self.df['Stellar Type'].to_numpy()
        
        # Preprare the x-axis for the fit
        #valid_color_range = (BP_RPmag >= 0.2) * (BP_RPmag <= 1.8)
        valid_color_range = np.full(BP_RPmag.shape, True)
        valid_stellar_type = (stellar_type == 'A') + (stellar_type == 'F') + (stellar_type == 'G') + (stellar_type == 'K') + (stellar_type == 'M')
        #xfit = np.linspace(0.2, 1.8)
        xfit = np.linspace(np.min(BP_RPmag[valid_stellar_type]), np.max(BP_RPmag[valid_stellar_type]))
        X = (BP_RPmag[valid_color_range*valid_stellar_type])[:, np.newaxis]

        # Upload the coefficient from PHANGS-Ha observations
        if draw_fit :
            
            list_of_npy = sorted(Path('/Users/arazza/DAS/Observations/NB_DR3_pipeline').glob('*.npy'))
            coeffs = [np.load(npy_file) for npy_file in list_of_npy]
            draw_fit_str = '_draw_fit'

        else :
        
            draw_fit_str = ''

        
        print('\n\nRemaining star to compute the CC relation : %s\n\n'%len(BP_RPmag[valid_color_range*valid_stellar_type]))

        # Plot Rc-Ha
        
        fig, ax1 = plt.subplots(1, 1, figsize=(7,7)) 

        sns.scatterplot(x='GBP-GRP', y='Rc-Ha', data=self.df, hue='Stellar Type', ax=ax1, zorder=2)

        leg1 = ax1.legend(loc='lower right', fontsize=9)

        ax1.errorbar(BP_RPmag, Rc_Ha, xerr=BP_RPmag_err, yerr=Rc_Ha_err, fmt='none', ms=4, alpha=0.5, ecolor='lightgray', zorder=1)

        ax1.set_xlabel(r'$(G_{\mathrm{BP}}-G_{\mathrm{RP}})$ [mag]')
        BB_filt_label = r'R_\mathrm{c}'
        NB_filt_label = r'\mathrm{H}\alpha'
        ax1.set_ylabel(r'$(%s-%s)$ [mag]'%(BB_filt_label,NB_filt_label))

        ## Linear regression
        #self.model = LinearRegression(fit_intercept=True)
        # Linear regression (order 4)
        model = make_pipeline(PolynomialFeatures(degree=4), LinearRegression(fit_intercept=False))
        model.fit(X, Rc_Ha[valid_color_range*valid_stellar_type])
        Xfit = xfit[:, np.newaxis]
        yfit = model.predict(Xfit)
        
        line1, = ax1.plot(xfit, yfit, color='tab:red', ls='--', lw=1.0)

        # Compute the error of the linear regression
        y_predict = model.predict(X)
        y = Rc_Ha[valid_color_range*valid_stellar_type]
        x = abs(y - y_predict)**2.
        y_std = np.sqrt(np.mean(x))
        
        # Plot the polynomial coefficients and the standard error
        coefs = model['linearregression'].coef_
        label = r'$y = %+.3f %+.3f x %+.3f x^2 %+.3f x^3 %+.3f x^4$'%(coefs[0], coefs[1], coefs[2], coefs[3], coefs[4])
        leg2 = ax1.legend([line1],[label], loc='upper left', fontsize=11) 
        ax1.add_artist(leg1)

        # Save the polynomial coefficients and the linear regression error in a Pands Series
        joint_coeff_err = np.hstack((coefs, y_std))
        Rc_Ha_coef = pd.Series(data=joint_coeff_err, index=['a0', 'a1', 'a2', 'a3', 'a4','y_std'], name='Rc-Ha')

        # Overplot the PHANGS-Ha data CC relation
        if draw_fit :
            
            slopes = [np.random.normal(loc=coeffs[0][2], scale=coeffs[0][3]) for i in range(1000)]
            intercepts = [np.random.normal(loc=coeffs[0][0], scale=coeffs[0][1]) for i in range(1000)]
            lines = np.array([slope * xfit + intercept for slope, intercept in zip(slopes, intercepts)])
            lo, hi = np.percentile(lines, (5., 95.), axis=0)    # 95 interval confidence
            
            slope = coeffs[0][2]
            intercept = coeffs[0][0]
            line1, = ax1.plot(xfit, slope * xfit + intercept, color='k', ls='--', lw=0.8)
            ax1.fill_between(xfit, lo, hi, color='k', alpha=0.10)
            leg2 = ax1.legend([line1],[r'PHANGS-H$\alpha$'], loc='center right', frameon=False, fontsize=11) 
            ax1.add_artist(leg1)

        #ax1.text(0.98, 0.12, r'$%s = %s-\alpha-\beta\, (G_{\mathrm{BP}}-G_{\mathrm{RP}})$'%(BB_filt_label,NB_filt_label), color='tab:red', ha='right', transform=ax1.transAxes)
        #ax1.text(0.98, 0.04, r'$\alpha = %s,\,\beta = %s$' %(round(model.intercept_, 2),round(model.coef_[0], 2)), color='tab:red', ha='right', transform=ax1.transAxes)
        
        ax1.axvline(x=0.2, color='k', ls=':', lw=0.5)
        ax1.axvline(x=1.8, color='k', ls=':', lw=0.5) 
         
        plt.subplots_adjust(right=0.90)

        fig.savefig('GaiaSPSS_calib_Rc_Ha%s.png'%draw_fit_str)


        # Plot r-Ha657
        
        fig, ax2 = plt.subplots(1, 1, figsize=(7,7)) 

        sns.scatterplot(x='GBP-GRP', y='r-Ha657', data=self.df, hue='Stellar Type', ax=ax2, zorder=2)

        leg1 = ax2.legend(loc='lower right', fontsize=9)

        ax2.errorbar(BP_RPmag, r_Ha657, xerr=BP_RPmag_err, yerr=r_Ha657_err, fmt='none', ms=4, alpha=0.5, ecolor='lightgray', zorder=1)

        ax2.set_xlabel(r'$(G_{\mathrm{BP}}-G_{\mathrm{RP}})$ [mag]')
        BB_filt_label = 'r'
        NB_filt_label = r'\mathrm{H}\alpha 657'
        ax2.set_ylabel(r'$(%s-%s)$ [mag]'%(BB_filt_label,NB_filt_label))

        ## Linear regression
        #model = LinearRegression(fit_intercept=True)
        # Linear regression (order 4)
        model = make_pipeline(PolynomialFeatures(degree=4), LinearRegression(fit_intercept=False))
        model.fit(X, r_Ha657[valid_color_range*valid_stellar_type])
        Xfit = xfit[:, np.newaxis]
        yfit = model.predict(Xfit)
        
        line1, = ax2.plot(xfit, yfit, color='tab:red', ls='--', lw=1.0)

        # Compute the error of the linear regression
        y_predict = model.predict(X)
        y = r_Ha657[valid_color_range*valid_stellar_type]
        x = abs(y - y_predict)**2.
        y_std = np.sqrt(np.mean(x))
        
        # Plot the polynomial coefficients and the standard error
        coefs = model['linearregression'].coef_
        label = r'$y = %+.3f %+.3f x %+.3f x^2 %+.3f x^3 %+.3f x^4$'%(coefs[0], coefs[1], coefs[2], coefs[3], coefs[4])
        leg2 = ax2.legend([line1],[label], loc='upper left', fontsize=11) 
        ax2.add_artist(leg1)

        # Save the polynomial coefficients and the linear regression error in a Pands Series
        joint_coeff_err = np.hstack((coefs, y_std))
        r_Ha657_coef = pd.Series(data=joint_coeff_err, index=['a0', 'a1', 'a2', 'a3', 'a4','y_std'], name='r-Ha657')

        # Overplot the PHANGS-Ha data CC relation
        if draw_fit :
        
            slopes = [np.random.normal(loc=coeffs[1][2], scale=coeffs[1][3]) for i in range(1000)]
            intercepts = [np.random.normal(loc=coeffs[1][0], scale=coeffs[1][1]) for i in range(1000)]
            lines = np.array([slope * xfit + intercept for slope, intercept in zip(slopes, intercepts)])
            lo, hi = np.percentile(lines, (5., 95.), axis=0)    # 95 interval confidence
            
            slope = coeffs[1][2]
            intercept = coeffs[1][0]
            line1, = ax2.plot(xfit, slope * xfit + intercept, color='k', ls='--', lw=0.8)
            ax2.fill_between(xfit, lo, hi, color='k', alpha=0.10)
            leg2 = ax2.legend([line1],[r'PHANGS-H$\alpha$'], loc='center right', frameon=False, fontsize=11)
            ax2.add_artist(leg1)

        #ax2.text(0.98, 0.12, r'$%s = %s-\alpha-\beta\, (G_{\mathrm{BP}}-G_{\mathrm{RP}})$'%(BB_filt_label,NB_filt_label), color='tab:red', ha='right', transform=ax2.transAxes)
        #ax2.text(0.98, 0.04, r'$\alpha = %s,\,\beta = %s$' %(round(model.intercept_, 2),round(model.coef_[0], 2)), color='tab:red', ha='right', transform=ax2.transAxes)

        ax2.axvline(x=0.2, color='k', ls=':', lw=0.5)
        ax2.axvline(x=1.8, color='k', ls=':', lw=0.5)

        plt.subplots_adjust(right=0.90)
        
        fig.savefig('GaiaSPSS_r_Ha657%s.png'%draw_fit_str)


        # Plot r-Ha663
        
        fig, ax3 = plt.subplots(1, 1, figsize=(7,7)) 

        sns.scatterplot(x='GBP-GRP', y='r-Ha663', data=self.df, hue='Stellar Type', ax=ax3, zorder=2)

        leg1 = ax3.legend(loc='lower right', fontsize=9)

        ax3.errorbar(BP_RPmag, r_Ha663, xerr=BP_RPmag_err, yerr=r_Ha663_err, fmt='none', ms=4, alpha=0.5, ecolor='lightgray', zorder=1)

        ax3.set_xlabel(r'$(G_{\mathrm{BP}}-G_{\mathrm{RP}})$ [mag]')
        BB_filt_label = 'r'
        NB_filt_label = r'\mathrm{H}\alpha 663'
        ax3.set_ylabel(r'$(%s-%s)$ [mag]'%(BB_filt_label,NB_filt_label))

        ## Linear regression
        #model = LinearRegression(fit_intercept=True)
        # Linear regression (order 4)
        model = make_pipeline(PolynomialFeatures(degree=4), LinearRegression(fit_intercept=False))
        model.fit(X, r_Ha663[valid_color_range*valid_stellar_type])
        Xfit = xfit[:, np.newaxis]
        yfit = model.predict(Xfit)
        
        line1, = ax3.plot(xfit, yfit, color='tab:red', ls='--', lw=1.0)

        # Compute the error of the linear regression
        y_predict = model.predict(X)
        y = r_Ha663[valid_color_range*valid_stellar_type]
        x = abs(y - y_predict)**2.
        y_std = np.sqrt(np.mean(x))
        
        # Plot the polynomial coefficients and the standard error
        coefs = model['linearregression'].coef_
        label = r'$y = %+.3f %+.3f x %+.3f x^2 %+.3f x^3 %+.3f x^4$'%(coefs[0], coefs[1], coefs[2], coefs[3], coefs[4])
        leg2 = ax3.legend([line1],[label], loc='upper left', fontsize=11) 
        ax3.add_artist(leg1)

        # Save the polynomial coefficients and the linear regression error in a Pands Series
        joint_coeff_err = np.hstack((coefs, y_std))
        r_Ha663_coef = pd.Series(data=joint_coeff_err, index=['a0', 'a1', 'a2', 'a3', 'a4','y_std'], name='r-Ha663')

        # Overplot the PHANGS-Ha data CC relation
        if draw_fit :
        
            slopes = [np.random.normal(loc=coeffs[2][2], scale=coeffs[2][3]) for i in range(1000)]
            intercepts = [np.random.normal(loc=coeffs[2][0], scale=coeffs[2][1]) for i in range(1000)]
            lines = np.array([slope * xfit + intercept for slope, intercept in zip(slopes, intercepts)])
            lo, hi = np.percentile(lines, (5., 95.), axis=0)    # 95 interval confidence
            
            slope = coeffs[2][2]
            intercept = coeffs[2][0]
            line1, = ax3.plot(xfit, slope * xfit + intercept, color='k', ls='--', lw=0.8)
            ax3.fill_between(xfit, lo, hi, color='k', alpha=0.10)
            leg2 = ax3.legend([line1],[r'PHANGS-H$\alpha$'], loc='center right', frameon=False, fontsize=11) 
            ax3.add_artist(leg1)

        #ax3.text(0.98, 0.12, r'$%s = %s-\alpha-\beta\, (G_{\mathrm{BP}}-G_{\mathrm{RP}})$'%(BB_filt_label,NB_filt_label), color='tab:red', ha='right', transform=ax3.transAxes)
        #ax3.text(0.98, 0.04, r'$\alpha = %s,\,\beta = %s$' %(round(model.intercept_, 2),round(model.coef_[0], 2)), color='tab:red', ha='right', transform=ax3.transAxes)
        
        ax3.axvline(x=0.2, color='k', ls=':', lw=0.5)
        ax3.axvline(x=1.8, color='k', ls=':', lw=0.5)
        
        plt.subplots_adjust(right=0.90)

        fig.savefig('GaiaSPSS_r_Ha663%s.png'%draw_fit_str)

        # Create the DataFrame with the coefficients and linear regression error and save it in CSV
        coef_df = pd.concat([Rc_Ha_coef, r_Ha657_coef, r_Ha663_coef], axis=1)
        coef_df.to_csv(Path('.').joinpath('CC_relation.csv'), index_label='coefs')
        
    


class SpecPhotStdStar (object) :
    '''
    Class to store the ascii files with the fluxes of each SPSS from Gaia SPSS v2.
    '''
    
    def __init__ (self, specfile) :

        self.spec = ascii.read(specfile)


    def synthetic_magnitude (self, tr_file, filt) :
        '''
        Method to compute the synthetic magnitude for
        ESO844, ESO856, Ha657, Ha663 and LCOSITe3_r
        filters.
        '''
        
        # convert wavelength and flux arrays for the SPSS spectrum
        
        if 'NA' in self.spec['Flux'] :
            valid = self.spec['Flux'] != 'NA'
            wave = self.spec['Wave'][valid]
            flux = (self.spec['Flux'][valid]).astype('float32')
            err = (self.spec['Error'][valid]).astype('float32')
        else :
            wave = self.spec['Wave']
            flux = (self.spec['Flux']).astype('float32')
            err = (self.spec['Error']).astype('float32')

        wave = (wave * u.nm).to(u.angstrom).value
        flux = (flux * u.watt / u.m**2 / u.nm).to(u.erg / u.cm**2. / u.second / u.angstrom).value
        err = (err * u.watt / u.m**2 / u.nm).to(u.erg / u.cm**2. / u.second / u.angstrom).value

        # Load the filter trasmission

        wave_f, T_f = np.loadtxt(str(tr_file), dtype='float', unpack=True)
        
        # Correct LCO trasmission filters (they are given in percentage of transmission vs. nm)
        
        if (filt == 'Ha657') + (filt == 'Ha663') :
            wave_f = wave_f * 10.
            T_f = T_f / 100. 

        # Interpolating the standard star spectrum to match the transmission filter
        '''
        Check if the spectrophotometric standard star spectrum is upsampled compared to that of the filter transmission.
        specphot sampling > filter sampling --> intrpolate the filter 
        specphot sampling < filter sampling --> intrpolate the specphot spectrum (common case!!)
        '''
        if wave[1]-wave[0] < wave_f[1]-wave_f[0] :
            
            mask = (wave>=np.min(wave_f))&(wave<=np.max(wave_f))
            flux_f = flux[mask]
            err_f = err[mask]
            T_f = np.interp(wave[mask], wave_f, T_f)
            wave_arr = wave[mask].copy()
        
        else :
            
            flux_f = np.interp(wave_f, wave, flux)
            err_f = np.interp(wave_f, wave, err)
            wave_arr = wave_f.copy()
        
        # convolve standard star spectrum with filter transmission
        
        W_f = np.max (np.cumsum (T_f*wave_arr))
        #f0_f = np.max (np.cumsum (T_f*flux_f*wave_arr)) / W_f

        # Montecarlo to compute the flux error
        
        f0_arr = []
        
        for i in range(1000) :
            
            flux_f_norm = np.random.normal(loc=flux_f, scale=err_f)
            f0_arr.append(np.max (np.cumsum (T_f*flux_f_norm*wave_arr)) / W_f)
        
        f0_arr = np.array(f0_arr)

        f0_f = np.mean(f0_arr)
        f0_f_err = np.std(f0_arr)

        print('\nIntegraated flux in filter %s : %s +/- %s'%(filt,f0_f,f0_f_err))
                
        # Transformation from flux_lbd to flux_nu
        
        lbd_p = np.sqrt(W_f / np.max(np.cumsum(T_f/wave_arr)))
        print('\nPivot wavelength of filter %s : %s'%(filt,lbd_p))
        
        c = const.c.to('u.angstrom/u.second').value
        f0_nu = lbd_p**2./c*f0_f
        f0_nu_err = lbd_p**2./c*f0_f_err

        # synthetic magnitudes computation
        
        synth_mag = -2.5*np.log10(f0_nu) - 48.600
        synth_mag_err = 1.086 * f0_nu_err / f0_nu

        print('\nSynthetic magnitude of filter %s : %s +/- %s'%(filt,synth_mag,synth_mag_err))

        return synth_mag, synth_mag_err
        


if __name__ == "__main__":
    
    start_time = timeit.default_timer()

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--draw_fit', action='store_true', default=False, help='Overplot the fit from PHANGS-Ha data')
    
    args = parser.parse_args()

    path_to_tr_filt = Path('/Users/arazza/DAS/Observations/NB_DR3_pipeline/filter_transmission/')
        
    tr_d = {
        'Rc' : path_to_tr_filt.joinpath('LaSilla_WFI.ESO844.dat.txt'),
        'Ha' : path_to_tr_filt.joinpath('LaSilla_WFI.ESO856.dat.txt'),
        'r'  : path_to_tr_filt.joinpath('LCO_SITe3.r.dat.txt'),
        'Ha657': path_to_tr_filt.joinpath('Ha657BP6.Data.txt'),
        'Ha663': path_to_tr_filt.joinpath('Ha663BP6.Data.txt'),
    }
    
    gaia_spss = GaiaSPSS('SPSSpublicV2.csv') 
    
    # loop over the filters
    
    for filt in ['Rc', 'Ha', 'r', 'Ha657', 'Ha663'] :

        #if filt != 'Rc' :
        #    continue
        
        sm = []
        sm_err = []

        for specfile in gaia_spss.df['FILENAME'] :

            #if specfile != 'V2.SPSS010.ascii' :
            #    continue
                
            spss = SpecPhotStdStar(Path('SPSSpublicV2').joinpath(specfile))

            mag, mag_err = spss.synthetic_magnitude(tr_d[filt], filt)

            sm.append(mag)

            sm_err.append(mag_err)

        synth_mag = pd.Series(data=sm, index=gaia_spss.df.index, name='%s_synthetic'%filt)

        synth_mag_err = pd.Series(data=sm_err, index=gaia_spss.df.index, name='%s_syntERR'%filt)
            
        gaia_spss.df = pd.concat([gaia_spss.df, synth_mag, synth_mag_err], axis=1)
    
    gaia_spss.update_data_frame()

    gaia_spss.group_stellar_type()

    gaia_spss.sort_by_stellar_type()

    gaia_spss.plot_cc(args.draw_fit)



    elapsed = timeit.default_timer() - start_time
    m, s = divmod(elapsed, 60)
    h, m = divmod(m, 60)
    print ('Process terminated in %d:%02d:%02d'%(h, m, s))


