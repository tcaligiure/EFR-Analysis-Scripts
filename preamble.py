import os
import h5py
import csv
import sys
import warnings
from tqdm import tqdm, trange
import time
from IPython.display import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import matplotlib.transforms as transforms
import pandas as pd
import re
import imageio

import ipywidgets as widgets
from ipywidgets import interact, interactive, fixed, interact_manual

from pathlib import Path as p

pylab.style.use('classic')
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (18, 8),
         'axes.labelsize':22,
         'axes.titlesize':22,
         'xtick.labelsize':18,
         'ytick.labelsize':18,
         'axes.formatter.useoffset':'False',
          'lines.markersize' : 10,
          'xtick.major.size': 10,
          'xtick.minor.size': 4,
         'xtick.major.width': 1.5,
          'xtick.minor.width': 1,
         'xtick.direction': 'out',
          'ytick.major.size': 10,
          'ytick.minor.size': 4,
         'ytick.major.width': 1.5,
          'ytick.minor.width': 1,
         'ytick.direction': 'out',
         'axes.titlepad':20}
pylab.rcParams.update(params)
pylab.mpl.rc('figure', facecolor = 'white')

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import ScalarFormatter, MultipleLocator, FormatStrFormatter, AutoMinorLocator
from matplotlib import cm
from matplotlib.colors import LogNorm
from matplotlib.colors import Normalize 
import matplotlib.colors as colors

y_formatter = ScalarFormatter(useOffset=False)
x_formatter = ScalarFormatter(useOffset=False)

from scipy.signal import find_peaks
from scipy.signal import peak_widths
from scipy.signal import peak_prominences
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

from lmfit.models import PolynomialModel, BreitWignerModel, QuadraticModel
from scipy.optimize import curve_fit
import sympy as sym

# define some useful functions 
def find_nearest(a, a0):
    "Element in nd array `a` closest to the scalar value `a0`"
    idx = np.abs(a - a0).argmin()
    return a.flat[idx]

def find_nearest_pos(a, a0):
    "position in nd array `a` closest to the scalar value `a0`"
    idx = np.abs(a - a0).argmin()
    return idx

def printstruct(name,obj):
    print(name)
    print(list(obj.attrs.items()))
    
def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

def tryint(s):
    try:
        return int(s)
    except:
        return s
    
def repeat(func, n, x):
    for i in range(n):
        x = func(x)
    return x

def sort(lst):
    tm010 = [] 
    for i in range(len(lst)-1):
        if direction == 'up':
            if lst[i][1] > lst[i+1][1]:
                tm010.append(lst[i])
        elif direction == 'down':
            if lst[i][1] < lst[i+1][1] and lst[i][1] != 0:
                tm010.append(lst[i])
    return tm010

def peakfinder(data, phi, fstart, bw, p_width, prom, Wlen, rel, Height):
    '''finds central freq and 3db peak widths at some phi position given some parameters. Then calculate Q-estimate
    [0]:f0, [1]: 3db widths [2]:Q-estimate. Select True if using zoom_data, False otherwise'''
    # Find peaks
    peaks = find_peaks(
        data[phi], prominence=prom, height=Height, width=p_width, rel_height=rel, wlen=Wlen
    )
    prominences, left_bases, right_bases = peak_prominences(data[phi], peaks[0])

    # Create constant offset as a replacement for prominences
    offset = np.full_like(prominences, 3)

    # Calculate widths at x[peaks] - offset * rel_heigh
    widths, h_eval, left_ips, right_ips = peak_widths(
        data[phi], peaks[0], rel_height=1, prominence_data=(offset, left_bases, right_bases))
        
    f0 = peaks[0]*bw + fstart #central f 
    w3db = widths*bw  #3db width
    Q = f0/w3db #Q-estimate
        
    return [f0, w3db, Q, phi, peaks[1]["peak_heights"]]

def Q_fit(magdata,freq_list,old_f0,f0_list,meta_list,phi_list,phi,nlw,plot): 
    Qfit = [] 
    
    phi_pos = int(find_nearest_pos(np.asarray(phi_list),phi)) #this finds the neareast poisiton in the philist to the value phi 
    phi_val = int(find_nearest(np.asarray(phi_list),phi)) #this finds the nearest value in the philist to the value phi
    phi_meta = int(find_nearest_pos(np.asarray(meta_list),phi)) #this finds the nearest position in the metalist

    for mode in range(len(f0_list[phi_pos])):
        
        modepos = find_nearest_pos(old_f0[phi_meta],f0_list[phi_pos])
        minpos = find_nearest_pos(freq_list,f0_list[phi_pos][mode]-nlw*w3db[phi_meta][modepos])
        maxpos = find_nearest_pos(freq_list,f0_list[phi_pos][mode]+nlw*w3db[phi_meta][modepos])

        #input these positions into mag_data
        power_data = magdata**2
        reduced_data =power_data[minpos:maxpos,phi_val]  #list of reduced data for each mode
        modef = freq_list[minpos:maxpos] #list of mode frequencies in GHz for each mode

        resonator = BreitWignerModel() #fano-model
        background = QuadraticModel() 
        model = resonator + background #composite model

        #start the fit 
        amp_guess = 10 ** np.asarray(peak_height_list_pow, dtype=object)[phi_meta][modepos] # guess based on peak heights
        pars = model.right.guess(reduced_data, x=modef) + model.left.guess(reduced_data, x=modef) #guess params
        pars['amplitude'].set(value=amp_guess, vary=True, expr='') # set inital val 
        pars['sigma'].set(value=w3db[phi_meta][modepos], vary=True, expr='')
        pars['center'].set(value=f0_list[phi_pos][0], vary=True, expr='')
        pars['a'].set(value=-0.001, vary=True, expr='')
        pars['q'].set(value=-0.01, vary=True, expr='')
        fit_pars = model.fit(reduced_data, pars, x=modef) #fit using guess params
        final_fit = fit_pars.best_fit 
        Q=fit_pars.params['center'].value/fit_pars.params['sigma'].value
        Qfit.append(round(Q))

        if plot == True:
            #plot mode and fit
            fig,ax=plt.subplots(1,figsize=(13,8))
            ax.plot(modef, 10*np.log10(np.abs(reduced_data)),label=r'Data')
            ax.plot(modef,10*np.log10(np.abs(final_fit)),label="$Q_L$="+str(round(Q)) + '\n' + "$f_0$="+str(round(fit_pars.values['center'],3))
                    + ' GHz' + '\n' + '$q$='+str(round(fit_pars.values['q'],2)))
            plt.ylabel('|S21| (dB)',fontsize=30)
            plt.xlabel('Frequency (GHz)',fontsize=30)
            plt.title("Phi:"+str(phi_val),fontsize=30)
            ax.legend()
            ax.xaxis.set_major_formatter(x_formatter) 
            ax.tick_params(axis='both', which='major', labelsize=20)
            fig.tight_layout()
    return Qfit, 10*np.log10(final_fit.max()) #peak trans in dB

def peak_finder_slider(data, params, freq_list, phi, xmin, xmax, s_param):
    '''find the peaks based on some parameters and then slide between tuning steps'''
    # Find peaks
    minpos = find_nearest_pos(freq_list,xmin)
    maxpos = find_nearest_pos(freq_list,xmax)
    data = data[phi][minpos:maxpos]
    freq_list = freq_list[minpos:maxpos]
    # peaks = find_peaks(data, width=peak_width, prominence=prom, rel_height=rel, wlen=window, height=Height)
    peaks = find_peaks(data, width=params[0], prominence=params[1], rel_height=params[3], wlen=params[2], height=params[4])
    prominences, left_bases, right_bases = peak_prominences(data, peaks[0])

    # Create constant offset as a replacement for prominences
    offset = np.full_like(prominences,3)

    # Calculate widths at x[peaks] - offset * rel_height
    widths, h_eval, left_ips, right_ips = peak_widths(data, peaks[0], rel_height=1,prominence_data=(offset, left_bases, right_bases))

    fig,ax = plt.subplots(1,figsize=(13,8))

    plt.plot(freq_list, data)
    plt.plot(freq_list[peaks[0]], data[peaks[0]], "x", color = "C2",markersize=10)
    plt.xlabel(r'Frequency (GHz)')
    plt.xlim(xmin,xmax)
    if s_param == 'S21':
        plt.ylabel('$|S21|$ (dB)')
    if s_param == 'S11':
        plt.ylabel('$|S11|$ (dB)')
    if s_param == 'S22':
        plt.ylabel('$|S22|$ (dB)')
    ax.xaxis.set_major_formatter(x_formatter)  
    plt.show()
    f0 = freq_list[peaks[0]]
    w3db = (375000.0*widths) #vna_rbw = 375000.0
    Q = f0*1e9/w3db
    
    return 'f_c=', f0,'Q_3db=', Q

def peak_finder_sliderNeg(data, freq_list, phi, xmin, xmax, s_param):
    '''find the peaks based on some parameters and then slide between tuning steps'''
    # Find peaks
    minpos = find_nearest_pos(freq_list,xmin)
    maxpos = find_nearest_pos(freq_list,xmax)
    data = -1 * data[phi][minpos:maxpos]
    freq_list = freq_list[minpos:maxpos]
    peaks = find_peaks(data, width=peak_width, prominence=prom, rel_height=rel, wlen=window, height=Height)
    prominences, left_bases, right_bases = peak_prominences(data, peaks[0])

    # Create constant offset as a replacement for prominences
    offset = np.full_like(prominences,3)

    # Calculate widths at x[peaks] - offset * rel_height
    widths, h_eval, left_ips, right_ips = peak_widths(data, peaks[0], rel_height=1,prominence_data=(offset, left_bases, right_bases))

    fig,ax = plt.subplots(1,figsize=(13,8))

    plt.plot(freq_list, -1 * data)
    plt.plot(freq_list[peaks[0]], -1 * data[peaks[0]], "x", color = "C2",markersize=10)
    plt.xlabel(r'Frequency (GHz)')
    plt.xlim(xmin,xmax)
    if s_param == 'S21':
        plt.ylabel('$|S21|$ (dB)')
    if s_param == 'S11':
        plt.ylabel('$|S11|$ (dB)')
    if s_param == 'S22':
        plt.ylabel('$|S22|$ (dB)')
    #plt.ylim(data[])
    ax.xaxis.set_major_formatter(x_formatter)   
    plt.show()
    f0 = freq_list[peaks[0]]
    w3db = (vna_rbw*widths) #vna_rbw = 375000.0
    Q = f0*1e9/w3db
    
    return 'f_c=', f0,'Q_3db=', Q

def peak_finder_phi(data, params, freq_list, phi, xmin, xmax, plot, vna_rbw):
    '''find the peaks based on some parameters and then slide between tuning steps'''
    # Find peaks
    minpos = find_nearest_pos(freq_list,xmin)
    maxpos = find_nearest_pos(freq_list,xmax)
    data = data[phi][minpos:maxpos]
    freq_list = freq_list[minpos:maxpos]
    peaks = find_peaks(data, width=params[0], prominence=params[1], rel_height=params[3], wlen=params[2], height=params[4])
    prominences, left_bases, right_bases = peak_prominences(data, peaks[0])

    # Create constant offset as a replacement for prominences
    offset = np.full_like(prominences,3)

    # Calculate widths at x[peaks] - offset * rel_height
    widths, h_eval, left_ips, right_ips = peak_widths(data, peaks[0], rel_height=1,prominence_data=(offset, left_bases, right_bases))
    if plot:
        fig,ax = plt.subplots(1,figsize=(13,8))

        plt.plot(freq_list, data)
        plt.plot(freq_list[peaks[0]], data[peaks[0]], "x", color = "C2",markersize=10)
        plt.xlabel(r'Frequency (GHz)')
        plt.ylabel('$|S_{21}|$ (dB)')
        plt.xlim(xmin,xmax)
        ax.xaxis.set_major_formatter(x_formatter)   
        plt.show()
    f0 = freq_list[peaks[0]]
    w3db = (vna_rbw*widths) #vna_rbw = 375000.0
    Q = f0*1e9/w3db
    return [f0, w3db, Q, phi, peaks[1]["peak_heights"]]

def power_beta(on,off):
    return (1-np.sqrt(10**(on/10)/10**(off/10)))/(1+np.sqrt(10**(on/10)/10**(off/10)))

def hdf5_parser(fname):
    f = h5py.File(fname,'r')
    freq = f['Freq'][:]/1e9 #in GHz
    total_data = f['VNA']
    mag_data = np.sqrt(total_data[:,0]**2 + total_data[:,1]**2).T
    mag_data_db = 20*np.log10(mag_data)
    phase_data = np.angle(total_data[:,0] + 1j*total_data[:,1])
    file_dict = {}
    for key,val in f['Freq'].attrs.items():
        file_dict[key] = val
    return freq, mag_data, mag_data_db, phase_data, file_dict

def ato_hdf5_parser(fname):
    '''If time==True then add time stamps to dict Same with temp and capacitance'''
    
    f = h5py.File(fname,'r')
    freq = f['Freq'][:]/1e9 #in GHz
    ato_positions = sorted(f.keys(), key=lambda item: (int(item.partition(' ')[0]) if item[0].isdigit() else float('inf'), item))[:-1]
    phi = np.asarray([int(i) for i in ato_positions])
    total_data = np.array([f[ref][:] for ref in ato_positions])
    
    mag_data = np.sqrt(total_data[:,:,0]**2 + total_data[:,:,1]**2).T
    mag_data_db = 20*np.log10(mag_data)
    
    phase_data = np.angle(total_data[:,:,0] + 1j*total_data[:,:,1]).T
    
    file_dict = {}
    for k in f['Freq'].attrs.keys():
        file_dict[k] = f['Freq'].attrs[k]
            
    return mag_data, mag_data_db, phase_data, freq, phi, file_dict

def Q_fit(magdata,f_c,f0_list,reduced_f0_list,reduced_phi_list,nlw,plot,save): 
    Q_fit_list = np.array([])
    Q_3db_guess = np.array([])
    peak_height_list = np.array([])
    nearest_pos = find_nearest_pos(np.concatenate(f0_list),f_c)
    idx = np.where(reduced_f0_list==f_c)[0][0]
    phi_pos = int(reduced_phi_list[idx])
    w3db_guess = np.concatenate(w3dblist)[nearest_pos]
    
    minpos = find_nearest_pos(freq,f_c-nlw*w3db_guess/1e9)
    maxpos = find_nearest_pos(freq,f_c+nlw*w3db_guess/1e9)
    #input these positions into mag_data
    reduced_data = magdata[minpos:maxpos,phi_pos]**2
    modef = freq[minpos:maxpos] #list of mode frequencies in GHz for each mode
    resonator = BreitWignerModel() #fano-model
    background = QuadraticModel() 
    model = resonator + background #composite model

    #start the fit 
    amp_guess = np.concatenate(peak_height_list_pow)[nearest_pos] # guess based on peak heights
    pars = model.right.guess(reduced_data, x=modef) + model.left.guess(reduced_data, x=modef) #guess params
    pars['amplitude'].set(value=amp_guess, vary=True, expr='') # set inital val 
    pars['sigma'].set(value=w3db_guess/1e9, vary=True, expr='')
    pars['center'].set(value=f_c, vary=True, expr='')
    pars['a'].set(value=-0.1, vary=True, expr='')
    pars['q'].set(value=-0.01, vary=True, expr='')
    fit_pars = model.fit(reduced_data, pars, x=modef) #fit using guess params
    final_fit = fit_pars.best_fit 
    Q=fit_pars.params['center'].value/fit_pars.params['sigma'].value
    Q_fit_list = np.append(Q_fit_list,Q)
    Q_3db_guess = np.append(Q_3db_guess,np.concatenate(Qlist)[nearest_pos])
    peak_height_db = 10*np.log10(final_fit.max())
    peak_height_list = np.append(peak_height_list,peak_height_db)
    f_c_fit = round(fit_pars.values['center'],3)
    q = round(fit_pars.values['q'],2)

    if plot == True:
        #plot mode and fit
        fig,ax=plt.subplots(1,figsize=(13,8))
        ax.plot(modef, 10*np.log10(np.abs(reduced_data)),label=r'Data')
        ax.plot(modef,10*np.log10(np.abs(final_fit)),label=f"$Q_L$={round(Q)} \n $f_0$={f_c_fit} GHz \n $q$={q}")
                
               # "$Q_L$="+str(round(Q)) + '\n' + "$f_0$="+str(round(fit_pars.values['center'],3))
              #  + ' GHz' + '\n' + '$q$='+str(round(fit_pars.values['q'],2))
        plt.ylabel(r'$|S_{21}|$ (dB)',fontsize=25)
        plt.xlabel('Frequency (GHz)',fontsize=25)
        ax.legend()
        ax.xaxis.set_major_formatter(x_formatter) 
        ax.tick_params(axis='both', which='major', labelsize=20)
        fig.tight_layout()
        if save:
            save_dir = p.cwd()/'Q_fits'
            p(save_dir).mkdir(parents=True, exist_ok=True)
            pp = PdfPages(save_dir/("TM_010 phi=%s.pdf"%phi_pos))
            plt.savefig(pp, format='pdf', dpi=600)
            pp.close()
            plt.close()           
    return Q_fit_list, Q_3db_guess, peak_height_list #peak trans in dB

def Q_fit_all(magdata,freq_list,phi,f0_list,w3db,peak_height,nlw,plot): 
    Q_list = np.array([])
    peak_height_list = np.array([])
    for idx,f_c in enumerate(f0_list):
        minpos = find_nearest_pos(freq_list,f_c-nlw*w3db[idx]/1e9)
        maxpos = find_nearest_pos(freq_list,f_c+nlw*w3db[idx]/1e9)
        #input these positions into mag_data
        power_data = magdata[phi]**2
        reduced_data = power_data[minpos:maxpos]  #list of reduced data for each mode
        modef = freq_list[minpos:maxpos] #list of mode frequencies in GHz for each mode
        resonator = BreitWignerModel() #fano-model
        background = QuadraticModel() 
        model = resonator + background #composite model

        #start the fit 
        amp_guess = 10 ** np.asarray(peak_height[idx], dtype=object) # guess based on peak heights
        pars = model.right.guess(reduced_data, x=modef) + model.left.guess(reduced_data, x=modef) #guess params
        pars['amplitude'].set(value=amp_guess, vary=True, expr='') # set inital val 
        pars['sigma'].set(value=w3db[idx]/1e9, vary=True, expr='')
        pars['center'].set(value=f_c, vary=True, expr='')
        pars['a'].set(value=-0.1, vary=True, expr='')
        pars['q'].set(value=-0.01, vary=True, expr='')
        fit_pars = model.fit(reduced_data, pars, x=modef) #fit using guess params
        final_fit = fit_pars.best_fit 
        Q=fit_pars.params['center'].value/fit_pars.params['sigma'].value
        Q_list = np.append(Q_list,Q)
        peak_height_db = 10*np.log10(final_fit.max())
        peak_height_list = np.append(peak_height_list,peak_height_db)
        
        if plot == True:
            #plot mode and fit
            fig,ax=plt.subplots(1,figsize=(13,8))
            ax.plot(modef, 10*np.log10(np.abs(reduced_data)),label=r'Data')
            ax.plot(modef,10*np.log10(np.abs(final_fit)),label="$Q_L$="+str(round(Q)) + '\n' + "$f_0$="+str(round(fit_pars.values['center'],3))
                    + ' GHz' + '\n' + '$q$='+str(round(fit_pars.values['q'],2)))
            plt.ylabel('|S21| (dB)',fontsize=30)
            plt.xlabel('Frequency (GHz)',fontsize=30)
            ax.legend()
            ax.xaxis.set_major_formatter(x_formatter) 
            ax.tick_params(axis='both', which='major', labelsize=20)
            fig.tight_layout()
    return Q_list, peak_height_list#peak trans in dB
