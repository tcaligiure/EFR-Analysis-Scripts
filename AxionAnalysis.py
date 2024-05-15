import preamble as pre
import os
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

def data_processing(f):
    # Sorting data
    step_size = f['Freq'].attrs.__getitem__('ato_step')
    step_no = int(((len(sorted(f.keys())) - 1)/3)-1)
    fit_range = int(step_no - 1)
    nums = np.arange(0, step_no, 1)
    numbers = np.arange(0, (step_no*step_size)+step_size, step_size)
    
    total_data21 = np.array([f[(str(ref) + '_s21')][:] for ref in numbers])
    total_data11 = np.array([f[(str(ref) + '_s11')][:] for ref in numbers])
    total_data22 = np.array([f[(str(ref) + '_s22')][:] for ref in numbers])
    
    # Parameters
    vna_rbw = f['Freq'].attrs.__getitem__('vna_rbw')
    freq = f['Freq'][:]/1e9 #in GHz
    ato_positions = nums
    phi = np.asarray([int(i) for i in ato_positions])
    rad = 2 * (1.92) * np.sin(phi / 27)
    
    mag_data21 = np.sqrt(total_data21[:,:,0]**2 + total_data21[:,:,1]**2).T
    mag_data_db21 = 20*np.log10(mag_data21)
    mag_data11 = np.sqrt(total_data11[:,:,0]**2 + total_data11[:,:,1]**2).T
    mag_data_db11 = 20*np.log10(mag_data11)
    mag_data22 = np.sqrt(total_data22[:,:,0]**2 + total_data22[:,:,1]**2).T
    mag_data_db22 = 20*np.log10(mag_data22)
    
    return mag_data21, mag_data11, mag_data22, mag_data_db21, mag_data_db11, mag_data_db22, freq, phi, rad

def mode_map(mag_data_db21, freq, rad, exp_title):
    fig, ax = plt.subplots(figsize=(12,8))
    v_min = -50
    v_max = -5
    colour = plt.imshow(mag_data_db21, extent=[rad[0], rad[-1],freq[0],freq[-1]],aspect='auto',cmap=plt.cm.get_cmap('viridis'), 
                        origin='lower', norm=Normalize(vmin=v_min, vmax=v_max))
    ax.set_xlabel(r'Radial Pos (in)')
    ax.set_ylabel(r'Frequency (GHz)')

    cb=fig.colorbar(colour, ax=ax, pad=0.05)
    cb.set_label('$|S_{21}|$ (dB)')

    # plt.title(exp_name)
    plt.tight_layout()

    # pp = PdfPages(exp_name+".pdf")
    # plt.savefig(pp, format='pdf', dpi=600)
    # pp.close()
    plt.title(exp_title)
    plt.savefig('mode_map.PNG', dpi=300,facecolor='w')
    
    plt.show()
    
def S21_slider(mag_data_db21, freq, phi):
    # Parameters for slider function in preamble
    peak_width = [0,20]
    prom = 10
    window =300
    rel = 0.01
    Height = -20
    
    params = np.array([peak_width, prom, window, rel, Height], dtype = object)

    interact(lambda phi: pre.peak_finder_slider(mag_data_db21.T,params,freq,phi,np.min(freq),np.max(freq),s_param='S21'),phi=(0,phi[-1],1))
    
def S21_plot(mag_data_db21, freq):
    # Plotting first step of S21
    fig, ax = plt.subplots(figsize=(12,8))
    
    ax.scatter(freq, mag_data_db21.T[0])
    ax.set_xlabel('Freq (GHz)')
    ax.set_ylabel('$|S21|$ (dB)')
    plt.title('$|S21|$ vs Frequency')
    plt.savefig('s21_vFreq.PNG', dpi=300,facecolor='w')
    plt.show()
    
def Loaded_Q(mag_data_db21, freq, phi, f, exp_title):
    # Same parameters as earlier
    peak_width = [0,20]
    prom = 10
    window =300
    rel = 0.01
    Height = -20
    
    params = np.array([peak_width, prom, window, rel, Height], dtype = object)
    
    f0list,w3dblist,Qlist,philist,peak_height_list,peak_height_list_pow = [],[],[],[],[],[]
    
    f1points = pre.find_nearest_pos(freq,np.min(freq))
    f2points = pre.find_nearest_pos(freq,np.max(freq)) + 1
    reduced_f = freq[f1points:f2points]

    for i in range(phi[-1]-0):
        if pre.peak_finder_phi(mag_data_db21[f1points:f2points,0:phi[-1]].T,params,
                               reduced_f,i,np.min(freq),np.max(freq),0,f['Freq'].attrs.__getitem__('vna_rbw'))[0].size != 0:
            f0, w3db, Q, pos, peak_height_db = pre.peak_finder_phi(mag_data_db21[f1points:f2points,0:phi[-1]].T,params,reduced_f,i,
                                                                   np.min(freq),np.max(freq),0, f['Freq'].attrs.__getitem__('vna_rbw'))
            f0list.append(f0)
            w3dblist.append(w3db)
            Qlist.append(Q) 
            philist.append(pos+0)
            peak_height_list.append(peak_height_db) 
            peak_height_list_pow.append(10**(peak_height_db/20)) 
    TM010_Q = np.array([i[0] for i in Qlist])
    TM010_f0 = np.array([i[0] for i in f0list])
    
    fig,ax = plt.subplots(1,figsize=(13,8))

    plt.scatter(TM010_f0,TM010_Q)
    plt.title(f'{exp_title} Loaded Q')
    plt.ylabel('$Q_L$')
    plt.xlabel('Frequency (GHz)')
    plt.savefig('LoadedQ_zoom.PNG', dpi=300,facecolor='w')
    
    return TM010_Q, TM010_f0

def sT0_step(mag_data_db11, mag_data_db22, index11, index22, step, freq, mode):
    S11_index = index11[step][mode]
    S22_index = index22[step][mode]

    S11_index = S11_index.astype(np.int64)
    S22_index = S22_index.astype(np.int64)
    
    peak11_val = mag_data_db11.T[step][S11_index]
    peak22_val = mag_data_db22.T[step][S22_index]
    
    s110_i = np.polyfit(freq, mag_data_db11, deg = 0)
    s220_i = np.polyfit(freq, mag_data_db22, deg = 0)
    s110_i = s110_i.T
    s220_i = s220_i.T
    
    dip_mag11 = np.abs(peak11_val) - s110_i[step][0]
    dip_mag22 = np.abs(peak22_val) - s220_i[step][0]
    
    s110 = 10 ** ((-1 * dip_mag11) / 20)
    s220 = 10 ** ((-1 * dip_mag22) / 20)
    
    return s110 + s220
    
def unloaded_Q(mag_data_db11, mag_data_db22, phi, freq, TM010_Q, TM010_f0, mode, exp_title):
    fit_range = len(phi) - 1
    freq_list = freq
    
    s110_i = np.polyfit(freq, mag_data_db11, deg = 0)
    s220_i = np.polyfit(freq, mag_data_db22, deg = 0)

    inds11 = allPeaks(mag_data_db11.T, freq, phi)
    inds22 = allPeaks(mag_data_db22.T, freq, phi)
    
    sT0 = []
    f_list = []
    for phi in phi:
        try:
            sT0.append(sT0_step(mag_data_db11, mag_data_db22, inds11, inds22, phi, freq, mode))
            f_list.append(freq[inds11[phi][mode]])
        except IndexError:
            sT0.append(np.nan)
            f_list.append(np.nan)

    unloaded_Q = (2 * TM010_Q) / np.array(sT0[0:-1]) # Fix the problem w loaded Q yielding len of phi - 1
    
    fig,ax = plt.subplots(1,figsize=(13,8))

    plt.scatter(f_list[0:-1], unloaded_Q)
    plt.ylabel('$Q_u$')
    plt.xlabel('Frequency (GHz)')
    plt.title(f'{exp_title}, $Q_u$ for Mode {mode}')

    plt.show()
    
    return unloaded_Q, f_list

def TM010_Qu(mag_data_db11, mag_data_db22, phi, freq, TM010_Q, TM010_f0, exp_title):
    fit_range = len(phi) - 1
    
    s110_i = np.polyfit(freq, mag_data_db11, deg = 0)
    s220_i = np.polyfit(freq, mag_data_db22, deg = 0)

    data11 = np.abs(mag_data_db11.T)
    data22 = np.abs(mag_data_db22.T)
    freq_list = freq
    
    index11 = indFinder_11(mag_data_db11, phi, 0)
    index22 = indFinder_22(mag_data_db22, phi, 0)
        
    TM010_S11_index = np.empty(fit_range)
    for i in range(fit_range):
        TM010_S11_index[i] = index11[i][0]

    TM010_S22_index = np.empty(fit_range)
    for i in range(fit_range):
        TM010_S22_index[i] = index22[i][0]

    TM010_S11_index = TM010_S11_index.astype(np.int64)
    TM010_S22_index = TM010_S22_index.astype(np.int64)

    peak11_vals = np.empty(fit_range)
    peak22_vals = np.empty(fit_range)

    for i in range(fit_range):
        peak11_vals[i] = mag_data_db11.T[i][TM010_S11_index[i]]
        peak22_vals[i] = mag_data_db22.T[i][TM010_S22_index[i]]

    s110_i = s110_i.T
    s220_i = s220_i.T

    dip_mag11 = np.empty(fit_range)
    dip_mag22 = np.empty(fit_range)

    for i in range(fit_range):
        dip_mag11[i] = np.abs(peak11_vals[i]) + s110_i[i][0]
        dip_mag22[i] = np.abs(peak22_vals[i]) + s220_i[i][0] 

    s110 = np.empty(fit_range)
    s220 = np.empty(fit_range)

    for i in range(fit_range):
        s110[i] = 10 ** ((-1 * dip_mag11[i]) / 20)
        s220[i] = 10 ** ((-1 * dip_mag22[i]) / 20)

    sT0 = s110 + s220

    Qu = (2 * TM010_Q) / sT0

    fig,ax = plt.subplots(1,figsize=(13,8))

    plt.scatter(TM010_f0,Qu)
    plt.ylabel('$Q_u$')
    plt.xlabel('Frequency (GHz)')
    plt.title(f'{exp_title} TM010 $Q_u$')
    plt.savefig('TM010_Qu.PNG', dpi=300,facecolor='w')

    plt.show()
    
    return Qu, s110, s220, sT0    
    
def betas(TM010_Qu, TM010_f0, phi, s110, s220, sT0, exp_title):
    fit_range = len(phi) - 1
    
    beta1_num = np.empty(fit_range)
    beta2_num = np.empty(fit_range)
    for i in range(fit_range):
        beta1_num[i] = 1 - s110[i]
        beta2_num[i] = 1 - s220[i]
        
    beta1 = np.empty(fit_range)
    beta2 = np.empty(fit_range)
    for i in range(fit_range):
        beta1 = beta1_num / sT0
        beta2 = beta2_num / sT0
        
    fig,ax = plt.subplots(1,figsize=(13,8))

    plt.scatter(TM010_f0, beta1, c = 'mediumspringgreen', label = '\u03B2$_{1}$')
    plt.scatter(TM010_f0, beta2, c = 'darkorchid', label = '\u03B2$_{2}$')
    plt.title(f'{exp_title} TM010 Beta Parameters')
    plt.ylabel('Beta Values')
    plt.xlabel('Frequency (GHz)')
    plt.legend(loc = 'best')
    plt.savefig('betaParams.PNG', dpi=300,facecolor='w')
    
    return beta1, beta2

def Q_params(TM010_Qu, TM010_f0, phi, beta1, beta2, exp_title):
    fit_range = len(phi) - 1
    Q1 = np.empty(fit_range)
    Q2 = np.empty(fit_range)

    for i in range(fit_range):
        Q1[i] = TM010_Qu[i] / beta1[i]
        Q2[i] = TM010_Qu[i] / beta2[i]
        
    fig,ax = plt.subplots(1,figsize=(13,8))   
        
    plt.scatter(TM010_f0, Q1, c = 'mediumspringgreen', label = '$Q_{1}$')
    plt.scatter(TM010_f0, Q2, c = 'darkorchid', label = '$Q_{2}$')
    plt.title(f'{exp_title} TM010 Q Parameters')
    plt.ylabel('Q Values')
    plt.xlabel('Frequency (GHz)')
    plt.legend(loc = 'best')
    plt.savefig('QParams.PNG', dpi=300,facecolor='w')
    
    return Q1, Q2

def Q_check(TM010_Q, TM010_f0, Qu, phi, Q1, Q2, exp_title):
    fit_range = len(phi) - 1
    Qarg = np.empty(fit_range)

    for i in range(fit_range):
        Qarg[i] = (1/Q1[i]) + (1/Q2[i]) + (1/Qu[i])
        
    Calc_Q = np.empty(fit_range)

    for i in range(fit_range):
        Calc_Q[i] = 1/Qarg[i]
        
    fig,ax = plt.subplots(1,figsize=(13,8))

    plt.scatter(TM010_f0, TM010_Q, label = 'Measured Q', c = 'cyan', marker = 's', s = 35)
    plt.scatter(TM010_f0, Calc_Q, label = 'Calculated Q', c = 'red', marker = 'x')
    plt.title(f'{exp_title} Calculated vs Measured Loaded $Q$')
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('$Q_{L}$')
    plt.legend(loc = 'best')

    plt.savefig('Qcalc_vs_Qmeas.PNG', dpi=300,facecolor='w')
    
def S11_plot(mag_data_db11, freq):
    fig, ax = plt.subplots(1, 1, figsize = (12, 4), dpi = 300)

    plt.scatter(freq, mag_data_db11.T[0])
    plt.xlabel('Freq (GHz)')
    plt.ylabel('$|S11|$ (dB)')
    plt.savefig('s11_vFreq.PNG', dpi=300,facecolor='w')
    plt.show()
    
def S11_slider(mag_data_db11, freq, phi):
    peak_width = [0,20]
    prom = 10
    window =300
    rel = 0.01
    Height = -20
    
    params = np.array([peak_width, prom, window, rel, Height], dtype = object)

    interact(lambda phi: pre.peak_finder_slider(mag_data_db11.T,params,freq,phi,np.min(freq),np.max(freq),s_param='S11'),phi=(0,phi[-1],1))
    
def S22_plot(mag_data_db22, freq):
    fig, ax = plt.subplots(1, 1, figsize = (12, 4), dpi = 300)

    plt.scatter(freq, mag_data_db22.T[0])
    plt.xlabel('Freq (GHz)')
    plt.ylabel('$|S22|$ (dB)')
    plt.savefig('s22_vFreq.PNG', dpi=300,facecolor='w')
    plt.show()
    
def S22_slider(mag_data_db22, freq, phi):
    peak_width = [0,20]
    prom = 10
    window =300
    rel = 0.01
    Height = -20
    
    params = np.array([peak_width, prom, window, rel, Height], dtype = object)

    interact(lambda phi: pre.peak_finder_slider(mag_data_db22.T,params,freq,phi,np.min(freq),np.max(freq),s_param='S22'),phi=(0,phi[-1],1))
    
def q_err_test(TM010_Q, TM010_f0, TM010_Qu, phi, Q1, Q2):
    fig, ax = plt.subplots(1, 1, figsize = (13, 8), dpi = 300)
    
    def Ql_form(c1, c2):
        arg = (1/TM010_Q) + (1/(Q1*c1)) + (1/(Q2*c2))
        return (1/arg)
    
    plt.scatter(TM010_f0, TM010_Q, label = 'Measured $Q_L$', color = 'navy')
    plt.scatter(TM010_f0, Ql_form(0.9, 0.9), label = '$Q_L$, dec. $Q_{1,2}$', color = 'royalblue')
    plt.scatter(TM010_f0, Ql_form(1.1, 1.1), label = '$Q_L$, inc. $Q_{1,2}$', color = 'slategrey')
    plt.scatter(TM010_f0, Ql_form(0.9, 1.1), label = '$Q_L$, alt. $Q_{1,2}$', color = 'slateblue')
    plt.title('$Q_{L}$ with Coupling Error')
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('$Q_{L}$')
    plt.legend(loc = 'best', frameon = False)
    
    plt.show()
    
def indFinder_11(mag_data_db11, phi, mode):
    data11 = np.abs(mag_data_db11.T)

    heights = np.median(data11, axis=1) + 2 
    prominences = np.median(data11, axis=1) 
    rel_heights = 0.1 * np.max(data11, axis=1)

    index11 = np.empty(len(phi), dtype=object)

    for i in phi:
        peaks, _ = find_peaks(data11[i], height=heights[i], width=[0, 30], prominence=prominences[i])

        rel = rel_heights[i] / np.max(data11[i])

        peaks, _ = find_peaks(data11[i], height=heights[i], width=[0, 30], prominence=prominences[i], rel_height=rel)

        index11[i] = peaks
        
    stop_ind = None
    for i, element in enumerate(index11):
        if len(element) < (mode+1):
            stop_ind = i
            break
            
    if stop_ind == None:
        stop_ind = len(phi) - 1
        
    return index11
    
def indFinder_22(mag_data_db22, phi, mode):
    data22 = np.abs(mag_data_db22.T)

    heights = np.median(data22, axis=1) + 2 
    prominences = np.median(data22, axis=1) 
    rel_heights = 0.1 * np.max(data22, axis=1)

    index22 = np.empty(len(phi), dtype=object)

    for i in phi:
        peaks, _ = find_peaks(data22[i], height=heights[i], width=[0, 30], prominence=prominences[i])

        rel = rel_heights[i] / np.max(data22[i])

        peaks, _ = find_peaks(data22[i], height=heights[i], width=[0, 30], prominence=prominences[i], rel_height=rel)

        index22[i] = peaks
        
    return index22

def allPeaks(data, freq_list, phi):
    peak_indices_for_all_phi = []
    
    peak_width = [0,20]
    prom = 1
    window = 300
    rel = 0.01
    Height = -5

    xmin = np.min(freq_list)
    xmax = np.max(freq_list)
    minpos = pre.find_nearest_pos(freq_list,xmin)
    maxpos = pre.find_nearest_pos(freq_list,xmax)

    for phi in phi:
        data_slice = -1 * data[phi][minpos:maxpos]
        freq_list_slice = freq_list[minpos:maxpos]

        peaks = find_peaks(data_slice, width=peak_width, prominence=prom, rel_height=rel, wlen=window, height=Height)
        prominences, left_bases, right_bases = peak_prominences(data_slice, peaks[0])

        # Create constant offset as a replacement for prominences
        offset = np.full_like(prominences, 3)

        # Calculate widths at x[peaks] - offset * rel_height
        widths, h_eval, left_ips, right_ips = peak_widths(data_slice, peaks[0], rel_height=1, prominence_data=(offset, left_bases, right_bases))

        peak_indices_for_all_phi.append(peaks[0])
        all_inds = np.array(peak_indices_for_all_phi)

    return all_inds