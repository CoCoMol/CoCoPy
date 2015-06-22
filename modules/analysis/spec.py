#!/usr/bin/python
# -*- coding: utf-8 -*-

################################################################################
#
# CoCoPy - A python toolkit for rotational spectroscopy
#
# Copyright (c) 2013 by David Schmitz (david.schmitz@chasquiwan.de).
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of 
# this software and associated documentation files (the “Software”), to deal in the 
# Software without restriction, including without limitation the rights to use, 
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the 
# Software, and to permit persons to whom the Software is furnished to do so, 
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all 
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A 
# PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT 
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN 
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH 
# THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# 
# MIT Licence (http://mit-license.org/)
#
################################################################################

import numpy as np
import pylab as pl
import wfm_readout as readout
import scipy.integrate as inte
import copy
import os

'''
Todo:
    - 1 Comment all 
    - 2
    - 3
'''

class spectrum:
    '''
    The Spectrum class tries to cover everything, which is related to the measured
    microwave spectrum or FID. Support for different file formats are implemented.
    '''
    
    def __init__(self, fname='test.txt', name='Test'):
        '''
        Starts an empty spectrum instance.

        Parameters
        ----------
        fname (str): Filename of waveform file ('test.txt').
        name (str): name of the spec instance
        
        Returns
        -------
        none
        
        Notes
        -----
        none
        
        '''
        
        self.fid = np.array([[]])
        self.fid_params = dict(npoints=0, max_t=0.0, delta_t=0.0, trigger_dl=0.0, avg=1)    
        
        self.spec = np.array([[]])
        self.peaks = np.array([[]])
        self.spec2d = np.array([[]])
        
        self.stepsize = 0.0
        
        self.spec_params = dict(npoints=0, min_f=0.0, max_f=0.0, delta_f=0.0, window_f='rect', noise_level=0.2)
        
        self.view_params = dict(min_f=0.0, max_f=0.0)
        
        self.gen_params = dict(name='test', fname='test.txt')
        
        self.gen_params['name'] = name
        self.gen_params['fname'] = fname
        
    def read_fid(self, fname=''):
        '''
        Reads in an fid file. If the file extension is .wfm it treats the file as
        a Tektronix binary waveform file. Otherwise it assumes a text file.
        It safes the FID data in self.fid and filles the fid_params dict.
                
        Parameters
        ----------
        fname (str): Filename of waveform file (''). If no filename is specified, 
        it uses the default filename "test.txt".
        
        Returns
        -------
        none
        
        Notes
        -----
        none
        
        '''

        try:
            if fname == '':
                fname = self.gen_params['fname']
            else:
                self.gen_params['fname'] = fname
            f = open(fname, 'r')
            f.close()
        except IOError:
            print 'Cannot open: ', self.gen_params['fname']
        else:
            if self.gen_params['fname'].split('.')[len(self.gen_params['fname'].split(fname))-1] == 'wfm':
                h = readout.fidReadoutWfm(fname=self.gen_params['fname'])
                self.fid_params['avg'] = h.meas_params['averages']

            else:
                h = readout.fidReadoutTxt(fname=self.gen_params['fname'])
                
            self.fid = h.fid
            self.fid_params['npoints'] = h.meas_params['npoints']
            self.fid_params['max_t'] = h.meas_params['max_t']
            self.fid_params['delta_t'] = h.meas_params['delta_t']
            self.fid_params['trigger_dl'] = h.meas_params['trigger_dl']
            
    def direct_fid(self, fid):
        '''
        Takes a 2 column numpy array and saves it as an attribute (fid) of the object.
        
        Parameters
        ----------
        fid (float): 1 or 2 column array with the times and the 
        intensities.

        Returns
        -------
        none
        
        Notes
        -----
        none
        
        '''
        
        self.fid = fid
        self.fid_params['max_t'] = max(self.fid[:,0])
        self.fid_params['n_points'] = len(self.fid[:,0])
        self.fid_params['delta_t'] = abs(self.fid[0,0]-self.fid[1,0])
            
    def read_spec(self, fname=''):
        '''
        Reads in a txt spectrum file. Frequencies should be in MHz. 
        It safes the data in self.spec and filles the spec_params dict.
        
        Parameters
        ----------
        fname (str): Filename of waveform file (''). If no filename is specified, 
        it uses the default filename "test.txt".

        Returns
        -------
        none
        
        Notes
        -----
        none
        
        '''
        try:
            if fname == '':
                fname = self.gen_params['fname']
            else:
                self.gen_params['fname'] = fname
            f = open(fname, 'r')
            f.close()
        except IOError:
            print 'Cannot open: ', self.gen_params['fname']
        else:
            h = readout.specReadoutTxt(fname=self.gen_params['fname'])

            self.spec = h.spec
            self.spec_params['npoints'] = h.meas_params['npoints']
            self.spec_params['max_f'] = h.meas_params['max_f']
            self.spec_params['min_f'] = h.meas_params['min_f']
            self.spec_params['delta_f'] = h.meas_params['delta_f']

    def direct_spec(self, spec=np.array([[]])):
        '''
        Takes a 2 column numpy array and saves it as an attribute of the object.
        
        Parameters
        ----------
        spec (float): 2 column array with the frequencies and the 
        intensities.

        Returns
        -------
        none
        
        Notes
        -----
        none
        
        '''
        if len(spec) != 0:
            self.spec = spec
            
            self.spec_params['max_f'] = max(self.spec[:,0])
            self.spec_params['min_f'] = min(self.spec[:,0])
            self.spec_params['delta_f'] = abs(self.spec[0,0]-self.spec[1,0])
            self.spec_params['npoints'] = len(self.spec[:,0])


    def read_linelist(self, fname=''):
        '''
        Reads in a txt linelist file. Frequencies should be in MHz. 
        It creates a plottable spectrum.
        
        Parameters
        ----------
        fname (str): Filename of the linelist file (''). If no filename is specified, 
        it uses the default filename "test.txt".

        Returns
        -------
        none
        
        Notes
        -----
        none
        
        '''
        
        try:
            if fname == '':
                fname = self.gen_params['fname']
            else:
                self.gen_params['fname'] = fname
            f = open(fname, 'r')
            f.close()
        except IOError:
            print 'Cannot open: ', self.gen_params['fname']
        else:
            h = readout.linesReadoutTxt(fname=self.gen_params['fname'])

            self.spec = lines_to_spec(h.lines)
            self.spec_params['max_f'] = h.meas_params['max_f']
            self.spec_params['min_f'] = h.meas_params['min_f']

    def direct_linelist(self, linelist):
        '''
        Converts a linelist into a plottable spectrum. Frequencies should be in MHz.
        
        Parameters
        ----------
        linelist (float): 1 or 2 column array with the frequencies and the 
        intensities. Intensities are optional.

        Returns
        -------
        none
        
        Notes
        -----
        none
        
        '''
        
        self.spec = lines_to_spec(linelist)
        self.spec_params['max_f'] = max(self.spec[:,0])
        self.spec_params['min_f'] = min(self.spec[:,0])        
    
    def fid_to_spec(self, window_f = ' ', beta = 2, t_start = 0., t_end = 100.0):
        '''
        FFT with zero-padding of the FID.
        Saves the result in self.spec
        
        Parameters
        ----------
        window_f: is the applied window function. The following window functions are
        applicable: hanning, hamming, blackman, bartlett, kaiser (beta (float) is only used
        by the kaiser window)
        t_start (float): specifies the beginning of the FID. The corresponding data points
        will be cut away.
        t_end (float): specifies the end of the FID. The corresponding data points
        will be cut away.
                
        Returns
        -------
        none
        
        Notes
        -----
        none
        '''
        if len(self.fid) != 0:

            fid = slice_fid(self.fid, t_start, t_end)
            window_f = window_f.lower()   # Choose the window function (default: rect / none)    
            
            if window_f == 'hanning':
                fid[:,1] = fid[:,1] * np.hanning(len(fid))
                
            elif window_f == 'hamming':
                fid[:,1] = fid[:,1] * np.hamming(len(fid))
                
            elif window_f == 'blackman':
                fid[:,1] = fid[:,1] * np.blackman(len(fid))
            
            elif window_f == 'bartlett':
                fid[:,1] = fid[:,1] * np.bartlett(len(fid))
                
            elif window_f == 'kaiser':
                fid[:,1] = fid[:,1] * np.kaiser(len(fid), beta)
            
            h = (int(np.sqrt(len(fid))) + 1) ** 2   # Zero padding to n**2 length to enhance computing speed
            #spec = np.absolute(np.fft.rfft(fid[:,1], h)) ** 2 / h
            spec = np.absolute(np.fft.rfft(fid[:,1], h)) / h

            spec = spec[0:int(h/2)]
            freq = np.fft.fftfreq(h, np.abs(fid[2,0] - fid[1,0])) / 1.E6
            freq = freq[0:int(h/2)]  # Combine FFT and frequencies

            self.spec = np.column_stack([freq, spec])
            
            self.spec_params['npoints'] = len(spec)
            self.spec_params['max_f'] = np.max(freq)
            self.spec_params['min_f'] = np.min(freq)
            self.spec_params['delta_f'] = np.abs(freq[1]-freq[0])

    def slice_fid(self, t_start=0.0, t_end=100.):
        '''
        Slices the FID.
        
        Parameters
        ----------
        t_start (float): Start time in mus (0.0)
        t_end (float): End time in mus (100.0)
        
        Returns
        -------
        none
        
        Notes
        -----
        none
        
        '''

        self.fid = slice_fid(self.fid, t_start, t_end)
        self.fid_params['npoints'] = len(self.fid)
        self.fid_params['max_t'] = self.fid[-1,0]
        self.fid_params['delta_t'] = self.fid[1,0] - self.fid[0,0]
        

    def slice_spec(self, f_start=2000., f_end=9000.):
        '''
        Slices the spectrum into pieces.
        
        Parameters
        ----------
        f_start (float): Start frequency in MHz (2000.)
        f_end (float): End frequency in MHz (9000.)
        
        Returns
        -------
        none
        
        Notes
        -----
        none
        
        '''

        self.spec = slice_spec(self.spec, f_start, f_end)
        self.spec_params['npoints'] = len(self.spec)
        self.spec_params['min_f'] = self.spec[0,0]
        self.spec_params['max_f'] = self.spec[-1,0]
        self.spec_params['delta_f'] = self.spec[1,0] - self.spec[0,0]

        
            
        
    def mask_spectrum(self, lines=[], thresh=.2, new_val=0.):
        '''
        Deletes unwanted lines in the spectrum. Makes use of line list
        
        Parameters
        ----------
        lines (list, float): List of lines, which should be removed from the 
        spectrum ([])
        thresh (float): specifies the range (+-) of deleted points in MHz (0.2)
        new_val (float): specifies the new value for the deleted points (0.0)
        
        Returns
        -------
        none
        
        Notes
        -----
        none
        
        '''
        if len(self.spec) != 0:
            self.spec = mask_spectrum(self.spec, lines, thresh, new_val)
        
    def del_lines(self, linelist_path='/Users/davidschmitz/Documents/Programmierung/Python/linelists/', files=['background_line.txt'], thresh=.2, new_val=0.):
        '''
        Deletes unwanted lines in the spectrum. Makes use of one or more linelist 
        text files.
        
        Parameters
        ----------        
        lineliste_path (str): folderpath, where the linelist files are located (Path: /Users/davidschmitz/Documents/Programmierung/Python/linelists/)
        files (list, str): filenames of the linelist files (['background_line.txt'])
        thresh (float): specifies the range (+-) of deleted points in MHz (.2)
        new_val (float): specifies the new value for the deleted points (0.0)
        
        Returns
        -------
        none
        
        Notes
        -----
        none
        '''
    
        if len(self.spec) != 0:
            self.spec = del_lines(self.spec, linelist_path, files, thresh, new_val)      
                                
    def peakfinder(self, thresh=1.0E-6):
        '''
        Identifies the peaks above a certain intensity thershold (thresh).
        
        Parameters
        ----------        
        thresh (float): intensity threshold (1.0E-6)
        
        Returns
        -------
        none
        
        Notes
        -----
        '''
        if len(self.spec) != 0:
            self.peaks = peakfinder(thresh, self.spec)

    def noise_level(self):
        '''
        Determines the noise level of the spectrum.
        
        Parameters
        ----------
        none
        
        Returns
        -------
        none
        
        Notes
        -----
        none
        '''
        
        if len(self.spec) != 0:
            
            h_mean = np.mean(self.spec[:,1])
            h_std = np.std(self.spec[:,1])
            
            peaks = peakfinder(thresh=h_mean+10.*h_std, spec=self.spec)
            spec = mask_spectrum(spec=np.copy(self.spec), lines=peaks[:,0], new_val=h_mean)
            
            self.spec_params['noise_level'] = np.mean(spec[:,1])
            
            return np.mean(spec[:,1])
            
    def normalize(self):
        '''
        Normalizes the spectrum to the noise level by dividing the amplitudes 
        by 'noise_level'.
        
        Parameters
        ----------
        none
        
        Returns
        -------
        none
        
        Notes
        -----
        none
        '''
        
        if len(self.spec) != 0:        
            self.noise_level()
            self.spec[:,1] = self.spec[:,1] / self.spec_params['noise_level']

    def fast_spec(self, fname='', t_start = 0., t_end = 100., f_start=2000.0, f_end=8000.0):
        '''
        This function does the common routine: 
        1. Read fid
        2. Create spectrum
        
        Parameters
        ----------
        fname (float): Filename of waveform file
        t_start (float): Start value for the fid slice to generate the FFT (0.E-6)
        t_end (float): End value for the fid slice to generate the FFT (100.E-6)
        f_start (float): Start value for the sliced FFT in MHz (2000.0)
        f_end (float): End value for the sliced FFT in MHz (8000.0)
        
        Returns
        -------
        none
        
        Notes
        -----
        none
        '''
        
        if len(self.fid) == 1:
            self.read_fid(fname)
        if len(self.fid) > 1:
            self.fid_to_spec(t_start=t_start, t_end=t_end)
            self.slice_spec(f_start, f_end)
            
    def line_int_fft(self, freq=[0.0], thresh=.1):
        '''
        This function returns the intensity of a point in the FFT.
        
        Parameters
        ----------
        freq (list, float): List of frequencies for which the intensity should be 
        returned (Frequencies in MHz) ([0.0])
        thresh (float): specifies the range (+-) to look for the max in MHz (0.1)
        
        Returns
        -------
        intensity (1D-array, float): Array of the intensities
        
        Notes
        -----
        none
        '''
        if len(self.spec) != 0:
            intensity = np.array([])
            for x in freq:
                h = self.spec[np.where(abs(self.spec - x) < thresh)[0]]
                if len(h) == 0:
                    h = -1.0
                else:
                    h = max(h[:,1])
                intensity = np.hstack((intensity, h))
            
            return intensity
            
        else:
            return np.array([])
            
    def read_phase(self, freq=[0.0], zero=[0.0], t_start=0.0, t_end=100.0):
        '''
        Calculates the phase of a certain frequency in the FID. It calculates the
        Fourier coefficents for that specific frequency.
        
        Parameters
        ----------
        freq (list, float): List of frequencies for which the phase should be 
        calculated (Frequencies in MHz) ([0.0])
        zero (list, float; float): Introduced phase shifts for the calculated 
        phases ([0.0])
        t_start (float): Starting point for the FID (0.0)
        t_end (float): End of the FID (100.0)

        Returns
        -------
        phase (1D-array, float): Array of the phases
        
        Notes
        -----
        none
        '''
        if len(self.fid) > 0:
            if zero is not list:
                zero = np.ones(len(freq))*zero
            fid = slice_fid(self.fid, t_start, t_end)
            phase = np.array([])
            i = 0
            for x in freq:

                cos2 = np.cos(fid[:,0]*1.E6*x*2.*np.pi)
                sin2 = np.sin(fid[:,0]*1.E6*x*2.*np.pi)
        
                a = inte.trapz(cos2*fid[:,1], fid[:,0]) / np.abs(fid[-1,0]) * 2.
                b = inte.trapz(sin2*fid[:,1], fid[:,0]) / np.abs(fid[-1,0]) * 2.
                
		                
                h_phase = np.pi/2. - np.arctan2(b,a) - zero[i]
                if h_phase > np.pi:
                    h_phase = -np.pi + np.abs(h_phase - np.pi)
                if h_phase < -np.pi:
                    h_phase = np.pi - np.abs(h_phase + np.pi)
                
		#h_phase = np.arccos(a/np.sqrt(a**2+b**2))
		
                phase = np.hstack((phase, h_phase))
                i += 1
            return phase
        
        else:
            return np.array([])


    def read_intensity(self, freq=[0.0], t_start=0.0, t_end=100.0):
        '''
        Calculates the intensity of a certain frequency in the FID. It calculates 
        the Fourier coefficents for that specific frequency.
        
        Parameters
        ----------
        freq (list, float): List of frequencies for which the intensity should be 
        calculated (Frequencies in MHz)

        Returns
        -------
        intensity (1D-array, float): Array of the intensities
        
        Notes
        -----
        none
        '''
        if len(self.fid) > 0:
            fid = slice_fid(self.fid, t_start, t_end)
            intensity = np.array([])
            for x in freq:
                cos2 = np.cos(fid[:,0]*x*1.E6*2.*np.pi)
                sin2 = np.sin(fid[:,0]*x*1.E6*2.*np.pi)
        
                a = inte.trapz(cos2*fid[:,1], fid[:,0]) / abs(fid[-1,0]) * 2.
                b = inte.trapz(sin2*fid[:,1], fid[:,0]) / abs(fid[-1,0]) * 2.
       
                intensity = np.hstack((intensity, np.sqrt(a**2 + b**2)))
            
            return intensity
                
        else:
            return np.array([])

    def mixdown_fid(self, freqs,mix_to=1.,samplerate=10.e3,window_len_percent=20., t_start=0.0, t_end=100.0):
        '''
        Mixes down the signal at the input frequencies (freqs) down the the target
        output frequency (mix_to). The output are the mixed down fids (mixed_fids). The
        input fid is down-sampled to the samplerate and final output fid is smoothed
        using window size equal to window_len_percent% of the total number of points in the
        down-sampled FID.
                
        Parameters
        ----------
        freqs (1-D array, float):       Frequencies in MHz to mix down.
        mix_to (float):                 Mix down to this frequency (MHz). Default is 1 MHz.
        samplerate (float):             downsample the input FID to this samplerate. Default is 10 GHz.
        window_len_percent (float):     percent of points in mixed down FID for smoothing window. Default 20.%
        
        Returns
        -------
        mixed_fids (nD-array, float)
        
        Notes
        -----
        none
        
        '''

        
        if len(self.fid) > 0:
 
            fid_in = copy.copy(self.fid)
 
            dsp = int((1.e-6/self.fid_params['delta_t'])/samplerate)
            dspts = np.arange(0,len(fid_in[:,0]),dsp)
            mixwin = int((window_len_percent/100.)*len(dspts))
             
            mixed_fids = fid_in[dspts,0]

            for j in np.arange(0,len(freqs)):

                fmx = (freqs[j]-mix_to)*1.e6
                mfd = np.sin(2*np.pi*fid_in[dspts,0]*fmx)
                amx = fid_in[dspts,1]*mfd
                asm = smooth(amx,window_len=mixwin)

                mixed_fids = np.vstack((mixed_fids,asm))
            
            return mixed_fids
        
        else:
            return np.array([])

            
    def spectograph(self, stepsize=10.0, window='', f_min=2000., f_max=9000.):
        '''
        Plots the spectrum.
        
        Parameters
        ----------
        xlim (list, float): Plotting x-range of the frequencies in MHz ([0.0, 0.0]) 

        Returns
        -------
        none
        
        Notes
        -----
        none
        '''
        spec2d = np.array([])
        
        if len(self.fid) > 0:
            steps = int(self.fid[-1,0]/(stepsize * 1E-6))
            for i in np.arange(0, steps, 1):
                self.fid_to_spec(window_f=window, t_start= float(i) * stepsize, t_end= float(i+1) * stepsize)
                self.slice_spec(f_min, f_max)                
                
                if len(spec2d) == 0:
                    spec2d = self.spec[:,1]
                else:
                    spec2d = np.vstack((spec2d, self.spec[:,1]))
            
            self.spec2d = spec2d
            self.stepsize = stepsize
            
            
    def plot_spectograph(self, fig=1):
        '''
        Plots the spectrum.
        
        Parameters
        ----------
        xlim (list, float): Plotting x-range of the frequencies in MHz ([0.0, 0.0]) 

        Returns
        -------
        none
        
        Notes
        -----
        none
        '''
        if len(self.spec2d) > 0:
            pl.figure(fig)
            pl.imshow(np.log10(self.spec2d), cmap=pl.cm.coolwarm, aspect='auto', extent=[self.spec[0,0], self.spec[-1,0], len(self.spec2d[:,0])*self.stepsize, 0])
        
    def reduce_noise(self, stepsize=1.0):
        '''
        Plots the spectrum.
        
        Parameters
        ----------
        xlim (list, float): Plotting x-range of the frequencies in MHz ([0.0, 0.0]) 

        Returns
        -------
        none
        
        Notes
        -----
        none
        '''
        if len(self.fid) > 0:
            noise = np.array([])
            npoints = int(stepsize*1E-6 / self.fid_params['delta_t'])
            steps = int(self.fid_params['npoints'] / npoints) + 1
            cutout = self.fid[-npoints:len(self.fid),1]
            for i in np.arange(0, steps, 1):
                noise = np.hstack((cutout, noise))
                    
            noise = noise[-len(self.fid):len(noise)]
            
            self.fid[:,1] -= noise
    
    def plot(self, x_lim=[0.0, 0.0], fig=1, scale=1.0, offset=0.0, shift=0.0, label='', inline = False, rasterized=False):
        '''
        Plots the spectrum.
        
        Parameters
        ----------
        xlim (list, float): Plotting x-range of the frequencies in MHz ([0.0, 0.0]) 
        fig (int): id of plot figure (1)
        scale (float): scales the data by that value (1.0)
        offset (float): sets an offset to the y-data (0.0)
        shift (float): shifts the x-data
        label (str): Label of the line within the plot ('')

        Returns
        -------
        none
        
        Notes
        -----
        none
        '''
        
        if len(self.spec) > 0:
            pl.ion()
            pl.figure(fig)
            pl.plot(self.spec[:,0]+shift, self.spec[:,1]*scale+offset, label=label, rasterized=rasterized)
            if x_lim[0] != x_lim[1]:
                pl.xlim(x_lim)
            if inline == False:
                pl.show()
            
    def int_plot(self, freq, intens, fig=1):
        '''
        Connects to an existing plot. Some interaction is possible.
        Press button a: saves cursor data in freq and intens
        Press button m: stops interaction mode and closes the plot
        Press button r: prints cursor data
        
        Parameters
        ----------
        freq (list, float): In this variable the cursor x-data is stored.
        intens (list, float): In this variable the cursor y-data is stored.
        fig (int): id of plot figure (1)

        Returns
        -------
        freq (list, float): The list of frequencies.
        intens (list, float): The list of the corresponding intensities.
        
        Notes
        -----
        #Application:
        
        freq=list()
        intens=list()

        spec.int_plot(freq, intens)
        # no you can access freq and intens from the ipython console
        '''
        
        def remove_dublicates(seq1, seq2):
            seq1_new = []
            seq2_new = []
            i = 0
            for e in seq1:
                if e not in seq1_new:
                    seq1_new.append(e)
                    seq2_new.append(seq2[i])
                i += 1
            return seq1_new, seq2_new
            
        fig = pl.figure(fig)
        
        def on_key(event):
            if event.key == 'a':
                print('Cursor: ', event.xdata, event.ydata)
                freq.append(event.xdata)
                intens.append(event.ydata)
                os.system("echo '%s' | pbcopy" % '{0:.3f}'.format(float(event.xdata)))

            if event.key == 'f':
                print('Cursor: ', event.xdata, event.ydata)
                freq.append(event.xdata)
                intens.append(event.ydata)
                os.system("echo '%s' | pbcopy" % '{0:.3f}'.format(float(event.xdata)))

            if event.key == 'i':
                print('Cursor: ', event.xdata, event.ydata)
                freq.append(event.xdata)
                intens.append(event.ydata)
                os.system("echo '%s' | pbcopy" % '{0:.20f}'.format(float(event.ydata)))
                            
            if event.key == 'm':
                fig.canvas.mpl_disconnect(cif)
                pl.close()
                
            if event.key == 'r':
                os.system("echo '%s' | pbcopy" % event.xdata)
                
        cif = fig.canvas.mpl_connect('key_press_event', on_key)

        freq, intens = remove_dublicates(freq, intens)
        return freq, intens

################################################################################        

def lines_to_spec(lines, eps = .0001):
    '''
    help
    
    Parameters
    ----------
    lines (1D or 2D np.array, float):
    eps (float):

    Returns
    -------
    spectrum (2D-array, float): 1. col frequencies (MHz), 2. col. intensities
    
    Notes
    -----
    none
    '''
    
    if len(lines.shape) == 1:
        g = np.ones(len(lines))
        g = g.reshape((len(g), 1))
        lines = np.hstack((lines.reshape((len(lines), 1)), g))
            
    i = 0
    h = np.split(lines, len(lines))
    
    while i < len(h):   # Puts a zero next to each line
        h1 = [[h[i][0][0] - eps, 0.]]
        h2 = [[h[i][0][0] + eps, 0.]]        
        #h[i][0][1] = intensity        
        h[i] = np.vstack((h1, h[i], h2))
        i+=1
    
    return np.vstack(h[:])
    
def mask_spectrum(spec=np.array([[]]), lines=[], thresh=.2, new_val=0.):
    '''
    Deletes unwanted lines in the spectrum. Makes use of lines array
    
    Parameters
    ----------        
    '''
    for x in lines: # Delete line if within 200 kHz range
        spec[np.where(np.abs(spec[:,0] - x) < thresh), 1] = new_val
        
    return spec
    
    
def del_lines(spec=np.array([[]]), linelist_path='', files=[], thresh=.2, new_val=.0):
    '''
    Deletes unwanted lines in the spectrum. Makes use of one or more linelist 
    text files.
    
    
    Parameters
    ----------        
    lineliste_path (str): folderpath, where the linelist files are located
    files (list, str): filenames of the linelist files
    thresh (float): specifies the range (+-) of deleted points in MHz
    
    Path: /Users/davidschmitz/Documents/Programmierung/Python/linelists/
    '''

    lines = np.array([[]])
    
    for x in files:
        h = readout.linesReadoutTxt(fname=linelist_path + x)
        lines = h.lines[:,0]
        for x in lines: # Delete line if within 200 kHz range
            spec[np.where(np.abs(spec[:,0] - x) < thresh), 1] = 0.
            
    return spec

def find_bg_lines(peaks_sig, peaks_bg, thresh=0.5, bw=0.02):
    
    real_bg = np.array([])
    
    for x in peaks_sig[:]:
        a = np.where(np.abs(peaks_bg[:,0] - x[0]) < bw)[0]
        if len(a) > 1:
            a = np.where(np.abs(peaks_bg[:,0] - x[0]) < 0.02)[0]
        if len(a) == 1:
            if x[1] < peaks_bg[a, 1]:
                real_bg = np.append(real_bg, x[0])
            elif np.abs(x[1] - peaks_bg[a, 1])/max([x[1], peaks_bg[a, 1]]) < thresh:
                real_bg = np.append(real_bg, x[0])
    
    return real_bg
                

                    
def peakfinder(thresh=1E-6, spec=np.array([[]])):
    '''
    Identifies the peaks above a certain intensity thershold (thresh).
    
    
    Parameters
    ----------        
    thresh (float): intensity threshold (1.E-6)
    spec (2-column np.array): the spectrum under study: first column contains
    the frequencies, the second one the intensities ([[]])
    
    Returns
    -------
    peaks (2D-array, float): Array of the peaks
        
    Notes
    -----
    none
    '''
    
    di = np.diff(spec[:,1])
    spec = np.split(spec, [len(spec)-1])[0]
    
    d = 5; i = 2 * d + 1; flag = 0
    
    while i < ( len(di) - 2*d ):
        # zero crossing
        if (di[i-1] * di[i] < 0.):
            # slope of di positive
            if (di[i-1] > di[i]):
                h = spec[i-d:i+d, :]
                # Peak bigger than Threshold
                if np.max(h[:, 1]) >= thresh:
                    k = np.where(h[:, 1] == np.max(h[:, 1]))[0]
                    h = spec[i - 2*d + k : i + k, :]
                    if flag == 0:
                        # Init Peaklist
                        peaks = h[np.where(h[:, 1] == np.max(h[:, 1])), :][0]
                        flag = 1
                    else:
                        # Add peak to peaklist
                        peaks = np.vstack((peaks, h[np.where(h[:, 1] == np.max(h[:, 1])), :][0]))
                    i += d - 2
        i+=1
        
    #Delete double peaks
    flag = 0    
    
    for x in peaks[:]:
        if flag == 0:
            peaks_clean = np.array([[x[0], x[1]]])
            flag = 1
        else:
            if len(np.where(peaks_clean[:,0] - x[0] == 0.)[0]) == 0:
                peaks_clean = np.vstack((peaks_clean, x))
                
    return peaks_clean

def slice_fid(fid, t_start=0.0, t_end=100.):
    '''
    Slices the FID.
    
    Parameters
    ----------
    t_start (float): Start time in mus (0.0)
    t_end (float): End time in mus (100.0)
    fid (2-d array, float): Input FID
    
    Returns
    -------
    fid (2-d array, float): Cropped FID
    
    Notes
    -----
    none
    
    '''
    
    t_start = t_start * 1.E-6
    t_end = t_end * 1.E-6
    if t_end > t_start and t_start < fid[-1,0]:
        fid = fid[np.where(fid[:,0] > t_start)]
        fid = fid[np.where(fid[:,0] < t_end)]
        
        return fid
    else:
        print 'f_end has to be larger than f_start'  

def slice_spec(spec, f_start=2000., f_end=9000.):
    '''
    Slices the spectrum into pieces.
    
    Parameters
    ----------
    f_start (float): Start frequency in MHz (2000.)
    f_end (float): End frequency in MHz (9000.)
    spec (2d-array, float): Input spectrum
    
    Returns
    -------
    spec (2d-array, float): Cropped spectrum
    
    Notes
    -----
    none
    
    '''

    if f_end > f_start and f_start < spec[-1,0]:
        spec = spec[np.where(spec[:,0] > f_start)]
        spec = spec[np.where(spec[:,0] < f_end)]
        
        return spec
    else:
        print 'f_end has to be larger than f_start'  

def coadd_fid(fid1, fid2, invert=False):
    '''
    Coadds the FIDs fid1 and fid2. FIDs must have same length
    
    Parameters
    ----------
    fid1 (spectrum object): First spectrum object with FID loaded
    fid2 (spectrum object): Second spectrum object with FID loaded
    invert (boolean): Defines if the second FID should be added or substracted 
            
    Returns
    -------
    new (spectrum object): Coadded spectrum object
    
    Notes
    -----
    none
    '''
    
    new = spectrum()
    h = float(fid1.fid_params['avg']) / float(fid2.fid_params['avg']) 
    new.fid = copy.deepcopy(fid1.fid)
    new.fid[:,1] *= h
    if invert == False:
        new.fid[:,1] += fid2.fid[:,1]
    else:
        new.fid[:,1] -= fid2.fid[:,1]
    new.fid[:,1] /= (h+1.)
    new.fid_params = copy.copy(fid1.fid_params) 
    
    new.fid_params['avg'] = fid1.fid_params['avg'] + fid2.fid_params['avg']
    
    return new
    
def coadd_list(fnames=[], key=''):
    '''
    Coadds the FIDs of different waveform files. FIDs must have the same length.
    If no filenames are specified, it coadds all waveform files in the specified
    folder. If this is also not specified, it uses the waveform files in the current
    directory
    
    Parameters
    ----------
    fnames (list): List of waveform filenames ([])
    key (string): waveforms in a folder have to contain that substring
            
    Returns
    -------
    a (spectrum object): Coadded spectrum object
    
    Notes
    -----
    none
    '''
    import wfm_readout as wfm
    
    def coadd(fnames):
        flag = 0
        for f in fnames:
            if flag == 0:
                a = spectrum(f)
                a.read_fid()
                flag = 1
            else:
                b = spectrum(f)
                b.read_fid()
                a = coadd_fid(a, b)
        return a

    if len(fnames) > 0:
        return coadd(fnames)
    elif len(wfm.files_folder(key=key)) > 0:
        return coadd(wfm.files_folder(key=key))
    else:
        return []

def make_spec(freq, inten=[]):
    '''
    Creates a two column array with frequencies in first and intensities in 
    the second column. If no intensities are specified, they are set to 1.
    
    Parameters
    ----------
    freq (array like): list or one column array
    inten (array like): list or one column array ([])
            
    Returns
    -------
    spectrum (two column array)
    
    Notes
    -----
    none
    '''
    freq = np.array(freq)
    inten = np.array(inten)
    if len(inten) == 0:
        inten = np.ones((len(freq), 1))
    if len(np.shape(freq)) == 1:
        freq = np.reshape(freq, (len(freq), 1))
    if len(np.shape(inten)) == 1:
        inten = np.reshape(inten, (len(inten), 1))
    
    return np.hstack((freq, inten))
        
def smooth(x, window_len=10, window='hanning'):
        
    """smooth the data using a window with requested size.
   	 
   This method is based on the convolution of a scaled window with the signal.
   The signal is prepared by introducing reflected copies of the signal 
   (with the window size) in both ends so that transient parts are minimized
   in the begining and end part of the output signal.
    	
   input:
       x: the input signal 
       window_len: the dimension of the smoothing window
       window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
               flat window will produce a moving average smoothing.

   output:
       the smoothed signal
        
   example:

    	import numpy as np    
    	t = np.linspace(-2,2,0.1)
    	x = np.sin(t)+np.random.randn(len(t))*0.1
    	y = smooth(x)
    
    	see also: 
    
    	numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    	scipy.signal.lfilter
 
    	TODO: the window parameter could be the window itself if an array instead of a string   
    	"""
    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."
    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."
    if window_len < 3:
        return x
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"
    s=np.r_[2*x[0]-x[window_len:1:-1], x, 2*x[-1]-x[-1:-window_len:-1]]
    	#print(len(s))
    
    if window == 'flat': #moving average
        w = np.ones(window_len,'d')
    else:
        w = getattr(np, window)(window_len)
    y = np.convolve(w/w.sum(), s, mode='same')
    return y[window_len-1:-window_len+1]



    