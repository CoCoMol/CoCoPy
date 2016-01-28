#!/usr/bin/python
# -*- coding: utf-8 -*-

################################################################################
#
# CoCoPy - A python toolkit for rotational spectroscopy
#
# Copyright (c) 2016 by David Schmitz (david.schmitz@chasquiwan.de).
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
from analysis.util import ProgressBar
import itertools as it
import analysis.spfit as spfit
import analysis.spec as spec
import copy as cp
import os

'''
Todo:
    1. Comment all
'''

class Autofit:

    def __init__(self, prefix='test', rot=[3000., 1500., 1000],
        dip=[1., 1., 1.], temp=.3, min_f=2500., max_f=7000.):
        '''
        This class tries to cover everything, which is related to Autofit.
        '''

        self.pred = {
            'A': 3000., 'B': 1500., 'C': 1000., 'dipA': 1., 'dipB': 1.,
            'dipC': 1., 'temp': .3, 'min_f': 2500., 'max_f': 7000.,
            'freq': np.array([]), 'qn': np.array([]), 'intens': np.array([])
            }

        self.fit_qn = {
            '0': np.array([]), '1': np.array([]), '2': np.array([]),
            'lin': np.array([]), 'cat': np.array([]), 'freq': np.array([])
            }

        self.pos_qn = {'qn': np.array([]), 'score': np.array([])}

        self.limits = np.array([])

        self.triples = np.array([])

        self.triples_all = np.array([])

        self.COMB_KEYS = [
            'A', 'B', 'C', 'A+B', 'A+C', 'B+C', 'A-B', 'A-C', 'B-C', 'A+B+C',
            'A-B-C', 'A+B-C', 'A-B+C']

        self.comb = {
            'A': np.array([[1, 0, 0], [2, 0, 0]]),
            'B': np.array([[0, 1, 0], [0, 2, 0]]),
            'C': np.array([[0, 0, 1], [0, 0, 2]]),
            'A+B': np.array([[1, -1, 0], [2, -2, 0]]),
            'A+C': np.array([[1, 0, -1], [2, 0, -2]]),
            'B+C': np.array([[0, 1, -1], [0, 2, -2]]),
            'A-B': np.array([[1, 1, 0], [2, 2, 0]]),
            'A-C': np.array([[1, 0, 1], [2, 0, 2]]),
            'B-C': np.array([[0, 1, 1], [0, 2, 2]]),
            'A+B+C': np.array([[1, 1, -2], [2, 2, -4]]),
            'A-B-C': np.array([[2, 1, 1], [4, 2, 2]]),
            'A+B-C': np.array([[1, 1, 2], [2, 2, 4]]),
            'A-B+C': np.array([[1, 2, 1], [2, 4, 2]])
            }

        self.peaks = {
            'all': np.array([]), '0': np.array([]), '1': np.array([]),
            '2': np.array([])
            }

        self.result = np.array([])

        self.params = {
            'min_N_lines': 7, 'bw': .25, 'cutoff': 50, 'start': -1, 'stop': -1,
            'itera': -1, 'delta_rot': 50, 'delta_freq': 100, 'seed': 1000,
            'N_triples': 0, 'N_triples_total': 0
            }

        self.prefix = prefix
        self.report = prefix
        self.pred['A'] = rot[0]; self.pred['B'] = rot[1]; self.pred['C'] = rot[2]
        self.pred['dipA'] = dip[0]; self.pred['dipB'] = dip[1]
        self.pred['dipC'] = dip[2]; self.pred['temp'] = temp
        self.pred['min_f'] = min_f; self.pred['max_f'] = max_f

        self.make_pred()

    def make_pred(self):
        pred = spfit.spfitAsym(fname=self.prefix+'_p', A=self.pred['A'], B=self.pred['B'], C=self.pred['C'])

        pred.int_dipole['dipole'] = np.array([self.pred['dipA'], self.pred['dipB'], self.pred['dipC']])
        pred.int_params['temp'] = self.pred['temp']
        pred.int_params['fqlim'] = self.pred['max_f'] / 1000.

        pred.cat_min()

        i_min = min(np.where(pred.cat_content['freq'] > self.pred['min_f'])[0])
        #i_max = min(np.where(pred.cat_content['freq'] > self.pred['max_f'])[0])

        p_freq = pred.cat_content['freq'][i_min:]
        p_qn = pred.cat_content['qn'][i_min:]
        p_int = pred.cat_content['lgint'][i_min:]

        self.pred['freq'] = p_freq[np.argsort(p_int)][::-1]
        self.pred['qn'] = p_qn[np.argsort(p_int)][::-1]
        self.pred['intens'] = p_int[np.argsort(p_int)][::-1]


    def assign(self, A, B, C, bw=0):

        dipA = self.pred['dipA']; dipB = self.pred['dipB']
        dipC = self.pred['dipC']; temp = self.pred['temp']
        cutoff = self.params['cutoff']; peaks = self.peaks['all']
        if bw == 0:
            bw = self.params['bw']

        pred = spfit.spfitAsym(self.prefix+'_a', A=A, B=B, C=C, dipA=dipA, dipB=dipB, dipC=dipC, temp=temp)
        pred.int_params['fqlim'] = self.pred['max_f'] / 1000.

        pred.cat_min()

        if cutoff < 0:
            lines = np.where(pred.cat_content['lgint'] > cutoff)[0]

        else:
            h = pred.cat_content['lgint'][np.argsort(pred.cat_content['lgint'])][-cutoff:]
            lines = np.where(pred.cat_content['lgint'] > min(h))[0]

        qn = pred.cat_content['qn'][lines]
        lines = pred.cat_content['freq'][lines]

        ind_l = np.array([], dtype=int)
        ind_p = np.array([], dtype=int)

        i = 0
        for x in lines:
            h = np.where(np.abs(peaks[:,0] - x) < bw)[0]

            if len(h) == 1:
                h_ = h[0]
                if h_ not in ind_p:
                    ind_l = np.append(ind_l, i)
                    ind_p = np.append(ind_p, h_)

            if len(h) > 1:
                h_ = np.where(abs(peaks[:,0] - x) == min(abs(peaks[:,0] - x)))[0]
                if len(h_) > 1:
                    h_ = h_[0]
                if h_ not in ind_p:
                    ind_l = np.append(ind_l, i)
                    ind_p = np.append(ind_p, h_)
            i+=1

        omc = np.sqrt(np.sum((lines[ind_l] - peaks[ind_p][:,0]) ** 2.))

        return ind_p, ind_l, qn, omc

    def load_peaks(self, peaks):
        if type(peaks) == list or type(peaks) == np.ndarray:
            self.peaks['all'] = spfit.reshape_linelist(peaks)


    def lin_dep_debug(self, qn):

        freq = self.pred['freq']
        A = self.pred['A']; B = self.pred['B']; C = self.pred['C']

        result = np.zeros((13,4))
        chk = np.zeros((2,2))
        comb_trans = [[0,1], [0,2], [1,2]]
        m = 0
        for y in comb_trans:

            n = 0
            for key in self.COMB_KEYS:

                i = 0
                for x in self.comb[key]:
                    pred = spfit.spfitAsym(self.prefix+'_l', A=A+x[0], B=B+x[1], C=C+x[2], temp=self.pred['temp'])
                    pred.int_params['fqlim'] = self.pred['max_f'] / 1000.
                    pred.cat_min()

                    j = 0
                    for k in y:
                        freq_org = freq[np.where(self.pred['qn'] == qn[k])[0]]
                        #print freq_org
                        freq_shift = pred.cat_content['freq'][np.where(pred.cat_content['qn'] == qn[k])[0]]
                        #print freq_shift
                        chk[i,j] = freq_org - freq_shift

                        j += 1

                    i += 1

                result[n,m] = abs(np.linalg.det(chk))
                n += 1
                #print key, abs(np.linalg.det(chk))
            m += 1

        result[:,3] = result[:,0] + result[:,1] + result[:,2]

        return result

    def lin_dependence(self, N=10, max_K=2, diff_J=False, diff_tt=False):
        '''
        TODO: Implement max_K
        TODO: Implement diff_J
        TODO: Implement diff_tt
        '''

        qn = self.pred['qn']; freq = self.pred['freq']
        A = self.pred['A']; B = self.pred['B']; C = self.pred['C']

        if type(N) == int:
            qn_c = np.array(map(None, it.combinations(range(N), 3)))
        elif type(N) == list or type(N) == np.ndarray:
            qn_c = np.array(map(None, it.combinations(N, 3)))

        score = np.zeros(len(qn_c))

        j = 0
        for y in qn_c:
            chk = np.zeros((3,3))

            i = 0
            for x in np.eye(3):
                pred = spfit.spfitAsym(self.prefix+'_l', A=A+x[0], B=B+x[1], C=C+x[2], temp=self.pred['temp'])
                pred.int_params['fqlim'] = self.pred['max_f'] / 1000.
                pred.cat_min()

                for k in range(3):
                    chk[i,k] = freq[y[k]] - pred.cat_content['freq'][np.where(pred.cat_content['qn'] == qn[y[k]])[0]]
                i += 1

            score[j] = abs(np.linalg.det(chk))
            j += 1

        qn_c = qn_c[np.argsort(score)][::-1]
        score = score[np.argsort(score)][::-1]

        self.pos_qn['qn'] = qn[qn_c]
        self.pos_qn['score'] = score

        self.choose_qn(0)

        return self.fit_qn

    def choose_qn(self, n):
        h = spfit.standard_qn(self.pos_qn['qn'][n])
        self.fit_qn['lin'] = spfit.convert_qn_lin(h)
        self.fit_qn['cat'] = self.pos_qn['qn'][n]

        for i in range(3):
            self.fit_qn[str(i)] = h[i]

        freq = np.zeros(3)
        for i in range(3):
            freq[i] = self.pred['freq'][np.where(self.pred['qn'] == self.fit_qn['cat'][i])[0]]

        self.fit_qn['freq'] = freq

    def print_qn(self):
        if len(self.pos_qn['qn']) > 0:
            i = 0
            print 'N\tscore\t qn_0\t\t\t\t qn_1\t\t\t\t qn_2\n\n'
            for qn in self.pos_qn['qn']:
                h = '{}\t{:2.2f}\t{}\t{}\t{}'.format(i, self.pos_qn['score'][i], qn[0], qn[1], qn[2])
                h += '\n' + ((len(h)+16) * '-')

                print h
                i += 1

    def print_pred(self):
        if len(self.pred['qn']) > 0:
            i = 0
            print 'N\tfreq\t qn\t\t\t\tintensity\n\n'
            for qn in self.pred['qn']:
                h = '{}\t{:2.2f}\t{}\t{:2.2f}'.format(i, self.pred['freq'][i], qn, self.pred['intens'][i])
                h += '\n' + ((len(h)+13) * '-')

                print h
                i += 1


    def det_qn(self, qn):
        h = spfit.standard_qn(qn)
        self.fit_qn['lin'] = spfit.convert_qn_lin(h)
        self.fit_qn['cat'] = qn

        for i in range(3):
            self.fit_qn[str(i)] = h[i]

        self.triples = np.array([])
        self.triples_all = np.array([])



    def det_limits_rot(self, delta_rot=0, seed=0):

        if delta_rot != 0:
            self.params['delta_rot'] = delta_rot
        else:
            delta_rot = self.params['delta_rot']

        if seed != 0:
            self.params['seed'] = seed
        else:
            seed = self.params['seed']

        A = self.pred['A']; B = self.pred['B']; C = self.pred['C']
        A_h, B_h, C_h = self.create_random_seed(A, B, C, delta_rot, seed)

        freq_lim = np.zeros((3,2))
        freq_lim[:,0] = 1.E9
        for i in range(len(A_h)):

            pred = spfit.spfitAsym(fname=self.prefix+'_d', A=A_h[i], B=B_h[i], C=C_h[i], temp=self.pred['temp'])
            pred.int_params['fqlim'] = self.pred['max_f'] / 1000.
            pred.cat_min()

            for j in range(3):
                h = pred.cat_content['freq'][np.where(pred.cat_content['qn'] == self.fit_qn['cat'][j])[0]]
                if h < freq_lim[j,0]:
                    freq_lim[j,0] = h
                if h > freq_lim[j,1]:
                    freq_lim[j,1] = h

        self.limits = freq_lim

        return freq_lim


    def det_limits_spec(self, delta_freq=0):

        if delta_freq != 0:
            self.params['delta_freq'] = delta_freq
        else:
            delta_freq = self.params['delta_freq']

        delta = np.zeros(3)

        if type(delta_freq) == list or type(delta_freq) == np.ndarray:
            if len(delta_freq) == 3:
                delta[0] = delta_freq[0]; delta[1] = delta_freq[1]; delta[2] = delta_freq[2]
            else:
                delta[0] = delta_freq[0]; delta[1] = delta_freq[0]; delta[2] = delta_freq[0]
        else:
            delta[0] = delta_freq; delta[1] = delta_freq; delta[2] = delta_freq


        freq_lim = np.zeros((3,2))

        for i in range(3):
            h = self.pred['freq'][np.where(self.pred['qn'] == self.fit_qn['cat'][i])[0]]
            freq_lim[i,0] = h - delta[i]
            freq_lim[i,1] = h + delta[i]

        self.limits = freq_lim

        return freq_lim


    def det_peaks(self, intens=False):
        '''
        TODO: sort by intensity
        '''
        def pick_peaks(peaks, f_min, f_max):
            h1 = np.where(peaks[:,0] > f_min)[0]
            h2 = np.where(peaks[:,0][h1] < f_max)[0]

            return peaks[h1][h2]

        h = []
        for x in self.limits:
            h.append(pick_peaks(self.peaks['all'], x[0], x[1]))

        for i in range(3):
            self.peaks[str(i)] = h[i][np.argsort(abs(h[i][:,0] - self.pred['freq'][np.where(self.pred['qn'] == self.fit_qn['cat'][i])[0]]))]

        self.params['N_triples_total'] = len(self.peaks['0']) * len(self.peaks['1']) * len(self.peaks['2'])
        self.triples_all = np.array([])

        return len(self.peaks['0']), len(self.peaks['1']), len(self.peaks['2']), self.params['N_triples_total']

    def define_triples(self, start=-1, stop=-1, itera=-1):

        if stop < 0:
            if self.params['stop'] < 0:
                stop = self.params['N_triples_total'] + 1
                self.params['stop'] = stop
            else:
                stop = self.params['stop']
        else:
            self.params['stop'] = stop

        if start < 0:
            if self.params['start'] < 0:
                start = 0
                self.params['start'] = 0
            else:
                start = self.params['start']
        else:
            self.params['start'] = start

        if itera < 0:
            if self.params['itera'] > 0:
                itera = self.params['itera']
        else:
            self.params['itera'] = itera

        l0 = len(self.peaks['0'][:,0]); l1 = len(self.peaks['1'][:,0]);
        l2 = len(self.peaks['2'][:,0])

        if len(self.triples_all) == 0:
            triples = np.array([list(i) for i in it.product(range(l0), range(l1), range(l2))])
            h = [np.sum(triples[i]) for i in range(len(triples))]
            self.triples_all = triples[np.argsort(h)]
            triples = triples[np.argsort(h)]
        else:
            triples = self.triples_all

        if itera < 0 and self.params['itera'] < 0:
            triples = triples[start:stop]
        else:
            triples = triples[start::itera]

        self.params['N_triples'] = len(triples)
        self.triples = triples

        return len(triples)

    def del_files(self, del_all=False):
        import glob
        flist = glob.glob(self.prefix+'*')
        for x in flist:
            if del_all:
                os.remove(x)
            elif 'result' not in x and '.pkl' not in x:
                os.remove(x)

    def run_autofit(self, triples=-1, verbose=False):
        if triples == -1:
            triples = self.triples

        else:
            self.triples = triples

        p0 = self.peaks['0'][:,0]; p1 = self.peaks['1'][:,0];
        p2 = self.peaks['2'][:,0]

        qn = self.fit_qn['lin']
        A = self.pred['A']; B = self.pred['B']; C = self.pred['C']
        result = np.array([])

        if verbose == True:
            m = 0
            p = ProgressBar(len(triples))


        for x in triples:
            fit = spfit.spfitAsym(fname = self.prefix + '_f', A=A, B=B, C=C)
            fit.lin_content['freq'] = np.array([p0[x[0]], p1[x[1]], p2[x[2]]])
            fit.lin_content['qn'] = qn
            fit.par_params['errtst'] = 1000/0.02

            fit.fit_min()

            if verbose == True:
                if np.mod(m, 10) == 0:
                    p.animate(m)
                m += 1


            if fit.fit_content['err_mw_rms'] >= 0. and fit.fit_content['err_mw_rms'] < .1 and fit.fit_content['reject'] == False and fit.fit_content['err_new_rms'] < 1.:

                A = fit.par_fitpar['par'][0]; B = fit.par_fitpar['par'][1]
                C = fit.par_fitpar['par'][2]

                err = fit.fit_content['err_mw_rms']

                ind_p, ind_l, qn_, omc = self.assign(A, B, C)
                N = len(ind_p)

                if len(ind_p) >= self.params['min_N_lines']:

                    h = np.array([A, B, C, N, omc])
                    result = h if len(result) == 0 else np.vstack((result, h))

        self.result = result[np.argsort(result[:,3])][::-1]

        return self.result

    def write_report(self):
        fname = self.report + '_result.txt'
        if len(self.result):
            fmt = ('%.2f', '%.2f', '%.2f', '%d','%.2e')
            np.savetxt(fname, self.result, fmt=fmt)

    def refit(self, A, B, C, err, peaks, qn):
        fit = spfit.spfitAsym(fname=self.prefix+'_rf', A=A, B=B, C=C)
        fit.lin_content['freq'] = peaks
        fit.lin_content['qn'] = qn
        fit.par_params['errtst'] = err/0.02

        fit.fit_min()

        if fit.fit_content['err_mw_rms'] >= 0 and fit.fit_content['err_mw_rms'] < .1:
            rot = fit.var_fitpar['par']
            ind_p, ind_l, qn_, omc_ = self.assign(rot[0], rot[1], rot[2])
            A = fit.var_fitpar['par'][0]; B = fit.var_fitpar['par'][1]
            C = fit.var_fitpar['par'][2];
            fitted = fit.fit_content['N_fitted_lines']

            return A, B, C, ind_p, ind_l, qn_, fitted, omc_, True

        else:
            return A, B, C, [-1], [-1], [-1], -1, -1, False

    def create_random_seed(self, A, B, C, delta, N):

        if type(delta) == list or type(delta) == np.ndarray:
            if len(delta) == 3:
                dA = delta[0]; dB = delta[1]; dC = delta[2]
            else:
                dA = delta[0]; dB = delta[0]; dC = delta[0]
        else:
            dA = delta; dB = delta; dC = delta

        A_H = (np.random.rand(N)*2.-1.)*dA + A
        B_H = (np.random.rand(N)*2.-1.)*dB + B
        C_H = (np.random.rand(N)*2.-1.)*dC + C

        return self.check_const(A_H, B_H, C_H)

    def check_const(self, A, B, C):

        h = B-C; h = np.where(h[:] > 0)[0]; A = A[h]; B = B[h]; C=C[h]
        h = A-C; h = np.where(h[:] > 0)[0]; A = A[h]; B = B[h]; C=C[h]
        h = A-B; h = np.where(h[:] > 0)[0]; A = A[h]; B = B[h]; C=C[h]

        h = np.where(A[:] > 0)[0]; A = A[h]; B = B[h]; C=C[h]
        h = np.where(B[:] > 0)[0]; A = A[h]; B = B[h]; C=C[h]
        h = np.where(C[:] > 0)[0]; A = A[h]; B = B[h]; C=C[h]

        return A, B, C

    def read_result(self, path=''):

        if path == '':
            path = os.getcwd()

        if os.path.exists(path):
            self.path = path

            import glob
            filelist = glob.glob(os.path.join(self.path, '*result.txt'))

            result = np.array([])

            for x in filelist:
                h = np.loadtxt(x)
                if len(result) > 0:
                    result = np.vstack((result, h))
                else:
                    result = h

            if len(result):
                self.result = result[np.argsort(result[:,3])][::-1]
                return self.result
            else:
                return np.array([])

        else:
            return np.array([])

    def refine_result(self, n = 1000, err=.0, limit=0.05):
        #TODO: Add progress bar
        if len(self.result) > 0:
            if err == .0:
                err = self.params['bw']

            result = np.array([])

            for x in self.result[:n]:
                ind_p, ind_l, qn_, omc_ = self.assign(x[0], x[1], x[2])
                check = True; A_ = x[0]; B_ = x[1]; C_ = x[2]; N_=len(ind_p)

                while len(ind_p) >= self.params['min_N_lines'] and err > limit and check:
                    peaks_h = self.peaks['all'][ind_p][:,0]
                    qn_h = spfit.convert_qn_cat_lin(qn_[ind_l])
                    A_, B_, C_, ind_p, ind_l, qn_, N_, omc_, check = self.refit(A_, B_, C_, err, peaks_h, qn_h)
                    err /= 2.

                h = np.array([A_, B_, C_, N_, omc_])
                result = h if len(result) == 0 else np.vstack((result, h))

            h = cp.deepcopy(self)

            h.result = result[np.argsort(result[:,3])][::-1]

            return h

    def plot_result(self, n=1000, dim=.0):

        import matplotlib.pyplot as plt
        import matplotlib as mplib
        from mpl_toolkits.mplot3d import Axes3D

        if len(self.result) > 0:

            if dim == 0:
                dim = self.params['delta_rot']

            fig = plt.figure()

            ax = fig.add_subplot(111, projection='3d')
            plt.subplots_adjust(right=.8)

            A = self.pred['A']; B = self.pred['B']; C = self.pred['C']

            ax.scatter([A], [B], [C], marker='^', c='green', s=400, edgecolors=['none'])

            result = self.result[:n]

            norm = mplib.colors.Normalize(vmin=min(result[:,3]), vmax=max(result[:,3]))
            cmap = mplib.cm.jet_r

            co = mplib.cm.ScalarMappable(norm=norm, cmap=cmap)

            ax.scatter(result[:,0], result[:,1], result[:,2], marker='o', lw = 0, c=co.to_rgba(result[:,3]))


            ax.set_xlim(A-dim, A+dim)
            ax.set_ylim(B-dim, B+dim)
            ax.set_zlim(C-dim, C+dim)

            ax.xaxis.set_rotate_label(True)
            ax.yaxis.set_rotate_label(True)
            ax.zaxis.set_rotate_label(True)

            ax.set_xlabel('\nA (MHz)', linespacing=1)
            ax.set_ylabel('\nB (MHz)', linespacing=1)
            ax.set_zlabel('\nC (MHz)', linespacing=.5)

            ax.azim = -55
            ax.elev = 25

            #cx = fig.add_axes([0.9, 0.1, 0.03, 0.8])
            bx = fig.add_axes([0.85, 0.3, 0.03, 0.4])
            cb = mplib.colorbar.ColorbarBase(bx, cmap=cmap, norm=norm)
            #fig.set_tight_layout(True)
            cb.set_label('fitness: N. of lines')

            return fig, ax

    def print_result(self, start=0, stop=100):
        if len(self.result) > 0:
            print 'N\tA\tB\t C\t\tN_lines\t\t err\n\n'

            i = 0
            for x in self.result[start:stop]:
                h = '{:d}\t{:2.2f}\t{:2.2f}\t{:2.2f}\t\t{:d}\t\t{:2.2e}'.format(i, x[0], x[1], x[2], int(x[3]), x[4])
                h += '\n' + ((len(h)+27) * '-')
                print h
                i += 1

    def run_autofit_parallel(self, N):
        import multiprocessing

        def run_parallel(fname, i):
            task = load_task(fname, i)[0][0]

            task.report = task.prefix
            task.run_autofit(verbose=False)

            task.write_report()
            task.del_files()

        create_tasks(self, N, seq=True, ret=False)

        check = list([])
        for i in range(N):
            p = multiprocessing.Process(target=run_parallel, args=(self.prefix, i))
            p.start()
            check.append(p)

        for job in check:
            job.join()

        self.read_result()
        self.del_files(del_all=True)

    def easy_fit(self, n_dep=10, bw_rot=20.):
        self.lin_dependence(n_dep)
        self.det_limits_rot(bw_rot)
        self.det_peaks()
        self.define_triples()
        self.del_files()

        return self.params['N_triples_total']

    def plot_comparison(self, n=0, spec=np.array([]), scale=1.):
        import matplotlib.pyplot as plt

        if type(n) == int:
            n = list([n])

        if len(self.result) and len(spec):
            cutoff = self.params['cutoff']

            fig, ax = plt.subplots()

            line = ax.plot(spec[:,0], spec[:,1] * scale, rasterized=True, color='k')

            colors = list(['b', 'g', 'r', 'c', 'm', 'y', 'k'])

            k = 0
            for i in n[:7]:

                A = self.result[i][0]; B = self.result[i][1]
                C = self.result[i][2]

                ind_p, ind_l, qn, omc = self.assign(A, B, C)

                pred = spfit.spfitAsym(fname = self.prefix + '_a', A=A, B=B, C=C)
                pred.read_var()
                pred.read_int()
                pred.read_cat()

                if cutoff < 0:
                    lines = np.where(pred.cat_content['lgint'] > cutoff)[0]

                else:
                    h = pred.cat_content['lgint'][np.argsort(pred.cat_content['lgint'])][-cutoff:]
                    lines = np.where(pred.cat_content['lgint'] > min(h))[0]

                lines = pred.cat_content['freq'][lines]

                pred.make_spec_cat()

                ax.plot(pred.pred_spec['spec'][:,0], pred.pred_spec['spec'][:,1]*-1.,
                    color=colors[k], label=str(i))

                for j in range(len(ind_p)):
                    h = np.array([self.peaks['all'][ind_p[j]][0], lines[ind_l[j]]])
                    ax.plot(h, np.zeros(2), 'o-', color=colors[k])

                k += 1

            ax.set_xlabel('frequency (MHz)')
            ax.set_ylabel('intensity (arb. units)')
            ax.legend()

    def export_spfit_files(self, n=0, name=''):
        if name == '':
            name = self.prefix + '_export'

        if len(self.result) > 0:

            A = self.result[n][0]; B = self.result[n][1]
            C = self.result[n][2]

            dipA = self.pred['dipA']; dipB = self.pred['dipB']
            dipC = self.pred['dipC']

            ind_p, ind_l, qn_, omc_ = self.assign(A, B, C)

            peaks = self.peaks['all'][ind_p][:,0]
            qn = spfit.convert_qn_cat_lin(qn_[ind_l])

            expt = spfit.spfitAsym(fname=name, A=A, B=B, C=C, dipA=dipA,
                dipB=dipB, dipC=dipC)
            expt.lin_content['freq'] = peaks
            expt.lin_content['qn'] = qn
            expt.par_params['errtst'] = self.params['bw']/0.02

            expt.write_var()
            expt.write_par()
            expt.write_lin()
            expt.write_int()

################################################################################

class AutofitGWDG(Autofit):
    def run_autofit(self, triples=-1, verbose=False):
        if triples == -1:
            triples = self.triples

        else:
            self.triples = triples

        p0 = self.peaks['0'][:,0]; p1 = self.peaks['1'][:,0];
        p2 = self.peaks['2'][:,0]

        qn = self.fit_qn['lin']
        A = self.pred['A']; B = self.pred['B']; C = self.pred['C']
        result = np.array([])

        for x in triples:
            fit = spfit.spfitAsym(fname = self.prefix + '_f', A=A, B=B, C=C)
            fit.lin_content['freq'] = np.array([p0[x[0]], p1[x[1]], p2[x[2]]])
            fit.lin_content['qn'] = qn
            fit.par_params['errtst'] = 1000/0.02

            fit.fit_min()

            if fit.fit_content['err_mw_rms'] >= 0. and fit.fit_content['err_mw_rms'] < .1 and fit.fit_content['reject'] == False and fit.fit_content['err_new_rms'] < 1.:

                A = fit.par_fitpar['par'][0]; B = fit.par_fitpar['par'][1]
                C = fit.par_fitpar['par'][2]

                err = fit.fit_content['err_mw_rms']

                ind_p, ind_l, qn_, omc = self.assign(A, B, C)
                N = len(ind_p)

                if len(ind_p) >= self.params['min_N_lines']:

                    h = np.array([A, B, C, N, omc])
                    result = h if len(result) == 0 else np.vstack((result, h))

        self.result = result[np.argsort(result[:,3])][::-1]

        return self.result

    def write_report(self):
        fname = self.report + '_result.txt'
        if len(self.result) > 0:
            fmt = ('%.2f', '%.2f', '%.2f', '%d','%.2e')
            np.savetxt(fname, self.result, fmt=fmt)

################################################################################

def create_random_seed(A, B, C, delta, N):
    A_H = (np.random.rand(N)*2.-1.)*delta + A
    B_H = (np.random.rand(N)*2.-1.)*delta + B
    C_H = (np.random.rand(N)*2.-1.)*delta + C

    return check_const(A_H, B_H, C_H)

def check_const(A, B, C):

    h = B-C; h = np.where(h[:] > 0)[0]; A = A[h]; B = B[h]; C=C[h]
    h = A-C; h = np.where(h[:] > 0)[0]; A = A[h]; B = B[h]; C=C[h]
    h = A-B; h = np.where(h[:] > 0)[0]; A = A[h]; B = B[h]; C=C[h]

    h = np.where(A[:] > 0)[0]; A = A[h]; B = B[h]; C=C[h]
    h = np.where(B[:] > 0)[0]; A = A[h]; B = B[h]; C=C[h]
    h = np.where(C[:] > 0)[0]; A = A[h]; B = B[h]; C=C[h]

    return A, B, C

def create_tasks(afit, N_tasks=2, seq=True, ret=False):

    import pickle
    tasks = list([])
    N_total = afit.params['N_triples_total']
    with open(afit.prefix + '_tasks.pkl', 'wb') as output:

        if seq:
            for i in range(N_tasks):
                start = i * N_total/N_tasks
                stop = (i+1) * N_total/N_tasks
                h = cp.deepcopy(afit)
                h.define_triples(start=start, stop=stop, itera=-1)
                h.triples_all = np.array([])
                h.prefix = h.prefix + '_' + str(i)
                pickle.dump(h, output, pickle.HIGHEST_PROTOCOL)
                if ret:
                    tasks.append(h)
                del h

        else:
            for i in range(N_tasks):
                start = i
                itera = N_tasks
                h = cp.deepcopy(afit)
                h.define_triples(start=start, stop=-1, itera=itera)
                h.triples_all = np.array([])
                h.prefix = h.prefix + '_' + str(i)
                pickle.dump(h, output, pickle.HIGHEST_PROTOCOL)
                if ret:
                    tasks.append(h)
                del h

    return tasks

def load_task(name, tasks_id=-1):

    import pickle

    if '.pkl' in name:
        infile = open(name, 'rb')
    else:
        infile = open(name+'_tasks.pkl', 'rb')

    if type(tasks_id) == int:
        tasks_id = list([tasks_id])

    if type(tasks_id) == list or type(tasks_id) == np.ndarray:
        tasks_id = list(tasks_id)

    tasks = list([])
    i = 0
    while 1:
        try:
            if i in tasks_id or -1 in tasks_id:
                tasks.append(pickle.load(infile))
            else:
                dump = pickle.load(infile)
            i += 1
        except (EOFError):
            break
    infile.close()

    return tasks, i

################################################################################

class GwdgJobs():
    def __init__(self, job='test', path='', queue='mpi-short'):
        self.job = job
        self.queue = queue
        self.filelist = []
        self.message = False

        self.walltime = {
            'fat-short': 2, 'mpi-short': 2, 'fat': 48, 'mpi': 48,
            'fat-long': 120, 'mpi-long': 120
            }

        self.header = '#!/bin/sh\n\
        #BSUB -L /bin/sh\n#BSUB -q {}\n\
        #BSUB -n 1\n\
        #BSUB -R "span[hosts=1]"\n\
        #BSUB -W {}:00\n\
        #BSUB -M 8000000\n\
        #BSUB -R scratch\n'

        self.header = self.header.replace('    ', '')

        self.body = '\nexport PATH=$HOME/bin:$PATH\n\
        mkdir -p /scratch/${{USER}}\n\
        MYSCRATCH=`mktemp -d /scratch/${{USER}}/py.XXXXXXXX`\n\
        python autofit_gwdg.py {} {} $MYSCRATCH\n\
        rm -rf $MYSCRATCH'

        self.body = self.body.replace('    ', '')

        if path == '':
            path = os.getcwd()

        if os.path.exists(path):
            self.path = path

            if os.path.exists(os.path.join(self.path, job + '_tasks.pkl')):
                self.job = job
            else:
                print 'File: ' + os.path.join(self.path, job) + '_tasks.pkl does not exist. Please correct!'

        else:
            print 'Path: ' + path + ' does not exist. Please correct!'


    def del_files(self, ext='job', rm_batch = True):
        #
        import glob
        if len(glob.glob('*.' + ext)) > 0:
            for x in glob.glob('*.' + ext):
                os.remove(x)
        if rm_batch == True:
            if len(glob.glob('batch.sh')) > 0:
                os.remove('batch.sh')

    def write_job_files(self, queue=''):

        self.header = self.make_header(self.message, queue)
        self.del_files()

        #catch exception
        task, N = load_task(os.path.join(self.path, self.job + '_tasks.pkl'), 10000)

        for i in range(N):
            filename = 'AFIT_' + self.job + '_' + str(i) + '.job'

            f = open(os.path.join(self.path, filename), 'w+')
            filestr = self.header + self.make_body(i)
            f.write(filestr)
            f.close()

            self.filelist.append(filename)

        self.write_batch_file()


    def make_header(self, message=False, queue=''):
        #
        if message == True:
            self.header += '#BSUB -N\n'
        else:
            self.header += '#BSUB -o /dev/null sleep 5\n'

        if queue == '':
            queue = self.queue
        else:
            self.queue = queue

        return self.header.format(self.queue, self.walltime[self.queue])

    def make_body(self, i):
        return self.body.format(self.job, i)

    def write_batch_file(self):
        #TODO: Fix newline
        f = open(os.path.join(self.path, 'batch.sh'), 'w+')
        for x in self.filelist:
            f.write('bsub < ' + x + '\n')
        f.close()

################################################################################
