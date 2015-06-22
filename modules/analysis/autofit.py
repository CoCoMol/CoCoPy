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

import spfit
import numpy as np
import helper
import itertools as it

def assign(peaks, lines, bw):
    ind_l = np.array([], dtype=int)
    ind_p = np.array([], dtype=int)
    i = 0
    for x in lines: 
        a = np.where(np.abs( peaks[:,0] - x) < bw)[0]
        if len(a) == 1:
            ind_l = np.append(ind_l, i)
            ind_p = np.append(ind_p, a)
        if len(a) > 1:
            h = np.where(abs(peaks[a,0] - x) == min(abs(peaks[a,0]-x)))[0]
            if len(h) > 1:
                h = h[0]
            #h = np.where(peaks[a,1] == max(peaks[a,1]))[0]

            ind_l = np.append(ind_l, i)
            ind_p = np.append(ind_p, a[h])
        i+=1
        
    return ind_p, ind_l 

def assign_c(peaks, freq, bw):
    for x in lines: 
        a = np.where(np.abs( peaks[:,0] - x) < bw)[0]

    return a

def compare(A, B, C, dipA, dipB, dipC, peaks, bw, cutoff):
    a = spfit.spfitAsym(A=A, B=B, C=C, dipA=dipA, dipB=dipB, dipC=dipC)
    a.write_var()
    a.int_params['flags'] = '0000'
    a.int_params['fend'] = 20

    a.write_int()
    a.cat_run()
    a.read_cat()
    
    if cutoff < 0:
        lines = np.where(a.cat_content['lgint'] > cutoff)[0]

    else:
        lines = a.cat_content['lgint'][np.argsort(a.cat_content['lgint'])][-cutoff:]
        lines = np.where(a.cat_content['lgint'] > min(lines))[0]
        
    
    return assign(peaks, a.cat_content['freq'][lines], bw), a.cat_content, lines


def ABC_SCAN(A_R, B_R, C_R, dipA, dipB, dipC, peaks, bw, cutoff, N):
    result = np.array([])
    i = 0
    j=10
    print '\nScan, bw='+str(bw)+' MHz, Int. cutoff='+str(cutoff) + ', N. of lines='+str(N)
    p = helper.ProgressBar(len(A_R)*len(B_R)*len(C_R))

    for A in A_R:
        for B in B_R:
            for C in C_R:
                
                (ind_p, ind_l), a, lines = compare(A, B, C, dipA, dipB, dipC, peaks, bw, cutoff)

                if len(ind_p) > N:
                    fitness = np.sum((a['freq'][lines][ind_l]-peaks[ind_p])**2)/len(ind_p)**2                     
                    result = np.vstack((result, np.array([fitness, len(ind_p), A, B, C]))) if len(result) > 0 else np.array([fitness, len(ind_p), A, B, C])
                if np.mod(i, j) == 0:
                    p.animate(i+j)
                i+=1

    return result[:,2:], result[:,:2]

def ABC_SCANR(A_R, B_R, C_R, dipA, dipB, dipC, peaks, bw, cutoff, N_min, N_max=100):
    result = np.array([])
    i = 0
    j=10
    print '\nScan, bw='+str(bw)+' MHz, Int. cutoff='+str(cutoff) + ', N. of lines='+str(N_min)
    p = helper.ProgressBar(len(A_R))
    for A in A_R:
        (ind_p, ind_l), cat, lines = compare(A, B_R[i], C_R[i], dipA, dipB, dipC, peaks, bw, cutoff)
        
        if len(ind_p) >= N_max:
            ind_h = np.argsort(cat['lgint'][lines][ind_l])
            ind_p = ind_p[ind_h]
            ind_l = ind_l[ind_h]
            dump, ind_h = np.unique(ind_p, return_index=True)
            ind_l = ind_l[ind_h]
            ind_p = ind_p[ind_h]
            intens = cat['lgint'][lines][ind_l][np.argsort(cat['lgint'][lines][ind_l])][-N_max:]
            ind_i = np.where(cat['lgint'][lines][ind_l] >= min(intens))[0][:N_max]
                
        else:
            ind_i = np.arange(len(ind_p))

        if len(ind_p) >= N_min:
            fitness = np.sum((cat['freq'][lines][ind_l][ind_i]-peaks[ind_p][ind_i][:,0])**2)/len(ind_i)**2
            result = np.vstack((result, np.array([fitness, len(ind_i), A, B_R[i], C_R[i]]))) if len(result) > 0 else np.array([fitness, len(ind_i),A, B_R[i], C_R[i]])
        if np.mod(i, j) == 0:
            p.animate(i+j)
        i+=1
        
    return result[:,2:], result[:,:2]

def create_grid(A, B, C, bw, cube_size, sort=False):

    N = int(bw/cube_size)/2. - 0.5
    A_D = np.arange(A-N*cube_size, A+N*cube_size+0.01, cube_size)
    B_D = np.arange(B-N*cube_size, B+N*cube_size+0.01, cube_size)
    C_D = np.arange(C-N*cube_size, C+N*cube_size+0.01, cube_size)
    A_D, B_D, C_D = np.meshgrid(A_D, B_D, C_D)
    A_D = A_D.flatten(); B_D = B_D.flatten(); C_D = C_D.flatten()

    
    if sort == True:
        h = np.sqrt((A_D-A)**2 + (B_D-B)**2 + (C_D-C)**2)
        h = np.argsort(h)
        A_D = A_D[h]; B_D = B_D[h], C_D = C_D[h] 
    
    return A_D, B_D, C_D

def create_seed(A, B, C, delta, step):
    A_H = np.array([])
    B_H = np.array([])
    C_H = np.array([])
    
    for i in np.arange(len(A)):
        A_D = np.arange(A[i]-delta, A[i]+delta+0.01, step)
        B_D = np.arange(B[i]-delta, B[i]+delta+0.01, step)
        C_D = np.arange(C[i]-delta, C[i]+delta+0.01, step)
        
        A_D, B_D, C_D = np.meshgrid(A_D, B_D, C_D)
        
        A_H = np.append(A_H, A_D.flatten())
        B_H = np.append(B_H, B_D.flatten())
        C_H = np.append(C_H, C_D.flatten())
    return A_H, B_H, C_H
    
def create_random_seed(A, B, C, delta, N):
    A_H = (np.random.rand(N)*2.-1.)*delta + A
    B_H = (np.random.rand(N)*2.-1.)*delta + B
    C_H = (np.random.rand(N)*2.-1.)*delta + C

    return check_const(A_H, B_H, C_H)

def check_const(A,B,C):
    
    h = B-C; h = np.where(h[:] > 0)[0]; A = A[h]; B = B[h]; C=C[h]
    h = A-C; h = np.where(h[:] > 0)[0]; A = A[h]; B = B[h]; C=C[h]
    h = A-B; h = np.where(h[:] > 0)[0]; A = A[h]; B = B[h]; C=C[h]

    h = np.where(A[:] > 0)[0]; A = A[h]; B = B[h]; C=C[h]
    h = np.where(B[:] > 0)[0]; A = A[h]; B = B[h]; C=C[h]
    h = np.where(C[:] > 0)[0]; A = A[h]; B = B[h]; C=C[h]

    return A, B, C
    
def refine(par, dx):
    
    par_h = np.array([0,0,0])

    for x in par:
        par_h = np.vstack((par_h, [x[0]+dx, x[1], x[2]]))
        par_h = np.vstack((par_h, [x[0], x[1]+dx, x[2]]))
        par_h = np.vstack((par_h, [x[0], x[1], x[2]+dx]))
#        par_h = np.vstack((par_h, [x[0]+dx, x[1]+dx, x[2]]))
#        par_h = np.vstack((par_h, [x[0]+dx, x[1], x[2]+dx]))
#        par_h = np.vstack((par_h, [x[0], x[1]+dx, x[2]+dx]))
#        par_h = np.vstack((par_h, [x[0]+dx, x[1]+dx, x[2]+dx]))
        par_h = np.vstack((par_h, [x[0]-dx, x[1], x[2]]))
        par_h = np.vstack((par_h, [x[0], x[1]-dx, x[2]]))
        par_h = np.vstack((par_h, [x[0], x[1], x[2]-dx]))
#        par_h = np.vstack((par_h, [x[0]-dx, x[1]-dx, x[2]]))
#        par_h = np.vstack((par_h, [x[0]-dx, x[1], x[2]-dx]))
#        par_h = np.vstack((par_h, [x[0], x[1]-dx, x[2]-dx]))
#        par_h = np.vstack((par_h, [x[0]-dx, x[1]-dx, x[2]-dx]))

    return par_h[1:]


def fit(seed, dipA, dipB, dipC, peaks, bw, cutoff, N_min, N_max):

    for y in bw:

        fitness = np.array([])
        par = np.array([])
        i = 0
        j=10
        print '\nFit, bw='+str(y)+' MHz, Int. cutoff='+str(cutoff) + ', N. of lines='+str(N_min) +' - '+str(N_max) 
        p = helper.ProgressBar(len(seed))
        for x in seed:

            (ind_p, ind_l), cat, lines = compare(A=x[0], B=x[1], C=x[2], dipA=dipA, dipB=dipB, dipC=dipC, peaks=peaks, bw=y, cutoff=cutoff)
            
            if np.mod(i, j) == 0:
                p.animate(i+j)
            i+=1
            if len(ind_p) >= N_max:
                ind_h = np.argsort(cat['lgint'][lines][ind_l])
                ind_p = ind_p[ind_h]
                ind_l = ind_l[ind_h]
                dump, ind_h = np.unique(ind_p, return_index=True)
                ind_l = ind_l[ind_h]
                ind_p = ind_p[ind_h]
                intens = cat['lgint'][lines][ind_l][np.argsort(cat['lgint'][lines][ind_l])][-N_max:]
                ind_i = np.where(cat['lgint'][lines][ind_l] >= min(intens))[0][:N_max]
                
            else:
                ind_i = np.arange(len(ind_p))
            
            if len(ind_p) >= N_min:

                a = spfit.spfitAsym(fname = 'test_fit', A=x[0], B=x[1], C=x[2], dipA=dipA, dipB=dipB, dipC=dipC)

                a.lin_content['freq'] = peaks[ind_p][:,0][ind_i]
                a.lin_content['qn'] = spfit.convert_qn_cat_lin(cat['qn'][lines][ind_l][ind_i])
                a.par_params['errtst'] = y/0.02

                a.write_par()
                a.write_int()
                a.write_lin()
                a.fit_run()
                a.read_fit()
                a.read_var()

                if a.fit_content['err_mw_rms'] > 0.001 and a.fit_content['err_new_rms'] < 1000.:

                    fitness = np.vstack((fitness, np.array([a.fit_content['err_mw_rms'], len(ind_p)]))) if len(fitness) > 0 else np.array([a.fit_content['err_mw_rms'], len(ind_p)])
                    par = np.vstack((par, a.var_fitpar['par'])) if len(par) > 0 else np.array([a.var_fitpar['par']])

        seed = np.array(par)
    
    return par, fitness
    
def fit_c(seed, dipA, dipB, dipC, peaks, bw, cutoff, N_min, N_max):

    for y in bw:

        fitness = np.array([])
        par = np.array([])
        i = 0
        j=10
        print '\nFit, bw='+str(y)+' MHz, Int. cutoff='+str(cutoff) + ', N. of lines='+str(N_min) +' - '+str(N_max) 
        p = helper.ProgressBar(len(seed))
        for x in seed:

            (ind_p, ind_l), cat, lines = compare(A=x[0], B=x[1], C=x[2], dipA=dipA, dipB=dipB, dipC=dipC, peaks=peaks, bw=y, cutoff=cutoff)
            
            if np.mod(i, j) == 0:
                p.animate(i+j)
            i+=1
            if len(ind_p) >= N_max:
                ind_h = np.argsort(cat['lgint'][lines][ind_l])
                ind_p = ind_p[ind_h]
                ind_l = ind_l[ind_h]
                dump, ind_h = np.unique(ind_p, return_index=True)
                ind_l = ind_l[ind_h]
                ind_p = ind_p[ind_h]
                intens = cat['lgint'][lines][ind_l][np.argsort(cat['lgint'][lines][ind_l])][-N_max:]
                ind_i = np.where(cat['lgint'][lines][ind_l] >= min(intens))[0][:N_max]
                
            else:
                ind_i = np.arange(len(ind_p))
                
            
            if len(ind_p) >= N_min:
                
                for f in it.combinations([0,1,2,3,4],4):
                    
                    x1 = peaks[np.where(np.abs( peaks[:,0] - cat['freq'][lines][ind_l][ind_i][list(f)][0]) < y)[0], 0]
                    x2 = peaks[np.where(np.abs( peaks[:,0] - cat['freq'][lines][ind_l][ind_i][list(f)][1]) < y)[0], 0]
                    x3 = peaks[np.where(np.abs( peaks[:,0] - cat['freq'][lines][ind_l][ind_i][list(f)][2]) < y)[0], 0]
                    x4 = peaks[np.where(np.abs( peaks[:,0] - cat['freq'][lines][ind_l][ind_i][list(f)][3]) < y)[0], 0]
                    
                    for l in x1:
                        for m in x2:
                            for n in x3:
                                for o in x4:
                                    a = spfit.spfitAsym(fname = 'test_fit', A=x[0], B=x[1], C=x[2], dipA=dipA, dipB=dipB, dipC=dipC)
                                    
                                    a.lin_content['freq'] = np.array([l, m, n, o])
                                    a.lin_content['qn'] = spfit.convert_qn_cat_lin(cat['qn'][lines][ind_l][ind_i][list(f)])
                                    a.par_params['errtst'] = y/0.02
                    
                                    a.write_par()
                                    a.write_int()
                                    a.write_lin()
                                    a.fit_run()
                                    a.read_fit()
                                    a.read_var()
                                    
                                    if a.fit_content['err_mw_rms'] > 0.001 and a.fit_content['err_new_rms'] < 10.:
                                        fitness = np.vstack((fitness, np.array([a.fit_content['err_mw_rms'], len(ind_p)]))) if len(fitness) > 0 else np.array([a.fit_content['err_mw_rms'], len(ind_p)])
                                        par = np.vstack((par, a.var_fitpar['par'])) if len(par) > 0 else np.array([a.var_fitpar['par']])
                                    
            seed = np.array(par)
                        
        return par, fitness

def fit_d(seed, dipA, dipB, dipC, peaks, bw, cutoff, N_min, N_max):

    for y in bw:

        fitness = np.array([])
        par = np.array([])
        i = 0
        j=1
        print '\nFit, bw='+str(y)+' MHz, Int. cutoff='+str(cutoff) + ', N. of lines='+str(N_min) +' - '+str(N_max) 
        p = helper.ProgressBar(len(seed))
        for x in seed:

            (ind_p, ind_l), cat, lines = compare(A=x[0], B=x[1], C=x[2], dipA=dipA, dipB=dipB, dipC=dipC, peaks=peaks, bw=y, cutoff=cutoff)
            
            if np.mod(i, j) == 0:
                p.animate(i+j)
            i+=1
            if len(ind_p) >= N_min:

                ind_h = np.argsort(cat['lgint'][lines][ind_l])
                ind_p = ind_p[ind_h]
                ind_l = ind_l[ind_h]
                dump, ind_h = np.unique(ind_p, return_index=True)
                ind_l = ind_l[ind_h]
                ind_p = ind_p[ind_h]
                intens = cat['lgint'][lines][ind_l][np.argsort(cat['lgint'][lines][ind_l])][-N_max:]
                ind_i = np.where(cat['lgint'][lines][ind_l] >= min(intens))[0][:N_max]
                
                np.random.shuffle(ind_i)
                ind_i=ind_i[:4]

            
            if len(ind_p) >= N_min:
                                    
                h1 = np.where(np.abs( peaks[:,0] - cat['freq'][lines][ind_l][ind_i][0]) < y)[0]
                h2 = np.argsort(peaks[h1][:,1]); x1 = peaks[h1][h2][-4:,0]
                h1 = np.where(np.abs( peaks[:,0] - cat['freq'][lines][ind_l][ind_i][1]) < y)[0]
                h2 = np.argsort(peaks[h1][:,1]); x2 = peaks[h1][h2][-4:,0]
                h1 = np.where(np.abs( peaks[:,0] - cat['freq'][lines][ind_l][ind_i][2]) < y)[0]
                h2 = np.argsort(peaks[h1][:,1]); x3 = peaks[h1][h2][-4:,0]
                h1 = np.where(np.abs( peaks[:,0] - cat['freq'][lines][ind_l][ind_i][3]) < y)[0]
                h2 = np.argsort(peaks[h1][:,1]); x4 = peaks[h1][h2][-4:,0]
                
                for l in x1:
                    for m in x2:
                        for n in x3:
                            for o in x4:
                                a = spfit.spfitAsym(fname = 'test_fit', A=x[0], B=x[1], C=x[2], dipA=dipA, dipB=dipB, dipC=dipC)
                                
                                a.lin_content['freq'] = np.array([l, m, n, o])
                                a.lin_content['qn'] = spfit.convert_qn_cat_lin(cat['qn'][lines][ind_l][ind_i])
                                a.par_params['errtst'] = y/0.02
                                a.par_params['nitr'] = 20
                
                                a.write_par()
                                a.write_int()
                                a.write_lin()
                                a.fit_run()
                                a.read_fit()
                                a.read_var()
                                
                                if a.fit_content['err_mw_rms'] > 0.001 and a.fit_content['err_new_rms'] < 100.:
                                    fitness = np.vstack((fitness, np.array([a.fit_content['err_mw_rms'], len(ind_p)]))) if len(fitness) > 0 else np.array([a.fit_content['err_mw_rms'], len(ind_p)])
                                    par = np.vstack((par, a.var_fitpar['par'])) if len(par) > 0 else np.array([a.var_fitpar['par']])
                                    
            seed = np.array(par)
                        
        return par, fitness


def fit_b(seed, dipA, dipB, dipC, peaks, bw, cutoff, N_min, N_max):

    for y in bw:

        fitness = np.array([])
        par = np.array([])
        i = 0
        j=10
        print '\nFit, bw='+str(y)+' MHz, Int. cutoff='+str(cutoff) + ', N. of lines='+str(N_min) +' - '+str(N_max) 
        p = helper.ProgressBar(len(seed))
        for x in seed:

            (ind_p, ind_l), cat, lines = compare(A=x[0], B=x[1], C=x[2], dipA=dipA, dipB=dipB, dipC=dipC, peaks=peaks, bw=y, cutoff=cutoff)
            
            if np.mod(i, j) == 0:
                p.animate(i+j)
            i+=1
            if len(ind_p) >= N_max:
                ind_h = np.argsort(cat['lgint'][lines][ind_l])
                ind_p = ind_p[ind_h]
                ind_l = ind_l[ind_h]
                dump, ind_h = np.unique(ind_p, return_index=True)
                ind_l = ind_l[ind_h]
                ind_p = ind_p[ind_h]
                intens = cat['lgint'][lines][ind_l][np.argsort(cat['lgint'][lines][ind_l])][-N_max:]
                ind_i = np.where(cat['lgint'][lines][ind_l] >= min(intens))[0][:N_max]
                                    
            else:
                ind_i = np.arange(len(ind_p))
                
            
            if len(ind_p) >= N_min:
                                
                for l in it.combinations([0,1,2,3,4,5,6,7],5):
                    a = spfit.spfitAsym(fname = 'test_fit', A=x[0], B=x[1], C=x[2], dipA=dipA, dipB=dipB, dipC=dipC)
                    
                    a.lin_content['freq'] = peaks[ind_p][:,0][ind_i][list(l)]
                    a.lin_content['qn'] = spfit.convert_qn_cat_lin(cat['qn'][lines][ind_l][ind_i][list(l)])
                    a.par_params['errtst'] = y/0.02
    
                    a.write_par()
                    a.write_int()
                    a.write_lin()
                    a.fit_run()
                    a.read_fit()
                    a.read_var()
                    
                    if a.fit_content['err_mw_rms'] > 0.000001:
                        fitness = np.vstack((fitness, np.array([a.fit_content['err_mw_rms'], len(ind_p)]))) if len(fitness) > 0 else np.array([a.fit_content['err_mw_rms'], len(ind_p)])
                        par = np.vstack((par, a.var_fitpar['par'])) if len(par) > 0 else np.array([a.var_fitpar['par']])
                                
        seed = np.array(par)
                    
    return par, fitness


def find_duplicates(par, delta):
    par_h = np.array([])
    i=0
    ind = [0]
    
    for x in np.array(par):
        if len(par_h) > 0:
            sig=(par_h[:,0]-x[0])**2+(par_h[:,1]-x[1])**2+(par_h[:,2]-x[2])**2
            if len(np.where(sig[:] < delta)[0]) == 0:
                par_h = np.vstack((par_h, x))
                ind.append(i)
        else:
            par_h = np.array([x])
        
        i+=1
        
    return par_h, ind