# -*- coding: utf-8 -*-
"""
Based on work: Kantelhardt J.W. [et al.]. Multifractal detrended fluctuation analysis of nonstationary time series. // Physica A. 2002. V. 316. P. 87-114.
All mentioned steps described in this work.
"""

import sys
import math
import numpy as np
from matplotlib.pyplot import subplots
import pandas as pd
import scipy.optimize
import scipy.signal
import scipy.fftpack


def to_profile(Y):  # first step, determine the "profile"
    y_avg = np.mean(Y)  # mean value of Y
    Y_trend = np.cumsum(Y - y_avg)  # "profile"
    return Y_trend


def linfunc(X, Y, power):  # least squares polynomial fitted function
    p = np.polyfit(X, Y, power)
    Yfunc = np.polyval(p, X)
    return Yfunc


def dividing_by_segments(X, Y, S, power):  # second and third steps of algorithm
    N = Y.size
    F = []
    for s in S:  # for each s in S
        Ns = int(np.trunc(N/s))
        Fbuf = np.zeros(2 * Ns)  # F(v, s) ** 2
        for i in range(Ns):  # calculating values of F(v, s) ** 2 from 1 to Ns and from Ns + 1 to 2Ns
            Yn = linfunc(X[i * s : (i + 1) * s], Y[i * s : (i + 1) * s], power)
            Fbuf[i] = sum((Y[i * s : (i + 1) * s] - Yn) ** 2) / s
            Yn = linfunc(X[N - (i + 1) * s : N - i * s], Y[N - (i + 1) * s : N - i * s], power)
            Fbuf[i + Ns] = sum((Y[N - (i + 1) * s : N - i * s] - Yn) ** 2) / s
        F.append(Fbuf)
    return F


def qfunc(F, q):  # calculating q-th order fluctuation function, step 4
    Fq = np.zeros(len(F))
    N = len(F)
    if q != 0:
        for i in range(N):
            Ns = F[i].size
            f_buf = sum(F[i] ** (q / 2))
            Fq[i] = ((f_buf) / Ns) ** (1 / q)
    else:  # calculating for q == 0
        for i in range(N):
            Ns = F[i].size
            f_buf = sum([np.log(x) for x in F[i]])
            Fq[i] = np.exp(f_buf / (2 * Ns))
    return Fq


def Hq(Fq, S):  # calculating exponent h(q)
    Hq = np.zeros_like(Fq[:, 0])
    dHq = np.zeros_like(Fq[:, 0])
    Nf = Fq[:, 0].size
    for i in range(Nf):  # finding a best linear approximation for Fq[i]
        p, cov = np.polyfit(np.log(S), np.log(Fq[i]), deg=1, full=False, cov=True)
        Hq[i] = p[0]
        dHq[i] = np.sqrt(cov[0][0])
    return Hq, dHq


def Boltzman(Q, A1, A2, x0, dx):  # Q, H - variables, A1, A2, x0, dx - constant parameters, calculate Boltzman function
    return (A1 - A2) / (1 + np.exp((Q - x0) / dx)) + A2


def Boltzman_der(Q, A1, A2, x0, dx):  # first derivative of Boltzman function
    return - ((A1 - A2) * np.exp((Q - x0) / dx)) / (dx * (1 + np.exp((Q - x0) / dx)) ** 2)


def Boltzman_fitting(Q, H, sigma=None):  # fit Boltzman function
    return scipy.optimize.curve_fit(Boltzman, Q, H, sigma=sigma)


def H_der(Q, H, dH):  # calculating derivative using finite-difference approximations
    Nh = H.size - 1
    Hder = np.zeros(Nh)
    dHder = np.zeros(Nh)
    for i in range(Nh):
        Hder[i] = (H[i + 1] - H[i]) / (Q[i + 1] - Q[i])
        dHder[i] = math.sqrt(dH[i + 1] ** 2 + dH[i] ** 2) / (Q[i + 1] - Q[i])
    return Hder, dHder


def PSD_trend(X, Y, power):  # removes linear trend from sequence
    return Y - linfunc(X, Y, power)


def Psd(X, Y, k=0):  # calculate PSD and fractal dimension
    N = X.size - 1
    step = 0
    for i in range(N):
        step += X[i + 1] - X[i]
    step /= N
    if k != 0:
        Y = PSD_trend(X, Y, k)
    f, P = scipy.signal.periodogram(Y, step)  # frequencies and PSD
    param, cov = np.polyfit(np.log(f[1:]), np.log(P[1:]), deg=1, cov=True)
    D_error = math.sqrt(cov[0][0])
    LinP = param[0] * np.log(f[1:]) + param[1]  # linear approximation for PSD
    D = (5 + param[0]) / 2  # fractal dimension
    return f, P, LinP, D, D_error


def Boltzman_err(Q, A1, A2, x0, dx, dA1, dA2, dx0, ddx):  # calculates error of Boltzmann approximation
    A1_err = ((1 / (1 + np.exp((Q - x0) / dx))) * dA1) ** 2
    A2_err = ((1 - 1 / (1 + np.exp((Q - x0) / dx))) * dA2) ** 2
    x0_err = (((A1 - A2) * np.exp((Q - x0) / dx)) / (dx * (1 + np.exp((Q - x0) / dx)) ** 2) * dx0) ** 2
    dx_err = (((Q - x0) * (A1 - A2) * np.exp((Q - x0) / dx)) / ((dx ** 2) * (1 + np.exp((Q - x0) / dx)) ** 2) * ddx) ** 2
    return np.sqrt(A1_err + A2_err + x0_err + dx_err)


def WMandelbrot(t, b=1.5, d=1.8, M=50):  # Weierstrass-Mandelbrot function, for testing
    W = np.zeros_like(t)
    for j in range(len(t)):
        for i in range(-M, M + 1):
            W[j] += 1 / (b ** ((2 - d) * i)) * (1 - np.cos(b ** i * t[j]))
    return W


def sequence_type(test_counter, fname):  # choose sequence
    if test_counter == 0:  # sequence from .csv file
        data = pd.read_csv(fname)  # parse profile to numpy arrays, X and Y
        a = data.columns.tolist()
        X_with_str = np.array(data[a[0]][2:])
        Y_with_str = np.array(data[a[1]][2:])
        #X_with_comma = [x.replace(',', '.') for x in X_with_str]
        #Y_with_comma = [y.replace(',', '.') for y in Y_with_str]
        X = np.array([float(x) for x in X_with_str])
        Y = np.array([float(y) for y in Y_with_str])
    if test_counter == 1:  # Weierstrass-Mandelbrot testing sequence
        N = 4000
        t = np.linspace(0, 1, N)
        X = np.arange(0, N)
        Y = WMandelbrot(t)
    if test_counter == 2:  # binomial multifractal series
        n_max = 12
        Nbin = 2 ** n_max  # length of series
        Y = np.zeros(Nbin)
        a = 0.8  # parametr
        for i in range(Nbin):  # series generation
            Y[i] = a ** (bin(i).count('1')) * (1 - a) ** (n_max - bin(i).count('1'))
        X = np.arange(0, Nbin, 1)
    return X, Y


def mfdfa(fname, m, S, Q, trend_counter, test_counter):  # calculates all multifractal characteristics using MFDFA method
    X, Y = sequence_type(test_counter, fname)  # choosing a sequence
    if trend_counter == 0:  # creating the profile
        Y_trend = Y.copy()
    else:
        Y_trend = Y.copy()
        for count in range(trend_counter):
            Y_trend = to_profile(Y_trend)
    Fq = np.zeros((Q.size, S.size))  # create Fq array with zeros, equation (4)
    Nq = Q.size
    F = dividing_by_segments(X, Y_trend, S, m)  # performing steps 2 and 3
    for i in range(Nq):  # calculating q-th order fluctuation functions
        Fq[i] = qfunc(F, Q[i])
    H, dH = Hq(Fq, S)  # calculating h(q) exponents
    Fq2 = np.zeros((2, S.size))  # simple DFA method, calculate H = h(2)
    Fq2[0] = qfunc(F, 0)
    Fq2[1] = qfunc(F, 2)
    H02, dH02 = Hq(Fq2, S)
    H2 = H02[1]
    dH2 = dH02[1]
    '''
    if H2 - dH2 > 1.:
        H = H - 1
        H2 = H2 - 1
    '''
    Fq1 = np.zeros((2, S.size))
    Fq1[0] = qfunc(F, 0)
    Fq1[1] = qfunc(F, 1)
    H01, dH01 = Hq(Fq1, S)
    H1 = H01[1]
    dH1 = dH01[1]
    T = Q * H - 1  # calculating tau(q)
    Hder, dHder = H_der(Q, H, dH)  # calculating number derivative of h(q) and its error
    Alp = H[:-1] + Q[:-1] * Hder  # calculating alpha
    dAlp = np.sqrt(dH[:-1] ** 2)  # calculating error of alpha
    Falp = Q[:-1] * (Alp - H[:-1]) + 1  # calculating f(alpha)
    dFalp = Q[:-1] * np.sqrt(dAlp ** 2 + dH[:-1] ** 2)  # and its error
    return X, Y, Fq, F, H, dH, H2, dH2, H1, dH1, T, Alp, dAlp, Falp, dFalp


def plotting(S, Q, test_counter, X, Y, Fq, H, dH, T, Alp, dAlp, Falp, dFalp, k, f, P, LinP, D, D_error):  # plots calculated characteristics
    fig, ax = subplots(3, 2, figsize=(12, 12))
    ax[0][0].loglog(S, Fq.T)
    ax[0][0].set_xlabel('$log(s)$')
    ax[0][0].set_ylabel('$log(F_{q}(s))$')
    ax[0][0].legend()
    ax[0][1].errorbar(Q, H, yerr=dH, label='$h_{exp}(q)$')
    if test_counter == 0:
        p = Boltzman_fitting(Q, H, dH)
        ax[0][1].errorbar(Q, Boltzman(Q, p[0][0], p[0][1], p[0][2], p[0][3]), yerr=Boltzman_err(Q, p[0][0], p[0][1], p[0][2], p[0][3], np.sqrt(p[1][0][0]), np.sqrt(p[1][1][1]), np.sqrt(p[1][2][2]), np.sqrt(p[1][3][3])), label='$h_{th}(q)$')
    ax[0][1].set_xlabel('$q$')
    ax[0][1].set_ylabel('$h(q)$')
    ax[0][1].legend()
    ax[1][0].plot(Q, T)
    ax[1][0].set_xlabel('$q$')
    ax[1][0].set_ylabel('$\\tau(q)$')
    ax[1][0].legend()
    ax[1][1].plot(Alp, Falp, 'ks')
    ax[1][1].set_xlabel('$\\alpha$')
    ax[1][1].set_ylabel('$f(\\alpha)$')
    ax[1][1].legend()
    ax[2][0].loglog(f[1:], P[1:])
    ax[2][0].loglog(f[1:], np.exp(LinP), label='D_psd = ' + str(round(D, 3)) + '$\pm$' + str(round(D_error, 3)))
    ax[2][0].set_xlabel('$log(f)$')
    ax[2][0].set_ylabel('$log(PSD)$')
    ax[2][0].legend()
    ax[2][1].plot(X, Y, label='Original data')
    ax[2][1].plot(X, PSD_trend(X, Y, k), label='Detrended data')
    ax[2][1].set_xlabel('X')
    ax[2][1].set_ylabel('Y')
    ax[2][1].legend()


def main(args):
    fname = 'binomial.csv'
    m = 4  # order of detrending polynomial
    S = np.arange(50, 160, 10)  # sequence of segments
    Q = np.arange(-10, 10., 0.5)  # sequence of orders of fluctiation function
    trend_counter = 1  # times of calculating "profile" of data, 0 is recomended for fBm-sequences, 1 - for fGn-sequences, 2 and higher - if h(q) < 0
    test_counter = 0  # type of sequence
    X, Y, Fq, F, H, dH, H2, dH2, H1, dH1, T, Alp, dAlp, Falp, dFalp = mfdfa(fname, m, S, Q, trend_counter, test_counter)
    k = 2  # order of detrending polynomial in PSD-method
    f, P, LinP, D, D_error = Psd(X, Y, k)  # calculating fractal dimension from PSD
    plotting(S, Q, test_counter, X, Y, Fq, H, dH, T, Alp, dAlp, Falp, dFalp, k, f, P, LinP, D, D_error)
    fl = open(fname[:fname.rfind('.')] + '_PSD' + '.txt', 'w')  # saving fractal dimension and Hurst exponent in fname_PSD.txt file
    fl.write('D_psd = ' + str(round(D, 3)) + '$\pm$' + str(round(D_error, 3)) + '\n')
    fl.write('H(2)_mfdfa = ' + str(round(H2, 3)) + '$\pm$' + str(round(dH2, 3)) + '\n')
    fl.close()
    print('D_psd = ' + str(round(D, 3)) + '$\pm$' + str(round(D_error, 3)) + '\n')
    print('H(2)_mfdfa = ' + str(round(H2, 3)) + '$\pm$' + str(round(dH2, 3)) + '\n')
    Fqt = Fq.T
    # saving all calculated data in fname_MFDFA.csv file
    df_s_and_Fq = pd.DataFrame(np.concatenate((np.reshape(S, (len(S), 1)), Fqt), axis=1))
    df_s_and_Fq_names = ['s']
    for q in Q:
        df_s_and_Fq_names.append('F(' + str(q) + ')(s)')
    df_s_and_Fq.columns = np.array(df_s_and_Fq_names)
    df_q_and_Hq = pd.DataFrame(np.array([Q, H, dH, T]).T)
    df_q_and_Hq_names = ['q', 'h(q)', 'dh(q)', 'tau(q)']
    df_q_and_Hq.columns = np.array(df_q_and_Hq_names)
    df_a_and_Fa = pd.DataFrame(np.array([Alp, dAlp, Falp, dFalp]).T)
    df_a_and_Fa.columns = np.array(['alpha', 'd_alpha', 'f(alpha)', 'd_f(alpha)'])
    df_f_and_P = pd.DataFrame(np.array([f, P]).T)
    df_f_and_P.columns = np.array(['f', 'P'])
    df_sum = pd.concat([df_s_and_Fq, df_q_and_Hq, df_a_and_Fa, df_f_and_P], axis=1)
    df_sum.to_csv(fname[:fname.rfind('.')] + '_MFDFA' + '.csv', index=False, index_label=False)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
