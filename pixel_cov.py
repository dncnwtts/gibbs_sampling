import numpy as np
import matplotlib.pyplot as plt
import healpy as hp

nside = 8
npix = hp.nside2npix(nside)
ellmax = 3*nside-1

import real_alm as rlm
def get_matrix_emode_qu(ellmax, nside):
    len_ = (ellmax + 1)**2
    npix = hp.nside2npix(nside)
    matrix = np.zeros((len_, 3*npix), dtype='double')
    for r in range(len_):
        re_alm = np.zeros(len_, dtype='double')
        re_alm[r] = 1.0
        cx_alm = rlm.real2complex(re_alm)
        tuple_ = (cx_alm * 0, cx_alm, cx_alm * 0)    # Zero the T,B modes
        harmonic_tqu = hp.sphtfunc.alm2map(tuple_, nside, verbose=False,
                pixwin=True)
        matrix[r,0*npix:npix]   = harmonic_tqu[0] # I Stokes parameter
        matrix[r,1*npix:2*npix] = harmonic_tqu[1] # Q Stokes parameter
        matrix[r,2*npix:3*npix] = harmonic_tqu[2] # U Stokes parameter
    return matrix

def get_matrix_bmode_qu(ellmax, nside):
    len_ = (ellmax + 1)**2
    npix = hp.nside2npix(nside)
    matrix = np.zeros((len_, 3*npix), dtype='double')
    for r in range(len_):
        re_alm = np.zeros(len_, dtype='double')
        re_alm[r] = 1.0
        cx_alm = rlm.real2complex(re_alm)
        tuple_ = (cx_alm * 0, cx_alm * 0, cx_alm)    # Zero the E modes
        harmonic_tqu = hp.sphtfunc.alm2map(tuple_, nside, verbose=False, pixwin=True)
        matrix[r,0*npix:npix]   = harmonic_tqu[0] # I Stokes parameter
        matrix[r,1*npix:2*npix] = harmonic_tqu[1] # Q Stokes parameter
        matrix[r,2*npix:3*npix] = harmonic_tqu[2] # U Stokes parameter
    return matrix


def get_matrix_tmode_ii(ellmax, nside):
    len_ = (ellmax + 1)**2
    npix = hp.nside2npix(nside)
    matrix = np.zeros((len_, 3*npix), dtype='double')
    for r in range(len_):
        re_alm = np.zeros(len_, dtype='double')
        re_alm[r] = 1.0
        cx_alm = rlm.real2complex(re_alm)
        tuple_ = (cx_alm, cx_alm * 0, cx_alm * 0)    # Zero the E/B modes
        harmonic_tqu = hp.sphtfunc.alm2map(tuple_, nside, verbose=False, pixwin=True)
        matrix[r,0*npix:npix]   = harmonic_tqu[0] # I Stokes parameter
        matrix[r,1*npix:2*npix] = harmonic_tqu[1] # Q Stokes parameter
        matrix[r,2*npix:3*npix] = harmonic_tqu[2] # U Stokes parameter
    return matrix

def expand_cl2alm(cl):
    ellmax = cl.shape[0] - 1
    len_ = (ellmax + 1)**2
    ell, m = rlm.r2lm(np.arange(len_))
    return cl[ell]



YT = get_matrix_tmode_ii(ellmax, nside)
YE = get_matrix_emode_qu(ellmax, nside)
YB = get_matrix_bmode_qu(ellmax, nside)

YTI = YT[:,:npix]
YEQ, YEU = np.split(YE[:,npix:],2, axis=1)
YBQ, YBU = np.split(YB[:,npix:],2, axis=1)

def mat_from_cl(Cl_array):
    TT, EE, BB, TE, TB, EB = Cl_array
    TT_alm = np.diag(expand_cl2alm(TT))
    TE_alm = np.diag(expand_cl2alm(TE))
    EE_alm = np.diag(expand_cl2alm(EE))
    BB_alm = np.diag(expand_cl2alm(BB))
    TB_alm = np.diag(expand_cl2alm(TB))
    EB_alm = np.diag(expand_cl2alm(EB))

    II = YTI.T.dot(TT_alm.dot(YTI)) ## TT only
    IQ = YTI.T.dot(TE_alm.dot(YEQ) + TB_alm.dot(YBQ)) # TE, TB
    IU = YTI.T.dot(TE_alm.dot(YEU) + TB_alm.dot(YBU)) # TE, TB
    QI = IQ.T
    UI = IU.T
    QQ = YEQ.T.dot(EE_alm.dot(YEQ)) + YBQ.T.dot(EB_alm.dot(YEQ)) +\
         YEQ.T.dot(EB_alm.dot(YBQ)) + YBQ.T.dot(BB_alm.dot(YBQ))
    QU = YEQ.T.dot(EE_alm.dot(YEU)) + YBQ.T.dot(EB_alm.dot(YEU)) +\
         YEQ.T.dot(EB_alm.dot(YBU)) + YBQ.T.dot(BB_alm.dot(YBU))
    UQ = QU
    UU = YEU.T.dot(EE_alm.dot(YEU)) + YBU.T.dot(EB_alm.dot(YEU)) +\
         YEU.T.dot(EB_alm.dot(YBU)) + YBU.T.dot(BB_alm.dot(YBU))
    
    M = np.vstack( (np.hstack((II,IQ,IU)),
                    np.hstack((QI,QQ,QU)),
                    np.hstack((UI,UQ,UU))))
    return M

ell = np.arange(ellmax+1.)
C_l = np.zeros_like(ell)
C_l[2:] = 1./ell[2:]**2


Cl_array = np.array([C_l, 0*C_l, 0*C_l, 0*C_l, 0*C_l, 0*C_l])

M = mat_from_cl(Cl_array)
plt.imshow(M, vmin=-0.1, vmax=0.1, cmap='seismic')
plt.colorbar()
plt.show()

# Test that things are sort of right

Cl_array = np.array([C_l, 0.1*C_l, 0.001*C_l, 0.3*C_l, 0*C_l, 0*C_l])
M = mat_from_cl(Cl_array)
plt.imshow(M, vmin=-0.1, vmax=0.1, cmap='seismic')
plt.colorbar()
plt.show()

mu = np.zeros(3*npix)
map_ = np.random.multivariate_normal(mu, M)
plt.figure()
I,Q,U = np.split(map_, 3)
m = np.array([I,Q,U])
hp.mollview(I, sub=131)
hp.mollview(Q, sub=132)
hp.mollview(U, sub=133)
plt.show()

plt.figure()

clhat = hp.anafast(m)

for i, cl in enumerate(clhat):
    plt.plot(ell[2:], cl[2:], '.', color='C{0}'.format(i))
for i, cl in enumerate(Cl_array):
    plt.plot(ell[2:], cl[2:]*hp.pixwin(8)[2:24]**2, color='C{0}'.format(i))
plt.ylim([1e-7, 1])
plt.yscale('log')

