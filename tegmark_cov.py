import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
from scipy.special import legendre, lpmv
#'''
nside = 8
npix = hp.nside2npix(nside)
dirs = hp.pix2ang(nside, np.arange(npix))
z = np.zeros((npix, npix))
for i in range(npix):
    for j in range(i+1):
        d1 = np.array([dirs[0][i], dirs[1][i]])
        d2 = np.array([dirs[0][j], dirs[1][j]])
        z[i,j] = np.cos(hp.rotator.angdist(d1,d2))
        z[j,i] = np.cos(hp.rotator.angdist(d2,d1))
#'''
'''
M = np.array([[TT,TQ,TU],[TQ,QQ,QU],[TU,QU,UU]])
R = np.array([[np.ones_like(alphas),0*alphas,0*alphas],
	      [0*alphas,  np.cos(2*alphas), np.sin(2*alphas)], 
              [0*alphas, -np.sin(2*alphas), np.cos(2*alphas)]])
R[np.isnan(R)] = 0

Cov = np.zeros((3*768, 3*768))
for i in range(768):
    for j in range(768):
        Rot = R[:,:,i,j].dot(M[:,:,i,j].dot(R[:,:,j,i].T))
        for k in range(3):
            for l in range(3):
                Cov[i+k*768,j+l*768] = Rot[k,l]
'''
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
        if r % 100 == 0:
            print(r, len_)
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


def test_cov():
    nside = 8
    ell = np.arange(3.*nside)
    ellmax = int(ell.max())
    C_l = np.zeros_like(ell)
    C_l[2:] = 1./ell[2:]**2

    clee_alm = expand_cl2alm(C_l)
    clbb_alm = expand_cl2alm(C_l)
    cl_alm = np.hstack((clee_alm, clbb_alm))
    nalm = (ell.max()+1)**2
    yeb = np.vstack((get_matrix_emode_qu(int(ell.max()), nside), get_matrix_bmode_qu(int(ell.max()), nside)))
    yeb_cl = np.tile(cl_alm, yeb.shape[1]).reshape(yeb.shape[1], 2*int(nalm)).transpose()
    yeb_cl = yeb_cl*yeb
    mat = np.dot(yeb_cl.transpose(), yeb)
    return mat


def test_legendre():
    l = 5
    m = 0
    z = np.linspace(-1, 1, 1000)
    P_l = legendre(l)
    plt.plot(z, P_l(z))
    plt.plot(z, lpmv(m, l, z))
    plt.show()

    plt.figure()
    for m in range(-l, l+1):
        plt.plot(z, lpmv(m, l, z), label=m)
    plt.show()

    plt.figure()
    plt.plot(z, P(2, z))
    plt.plot(z, P2(2, z))
    plt.show()

def test_fs():
    plt.figure()
    z = np.linspace(-1, 1, 1000)
    l = 3
    plt.plot(z, P(l, z), label=r'$P_\ell(z)$')
    plt.plot(z, F10(l, z), label=r'$F_\ell^{10}(z)$')
    plt.plot(z, F12(l, z), label=r'$F_\ell^{12}(z)$')
    plt.plot(z, F22(l, z), label=r'$F_\ell^{22}(z)$')
    plt.legend(loc='best')
    plt.show()

    plt.figure()
    l = 2
    plt.plot(z, P(l, z), label=r'$P_\ell(z)$')
    plt.plot(z, F10(l, z), label=r'$F_\ell^{10}(z)$')
    plt.plot(z, F12(l, z), label=r'$F_\ell^{12}(z)$')
    plt.plot(z, F22(l, z), label=r'$F_\ell^{22}(z)$')
    plt.legend(loc='best')
    plt.show()

    return


def R_pixel(alpha):

    # For a single pixel...
    R = np.array([[1,0,0],
        [0, np.cos(2*alpha), np.sin(2*alpha)],
        [0,-np.sin(2*alpha), np.cos(2*alpha)]])
    return R

def get_alpha(r1, r2):
    rij = np.cross(r1, r2)
    rij = rij/np.sqrt(rij.dot(rij))
    z = np.array([0,0,1])
    if np.allclose(rij, np.array([0,0,0])):
        return 0
    ristar = np.cross(z, r1)
    ristar = ristar/np.sqrt(ristar.dot(ristar))
    cos = rij.dot(z)
    if cos > 0:
        alpha = np.arccos(rij.dot(ristar))
    else:
        alpha = -np.arccos(rij.dot(ristar))
    return alpha

def get_Rmat():
    npix = hp.nside2npix(8)
    R = np.zeros((3*npix, 3*npix))
    alpha = alpha_mat()
    plt.imshow(alpha)
    plt.figure()
    R = np.array([[np.ones_like(alpha),np.zeros_like(alpha),np.zeros_like(alpha)],
        [np.zeros_like(alpha), np.cos(2*alpha), np.sin(2*alpha)],
        [np.zeros_like(alpha),-np.sin(2*alpha), np.cos(2*alpha)]])
    #for i in range(len(alphas)):
    #    for j in range(len(alphas)):
    #        R[i:i+3,j:j+3] = R_pixel(alphas[i,j])
    #plt.imshow(R)
    #plt.show()
    return R


def alpha_test():
    d1 = hp.pix2vec(8, 700)
    d2 = hp.pix2vec(8, 600)
    alpha = get_alpha(d1, d2)
    print(alpha)

    alphas = alpha_mat()
    plt.imshow(alphas)
    plt.colorbar()
    plt.show()
    return


def alpha_mat():
    npix = hp.nside2npix(8)
    alphas = np.zeros((npix, npix))
    for i in range(npix):
        for j in range(i):
            d1 = hp.pix2vec(8, i)
            d2 = hp.pix2vec(8, j)
            alphas[i,j] = get_alpha(d1,d2)
            alphas[j,i] = -alphas[i,j]
    return alphas




def F10(l, z):
    t1 = l*z*P(l-1,z)/(1-z**2)
    t2 = (l/(1-z**2) + l*(l-1)/2)*P(l,z)
    den = np.sqrt((l-1)*l*(l+1)*(l+2))
    F = 2*(t1-t2)/den
    F[abs(z) == 1] = 0
    return F

def F12(l, z):
    t1 = (l+2)*z/(1-z**2)*P2(l-1,z)
    t2 = ((l-4)/(1-z**2) + l*(l-1)/2)*P2(l,z)
    den = (l-1)*l*(l+1)*(l+2)
    F = 2*(t1-t2)/den
    F[z == 1] = 0.5
    F[z == -1] = 0.5*(-1)**l
    return F

def F22(l, z):
    t1 = (l+2)*P2(l-1, z)
    t2 = (l-1)*z*P2(l, z)
    den = (l-1)*l*(l+1)*(l+2)*(1-z**2)
    F =  4*(t1-t2)/den
    F[z == 1] = -0.5
    F[z == -1] = 0.5*(-1)**l
    return F

def P(l, z):
    P_l = legendre(l)
    return P_l(z)

def P2(l, z):
    P_l = lpmv(2, l, z)
    return P_l

def test_M(show=False):
    ell = np.arange(3*8.)
    C_l = np.zeros_like(ell)
    C_l[2:] = 1./ell[2:]**2
    TT = TT_mat(C_l) # TT
    TQ = TQ_mat(C_l)  # TE
    TU = TU_mat(C_l*0) # BT
    QQ = QQ_mat(C_l, C_l) # E, B
    UU = UU_mat(C_l, C_l) # E, B
    QU = QU_mat(C_l*0) # EB

    if show:
        plt.subplot(3,3,1)
        plt.imshow(TT, vmin=-0.1, vmax=0.1)

        plt.subplot(332)
        plt.imshow(TQ, vmin=-0.1, vmax=0.1)
        plt.subplot(334)
        plt.imshow(TQ, vmin=-0.1, vmax=0.1)

        plt.subplot(333)
        plt.imshow(TU, vmin=-0.1, vmax=0.1)
        plt.subplot(337)
        plt.imshow(TU, vmin=-0.1, vmax=0.1)

        plt.subplot(336)
        plt.imshow(QU, vmin=-0.1, vmax=0.1)
        plt.subplot(338)
        plt.imshow(QU, vmin=-0.1, vmax=0.1)

        plt.subplot(335)
        plt.imshow(QQ, vmin=-0.1, vmax=0.1)
        plt.subplot(339)
        plt.imshow(UU, vmin=-0.1, vmax=0.1)

        plt.show()

    alphas = alpha_mat()
    M = np.array([[TT,TQ,TU],[TQ,QQ,QU],[TU,QU,UU]])
    R = np.array([[np.ones_like(alphas),0*alphas,0*alphas],
                  [0*alphas,  np.cos(2*alphas), np.sin(2*alphas)], 
                  [0*alphas, -np.sin(2*alphas), np.cos(2*alphas)]])
    R[np.isnan(R)] = 0
    
    Cov = np.zeros((3*768, 3*768))
    for i in range(768):
        for j in range(768):
            Rot = R[:,:,i,j].dot(M[:,:,i,j].dot(R[:,:,j,i].T))
            for k in range(3):
                for l in range(3):
                    Cov[i+k*768,j+l*768] = Rot[k,l]
    return Cov

def TT_mat(C_l):
    ell = np.arange(len(C_l))
    TT = np.zeros_like(z)
    for l in ell[2:]:
        TT += (2*l+1)/(4*np.pi)*C_l[l]*P(l,z)
    return TT

def TQ_mat(C_l):
    ell = np.arange(len(C_l))
    TQ = np.zeros_like(z)
    for l in ell[2:]:
        TQ -= (2*l+1)/(4*np.pi)*F10(l,z)*C_l[l]
    return TQ

def TU_mat(C_l):
    ell = np.arange(len(C_l))
    TU = np.zeros_like(z)
    for l in ell[2:]:
        TU -= (2*l+1)/(4*np.pi)*F10(l,z)*C_l[l]
    return TU

def QQ_mat(C_lE, C_lB):
    ell = np.arange(len(C_lE))
    QQ = np.zeros_like(z)
    for l in ell[2:]:
        QQ += (2*l+1)/(4*np.pi)*(F12(l,z)*C_lE[l] - F22(l,z)*C_lB[l])
    return QQ

def UU_mat(C_lE, C_lB):
    ell = np.arange(len(C_lE))
    UU = np.zeros_like(z)
    for l in ell[2:]:
        UU += (2*l+1)/(4*np.pi)*(F12(l,z)*C_lB[l] - F22(l,z)*C_lE[l])

    return UU

def QU_mat(C_l):
    ell = np.arange(len(C_l))
    QU = np.zeros_like(z)
    for l in ell[2:]:
        QU += (2*l+1)/(4*np.pi)*(F12(l,z) + F22(l,z))*C_l[l]
    return QU

if __name__ == '__main__':
    #test_legendre()
    #test_fs()
    #test_M()
    #alpha_test()
    #R = get_Rmat()
    Cov = test_M()
    plt.imshow(Cov, vmin=-0.1, vmax=0.1)
    plt.show()

    mat = test_cov()
    plt.figure()
    plt.imshow(mat, vmin=-0.1, vmax=0.1)
    plt.show()
