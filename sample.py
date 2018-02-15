import matplotlib.pyplot as plt
import numpy as np
import healpy as hp

from glob import glob


from scipy.stats import invgamma, invwishart
nside = 8
npix = hp.nside2npix(nside)
dirs = hp.pix2ang(nside, np.arange(npix))
z = np.zeros((npix, npix))
for i in range(npix):
    for j in range(i+1):
        d1 = np.array([dirs[0][i], dirs[1][i]])
        d2 = np.array([dirs[0][j], dirs[1][j]])
        z[i,j] = np.cos(hp.rotator.angdist(d1,d2))
        z[j,i] = z[i,j]

mask = hp.read_map('class_mask.fits')
mask = hp.ud_grade(mask, 8)
mask = np.where(mask < 0.5, 0, 1)

def test1():
    np.random.seed(0)
    nside = 8
    ell = np.arange(3.*nside)
    Cltrue = np.zeros_like(ell)
    Cltrue[2:] = 1/ell[2:]**2
    m = hp.synfast(Cltrue, nside)
    Clhat = hp.anafast(m)
    
    sigma_l = (2*ell+1)*Clhat
    sigma_l = Clhat
    
    Cl = np.linspace(5e-3,5,1000)
    
    sl = sigma_l[2]
    l = ell[2]
    y = np.exp(-(2*l+1)*sl/(2*Cl))/np.sqrt(Cl**(2*l+1))
    plt.semilogx(Cl, y/y.max())
    alpha = (2*l-1)/2
    beta = (2*l+1)*sl/2
    pdf = invgamma.pdf(Cl, alpha, scale=beta)
    plt.semilogx(Cl, pdf/pdf.max())
    pdf = invwishart.pdf(Cl, df=alpha*2, scale=beta*2)
    plt.semilogx(Cl, pdf/pdf.max())
    plt.axvline(Clhat[2])
    plt.axvline(Cltrue[2], color='k', linestyle='--')
    plt.savefig('test1.pdf')
    plt.show()
    plt.close()

    return

def test2():
    np.random.seed(0)
    nside = 8
    ell = np.arange(3.*nside)
    Cltrue = np.zeros_like(ell)
    Cltrue[2:] = 1/ell[2:]**2
    m = hp.synfast(Cltrue, nside)
    Clhat = hp.anafast(m)
    
    sigma_l = Clhat

    plt.plot(ell[2:], Cltrue[2:])
    plt.plot(ell[2:], Clhat[2:], '.')
    

    # Let's replace this with inverse-gamma so the transition to inverse-wishart
    # is simpler
    #for i in range(100):
    #    rho_l = np.random.randn(len(sigma_l)-2, len(sigma_l))
    #    plt.plot(ell, sigma_l/(rho_l**2).mean(axis=0), 'k.', alpha=0.1,
    #            zorder=-1)
    for l in ell[2:]:
        l = int(l)
        alpha = (2*ell[l]-1)/2
        beta = (2*ell[l]+1)*sigma_l[l]/2
        #cl_draw = invgamma.rvs(alpha, scale=beta, size=1000)
        cl_draw = invwishart.rvs(df=alpha*2, scale=beta*2, size=1000)
        plt.plot([l]*1000, cl_draw, 'k.', alpha=0.01, zorder=-1)

    plt.yscale('log')
    plt.savefig('test2.pdf')
    plt.show()

from scipy.special import legendre
def get_TT_cov(C_l, nside=8):
    C = np.zeros_like(z)
    for l in range(len(C_l)):
        P_l = legendre(l)
        C += C_l[l]*P_l(z)*(2*l+1)/(4*np.pi)
    return C

def map_sample(C_l, m, show=False, var=1e-2):
    npix = len(m)
    N = np.eye(npix)*var
    Ninv = np.linalg.inv(N)*var
    #inds = (mask == 0)
    #Ninv[inds] = 0
    #Ninv[:,inds] = 0
    #N[inds, inds] = 1e3

    S = get_TT_cov(C_l)
    I = np.eye(npix)
    SN = S.dot(Ninv)
    mu = S.dot(np.linalg.inv(S+N).dot(m))
    #mu = SN.dot(np.linalg.inv(I+SN).dot(m))


    Sigma  = np.linalg.inv(np.linalg.inv(S) + Ninv)
    #Sigma = S - SN.dot(np.linalg.inv(I + SN).dot(S))
    # Maybe the problem is when I'm sampling from the multivariate normal...
    # Graeme did a scipy linalg decomposition of the covariance, 
    #returned mean + L.dot(x) where x is a random standard Gaussian.
    if show:
        hp.mollview(mu, title=r'$\mu$', min=-2, max=2)
        plt.savefig('mu.pdf')
        plt.figure()
        plt.subplot(131)
        plt.imshow(N, vmin=-0.1, vmax=0.1, cmap='seismic')
        plt.title(r'$N$')
        plt.subplot(132)
        plt.imshow(S, vmin=-0.1, vmax=0.1, cmap='seismic')
        plt.title(r'$S(C_\ell)$')
        plt.subplot(133)
        plt.title(r'$(S^{-1}+N^{-1})^{-1}$')
        plt.imshow(Sigma, vmin=-0.1, vmax=0.1, cmap='seismic')
        plt.savefig('sample_cov.pdf')
    s = np.random.multivariate_normal(mu, Sigma)
    return s

def cl_sample(s):
    sigma_l = hp.anafast(s)
    ell = np.arange(len(sigma_l))
    cl_draw = []
    for l in ell[2:]:
        l = int(l)
        alpha = (2*ell[l]-1)/2
        beta = (2*ell[l]+1)*sigma_l[l]/2
        #cl_draw.append(invgamma.rvs(alpha, scale=beta))
        cl_draw.append(invwishart.rvs(df=alpha*2, scale=beta*2))
    return np.array(cl_draw)


def gibbs_sample(m, iters=20):
    #inds = (mask == 0)
    #m[inds] = 0
    Cl = hp.anafast(m)
    maps = []
    Cls = []
    for i in range(iters):
        s = map_sample(Cl, m)
        Cl = cl_sample(s)
        maps.append(s)
        Cls.append(Cl)
        print(i, s.std(), Cl.mean())
    return maps, Cls

from scipy.special import gamma
def stupid_test1():
    l = 2
    s_l = 2.
    C_ls = np.linspace(0,5,1000)
    P = s_l**(2*l-1)*np.exp(-s_l/(2*C_ls))/gamma(2*l-1)/2**(2*l-1)/np.sqrt(C_ls**(2*l+1))
    plt.plot(C_ls, P)
    plt.axvline(s_l/(2*l+1))
    plt.show()

    return

def stupid_test2():
    mask = hp.read_map('class_mask.fits')
    mask = hp.ud_grade(mask, 8)
    mask = np.where(mask < 0.5, 0, 1)
    inds = (mask == 0)
    npix = hp.nside2npix(8)

    N = np.eye(npix)
    N[inds, inds] = 1e3

    #mu = S(S+N)^{-1}m = S([I+SN^{-1}]N)^{-1}m = SN^{-1}(I+SN^{-1})^{-1}m
    #cov = (S^-1 +N^-1)^1 =  = S-SN^-1(I+SN^-1)S
    ell = np.arange(3*8.)
    npix = hp.nside2npix(8)
    C_l = np.zeros_like(ell)
    C_l[2:] = 10./ell[2:]**2
    n = np.random.multivariate_normal(np.zeros(npix), N)
    m = hp.synfast(C_l, 8) + n
    m[inds] = 0
    S = get_TT_cov(C_l)
    Ninv = np.linalg.inv(N)
    Ninv[inds] = 0
    Ninv[:,inds] = 0
    SNi = S.dot(Ninv)

    I = np.eye(npix)

    mu = SNi.dot(np.linalg.inv(I + SNi)).dot(m)
    cov = S - SNi.dot(np.linalg.inv(I+SNi)).dot(S)

    plt.subplot(221)
    plt.imshow(Ninv)
    plt.subplot(222)
    plt.imshow(S, vmin=-1, vmax=1, cmap='seismic')
    plt.subplot(223)
    plt.imshow(SNi, vmin=-1, vmax=1, cmap='seismic')
    plt.subplot(224)
    plt.imshow(cov, vmin=-1, vmax=1, cmap='seismic')

    hp.mollview(m, min=-5, max=5)
    m = np.random.multivariate_normal(mu, cov)
    # is there a way to convert the covariance matrix into C_l's? I guess I just
    # want a higher resolution version of this map.
    hp.mollview(m, min=-5, max=5)
    hp.mollview(mu, min=-5, max=5)

    return


if __name__ == '__main__':
    var = 1e-2
    #test1()
    #test2()

    ell = np.arange(3*8.)
    C_l = np.zeros_like(ell)
    C_l[2:] = 1./ell[2:]**2

    s_true = hp.synfast(C_l, 8)
    n = np.random.randn(hp.nside2npix(8))*var**0.5
    m = s_true + n

    #map_sample(C_l, m, show=True)

    maps, Cls = gibbs_sample(m, iters=20)
    plt.figure()
    plt.plot(ell[2:], C_l[2:], label='True')
    plt.plot(ell[2:], hp.anafast(s_true)[2:], '.', label=r'$\hat C_{\ell,\mathrm{true}}$')
    for i in range(len(Cls)):
        plt.plot(ell[2:], Cls[i], 'k.', alpha=0.5)
    plt.yscale('log')
    plt.legend(loc='best')
    plt.savefig('power_spectra.png')


    #for i in range(len(maps)):
    #    hp.mollview(maps[i], min=-2, max=2, title='', cbar=False)
    #    plt.savefig('map_{0}.png'.format(str(i).zfill(3)))
    #    plt.close()
   

    #C = get_TT_cov(C_l)
    #plt.imshow(C, vmin=-0.1, vmax=0.1, cmap='seismic')
    #plt.colorbar()
    #plt.savefig('covmat.pdf')
    #plt.figure()
    #d = np.diag(C)
    #rho = np.outer(d,d)
    #plt.plot(d)
    #plt.savefig('diag.pdf')
    #plt.figure()
    #plt.imshow(C/rho, vmin=-1, vmax=1, cmap='seismic')
    #plt.colorbar()
    #plt.savefig('rho.pdf')
    #plt.show()

    #plt.figure()
    #m = hp.synfast(C_l, 8)
    #n = np.random.randn(len(m))*0.1**0.5
    #m = m + n
    #s = map_sample(C_l, m)
    #plt.figure()
    #hp.mollview(m, sub=121, min=-2, max=2, title=r'$m$')
    #hp.mollview(s, sub=122, min=-2, max=2, title=r'$s\sim P(s|C_\ell, m)$')
    #plt.savefig('sample.pdf')
