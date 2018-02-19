import matplotlib.pyplot as plt
import numpy as np
import healpy as hp

from glob import glob

from pixel_cov import mat_from_cl
# mat_from_cl takes as an argument Cl_array = np.array([TT, EE, BB, TE, TB, EB])
# and by default uses nside = 8

cmap = plt.cm.seismic
cmap.set_under('white')
cmap.set_bad('gray')


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
    #plt.show()
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
    #plt.show()

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
    Ninv = np.linalg.inv(N)
    #inds = (mask == 0)
    #Ninv[inds] = 0
    #Ninv[:,inds] = 0
    #N[inds, inds] = 1e3

    S = get_TT_cov(C_l)
    I = np.eye(npix)
    SN = S.dot(Ninv)
    #mu = S.dot(np.linalg.inv(S+N).dot(m))
    # Weiner filter is 
    # mu = (S^-1+N^-1)^-1 N^-1 m
    #    = [S - SN^-1(I+SN^-1)^-1S]N^-1 m
    #mu = SN.dot(m) - SN.dot(np.linalg.inv(I+SN).dot(SN.dot(m)))
    # the paper I'm working with gives
    # mu =S(S+N)^-1 m=S[(I+SN^-1)N]^-1 m = SN^-1(I+SN^-1) m
    mu = SN.dot(np.linalg.inv(I+SN).dot(m))


    #Sigma  = np.linalg.inv(np.linalg.inv(S) + Ninv)
    Sigma = S - SN.dot(np.linalg.inv(I + SN).dot(S))
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


def maparr_sample(Cl_arr, map_arr, var=1e-2):
    m = np.concatenate((map_arr[0], map_arr[1], map_arr[2]))
    npix = len(map_arr[0])
    N = np.eye(3*npix)*var
    Ninv = np.linalg.inv(N)
    allmask = np.concatenate((mask,mask,mask))
    inds = (allmask == 0)
    Ninv[inds] = 0
    Ninv[:,inds] = 0
    N[inds, inds] = 1e3


    # Invert only the rows and columns that will be unmasked
    # Put it back into the matrix?

    S = mat_from_cl(Cl_arr)
    I = np.eye(3*npix)
    SN = S.dot(Ninv)
    mu = SN.dot(np.linalg.inv(I+SN).dot(m))

    imu, qmu, umu = np.split(mu,3)
    plt.figure()
    hp.mollview(imu, min=-250, max=250, cbar=False, title='', sub=131, cmap=cmap)
    hp.mollview(qmu, min=-0.5, max=0.5, cbar=False, title='', sub=132, cmap=cmap)
    hp.mollview(umu, min=-0.5, max=0.5, cbar=False, title='', sub=133, cmap=cmap)
    fnames = glob('meanarr_*')
    plt.savefig('meanarr_{0}.png'.format(str(len(fnames)).zfill(3)))
    plt.close()


    Sigma = S - SN.dot(np.linalg.inv(I + SN).dot(S))
    s = np.random.multivariate_normal(mu, Sigma)
    i,q,u = np.split(s,3)
    return np.array([i,q,u])

def cl_sample(s):
    c_l = hp.anafast(s)
    ell = np.arange(len(sigma_l))
    cl_draw = [0,0]
    for l in ell[2:]:
        l = int(l)
        alpha = (2*ell[l]-1)/2
        beta = (2*ell[l]+1)*c_l[l]/2
        #cl_draw.append(invgamma.rvs(alpha, scale=beta))
        cl_draw.append(invwishart.rvs(df=alpha*2, scale=beta*2))
    return np.array(cl_draw)

def clarr_sample(s):
    # s = np.array([I,Q,U])
    TT, EE, BB, TE, TB, EB = hp.anafast(s)
    tt_draw = [0,0]
    ee_draw = [0,0]
    bb_draw = [0,0]
    te_draw = [0,0]
    tb_draw = [0,0]
    eb_draw = [0,0]
    ell = np.arange(len(EE))
    for l in ell[2:]:
        l = int(l)
        # full wishart
        c_l = np.array([
            [TT[l], TE[l], 0*TB[l]],
            [TE[l], EE[l], 0*EB[l]],
            [0*TB[l], 0*EB[l], BB[l]]])
        alpha = (2*l-1)/2
        beta = (2*l+1)*c_l/2
        cl_draw = invwishart.rvs(df=alpha*2, scale=beta*2)
        tt_draw.append(cl_draw[0,0])
        ee_draw.append(cl_draw[1,1])
        bb_draw.append(cl_draw[2,2])
        te_draw.append(cl_draw[0,1])
        tb_draw.append(cl_draw[0,2])
        eb_draw.append(cl_draw[1,2])

        ## T/E Wishart
        #c_l = np.array([
        #    [TT[l], TE[l]],
        #    [TE[l], EE[l]]])
        #alpha = (2*l-1)/2
        #beta = (2*l+1)*c_l/2
        #cl_draw = invwishart.rvs(df=alpha*2, scale=beta*2)
        #tt_draw.append(cl_draw[0,0])
        #ee_draw.append(cl_draw[1,1])
        #te_draw.append(cl_draw[0,1])

        ## B Wishart
        #c_l = BB[l]
        #alpha = (2*l-1)/2
        #beta = (2*l+1)*c_l/2
        #cl_draw = invwishart.rvs(df=alpha*2, scale=beta*2)
        #bb_draw.append(cl_draw)
    tt = np.array(tt_draw)
    ee = np.array(ee_draw)
    bb = np.array(bb_draw)
    te = np.array(te_draw)
    #tb = np.array(tb_draw)
    #eb = np.array(eb_draw)
    #return np.array([tt,ee,bb,te,tb,eb])
    return np.array([tt,ee,bb,te,0*te,0*te])


def gibbs_sample_temp(m, iters=20):
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
        print(i, s.std(), Cl[2:].min())
    return maps, Cls

from scipy.special import gamma
def stupid_test1():
    l = 2
    s_l = 2.
    C_ls = np.linspace(0,5,1000)
    P = s_l**(2*l-1)*np.exp(-s_l/(2*C_ls))/gamma(2*l-1)/2**(2*l-1)/np.sqrt(C_ls**(2*l+1))
    plt.plot(C_ls, P)
    plt.axvline(s_l/(2*l+1))
    #plt.show()

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

def polsample_test(niter=1000):
    import camb
    from camb import model, initialpower
    pars = camb.CAMBparams()
    #This function sets up CosmoMC-like settings, with one massive neutrino and helium set using BBN consistency
    pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0, tau=0.06)
    pars.WantTensors = True
    pars.InitPower.set_params(ns=0.965, r=0.05)
    pars.set_for_lmax(2500, lens_potential_accuracy=0);
    results = camb.get_results(pars)
    powers =results.get_cmb_power_spectra(pars, CMB_unit='muK')

    var = 1e-4
    nside = 8
    lmax = 3*nside - 1
    ell = np.arange(lmax+1.)
    Cl_arr = powers['total'][:lmax+1].T
    Z = ell*(ell+1)/(2*np.pi)
    Z[:2] = 1.


    Cls = np.array([Cl_arr[0]/Z, Cl_arr[1]/Z, Cl_arr[2]/Z,\
                    Cl_arr[3]/Z, 0*Cl_arr[0], 0*Cl_arr[0]])
    plt.figure()
    for i in range(4):
        plt.subplot(2,2,i+1)
        plt.loglog(ell[2:], Cls[i][2:], color='C{0}'.format(i))

    s_true = hp.synfast(Cls, nside, new=True)
    m = np.copy(s_true)
    inds = (mask == 0)
    for i in range(3):
        m[i] = s_true[i] + np.random.randn(npix)*var**0.5
        m[i][inds] = 0
    Clhat_true = hp.anafast(s_true)
    Clhat_noise = hp.anafast(m)
    for i in range(4):
        plt.subplot(2,2,i+1)
        plt.loglog(ell[2:], Clhat_true[i][2:], '.', color='C{0}'.format(i))
    for i in range(4):
        plt.subplot(2,2,i+1)
        plt.loglog(ell[2:], Clhat_noise[i][2:], 'o', color='C{0}'.format(i))
    plt.savefig('1.png')
    #plt.show()

    plt.figure()
    plt.suptitle('Input')
    hp.mollview(s_true[0], min=-200,max=200,sub=131, title='')
    hp.mollview(s_true[1], min=-0.5,max=0.5,sub=132, title='')
    hp.mollview(s_true[2], min=-0.5,max=0.5,sub=133, title='')
    plt.savefig('2.png')
    #plt.show()


    plt.figure()
    Clhat = hp.anafast(m)
    s = np.copy(m)
    Cli = np.copy(Cls)
    for n in range(niter):
        print(n, niter)
        si = maparr_sample(Cli, m, var=var)
        Cli = clarr_sample(si)
        s = np.vstack((s, si))
        Clhat = np.vstack((Clhat, Cli))
        
        for j in range(4):
            plt.subplot(2,2,j+1)
            plt.plot(ell[2:], Cli[j][2:], '.', alpha=0.05,\
                    color='C{0}'.format(j))

        plt.figure()
        hp.mollview(si[0], min=-200, max=200, sub=131, title='', cbar=False,
                cmap=cmap)
        hp.mollview(si[1], min=-.5, max=.5, sub=132, title='', cbar=False,
                cmap=cmap)
        hp.mollview(si[2], min=-.5, max=.5, sub=133, title='', cbar=False,
                cmap=cmap)
        fnames = glob('realization_*')
        plt.savefig('realization_{0}.png'.format(str(len(fnames)).zfill(4)))
        plt.close()

    for i in range(4):
        plt.subplot(2,2,i+1)
        plt.plot(ell[2:], Cls[i][2:], 'k')
    for i in range(4):
        plt.subplot(2,2,i+1)
        plt.plot(ell[2:], Clhat_true[i][2:], 'k.')
        plt.yscale('log')
        plt.xscale('log')
    plt.savefig('3.png')
    #plt.show()

    np.save('map_samples.npy', s)
    np.save('cl_samples.npy', clhat)

    plt.figure()
    plt.suptitle('Output')
    hp.mollview(si[0], min=-200,max=200,sub=131, title='')
    hp.mollview(si[1], min=-0.5,max=0.5,sub=132, title='')
    hp.mollview(si[2], min=-0.5,max=0.5,sub=133, title='')

    plt.savefig('4.png')
    #plt.show()

    return


if __name__ == '__main__':
    polsample_test()
    '''
    var = 1e-2
    mask = hp.read_map('class_mask.fits')
    mask = hp.ud_grade(mask, 8)
    mask = np.where(mask < 0.5, 0, 1)
    inds = (mask == 0)
    #test1()
    #test2()

    ell = np.arange(3*8.)
    C_l = np.zeros_like(ell)
    C_l[2:] = 1./ell[2:]**2

    s_true = hp.synfast(C_l, 8, fwhm=2.5*hp.nside2resol(8))
    n = np.random.randn(hp.nside2npix(8))*var**0.5
    m = s_true + n
    m[inds] = 0

    #map_sample(C_l, m, show=True)

    maps, Cls = gibbs_sample_temp(m, iters=50)

    plt.figure()
    gb = hp.gauss_beam(2.5*hp.nside2resol(8), lmax=ell.max())
    plt.plot(ell[2:], C_l[2:]*gb[2:]**2 + 4*np.pi*var/hp.nside2npix(8), label='True')
    plt.plot(ell[2:], hp.anafast(s_true)[2:] + 4*np.pi*var/hp.nside2npix(8), '.', label=r'$\hat C_{\ell,\mathrm{true}}$')
    for i in range(len(Cls)):
        plt.plot(ell[2:], Cls[i][2:], 'k.', alpha=0.5, zorder=-1)
    plt.yscale('log')
    plt.legend(loc='best')
    plt.savefig('power_spectra.png')
    '''

