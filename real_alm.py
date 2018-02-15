"""HealPy alm arrays are ordered as follows.
In this case, ellmax = 5.

             m value ->
             0   1   2   3   4   5

ell    0     0
value  1     1   6
|      2     2   7  11
|      3     3   8  12  15
V      4     4   9  13  16  18
       5     5  10  14  17  19  20

Negative m values are not used; because the real-valued nature of the
temperature field constrains the negative m valued coefficients once the
positive-valued coefficients are known.

The desired ordering of real-valued coefficients is

                             m value ->
            -4  -3  -2  -1   0   1   2   3   4   

ell    0                     0
value  1                 1   2   3
|      2             4   5   6   7   8
|      3         9  10  11  12  13   14  15
V      4    16  17  18  19  20  21   22  23  24 

I could do something to try to mimic the HealPy ordering instead, but if I have
to rearrange the ordering to make them real-valued anyway, I might as well pick
an ordering that I'm used to (and that is easily extendable to higher ell,
without reordering the array).  Most of the time, I won't be working with
m-truncated instrument beams, which is what the HealPy ordering is very good
for.

We can use the same ordering for the real-valued harmonics as for the polarized
E and B modes, because (except for the monopole and dipole) there is a natural
correspondence between the two.
"""

from __future__ import division
import numpy as np
import healpy as hp
import sys
#import matplotlib.pyplot as mpl
import unittest





def lm2i(ellmax, ell, m):
    """
    Conversion from multipole ell and m (where m >= 0) to the index i
    used in healpy ordering of alm values.
    """
    i = m * (ellmax + 1) - ((m - 1) * m ) // 2 + ell - m
    return i


def i2lm(ellmax, i):
    """
    Conversion from index i used for healpy ordering of alm values 
    to multipole ell and m.
    """
    m = (np.floor(ellmax + 0.5 - np.sqrt((ellmax + 0.5)**2 - 2 * (i - ellmax - 1)))).astype(int) + 1
    ell = i - m * (ellmax + 1) + ((m - 1) * m) // 2 + m
    return (ell, m)


def ilength(ellmax, mmax):
    """
    The number of elements in a healpy alm array with a given ellmax and mmax
    """
    assert(ellmax >= mmax)
    assert(ellmax >= 0)
    assert(mmax >= 0)
    return (mmax + 1) * (ellmax + 1) - ((mmax + 1) * mmax) // 2


# Let's use the index r as the real-valued ordering integer.
# Note that we don't need to know ellmax.
def r2lm(r):
    """
    Conversion from real-valued ordering index r to multipole ell and m.
    -ell <= m <= ell
    ell >= 0
    """
    ell = np.floor(np.sqrt(r)).astype(int)
    m = r - ell**2 - ell
    return (ell, m)
    

# m is allowed to be negative, here.
def lm2r(ell, m):
    """
    Conversion from multipole ell and m to real-valued ordering index r.
    """
    r = ell**2 + ell + m
    return r


# I'm sure this can be optimized for speed, later.
# At low ell, we don't really care; it will be fast enough.
def complex2real(ellmax, mmax, cx_alm):
    """ 
    Convert an array from HealPy of complex-valued alms into an array
    of real-valued alms.
    """
    len_ = ilength(ellmax, mmax)
    assert(cx_alm.shape[0] == len_)
    #print "len = ", len
    re_alm = np.zeros((ellmax+1)**2, dtype='double')
    for i in range(len_):
        ell, m = i2lm(ellmax, i) 
        r = lm2r(ell, m)
        if m == 0:
            re_alm[r] = np.real(cx_alm[i])
        else:
            re_alm[r] = np.real(cx_alm[i]) * np.sqrt(2.0)
            r = lm2r(ell, -m)
            re_alm[r] = (-1)**m * np.imag(cx_alm[i]) * np.sqrt(2.0)
            
    return re_alm


def real2complex(re_alm, ellmax=-1, mmax=-1):
    """
    Convert an array of real-valued alms to a healpy array of complex-valued
    alms.
    """
    if ellmax < 0:
        ellmax = np.floor(np.sqrt(re_alm.shape[0] - 1)).astype(int)
    if mmax < 0:
        mmax = ellmax
    assert(re_alm.shape[0] >= (ellmax+1)**2)
    len_ = int(ilength(ellmax, mmax))
    cx_alm = np.zeros(len_, dtype='cfloat')
    for i in range(len_):
        ell, m = i2lm(ellmax, i) 
        r1 = lm2r(ell, m)
        if m == 0:
            cx_alm[i] = re_alm[r1]
        else:
            r2 = lm2r(ell, -m)
            cx_alm[i] = (re_alm[r1] + (-1)**m * 1.0j * re_alm[r2]) / np.sqrt(2.0)

    return cx_alm


def real2cl(re_alm):
    """ 
    Return raw cl values, from an array of real-valued alms.
    """
    cx_alm = real2complex(re_alm)
    cl = hp.sphtfunc.alm2cl(cx_alm)
    return cl 



class TestIndexing(unittest.TestCase):
    def my_test_i(self, ellmax, i):
        ell, m = i2lm(ellmax, i)
        self.assertTrue(m >= 0)
        self.assertTrue(ell >= 0)
        self.assertTrue(m <= ell)
        self.assertTrue(ell <= ellmax)
        i2 = lm2i(ellmax, ell, m)
        self.assertTrue(i == i2)

    def my_test_r(self, r):
        ell, m = r2lm(r)
        self.assertTrue(ell >= 0)
        self.assertTrue(abs(m) <= ell)
        r2 = lm2r(ell, m)
        self.assertTrue(r == r2)

    def my_test_lm(self, ellmax, ell, m):
        self.assertTrue(m <= ell)
        i = lm2i(ellmax, ell, m)
        ell2, m2 = i2lm(ellmax, i)
        #print "in:  ", i, ell, m
        #print "out: ", i, ell2, m2
        self.assertTrue(i >= 0)
        self.assertTrue(i < ilength(ellmax, ellmax))
        self.assertTrue(ell2 == ell)
        self.assertTrue(m2 == m)

    def my_test_rlm(self, ell, m):
        self.assertTrue(abs(m) <= ell)
        r = lm2r(ell, m)
        ell2, m2 = r2lm(r)
        self.assertTrue(r >= 0)
        self.assertTrue(r < (ell + 1)**2)
        self.assertTrue(ell2 == ell)
        self.assertTrue(m2 == m)

    def test_manual(self):
        self.assertEqual(ilength(5, 3), 18)
        self.assertEqual(ilength(5, 5), 21)
        self.assertEqual(ilength(0, 0), 1)
        self.assertEqual(ilength(1, 0), 2)
        self.assertEqual(ilength(1, 1), 3)

        self.assertEqual(lm2i(0,0,0), 0)
        self.assertEqual(lm2i(1,0,0), 0)
        self.assertEqual(lm2i(2,0,0), 0)

    def test_ilength(self):
        self.assertEqual(ilength(95, 95), 4656)

    def test_i_lowell(self):
        for ellmax in np.arange(0, 10):
            for i in range(ilength(ellmax, ellmax)):
                self.my_test_i(ellmax, i)

    def test_r_lowell(self):
        ellmax = 10
        for r in np.arange(ellmax**2):
            self.my_test_r(r)

    def test_i_highell(self):
        ellmax = 10000
        for i in np.arange(ilength(ellmax, ellmax) - 100, ilength(ellmax, ellmax)):
            self.my_test_i(ellmax, i)

    def test_r_highell(self):
        ellmax = 10000
        for r in np.arange(ellmax**2 - 100, ellmax**2):
            self.my_test_r(r)

    def test_lm_lowell(self):
        for ellmax in np.arange(0, 10):
            for ell in np.arange(ellmax + 1):
                for m in np.arange(ell + 1):
                    self.my_test_lm(ellmax, ell, m)

    def test_rlm_lowell(self):
        ellmax = 10
        for ell in np.arange(ellmax + 1):
            for m in np.arange(-ell, ell + 1):
                self.my_test_rlm(ell, m)

    def test_lm_highell(self):
        ellmax = 10000
        for ell in np.arange(4):
            for m in np.arange(ell + 1):
                self.my_test_lm(ellmax, ell, m)
                
        for ell in np.arange(ellmax - 1, ellmax + 1):
            for m in np.arange(ell + 1):
                self.my_test_lm(ellmax, ell, m)

    def test_rlm_highell(self):
        ellmax = 10000
        for ell in np.arange(ellmax - 1, ellmax + 1):
            for m in np.arange(-ell, ell + 1):
                self.my_test_rlm(ell, m)

    def test_array1(self):
        nside = 32
        npix = hp.nside2npix(nside)
        map_ = np.random.standard_normal(npix)
        cl, cx_alm = hp.anafast(map_, alm=True, use_weights=True)
        ellmax = cl.shape[0] - 1
        mmax = ellmax
        re_alm = complex2real(ellmax, mmax, cx_alm)
        cx_alm2 = real2complex(re_alm)
        self.assertTrue(np.max(np.abs(cx_alm2 - cx_alm)) < 1e-15)

    def test_array2(self):
        ellmax = 100
        mmax = ellmax
        len_ = (ellmax + 1)**2
        re_alm = np.arange(len_) + 1
        self.assertTrue(re_alm[0] != re_alm[1])
        self.assertTrue(re_alm[1] != re_alm[2])
        cx_alm = real2complex(re_alm)
        self.assertTrue(cx_alm.shape[0] == ilength(ellmax, mmax))
        re_alm2 = complex2real(ellmax, mmax, cx_alm)
        #print np.max(np.abs(re_alm2 - re_alm) / re_alm)
        self.assertTrue(np.min(re_alm2) > 0.5)
        self.assertTrue(np.max(np.abs(re_alm2 - re_alm) / re_alm) < 1e-14)

    def test_norm1(self):
        nside = 64
        npix = hp.nside2npix(nside)
        ellmax = 10 
        mmax = ellmax
        len_ = (ellmax + 1)**2

        for i in range(len_):
            re_alm = np.zeros(len_, dtype='double') 
            re_alm[i] = 1.0
            cx_alm = real2complex(re_alm)
            map_ = hp.alm2map(cx_alm, nside)
            unity = np.sum(map_**2) * np.pi * 4 / npix
            self.assertTrue(abs(unity - 1) < 1e-3)

    def test_orthonormal(self):
        nside = 64
        npix = hp.nside2npix(nside)
        #ellmax = 10 
        ellmax = 5
        mmax = ellmax
        len_ = (ellmax + 1)**2
        for i in range(len_):
            re_alm = np.zeros(len_, dtype='double') 
            re_alm[i] = 1.0
            cx_alm = real2complex(re_alm)
            map_ = hp.alm2map(cx_alm, nside)
            unity = np.sum(map_**2) * np.pi * 4 / npix
            self.assertTrue(abs(unity - 1) < 1e-3)
            for k in range(i+1, len_):
                re_alm = np.zeros(len_, dtype='double') 
                re_alm[k] = 1.0
                cx_alm = real2complex(re_alm)
                map2 = hp.alm2map(cx_alm, nside)
                dot = np.sum(map_ * map2) * np.pi * 4 / npix
                if abs(dot) > 1e-3:
                    print(i, k, dot)
                self.assertTrue(abs(dot) < 1e-3)

    def test_cl(self):
        ellmax = 10
        len_ = (ellmax + 1)**2
        alm = np.zeros(len_) + 2.0
        cl = real2cl(alm)
        for i in range(ellmax + 1):
            self.assertTrue(abs(cl[i] - 4.0) < 1e-14)

        alm = np.array([1, 2, 2, 2, 3, 3, 3, 3, 3])
        cl = real2cl(alm)
        self.assertTrue(abs(cl[0] - 1) < 1e-14)
        self.assertTrue(abs(cl[1] - 4) < 1e-14)
        self.assertTrue(abs(cl[2] - 9) < 1e-14)


def foo():
    # Scratch paper
    a = hp.sphtfunc.Alm()
    ellmax = 5
    ell = 0
    m = 0
    i = a.getidx(ellmax, ell, m)
    print(i)
    #ell, m = a.getlm(0, ellmax)
    #print ell, m
    #sys.exit()


def foo2():
    # Scratch paper
    nside = 32
    npix = hp.nside2npix(nside)
    map_ = np.random.standard_normal(npix)
    cl, cx_alm = hp.anafast(map_, alm=True, use_weights=True)
    print(cl.shape)
    ellmax = cl.shape[0] - 1
    mmax = ellmax
    print(cx_alm.shape)
    print(cx_alm)
    re_alm = complex2real(ellmax, mmax, cx_alm)
    cx_alm2 = real2complex(re_alm)
    #print re_alm.shape
    #print cx_alm2.shape
    #print cx_alm2
    print(np.max(np.abs(cx_alm2 - cx_alm)))
    #print (cx_alm - cx_alm2)[-20:]
    

def foo3():
    # Scratch paper 
    import matplotlib.pyplot as mpl
    nside = 64
    npix = hp.nside2npix(nside)
    ellmax = 10 
    mmax = ellmax
    len_ = (ellmax + 1)**2

    for i in range(len_):
        re_alm = np.zeros(len_, dtype='double') 
        re_alm[i] = 1.0
        cx_alm = real2complex(re_alm)
        map_ = hp.alm2map(cx_alm, nside)
        #hp.mollview(map)
        #mpl.show()
        #print np.ptp(map)
        #print map[0] - 1 / np.sqrt(np.pi * 4)
        print("1 = ", np.sum(map_**2) * np.pi * 4 / npix)
    


if __name__ == "__main__":
    #foo()
    #foo2()
    unittest.main()
    #foo3()

