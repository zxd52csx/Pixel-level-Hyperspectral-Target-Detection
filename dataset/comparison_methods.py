import numpy as np
import time
import random
from .ECEM_utils import Detector, dual_sigmoid
import copy
'''
Partial replication of the comparative methods referenced ECEM: "https://github.com/ruizhaocv/E_CEM-for-Hyperspectral-Target-Detection"
'''

def cem(img, tgt):
    # Basic implementation of the Constrained energy minimization (CEM)
    t1 = time.time()
    size = img.shape  # get the size of image matrix
    R = np.dot(img, img.T / size[1])  # R = X*X'/size(X,2);
    w = np.dot(np.linalg.pinv(R), tgt)  # w = (R+lamda*eye(size(X,1)))\d ;
    result = np.dot(w.T, img).T / np.dot(w.T, tgt)  # y=w'* X;
    print('CEM time comsumption:{}.'.format(time.time()-t1))
    return result

def cem_re(img, tgt):
    # Basic implementation of the Constrained energy minimization with regularization coefficient
    import random
    Lambda = 1e-6  # the regularization coefficient
    size = img.shape  # get the size of image matrix
    lamda = random.uniform(Lambda / (1 + Lambda), Lambda)  # random regularization coefficient
    R = np.dot(img, img.T / size[1])  # R = X*X'/size(X,2);
    w = np.dot(np.linalg.pinv((R + lamda * np.identity(size[0]))), tgt)  # w = (R+lamda*eye(size(X,1)))\d ;
    result = np.dot(w.T, img)  # y=w'* X;
    return result

def RSA(img, tgt):
    # Basic implementation of the Spectral angle measurement in the space whitened with Correlation matrix (R-SA**2)
    # Chang, Chein-I. "Hyperspectral target detection: Hypothesis testing, signal-to-noise ratio, and spectral angle 
    # theories." IEEE Transactions on Geoscience and Remote Sensing 60 (2021): 1-23.
    t1 = time.time()
    result = np.zeros([1, img.shape[1]])
    img_ = img.T[img.max(axis=0)>0].T
    
    size = img_.shape
    img_mean = np.mean(img_, axis=1)[:, np.newaxis]
    img0 = img_-img_mean.dot(np.ones((1, size[1])))
    R = img_.dot(img_.T)/size[1]
    y0 = (tgt).T.dot(np.linalg.pinv(R)).dot(img_)**2
    y1 = (tgt).T.dot(np.linalg.pinv(R)).dot(tgt)
    y2 = (img_.T.dot(np.linalg.pinv(R))*(img_.T)).sum(axis=1)[:, np.newaxis]
    result[0,:][img.max(axis=0)>0] = (y0/(y1*y2).T)[0,:]
    print('RSA time comsumption:{}.'.format(time.time()-t1))
    return result.T
    

def ace(img, tgt):
    t1 = time.time()
    # Basic implementation of the Adaptive Coherence/Cosine Estimator (ACE)
    # Manolakis, Dimitris, David Marden, and Gary A. Shaw. "Hyperspectral image processing for
    # automatic target detection applications." Lincoln laboratory journal 14, no. 1 (2003): 79-116.
    size = img.shape
    img_mean = np.mean(img, axis=1)[:, np.newaxis]
    img0 = img-img_mean.dot(np.ones((1, size[1])))
    R = img.dot(img.T)/size[1]
    # R = img0.dot(img0.T)/size[1]
    y0 = (tgt-img_mean).T.dot(np.linalg.pinv(R)).dot(img0)**2
    y1 = (tgt-img_mean).T.dot(np.linalg.pinv(R)).dot(tgt-img_mean)
    y2 = (img0.T.dot(np.linalg.pinv(R))*(img0.T)).sum(axis=1)[:, np.newaxis]
    result = y0/(y1*y2).T
    print('ACE time comsumption:{}.'.format(time.time()-t1))
    return result.T


def mf(img, tgt):
    # Basic implementation of the Matched Filter (MF)
    # Manolakis, Dimitris, Ronald Lockwood, Thomas Cooley, and John Jacobson. "Is there a best hyperspectral
    # detection algorithm?." In Algorithms and technologies for multispectral, hyperspectral, and ultraspectral
    # imagery XV, vol. 7334, p. 733402. International Society for Optics and Photonics, 2009.
    size = img.shape
    a = np.mean(img)
    k = (img-a).dot((img-a).T)/size[1]
    w = np.linalg.pinv(k).dot(tgt-a)
    result = w.T.dot(img-a)
    return result.T

def NAMD(img, tgt):
    t1 = time.time()
    # Basic implementation of the Normalized Adaptive Matched Detector (NAMD**2)
    # Chang, Chein-I. "Hyperspectral target detection: Hypothesis testing, signal-to-noise ratio, and spectral angle 
    # theories." IEEE Transactions on Geoscience and Remote Sensing 60 (2021): 1-23.
    size = img.shape
    a = np.mean(img)
    k = (img-a).dot((img-a).T)/size[1]
    w = np.linalg.pinv(k).dot(tgt-a)
    result = (w.T.dot(img-a)/ (tgt-a).T.dot(np.linalg.pinv(k)).dot(tgt-a))**2
    print('NAMD2 time comsumption:{}.'.format(time.time()-t1))
    return result.T

def sid(img, tgt):
    # Basic implementation of the Spectral Information Divergence (SID) detector
    # Chang, Chein-I. "An information-theoretic approach to spectral variability, similarity, and discrimination
    # for hyperspectral image analysis." IEEE Transactions on information theory 46, no. 5 (2000): 1927-1932.
    size = img.shape
    result = np.zeros((1, size[1]))
    for i in range(size[1]):
        pi = (img[:, i]/(img[:, i].sum())).reshape(-1, 1)+1e-20
        di = tgt/(tgt.sum())+1e-20
        sxd = (pi*np.log(abs(pi/di))).sum()
        sdx = (di*np.log(abs(di/pi))).sum()
        result[:, i] = 1/((sxd + sdx)/size[1] + 1e-12)
    return result.T

def sam(img, tgt):
    # Basic implementation of the Spectral Angle Mapper (SAM)
    # Kruse, Fred A., A. B. Lefkoff, J. W. Boardman, K. B. Heidebrecht, A. T. Shapiro, P. J. Barloon,
    # and A. F. H. Goetz. "The spectral image processing system (SIPS)—interactive visualization and analysis
    # of imaging spectrometer data." Remote sensing of environment 44, no. 2-3 (1993): 145-163.
    size = img.shape
    ld = np.sqrt(tgt.T.dot(tgt))
    result = np.zeros((1, size[1]))
    for i in range(size[1]):
        x = img[:, i]
        lx = np.sqrt(x.T.dot(x))
        cos_angle = x.dot(tgt)/ (lx*ld)
        angle = np.arccos(np.clip(cos_angle, -1, 1))
        result[:, i] = 1/(abs(angle) + 1e-12)
    return result.T


def wgn(x, snr):
    # add white Gaussian noise to x with specific SNR
    snr = 10**(snr/10)
    xpower = np.sum(x**2)/len(x)
    npower = xpower / snr
    return np.random.randn(len(x)) * np.sqrt(npower)


class hCEM(object):
    def __init__(self, lamb, tolerance, max_iter, display, imgH, imgW):
        self.lamb = lamb
        self.tol = tolerance
        self.max_iter = max_iter
        self.display = display
        self.imgH = imgH
        self.imgW = imgW
    def __call__(self, X, d):
        
        w = []
        Energy = []
        X = X.transpose(1,0).astype(np.float32)
        d = d.transpose(1,0).astype(np.float32)
        N = X.shape[1]
        D = X.shape[0]
        y_old = np.ones([1, N])
        weight = np.ones([1, N]).astype(np.float32)
        eye = np.eye(D)
        for T in range(self.max_iter):
            X = X * np.tile(weight, [X.shape[0],1])
            R = np.matmul(X,X.transpose(1,0)) / N
            R_inv = np.linalg.pinv(R + 0.0001*eye)
            nominator = np.matmul( R_inv, d)
            denominator = np.matmul(d.transpose(1,0), nominator)
            w.append(nominator / denominator)
            y = np.matmul(w[-1].transpose(1,0), X)
            weight = 1 - np.exp(-1*(self.lamb * y).clip(-5,None))

            weight[weight<0] = 1e-3

            res = (np.linalg.norm(y_old,2)**2)/N - (np.linalg.norm(y,2)**2)/N
            Energy.append(np.linalg.norm(y,2)**2/N)
            y_old = y

            if np.abs(res) < self.tol:
                break
        return y

def hcem(img, tgt, imgH, imgW):
    lamb = 200
    tolerance = 1e-6
    max_iter = 100
    display = False
    X = img.transpose(1,0)
    d = tgt.transpose(1,0)
    t1 = time.time()
    hCEMdetector = hCEM(lamb, tolerance, max_iter, display, imgH, imgW, )
    w = hCEMdetector(X,d)
    print('HCEM time comsumption:{}.'.format(time.time()-t1))
    return w.transpose()

class ECEM(Detector):

    def __init__(self):
        Detector.__init__(self)
        self.windowsize = [1/4, 2/4, 3/4, 4/4]  # window size
        self.num_layer = 10  # the number of detection layers
        self.num_cem = 6  # the number of CEMs per layer
        self.Lambda = 1e-6  # the regularization coefficient
        self.show_proc = True  # show the process or not

    def parmset(self, **parm):
        self.windowsize = parm['windowsize']  # parameters
        self.num_layer = parm['num_layer']
        self.num_cem = parm['num_cem']
        self.Lambda = parm['Lambda']
        self.show_proc = parm['show_proc']

    def setlambda(self):
        switch = {
            'san': 1e-6,
            'san_noise': 6e-2,
            'syn_noise': 5e-3,
            'cup': 1e-1
        }
        if self.name in switch:
            return switch[self.name]
        else:
            return 1e-10

    def cem(self, img, tgt):
        size = img.shape   # get the size of image matrix
        lamda = random.uniform(self.Lambda/(1+self.Lambda), self.Lambda)  # random regularization coefficient
        R = np.dot(img, img.T/size[1])   # R = X*X'/size(X,2);
        w = np.dot(np.linalg.inv((R+lamda*np.identity(size[0]))), tgt)  # w = (R+lamda*eye(size(X,1)))\d ;
        result = np.dot(w.T, img)  # y=w'* X;
        return result

    def ms_scanning_unit(self, winowsize):
        d = self.img.shape[0]
        winlen = int(d*winowsize**2)
        size = self.imgt.shape  # get the size of image matrix
        result = np.zeros(shape=(int((size[0]-winlen+1)/2+1), size[1]))
        pos = 0
        if self.show_proc: print('Multi_Scale Scanning: size of window: %d' % winlen)
        for i in range(0, size[0]-winlen+1, 2):  # multi-scale scanning
            imgt_tmp = self.imgt[i:i+winlen-1, :]
            result[pos, :] = self.cem(imgt_tmp, imgt_tmp[:, -1])
            pos += 1
        return result

    def cascade_detection(self, mssimg):   # defult parameter configuration
        size = mssimg.shape
        result_forest = np.zeros(shape=(self.num_cem, size[1]))
        for i_layer in range(self.num_layer):
            if self.show_proc: print('Cascaded Detection layer: %d' % i_layer)  # show the process of cascade detection
            for i_num in range(self.num_cem):
                result_forest[i_num,:] = self.cem(mssimg, mssimg[:, -1])
            weights = dual_sigmoid(np.mean(result_forest, axis=0))  # sigmoid nonlinear function
            mssimg = mssimg*weights
        result = result_forest[:, 0:-1]
        return result

    def detect(self, imgt, pool_num=4):
        # self.imgt = np.hstack((self.img, self.tgt))
        self.imgt = imgt
        self.img = imgt[:,:-1]
        
        # p = multiprocessing.Pool(pool_num)  # multiprocessing
        # results = self.ms_scanning_unit(self.windowsize[0])  # Multi_Scale Scanning
        results = []
        for i in range(len(self.windowsize)):
            result = self.ms_scanning_unit(self.windowsize[i])
            results.append(result)
        # p.close()
        # p.join()
        mssimg = np.concatenate(results, axis=0)
        cadeimg = self.cascade_detection(mssimg)  # Cascaded Detection
        result = np.mean(cadeimg, axis=0)[:self.imgt.shape[1]].reshape(-1, 1)
        return result
    

def ecem(img, tgt):
    ecem = ECEM()
    ecem.parmset(**{'windowsize': [1/4, 2/4, 3/4, 4/4],   # window size
                    'num_layer': 10,  # the number of detection layers
                    'num_cem': 6,  # the number of CEMs per layer
                    'Lambda': 1e-1,  # the regularization coefficient
                    'show_proc': True})  # show the process or not
    t1 = time.time()
    imgt =  np.hstack((img, tgt))
    result = ecem.detect(imgt=imgt ,pool_num=4)  # detection (we recomemend to use multi-thread processing to speed up detetion)
    print('ECEM time comsumption:{}.'.format(time.time()-t1))
    # ecem.show([result], ['E-CEM'])  # show
    return result


def classic_detectors(img, prior, row, col):
    tradtion_img = copy.copy(img)
    init_map_ace = ace(tradtion_img.T, prior[:, np.newaxis])
    init_map_NAMD2 = NAMD(tradtion_img.T, prior[:, np.newaxis])
    init_map_RSA = RSA(tradtion_img.T, prior[:, np.newaxis])
    init_map_cem = cem(tradtion_img.T, prior[:, np.newaxis])

    init_map_hcem = hcem(tradtion_img.T, prior[:,np.newaxis], row, col)
    init_map_ecem = ecem(tradtion_img.T, prior[:,np.newaxis])
    init_map_hcem = init_map_hcem / init_map_hcem.max()
    init_map_ecem = init_map_ecem / init_map_hcem.max()
    classic_result = [[init_map_ace, init_map_NAMD2, init_map_RSA, init_map_hcem, init_map_ecem, init_map_cem], 
                        [ 'ACE__', 'NAMD2_', 'RSA__','HCEM_', 'ECEM__','CEM__']]
    return classic_result