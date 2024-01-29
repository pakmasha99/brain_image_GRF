#from scipy.ndimage import gaussian_filter
from skimage import transform as sk_tf
from collections import namedtuple
import numpy as np
import numbers
import pdb
import torchvision.transforms as tf


def interval(obj, lower=None):
    """ Listify an object.

    Parameters
    ----------
    obj: 2-uplet or number
        the object used to build the interval.
    lower: number, default None
        the lower bound of the interval. If not specified, a symetric
        interval is generated.

    Returns
    -------
    interval: 2-uplet
        an interval.
    """
    if isinstance(obj, numbers.Number):
        if obj < 0:
            raise ValueError("Specified interval value must be positive.")
        if lower is None:
            lower = -obj
        return (lower, obj)
    if len(obj) != 2:
        raise ValueError("Interval must be specified with 2 values.")
    min_val, max_val = obj
    if min_val > max_val:
        raise ValueError("Wrong interval boudaries.")
    return tuple(obj)

#========CHANGED=========#
Transform = namedtuple("Transform", ["transform", "probability"]) #CHANGED!! https://stackoverflow.com/questions/16377215/how-to-pickle-a-namedtuple-instance-correctly 
#========================#
class Transformer(object):
    """ Class that can be used to register a sequence of transformations.
    """
    #Transform = namedtuple("Transform", ["transform", "probability"])

    def __init__(self):
        """ Initialize the class.
        """
        self.transforms = []

    def register(self, transform, probability=1):
        """ Register a new transformation.
        Parameters
        ----------
        transform: callable
            the transformation object.
        probability: float, default 1
            the transform is applied with the specified probability.
        """
        trf = Transform(transform=transform, probability=probability, ) #CHANGED as per the "Transform = namedtupleXXX" thing 
        self.transforms.append(trf)

    def __call__(self, arr):
        """ Apply the registered transformations.
        """
        transformed = torch.clone(arr)#arr.copy()
        for trf in self.transforms:
            if np.random.rand() < trf.probability:
                transformed = trf.transform(transformed)
                #print(f"trf applied : {trf}, dtype : {type(transformed)}") #use for finding bugs

        return transformed

    def __str__(self):
        if len(self.transforms) == 0:
            return '(Empty Transformer)'
        s = 'Composition of:'
        for trf in self.transforms:
            s += '\n\t- '+trf.__str__()
        return s


class Normalize(object):
    def __init__(self, mean=0.0, std=1.0, eps=1e-8):
        self.mean=mean
        self.std=std
        self.eps=eps

    def __call__(self, arr):
        return self.std * (arr - torch.mean(arr))/(torch.std(tnsr, unbiased = False) + self.eps) + self.mean

class Crop(object):
    """Crop the given n-dimensional array either at a random location or centered"""
    def __init__(self, shape, type="center", resize=False, keep_dim=False):
        """:param
        shape: tuple or list of int
            The shape of the patch to crop
        type: 'center' or 'random'
            Whether the crop will be centered or at a random location
        resize: bool, default False
            If True, resize the cropped patch to the inital dim. If False, depends on keep_dim
        keep_dim: bool, default False
            if True and resize==False, put a constant value around the patch cropped. If resize==True, does nothing
        """
        assert type in ["center", "random"]
        self.shape = shape
        self.copping_type = type
        self.resize=resize
        self.keep_dim=keep_dim

    def __call__(self, arr):
        #assert isinstance(arr, np.ndarray) #removed, cuz we're using torch tensor
        device = arr.get_device() #added, to chagne ot numpy to do sk_tf
        arr = np.array(arr.cpu())
        
        
        
        assert type(self.shape) == int or len(self.shape) == len(arr.shape), "Shape of array {} does not match {}".\
            format(arr.shape, self.shape)

        img_shape = np.array(arr.shape)
        if type(self.shape) == int:
            size = [self.shape for _ in range(len(self.shape))]
        else:
            size = np.copy(self.shape)
        indexes = []
        for ndim in range(len(img_shape)):
            if size[ndim] > img_shape[ndim] or size[ndim] < 0:
                size[ndim] = img_shape[ndim]
            if self.copping_type == "center":
                delta_before = (img_shape[ndim] - size[ndim]) / 2.0
            elif self.copping_type == "random":
                delta_before = np.random.randint(0, img_shape[ndim] - size[ndim] + 1)
            indexes.append(slice(int(delta_before), int(delta_before + size[ndim])))
        if self.resize:
            # resize the image to the input shape
            resized_arr = sk_tf.resize(arr[tuple(indexes)], img_shape, preserve_range=True) 
            resized_arr = torch.from_numpy(resized_arr).float().to(device) #added, to move back to gpu
            return resized_arr

        if self.keep_dim:
            mask = np.zeros(img_shape, dtype=np.bool)
            mask[tuple(indexes)] = True
            arr_copy = arr.copy()
            arr_copy[~mask] = 0
            arr_copy = torch.from_numpy(arr_copy).float().to(device) #added, to move back to gpu
            return arr_copy

        arr = torch.from_numpy(arr).float().to(device) #added, to move back to gpu

        return arr[tuple(indexes)]


class Cutout(object):
    """Apply a cutout on the images
    cf. Improved Regularization of Convolutional Neural Networks with Cutout, arXiv, 2017
    We assume that the square to be cut is inside the image.
    """
    def __init__(self, patch_size=None, value=0, random_size=False, inplace=False, localization=None):
        self.patch_size = patch_size
        self.value = value
        self.random_size = random_size
        self.inplace = inplace
        self.localization = localization

    def __call__(self, arr):
        device = device = arr.get_device() #ADDED
        img_shape = torch.tensor(arr.shape, device = device)        #CHANGED

        if type(self.patch_size) == int: #doesn't get used cuz not tuple
            size = [self.patch_size for _ in range(len(img_shape))]
        else:                            #gets used cuz tuple
            size = torch.tensor(self.patch_size, device = device)    #CHANGED

        assert len(size) == len(img_shape), "Incorrect patch dimension."
        indexes = []
        for ndim in range(len(img_shape)):
            if size[ndim] > img_shape[ndim] or size[ndim] < 0:
                size[ndim] = img_shape[ndim]
            if self.random_size:
                size[ndim] = np.random.randint(0, size[ndim])      #CHANGE (not changed yet, as not used)
            if self.localization is not None:
                delta_before = max(self.localization[ndim] - size[ndim]//2, 0)
            else:
                delta_before = torch.randint(low = 0, high = int(img_shape[ndim] - size[ndim]+1), size =(1,), device = device)[0] #CHANGED #fixed with int thingie
                

            indexes.append(slice(int(delta_before), int(delta_before + size[ndim])))
        if self.inplace:
            arr[tuple(indexes)] = self.value
            return arr
        else:
            arr_cut = arr.clone() #CHANGED
            arr_cut[tuple(indexes)] = self.value
            return arr_cut

class Flip(object):
    """ Apply a random mirror flip."""
    def __init__(self, axis=None):
        '''
        :param axis: int, default None
            apply flip on the specified axis. If not specified, randomize the
            flip axis.
        '''
        self.axis = axis

    def __call__(self, arr):
        if self.axis is None:
            device = arr.get_device()
            axis = torch.randint(low = 0, high = arr.ndim, size =(1,), device = device)[0]
        return torch.flip(arr, [axis]) #np.flip(arr, axis=(self.axis or axis))


class Blur(object):
    def __init__(self, snr=None, sigma=None):
        """ Add random blur using a Gaussian filter.
        Parameters
        ----------
        snr: float, default None
        the desired signal-to noise ratio used to infer the standard deviation
        for the noise distribution.
        sigma: float or 2-uplet
        the standard deviation for Gaussian kernel.
        """
        if snr is None and sigma is None:
            raise ValueError("You must define either the desired signal-to noise "
                             "ratio or the standard deviation for the noise "
                             "distribution.")
        self.snr = snr
        self.sigma = sigma

    def __call__(self, arr):
        arr = arr.float() #torch는 double tensor안쓰기 때문에, float32 tensor로 무조건 바꿔야함 
        
        sigma = self.sigma
        device = arr.get_device()
        if self.snr is not None:
            s0 = torch.std(arr, unbiased= False).item() #cpu 에서 보니 빼오기 
            sigma = s0/self.snr
        sigma = interval(sigma, lower=0)
        sigma_random = np.random.uniform(low=sigma[0], high=sigma[1], size=1)[0] #이거 numpy상에서 해도 괜찮을듯 
        
        ##MAJOR ADDITION
        kernel_size = 2*np.ceil((sigma_random -1)/2)+1 #make sure kernel-size is odd nubmered (unless padding을 asymmetric하게 해야할때가 나옴)
        opt_pad = int(((-1+kernel_size))/2)       
        arr = arr.unsqueeze(0) #add channel dimension (b/c torch 에 넣어서 돌릴 것이니) #이것은 single image가정할떄?
        arr = F.pad(arr,(opt_pad,)*6, mode = "replicate")#replicate the edge, because we know that the rest should be similar 

        blur = GaussianSmoothing(channels = 1, kernel_size=kernel_size, sigma=sigma_random, dim = 3).to(device)
        
        #print("위에서 channels 갯수, img_size등등을 args pass할때 주는 것으로 가져오게 하기! (shape을 매번 보게하는 것은 불편하니)")         
        return blur(arr)[0] #remove channel dim, as it should


class Noise(object):
    def __init__(self, snr=None, sigma=None, noise_type="gaussian"):
        """ Add random Gaussian or Rician noise.
           The noise level can be specified directly by setting the standard
           deviation or the desired signal-to-noise ratio for the Gaussian
           distribution. In the case of Rician noise sigma is the standard deviation
           of the two Gaussian distributions forming the real and imaginary
           components of the Rician noise distribution.
           In anatomical scans, CNR values for GW/WM ranged from 5 to 20 (1.5T and
           3T) for SNR around 40-100 (http://www.pallier.org/pdfs/snr-in-mri.pdf).
           Parameters
           ----------
           snr: float, default None
               the desired signal-to noise ratio used to infer the standard deviation
               for the noise distribution.
           sigma: float or 2-uplet, default None
               the standard deviation for the noise distribution.
           noise_type: str, default 'gaussian'
               the distribution of added noise - can be either 'gaussian' for
               Gaussian distributed noise, or 'rician' for Rice-distributed noise.
        """

        if snr is None and sigma is None:
            raise ValueError("You must define either the desired signal-to noise "
                             "ratio or the standard deviation for the noise "
                             "distribution.")
        assert noise_type in {"gaussian", "rician"}, "Noise muse be either Rician or Gaussian"
        self.snr = snr
        self.sigma = sigma
        self.noise_type = noise_type

    def __call__(self, arr):
        sigma = self.sigma
        if self.snr is not None:
            s0 = np.std(arr)
            sigma = s0 / self.snr
        sigma = interval(sigma, lower=0)
        #sigma_random = np.random.uniform(low=sigma[0], high=sigma[1], size=1)[0] ##change
        sigma_random = torch.cuda.FloatTensor(1).uniform_(0,1).item() #CHANGED

        #noise = np.random.normal(0, sigma_random, [2] + list(arr.shape)) ##change


        device = arr.get_device()
        noise = torch.cuda.FloatTensor(*([2]+list(arr.shape)), device = device).normal_()*sigma_random #이것이, cuda device에 만드는 것것!
        if self.noise_type == "gaussian":
            transformed = arr + noise[0]
        #elif self.noise_type == "rician": #not used (also, not chagned to tensor anyways)
        #    transformed = np.square(arr + noise[0])
        #    transformed += np.square(noise[1])
        #    transformed = np.sqrt(transformed)      
        return transformed
######below : COMPLETELY NEW, ADDED (FOR GAUSSIAN BLRU USING NN###
import math
import numbers
import torch
from torch import nn
from torch.nn import functional as F

class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / std) ** 2 / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups)
