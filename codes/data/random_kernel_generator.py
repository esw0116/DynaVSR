import numpy as np
from scipy.ndimage import measurements, interpolation
import torch
from torch.nn.functional import conv2d


class Degradation:
    def __init__(self, kernel_size, scale_factor, theta=0.0, sigma=[1.0, 1.0]):
        self.kernel_size = kernel_size
        self.scale = scale_factor

        self.theta = theta
        self.sigma = sigma
        self.build_kernel()

    def set_parameters(self, sigma, theta):
        self.sigma = sigma
        self.theta = theta

    def build_kernel(self):
        if self.sigma[0] == 0 and self.sigma[1] == 0:
            kernel = np.zeros((self.kernel_size, self.kernel_size))
            kernel[self.kernel_size // 2, self.kernel_size // 2] = 1
        else:
            kernel_radius = self.kernel_size // 2
            kernel_range = np.linspace(-kernel_radius, kernel_radius, self.kernel_size)

            # horizontal_range = kernel_range[None].repeat((self.kernel_size, 1))
            # vertical_range = kernel_range[:, None].repeat((1, self.kernel_size))
            horizontal_range, vertical_range = np.meshgrid(kernel_range, kernel_range)

            cos_theta = np.cos(self.theta)
            sin_theta = np.sin(self.theta)

            cos_theta_2 = cos_theta ** 2
            sin_theta_2 = sin_theta ** 2

            sigma_x_2 = 2.0 * (self.sigma[0] ** 2)
            sigma_y_2 = 2.0 * (self.sigma[1] ** 2)

            a = cos_theta_2 / sigma_x_2 + sin_theta_2 / sigma_y_2
            b = sin_theta * cos_theta * (1.0 / sigma_y_2 - 1.0 / sigma_x_2)
            c = sin_theta_2 / sigma_x_2 + cos_theta_2 / sigma_y_2

            gaussian = lambda x, y: np.exp((- (a * (x ** 2) + 2.0 * b * x * y + c * (y ** 2))))
            kernel = gaussian(horizontal_range, vertical_range)
            kernel = kernel / kernel.sum()
        self.kernel = kernel

    def kernel_shift(self, kernel):
        # There are two reasons for shifting the kernel:
        # 1. Center of mass is not in the center of the kernel which creates ambiguity. There is no possible way to know
        #    the degradation process included shifting so we always assume center of mass is center of the kernel.
        # 2. We further shift kernel center so that top left result pixel corresponds to the middle of the sfXsf first
        #    pixels. Default is for odd size to be in the middle of the first pixel and for even sized kernel to be at the
        #    top left corner of the first pixel. that is why different shift size needed between od and even size.
        # Given that these two conditions are fulfilled, we are happy and aligned, the way to test it is as follows:
        # The input image, when interpolated (regular bicubic) is exactly aligned with ground truth.

        # First calculate the current center of mass for the kernel
        current_center_of_mass = measurements.center_of_mass(kernel)

        # The second ("+ 0.5 * ....") is for applying condition 2 from the comments above
        wanted_center_of_mass = np.array(kernel.shape) // 2 + 0.5 * (self.scale - (kernel.shape[0] % 2))
        # wanted_center_of_mass = np.array(kernel.shape) / 2 + 0.5 * (np.array(sf)[0:2] - (kernel.shape[0] % 2))

        # Define the shift vector for the kernel shifting (x,y)
        shift_vec = wanted_center_of_mass - current_center_of_mass

        # Before applying the shift, we first pad the kernel so that nothing is lost due to the shift
        # (biggest shift among dims + 1 for safety)
        kernel = np.pad(kernel, np.int(np.ceil(np.max(shift_vec))) + 1, 'constant')

        # Finally shift the kernel and return
        return interpolation.shift(kernel, shift_vec)

    def get_kernel(self):
        return self.kernel

    def set_kernel_directly(self, kernel):
        self.kernel = kernel

    def apply(self, img):
        cu = False
        if img.is_cuda:
            cu = True

        if self.kernel.ndim == 2:
            shifted_kernel = self.kernel_shift(self.kernel)
            shifted_kernel = torch.from_numpy(shifted_kernel).float()
            shifted_kernel_length = shifted_kernel.shape[0]
            weights = shifted_kernel.repeat((3,1,1,1)) #.cuda()
            img = img #.cuda()
            pad_func = torch.nn.ReflectionPad2d(shifted_kernel_length // 2)
            dimension = 4
            if img.ndim == 3:
                # Single Image C(=3) H W
                dimension = 3
                img = img[None]
            img = pad_func(img)
            lr_img = conv2d(img, weights, groups=3, stride=int(self.scale))
        else:
            # self.kernel.ndim = 3 (T X 11 X 11)
            assert img.ndim == 4  # T C H W
            assert img.shape[0] == self.kernel.shape[0] or img.shape[0] == self.kernel.shape[0] + 2  # EDVR, DUF
            dimension = 4
            lr_img = []
            for i in range(img.shape[0]):
                if img.shape[0] == self.kernel.shape[0]:
                    shifted_kernel = self.kernel_shift(self.kernel[i])
                else:
                    shifted_kernel = self.kernel_shift(self.kernel[(i-1) % self.kernel.shape[0]])
                # shifted_kernel = np.stack(shifted_kernel, axis=0)
                shifted_kernel = torch.from_numpy(shifted_kernel).float()
                shifted_kernel_length = shifted_kernel.shape[0]
                weights = shifted_kernel.repeat((3,1,1,1)) #.cuda()
                img_slice = img[i:i+1] #.cuda()  # 1 C H W
                pad_func = torch.nn.ReflectionPad2d(shifted_kernel_length // 2)
                img_slice = pad_func(img_slice)
                lr_img_slice = conv2d(img_slice, weights, groups=3, stride=int(self.scale))  # 1 C H W
                lr_img.append(lr_img_slice)

            lr_img = torch.cat(lr_img, dim=0)

        if not cu:
            lr_img = lr_img.cpu()
        if dimension == 3:
            lr_img = lr_img[0]
        torch.cuda.empty_cache()
        return lr_img
