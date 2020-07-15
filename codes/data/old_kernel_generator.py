import torch
from torch.nn.functional import conv2d

base_kernel_dict = {
    'impulse': {2: [1], 4: [1]},
    'box': {2: [0.5, 0.5], 4: [0.25, 0.25, 0.25, 0.25]},
    'bilinear': {2: [0.125, 0.375, 0.375, 0.125],
                 4: [0.03125, 0.09375, 0.15625, 0.21875, 0.21875, 0.15625, 0.09375, 0.03125]},
    'bicubic': {2: [-0.01171875, -0.03515625, 0.11328125, 0.43359375, 0.43359375, 0.11328125, -0.03515625, -0.01171875],
                4: [-0.00170898, -0.01098633, -0.01831055, -0.01196289, 0.02270508, 0.09741211, 0.18188477, 0.2409668,
                    0.2409668, 0.18188477, 0.09741211, 0.02270508, -0.01196289, -0.01831055, -0.01098633, -0.00170898]},
    'lanczos': {2: [-0.00886333, -0.04194002, 0.1165001, 0.43430325, 0.43430325, 0.1165001, -0.04194002, -0.00886333],
                4: [-0.00106538, -0.0097517, -0.02038398, -0.01487788, 0.02459407, 0.0986585, 0.18311527, 0.2397111,
                    0.2397111, 0.18311527, 0.0986585, 0.02459407, -0.01487788, -0.02038398, -0.0097517, -0.00106538]}
}


class Degradation:
    def __init__(self, kernel_size, scale_factor, type=0.0, theta=0.0, sigma=[1.0, 1.0]):
        self.kernel_size = kernel_size
        self.scale = scale_factor

        self.basis_kernel = None
        self.b_kernel_size = 21
        # Type : Impulse, Box, Bilinear, Bicubic, Lanczos2
        self.type = type
        self.build_base_kernel()

        self.Gaussian_kernel = None
        self.G_kernel_size = 21
        self.theta = torch.tensor([theta]).cuda()
        self.sigma = torch.tensor(sigma).cuda()
        self.build_G_kernel()

        self.kernel = self.convolve_kernel()

    def set_parameters(self, sigma, theta):
        self.sigma = sigma
        self.theta = theta

    def build_base_kernel(self):
        kernel_center = self.b_kernel_size // 2
        kernel = torch.zeros((self.b_kernel_size, self.b_kernel_size)).cuda()
        '''
        if self.type < 0.2:
            type_name = 'impulse'
        elif self.type < 0.4:
            type_name = 'box'
        elif self.type < 0.6:
            type_name = 'bilinear'
        elif self.type < 0.8:
            type_name = 'bicubic'
        else:
            type_name = 'lanczos'
        '''
        if self.type < 0.5:
            type_name = 'impulse'
        else:
            type_name = 'bicubic'

        kernel_1d = torch.Tensor(base_kernel_dict[type_name][self.scale]).cuda()
        k_length = kernel_1d.size(0)
        kernel_2d = kernel_1d[:, None] * kernel_1d[None]
        l_length, r_length = (k_length - 1) // 2, (k_length + 2) // 2

        kernel[kernel_center - l_length:kernel_center + r_length,
        kernel_center - l_length:kernel_center + r_length] = kernel_2d
        kernel = kernel / kernel.sum()
        self.basis_kernel = kernel

    def build_G_kernel(self):
        if self.sigma[0] == 0 and self.sigma[1] == 0:
            kernel = self.sigma.new_zeros((self.G_kernel_size, self.G_kernel_size))
            kernel[self.G_kernel_size // 2, self.G_kernel_size // 2] = 1
        else:
            kernel_radius = self.G_kernel_size // 2
            kernel_range = torch.linspace(-kernel_radius, kernel_radius, self.G_kernel_size)
            if self.sigma.is_cuda:
                kernel_range = kernel_range.cuda()

            horizontal_range = kernel_range[None].repeat((self.G_kernel_size, 1))
            vertical_range = kernel_range[:, None].repeat((1, self.G_kernel_size))

            cos_theta = self.theta.cos()
            sin_theta = self.theta.sin()

            cos_theta_2 = cos_theta ** 2
            sin_theta_2 = sin_theta ** 2

            sigma_x_2 = 2.0 * (self.sigma[0] ** 2)
            sigma_y_2 = 2.0 * (self.sigma[1] ** 2)

            a = cos_theta_2 / sigma_x_2 + sin_theta_2 / sigma_y_2
            b = sin_theta * cos_theta * (1.0 / sigma_y_2 - 1.0 / sigma_x_2)
            c = sin_theta_2 / sigma_x_2 + cos_theta_2 / sigma_y_2

            gaussian = lambda x, y: (- (a * (x ** 2) + 2.0 * b * x * y + c * (y ** 2))).exp()

            kernel = gaussian(horizontal_range, vertical_range)
            kernel = kernel / kernel.sum()
        self.Gaussian_kernel = kernel

    def convolve_kernel(self):
        # Combine basis kernel, Gaussian kernel
        # We should flip the second kernel when combine two kernels, but we can skip it because Gaussian kernel is 180 rotation invariant
        base_kernel = self.basis_kernel.repeat((1, 1, 1, 1))
        Gaussian_kernel = self.Gaussian_kernel.repeat((1, 1, 1, 1))
        pad_length = self.G_kernel_size // 2
        final_kernel = conv2d(base_kernel, Gaussian_kernel, padding=pad_length)
        final_kernel = final_kernel.squeeze()
        center = final_kernel.size(0) // 2
        radius = self.kernel_size // 2
        final_kernel = final_kernel[center - radius:center + radius + 1, center - radius:center + radius + 1]
        final_kernel = final_kernel / final_kernel.sum()

        return final_kernel

    def set_kernel_directly(self, kernel):
        self.kernel = kernel

    def get_kernel(self):
        return self.kernel

    def get_features(self):
        return torch.reshape(self.kernel, (self.kernel_size ** 2,))

    def apply(self, img):
        cu = False
        if img.is_cuda:
            cu = True

        weights = self.kernel.repeat((3, 1, 1, 1)).cuda()
        img = img.cuda()
        pad_func = torch.nn.ReflectionPad2d(self.kernel_size // 2)
        dimension = 4
        if img.ndim == 3:
            # Single Image C H W
            dimension = 3
            img = img[None]
        img = pad_func(img)
        lr_img = conv2d(img, weights, groups=3, stride=int(self.scale))

        if not cu:
            lr_img = lr_img.cpu()
        if dimension == 3:
            lr_img = lr_img[0]

        return lr_img