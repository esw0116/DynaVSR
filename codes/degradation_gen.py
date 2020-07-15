import os
import math
import numpy as np
import torch


def main():
    # Set config
    mode = 'REDS'  # 'REDS'
    save_dir = '../experiments/pretrained_models'

    num_gen = 400 if mode == 'REDS' else 171

    generation = np.zeros((num_gen, 4))

    for i in range(num_gen):
        degrade_type = np.random.random_sample()
        sigma_x = 0.001 + np.random.random_sample() * 4
        sigma_y = 0.001 + np.random.random_sample() * 4
        theta = np.random.random_sample() * math.pi / 2
        generation[i] = [degrade_type, sigma_x, sigma_y, theta]

    gen_tensor = torch.from_numpy(generation)
    save_path = os.path.join(save_dir, mode+'.pth')
    torch.save(gen_tensor, save_path)


if __name__ == '__main__':
    main()
