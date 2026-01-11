import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image

class ImageProcessor:
    def __init__(self):
        self.sobel_kernels = self._get_sobel_kernels()

    def _get_sobel_kernels(self):
        kernels = [[[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                   [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]
        kernels = np.asarray(kernels)  # 2,3,3
        kernels = np.expand_dims(kernels, 1).repeat(3, axis=1)
        kernels = torch.from_numpy(kernels.astype(np.float32))
        return kernels

    def _apply_sobel(self, x):
        x = x.float()
        self.sobel_kernels = self.sobel_kernels.to(x.device)
        sob = F.conv2d(x, self.sobel_kernels, stride=1, padding=1)
        return torch.sum(sob, dim=1, keepdim=True).repeat(1, 3, 1, 1)

    def _apply_sobel2(self, x):
        x = x.float()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(x.device)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(x.device)

        b, c, h, w = x.shape
        sobel_output = torch.zeros((b, c, h, w), dtype=torch.float32, device=x.device)

        for i in range(c):
            channel = x[:, i:i + 1, :, :]
            grad_x = F.conv2d(channel, sobel_x, padding=1)
            grad_y = F.conv2d(channel, sobel_y, padding=1)
            grad = grad_x + grad_y
            sobel_output[:, i, :, :] = grad.squeeze(1)

        return sobel_output

    def apply_sobel_filter(self, x):
        x = x.float()
        x = x / 255.0
        batch_size, channels, height, width = x.shape

        sobel_x = torch.tensor([[-1.0, 0.0, 1.0],
                                [-2.0, 0.0, 2.0],
                                [-1.0, 0.0, 1.0]]).unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1, 1)
        sobel_y = torch.tensor([[-1.0, -2.0, -1.0],
                                [0.0, 0.0, 0.0],
                                [1.0, 2.0, 1.0]]).unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1, 1)

        sobel_x = sobel_x.to(x.device)
        sobel_y = sobel_y.to(x.device)

        grad_x = F.conv2d(x, sobel_x, padding=1, groups=channels)
        grad_y = F.conv2d(x, sobel_y, padding=1, groups=channels)

        grad_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2)

        return grad_magnitude

    def _convert_to_hsv(self, x):
        batch_size = x.shape[0]
        hsv_images = []
        for i in range(batch_size):
            img = x[i].permute(1, 2, 0).cpu().numpy()  # CHW to HWC
            hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            hsv_images.append(torch.from_numpy(hsv_img).permute(2, 0, 1))  # HWC to CHW
        return torch.stack(hsv_images)

    def _convert_to_lab(self, x):
        batch_size = x.shape[0]
        lab_images = []
        for i in range(batch_size):
            img = x[i].permute(1, 2, 0).cpu().numpy()  # CHW to HWC
            lab_img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            lab_images.append(torch.from_numpy(lab_img).permute(2, 0, 1))  # HWC to CHW
        return torch.stack(lab_images)

    def process_image(self, x, mode='sobel'):
        if mode == 'sobel':
            return self._apply_sobel(x)
        if mode == 'sobel2':
            return self._apply_sobel2(x)
        if mode == 'sobel3':
            return self.apply_sobel_filter(x)
        elif mode == 'hsv':
            return self._convert_to_hsv(x)
        elif mode == 'lab':
            return self._convert_to_lab(x)
        else:
            raise ValueError("Mode must be 'sobel', 'hsv' or 'lab'")