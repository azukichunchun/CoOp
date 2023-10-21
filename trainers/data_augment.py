from PIL import Image
import numpy as np
import random
import torch.nn as nn

class BlockShuffle(nn.Module):
    def __init__(self, grid):
        super().__init__()
        self.grid = grid
    
    def forward(self, img):
        s = int(img.size[0]/self.grid)
        tile = [img.crop(np.array([s * (n % self.grid), s * int(n / self.grid), 
                                   s * (n % self.grid + 1), s * (int(n / self.grid) + 1)]).astype(int)) 
                for n in range(self.grid**2)]
        
        random.shuffle(tile)
        
        dst = Image.new('RGB', (int(s * self.grid), int(s * self.grid)))
        for i, t in enumerate(tile):
            dst.paste(t, (i % self.grid * s, int(i / self.grid) * s)) 
            
        return dst
    
    
class PhaseMasking(nn.Module):
    def __init__(self, ratio):
        super().__init__()
        self.ratio = ratio
        
    def forward(self, img):
        
        img = np.array(img)
        
        img_fft = np.fft.fft2(img, axes=(0, 1))
        img_abs, img_pha = np.abs(img_fft), np.angle(img_fft)
        
        # put zero randomly into phase matrix
        zero_indices = np.random.choice(img_pha.size, int(img_pha.size*self.ratio), replace=False)
        np.put(img_pha, zero_indices, 0)

        img = img_abs * np.e ** (1j * img_pha)
        img = np.real(np.fft.ifft2(img, axes=(0, 1)))
        img = np.uint8(np.clip(img, 0, 255))
            
        dst = Image.fromarray(img)
            
        return dst