"""
MixGen: A New Multi-Modal Data Augmentation
https://arxiv.org/abs/2206.08358
Apache-2.0 License, Copyright 2022 Amazon
"""
import random
import numpy as np
import torch
import pdb

def mixgen(image, text, num, lam=0.5):
    # default MixGen
    for i in range(num):
        # image mixup
        image[i,:] = lam * image[i,:] + (1 - lam) * image[i+num,:]
        # text concat
        text[i] = text[i] + " " + text[i+num]
    return image, text


def mixgen_batch(image, text, num, lam=0.5):
    batch_size = image.size()[0]
    index = np.random.permutation(batch_size)
    for i in range(batch_size):
        # image mixup
        image[i,:] = lam * image[i,:] + (1 - lam) * image[index[i],:]
        # text concat
        #text[i] = text[i] + " " + text[index[i]]
    return image#, text

def mixgen_pt(image, prompt, tokenized_prompts, label, num, lam=0.5):
    batch_size = image.size()[0]
    token_size = prompt.shape[1]
    
    image_aug = []
    prompt_aug = []
    tokenized_prompts_aug = []
    
    index = np.random.permutation(batch_size)
    image_aug = lam * image + (1-lam) * image[index, :]
    y_a, y_b = label, label[index]
    
    for i, j in zip(y_a, y_b):
        ctx_length = tokenized_prompts[i].nonzero().size(0) - 1 # remove EOS elems
        prompt_aug.append(torch.cat((prompt[i, :ctx_length], prompt[j, 1:token_size-ctx_length+1])))
        tokenized_prompts_aug.append(torch.cat((tokenized_prompts[i, :ctx_length], tokenized_prompts[j, 1:token_size-ctx_length+1])))
    
    prompt_aug = torch.stack(prompt_aug)
    tokenized_prompts_aug = torch.stack(tokenized_prompts_aug)
    
    return image_aug, prompt_aug, tokenized_prompts_aug, y_a, y_b
    