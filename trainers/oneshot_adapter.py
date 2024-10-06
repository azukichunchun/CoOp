import os.path as osp

import string
import random
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.transforms import ToPILImage

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.data.transforms.transforms import AVAI_CHOICES, build_transform

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from .imagenet_templates import IMAGENET_TEMPLATES, IMAGENET_TEMPLATES_SELECT

import pickle
import pdb

_tokenizer = _Tokenizer()


CUSTOM_TEMPLATES = {
    'OxfordPets': 'a photo of a {}, a type of pet.',
    'OxfordFlowers': 'a photo of a {}, a type of flower.',
    'FGVCAircraft': 'a photo of a {}, a type of aircraft.',
    'DescribableTextures': '{} texture.',
    'EuroSAT': 'a centered satellite photo of {}.',
    'StanfordCars': 'a photo of a {}.',
    'Food101': 'a photo of {}, a type of food.',
    'SUN397': 'a photo of a {}.',
    'Caltech101': 'a photo of a {}.',
    'UCF101': 'a photo of a person doing {}.',
    'ImageNet': 'a photo of a {}.',
    'ImageNetSketch': 'a photo of a {}.',
    'ImageNetV2': 'a photo of a {}.',
    'ImageNetA': 'a photo of a {}.',
    'ImageNetR': 'a photo of a {}.'
}

AVAI_CHOICES = [t for t in AVAI_CHOICES if t not in ["center_crop", "imagenet_policy", "cifar10_policy", "svhn_policy"]]


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)
    
    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location='cpu').eval()
        state_dict = None
    
    except RuntimeError:
        state_dict = torch.load(model_path, map_location='cpu')
    
    model = clip.build_model(state_dict or model.state_dict())

    return model


class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4):
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.fc(x)
        return x
    
    
class TextEncoder(nn.Module):

    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.cfg = cfg
        self.classnames = classnames
        self.clip_model = clip_model
        self.dtype = clip_model.dtype
    
        self.save = True
        

    def generate_random_string(self, length=3):
        characters = string.ascii_letters + string.punctuation
        return ''.join(random.choice(characters) for _ in range(length))


    def insert_random_strings_in_text(self, text, length):
        words = text.split()
        result = []
        for i in range(len(words) - 1):
            result.append(words[i])
            result.append(self.generate_random_string(length))
        result.append(words[-1])
        return ' '.join(result)


    def forward(self, augment_num=0, augment_length=3):
        temp = CUSTOM_TEMPLATES[self.cfg.DATASET.NAME]  
        prompts = [temp.format(c.replace('_', ' ')) for c in self.classnames]

        if augment_num > 0:
            prompts_aug = []
            for p in prompts:
                prompt_aug_ = []
                for _ in range(augment_num):
                    prompt_aug_.append(self.insert_random_strings_in_text(text=p, length=augment_length))
                prompts_aug.append(prompt_aug_)
            prompts = prompts_aug

        prompts = torch.cat([clip.tokenize(p) for p in prompts])

        prompts = prompts.to('cuda')
        text_features = self.clip_model.encode_text(prompts)
        x = text_features

        if self.save == True:
            with open(osp.join(self.cfg.OUTPUT_DIR, "txt_features_before_train.pkl"), "wb") as f:
                pickle.dump(x, f)
            self.save = False
        
        return x


class CustomCLIP(nn.Module):

    def __init__(self, cfg, classnames, clip_model, augment_num, augment_text_length, tfms_num):
        super().__init__()
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(cfg, classnames, clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.adapter = Adapter(512, 4).to(clip_model.dtype)
        self.cfg = cfg
        self.classnames = classnames

        self.augment_num = augment_num
        self.augment_text_length = augment_text_length
        self.tfms_num = tfms_num

        avai_choice = random.sample(AVAI_CHOICES, self.tfms_num) # データ拡張方法をランダムに選ぶ
        self.tfms = build_transform(self.cfg, is_train=True, choices=avai_choice)

        self.save = True

    def get_entropy(self, input_):
        bs = input_.size(0)
        epsilon = 1e-5
        entropy = -input_ * torch.log(input_ + epsilon)
        entropy = torch.sum(entropy, dim=1)
        return entropy 


    def compute_im_loss(self, txt_features, img_features):

        logits = self.logit_scale * img_features @ txt_features.T

        softmax_out = nn.Softmax(dim=1)(logits)
        entropy_loss = torch.mean(self.get_entropy(softmax_out))
        msoftmax = softmax_out.mean(dim=0)
        gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + 1e-6))
        im_loss = entropy_loss - gentropy_loss
        return im_loss


    def compute_transport_loss(self, txt_features, img_features):

        sim_t = img_features @ txt_features.T
        logits = self.logit_scale * sim_t

        s_dist = F.softmax(logits, dim=1)
        t_dist = F.softmax(logits, dim=0)
        cost = 1 - sim_t
        s_cost = (cost * s_dist).sum(1).mean()
        t_cost = (cost * t_dist).sum(0).mean()
        return s_cost + t_cost            


    def augment_images(self, image):
        
        to_pil = ToPILImage()
        img_aug = []
        for img in image:
            img = to_pil(img)
            img_aug.extend([self.tfms(img) for _ in range(self.augment_num)])
        
        return torch.stack(img_aug).to(image.device)


    def forward(self, image, label=None):
        # 言語特徴量のデータ拡張
        text_features = self.text_encoder(self.augment_num, self.augment_text_length)
        
        # 画像特徴量のデータ拡張
        if self.augment_num > 0:
            image = self.augment_images(image)
            label = label.repeat_interleave(self.augment_num)
        
        image_features = self.image_encoder(image.type(self.dtype))
        
        if self.save == True:
            with open(osp.join(self.cfg.OUTPUT_DIR, "img_features_before_train.pkl"), "wb") as f:
                pickle.dump(image_features, f)
            self.save = False

        x = self.adapter(image_features)

        ratio = 0.2
        image_features = ratio * x + (1 - ratio) * image_features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        # 輸送損失と相互情報量損失の計算
        transport_loss = self.compute_transport_loss(text_features, image_features)
        im_loss = self.compute_im_loss(text_features, image_features)

        cross_entropy_loss = F.cross_entropy(logits, label)
       
        if self.adapter.training:
            total_loss = cross_entropy_loss + transport_loss + im_loss
            return total_loss
        
        else:
            return logits

    def inference(self, image):
        image_features = self.image_encoder(image.type(self.dtype))
        x = self.adapter(image_features)

        ratio = 0.2
        image_features = ratio * x + (1 - ratio) * image_features

        text_features = self.text_encoder()

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits


@TRAINER_REGISTRY.register()
class OneShot_Adapter(TrainerX):
    """ CLIP-Adapter """

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        self.templates = IMAGENET_TEMPLATES
        # add custom-made prompt
        if cfg.DATASET.NAME != "ImageNet":
            self.templates += [CUSTOM_TEMPLATES[cfg.DATASET.NAME]]


        print(f'Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})')
        clip_model = load_clip_to_cpu(cfg)
        clip_model.float()


        print('Building custom CLIP')
        self.model = CustomCLIP(cfg, classnames, clip_model, 
                                augment_num=16, augment_text_length=3, tfms_num=3)


        print('Turning off gradients in both the image and the text encoder')
        for name, param in self.model.named_parameters():
            if 'adapter' not in name:
                param.requires_grad_(False)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.adapter, cfg.MODEL.INIT_WEIGHTS)

        clip_model.to(self.device)
        self.text_features_template = []
        self.text_features_template_label = []
        for i, temp in enumerate(self.templates):
            prompts = [temp.format(c.replace("_", " ")) for c in classnames]
            prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(self.device)
            text_features = clip_model.encode_text(prompts)
            self.text_features_template.append(text_features)
        self.text_features_template = torch.stack(self.text_features_template)

        with open(osp.join(self.cfg.OUTPUT_DIR, "text_features_template.pkl"), "wb") as f:
            pickle.dump(self.text_features_template, f)

        
        self.model.to(self.device)
        # NOTE: only give text_encoder.adapter to the optimizer
        self.optim = build_optimizer(self.model.adapter, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        

        self.register_model('clip_adapter', self.model.adapter, self.optim, self.sched)

        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f'Multiple GPUs detected (n_gpus={device_count}), use all of them!')
            self.model = nn.DataParallel(self.model)


    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        #output = self.model(image, label=label, augment_num=10, augment_text_length=3, tfms_num=3)
        loss = self.model(image, label=label)
        
        self.model_backward_and_update(loss)
    
        loss_summary = {
            'loss': loss.item()
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary
    

    def parse_batch_train(self, batch):
        input = batch['img']
        label = batch['label']
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label
    

    def model_inference(self, input):
        return self.model.inference(input)
    

    def load_model(self, directory, epoch=None):
        if not directory:
            print(
                'Note that load_model() is skipped as no pretrained model is given'
            )
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = 'model-best.pth.tar'

        if epoch is not None:
            model_file = 'model.pth.tar-' + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError(
                    'Model not found at "{}"'.format(model_path)
                )

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint['state_dict']
            epoch = checkpoint['epoch']
            
            # Ignore fixed token vectors
            if 'token_prefix' in state_dict:
                del state_dict['token_prefix']
            
            if 'token_suffix' in state_dict:
                del state_dict['token_suffix']

            print(
                'Loading weights to {} '
                'from "{}" (epoch = {})'.format(name, model_path, epoch)
            )
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
