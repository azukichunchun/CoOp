import os.path as osp
import time
import pdb
import copy
from tqdm import tqdm
import datetime
import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.utils import (
    MetricMeter, AverageMeter, load_checkpoint, load_pretrained_weights
)

from sklearn_extra.cluster import KMedoids
from sklearn.linear_model import LinearRegression
from mmd import MMD_loss
from mixgen import mixgen_pt, mixgen_batch
from sklearn.manifold import TSNE
from scipy.sparse.csgraph import connected_components
import matplotlib.pyplot as plt
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from .transforms import get_tuning_transform

from contrastive import Proximity, Con_Proximity

_tokenizer = _Tokenizer()

def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)
    
    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        
        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.DOCOOP.N_CTX
        ctx_init = cfg.TRAINER.DOCOOP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        vis_dim = clip_model.visual.output_dim
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            if cfg.TRAINER.DOCOOP.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.vis_dim = vis_dim
        self.dtype = dtype
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.DOCOOP.CLASS_TOKEN_POSITION

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                ctx,     # (n_cls, n_ctx, dim)
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        )

        return prompts


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        
        self.n_cls = self.prompt_learner.n_cls
        self.vis_dim =self.prompt_learner.vis_dim
        self.dtype = clip_model.dtype
        self.criterion_conprox = Con_Proximity(num_classes=self.n_cls, 
                                               feat_dim=self.vis_dim,
                                               dtype=self.dtype,
                                               use_gpu=torch.cuda.is_available())

        self.criterion_prox = Proximity(num_classes=self.n_cls, 
                                        feat_dim=self.vis_dim,
                                        dtype=self.dtype,
                                        use_gpu=torch.cuda.is_available())

        self.logit_scale = clip_model.logit_scale
        self.cfg = cfg

    def forward(self, image):
        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts

        image_features = self.image_encoder(image.type(self.dtype))
        text_features = self.text_encoder(prompts, tokenized_prompts)

        image_features_norm = image_features.norm(dim=-1, keepdim=True)
        text_features_norm = text_features.norm(dim=-1, keepdim=True)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        if self.prompt_learner.training:
            return image_features, text_features, image_features_norm, text_features_norm
        return image_features, text_features
        
@TRAINER_REGISTRY.register()
class DoCoOp(TrainerX):

    def __init__(self, cfg):
        self.check_cfg(cfg)

        if torch.cuda.is_available() and cfg.USE_CUDA:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
       
        # Save as attributes some frequently used variables
        self.start_epoch = self.epoch = 0
        self.max_epoch = cfg.OPTIM.MAX_EPOCH
        self.output_dir = cfg.OUTPUT_DIR

        self.cfg = cfg
        self.build_data_loader()
        
        # LinearRegression
        lr = LinearRegression()
        points = np.array([[1, self.cfg.TRAINER.DOCOOP.LAMBDA_PROX],
                           [16, 1e-5]])
        X = points[:, 0].reshape(-1, 1)
        Y = points[:, 1]
        self.lr = lr.fit(X, Y)
        self.ot_weight = self.lr.predict([[self.cfg.DATASET.NUM_SHOTS]])[0]

        super().__init__(cfg)
        
      
    def check_cfg(self, cfg):
        assert cfg.TRAINER.DOCOOP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.DOCOOP.PREC == "fp32" or cfg.TRAINER.DOCOOP.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            param.requires_grad_(False)
        for name, param in self.model.named_parameters():
            if "prompt_learner" in name or "criterion" in name:
                param.requires_grad_(True)
            
        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")
        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)
        
        self.model.to(self.device)
        
        # Setup PC Loss
        ## set config file
        from yacs.config import CfgNode as CN
        optim_copy = cfg.OPTIM.copy()
        optim_copy["LR"] = self.cfg.TRAINER.DOCOOP.LR_CONPROX
        CFG_CONPROX = CN(init_dict=optim_copy)
        optim_copy["LR"] = self.cfg.TRAINER.DOCOOP.LR_PROX
        CFG_PROX = CN(init_dict=optim_copy)

        self.optim_conprox = build_optimizer(self.model.criterion_conprox.parameters(), CFG_CONPROX)
        self.optim_prox = build_optimizer(self.model.criterion_prox.parameters(), CFG_PROX)
        self.sched_conprox = build_lr_scheduler(self.optim_conprox, CFG_CONPROX)
        self.sched_prox = build_lr_scheduler(self.optim_prox, CFG_PROX)
        
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)
        self.register_model("conprox", self.model.criterion_conprox, self.optim_conprox, self.sched_conprox)
        self.register_model("prox", self.model.criterion_prox, self.optim_prox, self.sched_prox)

        self.scaler = GradScaler() if cfg.TRAINER.DOCOOP.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def get_entropy(self, input_):
        bs = input_.size(0)
        epsilon = 1e-5
        entropy = -input_ * torch.log(input_ + epsilon)
        entropy = torch.sum(entropy, dim=1)
        return entropy 

    def compute_im_loss(self, logits):
        softmax_out = nn.Softmax(dim=1)(logits)
        entropy_loss = torch.mean(self.get_entropy(softmax_out))
        msoftmax = softmax_out.mean(dim=0)
        gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + 1e-6))
        im_loss = entropy_loss - gentropy_loss
        return im_loss

    def compute_transport_loss(self, logits, sim_t):
        s_dist = F.softmax(logits, dim=1) # across class
        t_dist = F.softmax(logits, dim=0) # across data
        cost = 1 - sim_t
        s_cost = (cost * s_dist).sum(1).mean()
        t_cost = (cost * t_dist).sum(0).mean()
        #return s_cost + t_cost
        return t_cost

    def forward_backward(self, batch):
        
        images, labels = self.parse_batch_train(batch)
        
        image_features_scaled, text_features_scaled, image_norm, txt_norm = self.model(images)
        image_features = image_features_scaled * image_norm
        text_features = copy.deepcopy(text_features_scaled.detach())
        
        # Cross-Entropy Loss
        logit_scale = self.model.logit_scale.exp()
        logits = image_features_scaled @ text_features_scaled.t()
        logits_scaled = logit_scale * logits
        
        cross_entropy_loss = F.cross_entropy(logits_scaled, labels)

        # # Distribution Loss
        # distribution_loss_list = []
        # ot_losses = [self.compute_transport_loss(k.unsqueeze(0), v.unsqueeze(0)) for k, v in zip(logits_scaled, logits)]
        # mi_losses = [self.compute_im_loss(k.unsqueeze(0)) for k in logits_scaled]
        
        # if self.cfg.TRAINER.DOCOOP.ADJUST_WEIGHT:
        #     distribution_loss_list = list(map(lambda v: self.ot_weight * v[0] + 
        #                                 0.1 * self.ot_weight * v[1], zip(ot_losses, mi_losses)))
        # else:
        #     distribution_loss_list = list(map(lambda v: self.cfg.TRAINER.DOCOOP.LAMBDA_OT * v[0] + 
        #                                 self.cfg.TRAINER.DOCOOP.LAMBDA_MI * v[1], zip(ot_losses, mi_losses)))

        # distribution_loss = max(distribution_loss_list)

        # Prox Loss
        text_labels = torch.arange(self.dm.num_classes).to(self.device)
        prox_loss = self.model.criterion_prox(torch.cat((image_features, text_features)), 
                                        torch.cat((labels, text_labels)))
        prox_loss *= self.cfg.TRAINER.DOCOOP.LAMBDA_PROX
        
        # ConProx Loss
        conprox_loss = self.model.criterion_conprox(torch.cat((image_features, text_features)), 
                                        torch.cat((labels, text_labels)))
        conprox_loss *= self.cfg.TRAINER.DOCOOP.LAMBDA_CONPROX
        
        if self.epoch < self.cfg.TRAINER.DOCOOP:
            loss = cross_entropy_loss   
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            if (self.batch_idx + 1) == self.num_batches:
                self.sched.step()

        else:
            loss = cross_entropy_loss + prox_loss - conprox_loss
            
            self.optim.zero_grad()
            self.optim_conprox.zero_grad()
            self.optim_prox.zero_grad()

            loss.backward()
            self.optim.step()

            for param in self.model.criterion_conprox.parameters():
                param.grad.data *= (1. / self.cfg.TRAINER.DOCOOP.LAMBDA_CONPROX)

            for param in self.model.criterion_prox.parameters():
                param.grad.data *= (1. / self.cfg.TRAINER.DOCOOP.LAMBDA_PROX)

            self.optim_conprox.step()
            self.optim_prox.step()

            if (self.batch_idx + 1) == self.num_batches:
                self.sched.step()
                self.sched_prox.step()
                self.sched_conprox.step()
 
        loss_summary = {
            "loss(ce)": cross_entropy_loss.item(),
            #"loss(d)" : distribution_loss.item(),
            "loss(prox)": prox_loss.item(),
            "loss(conprox)": conprox_loss.item(),
            "acc": compute_accuracy(logits, labels)[0].item(),
        }

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
            
    @torch.no_grad()
    def test(self, split=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader

        print(f"Evaluate on the *{split}* set")

        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label = self.parse_batch_test(batch)
            image_features, text_features = self.model_inference(input)
            logit_scale = self.model.logit_scale.exp()
            logits = logit_scale * image_features @ text_features.t()
            self.evaluator.process(logits, label)

        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0]


    def embedding_feature(self, VISUAL_MAX_NUM=2000):
        umap = TSNE()
        # extract feature from visual encoder
        image_features = []
        text_features = []
        image_labels = []
        self.set_model_mode("eval")
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.test_loader):
                
                images, labels = self.parse_batch_train(batch)
                
                image_feature, text_feature = self.model_inference(images) # clip visual encoder
                image_features.append(image_feature)
                text_features.append(text_feature)
                image_labels.append(labels)
        
        image_features = torch.cat(image_features)
        image_labels = torch.cat(image_labels)
        text_features = text_features[0][:len(self.dm.dataset.classnames)]
        
        if len(image_features) > VISUAL_MAX_NUM:
            sample_idx = torch.randint(0, len(image_features), (VISUAL_MAX_NUM, ))
            image_features = image_features[sample_idx]
            image_labels = image_labels[sample_idx]
     
        def scaling(data):
            min_val = data.min(0, keepdim=True)[0]
            max_val = data.max(0, keepdim=True)[0]
            return (data - min_val) / (max_val - min_val)
        
         
        features = torch.cat((scaling(image_features), scaling(text_features)))
        embeddings = umap.fit_transform(features.cpu())
        
        # calc distance
        d_img_txt = []
        d_txt_txt = []
        for label in range(self.dm.num_classes):
            img_src = image_features[image_labels==label]
            txt_src = text_features[label].unsqueeze(0)
            
            d_img_txt.append(torch.norm(img_src-txt_src, dim=1).mean().item())
            d_txt_txt.append(torch.norm(text_features-txt_src, dim=1).mean().item())
        
        print(f"dist(img vs txt): {np.mean(d_img_txt)}±{np.std(d_img_txt)}")
        print(f"dist(txt vs txt): {np.mean(d_txt_txt)}±{np.std(d_txt_txt)}")
        
        # visualize
        num_class = len(self.dm.dataset.classnames)
        classnames = self.dm.dataset.classnames
        
        image_embeddings = embeddings[:-num_class]
        text_embeddings = embeddings[len(image_embeddings):]
        
        plt.figure(figsize=(8, 8))
        cmap = plt.cm.get_cmap("tab10", num_class)
        for i in range(num_class):
            indices = (image_labels.cpu() == i).numpy()
            plt.scatter(image_embeddings[indices, 0], image_embeddings[indices, 1], s=10, alpha=1.0, c=[cmap(i)])
        for i in range(num_class):       
            plt.scatter(text_embeddings[i, 0], text_embeddings[i, 1], label=str(i), s=200, marker="*", alpha=1.0, c=[cmap(i)], edgecolors="black")
        plt.legend()
        plt.savefig(f'./output/plots/tsne_DoCoOp_{self.cfg.DATASET.NAME}_{self.cfg.DATASET.NUM_SHOTS}_{self.cfg.OPTIM.MAX_EPOCH}.png', bbox_inches="tight")