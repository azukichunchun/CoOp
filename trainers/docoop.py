import os.path as osp
import time
import pdb
import copy
from tqdm import tqdm
import datetime
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

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from .transforms import get_tuning_transform

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

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image):
        image_features = self.image_encoder(image.type(self.dtype))

        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        #logit_scale = self.logit_scale.exp()
        #logits = logit_scale * image_features @ text_features.t()

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
        #pdb.set_trace()
        # Copy dataloader_x for data transform
        self.train_loader_c = copy.deepcopy(self.train_loader_x)
        transform_tune = get_tuning_transform()
        self.train_loader_c.transform = transform_tune
        
        # KMedoids
        self.km = KMedoids(n_clusters=1, metric="euclidean")
        
        # MMD
        self.mmd = MMD_loss()

        # LinearRegression
        lr = LinearRegression()
        points = np.array([[1, self.cfg.TRAINER.DOCOOP.LAMBDA_OT],
                           [16, 0.01]])
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
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)
        
        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)

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
        s_dist = F.softmax(logits, dim=1)
        t_dist = F.softmax(logits, dim=0)
        cost = 1 - sim_t
        s_cost = (cost * s_dist).sum(1).mean()
        t_cost = (cost * t_dist).sum(0).mean()
        return s_cost + t_cost

    def run_epoch(self):
        self.set_model_mode("train")
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        self.num_batches = len(self.train_loader_x)

        end = time.time()
        for self.batch_idx, batch in enumerate(zip(self.train_loader_x, self.train_loader_c)):
            data_time.update(time.time() - end)
            loss_summary = self.forward_backward(batch)
            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            meet_freq = (self.batch_idx + 1) % self.cfg.TRAIN.PRINT_FREQ == 0
            only_few_batches = self.num_batches < self.cfg.TRAIN.PRINT_FREQ
            if meet_freq or only_few_batches:
                nb_remain = 0
                nb_remain += self.num_batches - self.batch_idx - 1
                nb_remain += (
                    self.max_epoch - self.epoch - 1
                ) * self.num_batches
                eta_seconds = batch_time.avg * nb_remain
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                info = []
                info += [f"epoch [{self.epoch + 1}/{self.max_epoch}]"]
                info += [f"batch [{self.batch_idx + 1}/{self.num_batches}]"]
                info += [f"time {batch_time.val:.3f} ({batch_time.avg:.3f})"]
                info += [f"data {data_time.val:.3f} ({data_time.avg:.3f})"]
                info += [f"{losses}"]
                info += [f"lr {self.get_current_lr():.4e}"]
                info += [f"eta {eta}"]
                print(" ".join(info))

            n_iter = self.epoch * self.num_batches + self.batch_idx
            for name, meter in losses.meters.items():
                self.write_scalar("train/" + name, meter.avg, n_iter)
            self.write_scalar("train/lr", self.get_current_lr(), n_iter)

            end = time.time()


    def forward_backward(self, batch):
        
        images_n, labels_n, images_c, _ = self.parse_batch_train(batch)
        
        prec = self.cfg.TRAINER.DOCOOP.PREC
        if prec == "amp":
            with autocast():
                image_features_n, text_features_n = self.model(images_n)
                print("autocast")
                # Cross-Entropy Loss
                logit_scale = self.model.logit_scale.exp()
                logits = image_features_n @ text_features_n.t()
                logits_scaled = logit_scale * logits
                cross_entropy_loss = F.cross_entropy(logits_scaled, labels_n)
                
                # Distribution Loss
                image_features_c, _ = self.model(images_c)
                logits_c = image_features_c @ text_features_c.t()
                logits_scaled_c = logit_scale * logits_c
            
                transfer_loss = self.compute_transport_loss(logits_scaled_c, logits_c)
                mi_loss = self.compute_im_loss(logits_scaled_c)
                
                loss = cross_entropy_loss + self.cfg.TRAINER.DOCOOP.LAMBDA_OT * transfer_loss + self.cfg.TRAINER.DOCOOP.LAMBDA_MI * mi_loss

            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            image_features_n, text_features_n = self.model(images_n)
            
            # Cross-Entropy Loss
            logit_scale = self.model.logit_scale.exp()
            logits = image_features_n @ text_features_n.t()
            logits_scaled = logit_scale * logits
            cross_entropy_loss = F.cross_entropy(logits_scaled, labels_n)
            
            # Distribution Loss
            image_features_c, text_features_c = self.model(images_c)
            logits_c = image_features_c @ text_features_c.t()
            logits_scaled_c = logit_scale * logits_c
        
            ot_loss = self.compute_transport_loss(logits_scaled_c, logits_c)
            mi_loss = self.compute_im_loss(logits_scaled_c)
            
            # distribution_loss = self.cfg.TRAINER.DOCOOP.LAMBDA_OT * ot_loss + self.cfg.TRAINER.DOCOOP.LAMBDA_MI * mi_loss
            # images_c = images_c[0]
            # images_c = torch.unsqueeze(images_c, dim=0)
            distribution_loss_list = []
            for image_c in images_c:
                image_c = torch.unsqueeze(image_c, dim=0)
                image_features_c, text_features_c = self.model(image_c)
                logits_c = image_features_c @ text_features_c.t()
                logits_scaled_c = logit_scale * logits_c
            
                ot_loss = self.compute_transport_loss(logits_scaled_c, logits_c)
                mi_loss = self.compute_im_loss(logits_scaled_c)
                
                if self.cfg.TRAINER.DOCOOP.ADJUST_WEIGHT:
                    # calculate weight from num_shots
                    distribution_loss_list.append(self.ot_weight * ot_loss + 0.1 * self.ot_weight * mi_loss)
                else:
                    distribution_loss_list.append(self.cfg.TRAINER.DOCOOP.LAMBDA_OT * ot_loss + self.cfg.TRAINER.DOCOOP.LAMBDA_MI * mi_loss)
                    
            
            distribution_loss = max(distribution_loss_list)
            
            # image_features_c, text_features_c = self.model(images_c)
            
            # image_features_c = np.array([d.cpu().numpy() for d in image_features_c])
            # text_features_c = text_features_c.mean(axis=0, keepdim=True)
            # text_features_c = np.array([d.cpu().detach().numpy() for d in text_features_c])
            
            # image_features_km = self.km.fit(image_features_c).cluster_centers_
            # text_features_km = self.km.fit(text_features_c).cluster_centers_
            
            # image_features_km = torch.tensor(image_features_km, dtype=self.model.dtype, requires_grad=True).to(self.device)            
            #text_features_km = torch.tensor(text_features_km, dtype=self.model.dtype, requires_grad=False).to(self.device)            
            
            # logits_c = image_features_km @ text_features_c.t()
            # logits_scaled_c = logit_scale * logits_c
        
            # ot_loss = self.compute_transport_loss(logits_scaled_c, logits_c)
            #mi_loss = self.compute_im_loss(logits_scaled_c)
            # distribution_loss = self.cfg.TRAINER.DOCOOP.LAMBDA_OT * ot_loss + self.cfg.TRAINER.DOCOOP.LAMBDA_MI * mi_loss
            
            # distribution_loss = self.cfg.TRAINER.DOCOOP.LAMBDA_OT * self.mmd(image_features_km, text_features_c)
            loss = cross_entropy_loss + distribution_loss         
            
            self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(logits, labels_n)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        batch_n, batch_c = batch
        input_n = batch_n["img"]
        label_n = batch_n["label"]
        input_c = batch_c["img"]
        label_c = batch_c["label"]
        input_n = input_n.to(self.device)
        label_n = label_n.to(self.device)
        input_c = input_c.to(self.device)
        label_c = label_c.to(self.device)
        return input_n, label_n, input_c, label_c

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