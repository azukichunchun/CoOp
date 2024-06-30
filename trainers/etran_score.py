import torch
import torch.nn as nn
from tqdm import tqdm
from itertools import chain
import numpy as np
import random
import os
import copy
import pickle
import pdb
from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.model import convert_weights

from .coop import load_clip_to_cpu
from .imagenet_templates import IMAGENET_TEMPLATES, IMAGENET_TEMPLATES_SELECT

from etran.metrics import Energy_Score, LDA_Score

CUSTOM_TEMPLATES = {
    "OxfordPets": "a photo of a {}, a type of pet.",
    "OxfordFlowers": "a photo of a {}, a type of flower.",
    "FGVCAircraft": "a photo of a {}, a type of aircraft.",
    "DescribableTextures": "{} texture.",
    "EuroSAT": "a centered satellite photo of {}.",
    "StanfordCars": "a photo of a {}.",
    "Food101": "a photo of {}, a type of food.",
    "SUN397": "a photo of a {}.",
    "Caltech101": "a photo of a {}.",
    "UCF101": "a photo of a person doing {}.",
    "ImageNet": "a photo of a {}.",
    "ImageNetSketch": "a photo of a {}.",
    "ImageNetV2": "a photo of a {}.",
    "ImageNetA": "a photo of a {}.",
    "ImageNetR": "a photo of a {}.",
}

@TRAINER_REGISTRY.register()
class ZeroshotCLIP_ETran(TrainerX):
    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        clip_model.to(self.device)

        temp = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
        prompts = [temp.format(c.replace("_", " ")) for c in classnames]
        print(f"Prompts: {prompts}")
        prompts = torch.cat([clip.tokenize(p) for p in prompts])
        prompts = prompts.to(self.device)

        with torch.no_grad():
            text_features = clip_model.encode_text(prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        self.text_features = text_features
        self.clip_model = clip_model

    def model_inference(self, image):
        image_features = self.clip_model.encode_image(image)

        image_features_without_norm = copy.deepcopy(image_features)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        logit_scale = self.clip_model.logit_scale.exp()
        logits = logit_scale * image_features @ self.text_features.t()
        return image_features, image_features_without_norm, logits
    

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

        energy_scores = []
        energy_logits_scores = []
        etran_logits_scores = []
        etran_scores = []

        features = []
        logits = []
        labels = []
        impath = []

        logits_max = []

        for batch_idx, batch in enumerate(tqdm(self.test_loader)):
            input, label = self.parse_batch_test(batch)

            feature, feature_without_norm, logit = self.model_inference(input)

            features.append(feature_without_norm)
            logits.append(logit)
            labels.append(label)
            impath.append(batch['impath'])

            logits_max.append(torch.max(logit, axis=1)[0])

            self.evaluator.process(logit, label)

        features = np.array([features.detach().cpu().numpy().tolist() for features in list(chain.from_iterable(features))])
        logits = np.array([logit.detach().cpu().numpy().tolist() for logit in list(chain.from_iterable(logits))])
        labels = np.array([label.detach().cpu().numpy().tolist() for label in list(chain.from_iterable(labels))])
        logits_max = np.array([logit_max.detach().cpu().numpy().tolist() for logit_max in list(chain.from_iterable(logits_max))])

        impath = np.array(list(chain.from_iterable(impath)))

        energy_scores = Energy_Score(features, 0.5, tail="").tolist()
        energy_logits_scores = Energy_Score(logits, 0.5, tail="").tolist()

        lda_scores = LDA_Score(features, labels)
        lda_logits_scores = LDA_Score(logits, labels)

        assert len(energy_scores) == len(logits_max)
        np.save(os.path.join(self.cfg.OUTPUT_DIR, "logits.npy"), np.array(logits))
        np.save(os.path.join(self.cfg.OUTPUT_DIR, "labels.npy"), np.array(labels))

        # energyをソートし100点ずつで精度を測る
        energy_sort_accuracy = []
        energy_scores = np.array(energy_scores)
        energy_score_idx = np.argsort(energy_scores)
        energy_score_idx = np.array_split(energy_score_idx, indices_or_sections=20, axis=0)
        for idx in energy_score_idx:
            logits_sort = logits[idx]
            labels_sort = labels[idx]
           
            impath_sort = tuple(random.sample(impath[idx].tolist(), 4)) #
            energy_score_median = np.median(energy_scores[idx])
            
            pred = np.argmax(logits_sort, axis=1)
            accuracy = np.mean(pred == labels_sort)
            
            energy_sort_accuracy.append(tuple([energy_score_median, accuracy, impath_sort]))
       
        # save
        print("Save energy scores")
        np.save(os.path.join(self.cfg.OUTPUT_DIR, "features.npy"), np.array(features))
        np.save(os.path.join(self.cfg.OUTPUT_DIR, "energy_scores.npy"), np.array(energy_scores))
        np.save(os.path.join(self.cfg.OUTPUT_DIR, "energy_logits_scores.npy"), np.array(energy_logits_scores))
        np.save(os.path.join(self.cfg.OUTPUT_DIR, "lda_scores.npy"), np.array(lda_scores))
        np.save(os.path.join(self.cfg.OUTPUT_DIR, "lda_logits_scores.npy"), np.array(lda_logits_scores))

        with open(os.path.join(self.cfg.OUTPUT_DIR, "energy_sort_accuracy.pkl"), "wb") as f:
            pickle.dump(energy_sort_accuracy, f)

        features = []
        energy_scores = []
        impath = []
        # energyスコアとimpathの対応表を作る（足りていないtrainとvalを追加）
        for batch_idx, batch in enumerate(tqdm(self.train_loader_x)):
            input, label = self.parse_batch_test(batch)

            _, feature_without_norm, _ = self.model_inference(input)
            features.append(feature_without_norm)
            impath.append(batch['impath'])

        for batch_idx, batch in enumerate(tqdm(self.val_loader)):
            input, label = self.parse_batch_test(batch)

            _, feature_without_norm, _ = self.model_inference(input)
            features.append(feature_without_norm)
            impath.append(batch['impath'])

        for batch_idx, batch in enumerate(tqdm(self.test_loader)):
            input, label = self.parse_batch_test(batch)

            _, feature_without_norm, _ = self.model_inference(input)
            features.append(feature_without_norm)
            impath.append(batch['impath'])

        features = np.array([features.detach().cpu().numpy().tolist() for features in list(chain.from_iterable(features))])
        impath = np.array(list(chain.from_iterable(impath)))

        energy_scores = Energy_Score(features, 0.5, tail="").tolist()
        assert len(energy_scores) == len(impath)
        print(f"energy score size is {len(energy_scores)}")
        # energyスコアとimpathの対応表を作る
        print("Save energy scores list")
        energy_score_list = {p: e for p, e in zip(impath, energy_scores)}
        energy_logit_score_list = {p: e for p, e in zip(impath, energy_logits_scores)}

        root = self.cfg.DATASET.ROOT
        dataset_dir = self.cfg.OUTPUT_DIR.split('/')[-1]
        with open(os.path.join(root, dataset_dir, "energy_score_list.pkl"), "wb") as f:
            pickle.dump(energy_score_list, f)
        with open(os.path.join(root, dataset_dir, "energy_logit_score_list.pkl"), "wb") as f:
            pickle.dump(energy_logit_score_list, f)

        results = self.evaluator.evaluate()
        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0]