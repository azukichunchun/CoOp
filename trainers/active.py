import os
import pickle
import torch
import torch.nn as nn
import numpy as np
from scipy.stats import entropy as e
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
import random
from tqdm import tqdm

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.data.datasets import Datum

from clip import clip
from clip.model import convert_weights

import pdb
from .coop import load_clip_to_cpu
from .imagenet_templates import IMAGENET_TEMPLATES, IMAGENET_TEMPLATES_SELECT

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
class ActiveLearning(TrainerX):
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

        query = self.pseudo_labeled_balancing()
        self.dump_splited_dataset(query)


    def model_inference(self, image):
        image_features = self.clip_model.encode_image(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logit_scale = self.clip_model.logit_scale.exp()
        logits = logit_scale * image_features @ self.text_features.t()
        return logits
    

    def entropy_c(self, unlabeled_loader, category_size, sample_size_by_category=20):

        entropies = []
        indices = []
        preds = []
        img_features = []
        for batch_idx, batch in tqdm(enumerate(unlabeled_loader)):
            index = batch["index"]
            images = batch["img"]

            img_feature = self.clip_model.encode_image(images.to(self.device).type(self.clip_model.dtype))
            img_feature = img_feature.detach().cpu().numpy() # detach to avoid cuda out of memory
            img_features.extend(img_feature)

            logits = self.model_inference(images.to(self.device).type(self.clip_model.dtype))

            preds.extend(logits.argmax(axis=1).detach().cpu().numpy())
            entropies.extend(e(logits.detach().cpu().numpy(), axis=1))
            indices.extend(index.numpy())

        entropies = np.array(entropies)
        indices = np.array(indices)
        preds = np.array(preds)

        # 画像特徴量をクラスタリング
        kmeans = KMeans(n_clusters=category_size, random_state=42)
        img_features = np.array(img_features)
        kmeans.fit(img_features)

        # 重心から近い順にサンプリング
        distances = pairwise_distances(kmeans.cluster_centers_, img_features)
        sample_idx = []
        for d in distances:
            d_minid = np.argsort(d)
            d_minid = d_minid[:sample_size_by_category]
            sample_idx.extend(d_minid)
        sample_idx = np.array(list(set(sample_idx)))
        
        entropies = entropies[sample_idx]
        indices = indices[sample_idx]
        preds = preds[sample_idx]

        # 不確実性が高い(低い？)順にソート
        entropies_sort_ids = np.argsort(entropies)#[::-1]

        index_sort = indices[entropies_sort_ids]
        entropy_sort = entropies[entropies_sort_ids]
        preds_sort = preds[entropies_sort_ids]

        num_samples = int(1.0 * len(entropies))

        output = {"index": index_sort[:num_samples], 
                  "preds": preds_sort[:num_samples]}

        return output

    def entropy(self, unlabeled_loader, gamma):

        entropies = []
        indices = []
        preds = []

        for batch_idx, batch in tqdm(enumerate(unlabeled_loader)):
            index = batch["index"]
            images = batch["img"]

            logits = self.model_inference(images.to(self.device).type(self.clip_model.dtype))

            preds.extend(logits.argmax(axis=1).detach().cpu().numpy())
            entropies.extend(e(logits.detach().cpu().numpy(), axis=1))
            indices.extend(index.numpy())
        
        entropies = np.array(entropies)
        indices = np.array(indices)
        preds = np.array(preds)

        entropies_sort_ids = np.argsort(entropies)[::-1]

        index_sort = indices[entropies_sort_ids]
        entropy_sort = entropies[entropies_sort_ids]
        preds_sort = preds[entropies_sort_ids]
        
        num_samples = int(gamma * len(entropies))

        output = {"index": index_sort[:num_samples], 
                  "preds": preds_sort[:num_samples]}

        return output


    def random(self, unlabeled_loader, gamma):

        indices = []
        preds = []
        
        for batch_idx, batch in tqdm(enumerate(unlabeled_loader)):
            index = batch["index"]
            images = batch["img"]

            logits = self.model_inference(images.to(self.device).type(self.clip_model.dtype))

            preds.extend(logits.argmax(axis=1).detach().cpu().numpy())
            indices.extend(index.numpy())
        
        indices = np.array(indices)
        preds = np.array(preds)
        
        num_samples = int(gamma * len(indices))
        sample_ids = random.sample(range(len(indices)), k=num_samples)
        sample_ids = np.array(sample_ids)

        output = {"index": indices[sample_ids], 
                  "preds": preds[sample_ids]}

        return output


    def coreset(self, unlabeled_dataset, gamma, num_clusters):
        pass

    def badge(self):
        pass


    def pseudo_labeled_balancing(self, gamma=0.4, R=1, strategy="entropy_c", sampling_mode="filled", sampling_num=None):

        category_size = self.dm.num_classes
        unlabeled_loader = self.dm.train_loader_x
        
        for r in range(R):
            # Select informative subsets
            if strategy == "entropy":
                output = self.entropy(unlabeled_loader, gamma)

            elif strategy == "entropy_c":
                output = self.entropy_c(unlabeled_loader, category_size, sample_size_by_category=200)

            elif strategy == "random":
                output = self.random(unlabeled_loader, gamma)

            elif strategy == "coreset":
                output = self.coreset()
            elif strategy == "badge":
                output = self.badge

            p_pred = output["preds"]
            index = output["index"]

            query = []
            # あらかじめ決められた数のみサンプリング
            if sampling_mode == "fixed":

                if sampling_num is not None:
                    category_size = sampling_num

                for k in range(category_size):
                    candidates = [i for i, y_hat in enumerate(p_pred) if y_hat == k]
                    if len(candidates)==0:
                        assert f"class{k} has no candidate."
                    print(len(candidates))
                    # Select one sample from candidates
                    selected_id = candidates[np.random.randint(len(candidates))]
                    dataset_id = index[selected_id]
                    query.append(self.dm.dataset.train_x[dataset_id])

            # 全てのカテゴリにデータが含まれるまでサンプリング
            elif sampling_mode == "filled":
                while len(set([d.label for d in query])) < self.dm.num_classes:
                    #pdb.set_trace()
                    unsampled_category =  [i for i in range(self.dm.num_classes) if i not in set([d.label for d in query])]
                    for k in unsampled_category:
                        candidates = [i for i, y_hat in enumerate(p_pred) if y_hat == k]
                        assert len(candidates)!=0, f"class{k} has no candidate."
                        print(len(candidates))
                        # Select one sample from candidates
                        selected_id = candidates[np.random.randint(len(candidates))]
                        dataset_id = index[selected_id]
                        query.append(self.dm.dataset.train_x[dataset_id])


        return query
    
    def dump_splited_dataset(self, query):
        num_shots = 1
        sample_seed = self.cfg.DATASET.SAMPLE_SEED
        seed = self.cfg.SEED

        dataset_name = self.cfg.OUTPUT_DIR.split("/")[-1]+"_active"
        split_path = os.path.join(self.cfg.DATASET.ROOT, dataset_name, "split_fewshot")

        if not os.path.exists(split_path):
            os.makedirs(split_path, exist_ok=True)

        preprocessed = os.path.join(split_path, f"shot_{num_shots}-seed_{seed}-active.pkl")

        with open(preprocessed, "wb") as file:
            pickle.dump(query, file, protocol=pickle.HIGHEST_PROTOCOL)
 