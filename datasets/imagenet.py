import os
import pickle
from collections import OrderedDict
import random

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import listdir_nohidden, mkdir_if_missing, set_random_seed

from .oxford_pets import OxfordPets


@DATASET_REGISTRY.register()
class ImageNet(DatasetBase):

    dataset_dir = "imagenet"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "images")
        self.preprocessed = os.path.join(self.dataset_dir, "preprocessed.pkl")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)
    
        if os.path.exists(self.preprocessed):
            with open(self.preprocessed, "rb") as f:
                preprocessed = pickle.load(f)
                train = preprocessed["train"]
                test = preprocessed["test"]
        else:
            text_file = os.path.join(self.dataset_dir, "classnames.txt")
            classnames = self.read_classnames(text_file)
            train = self.read_data(classnames, "train")
            # Follow standard practice to perform evaluation on the val set
            # Also used as the val set (so evaluate the last-step model)
            test = self.read_data(classnames, "val")

            preprocessed = {"train": train, "test": test}
            with open(self.preprocessed, "wb") as f:
                pickle.dump(preprocessed, f, protocol=pickle.HIGHEST_PROTOCOL)

        num_shots = cfg.DATASET.NUM_SHOTS
        if num_shots >= 1:
            seed = cfg.SEED
            sample_seed = cfg.DATASET.SAMPLE_SEED
            preprocessed = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}-{sample_seed}.pkl")

            if cfg.DATALOADER.ENERGY.USE_ENERGY:

                # energyとpathのリストを読み込む
                with open(os.path.join(self.dataset_dir, 'energy_score_list.pkl'), "rb") as file:
                    path_to_energy = pickle.load(file)

                train = self.generate_fewshot_dataset_based_on_energy(path_to_energy, train, target=cfg.DATALOADER.ENERGY.USAGE_RANK, num_shot=num_shots)
                val = self.generate_fewshot_dataset_based_on_energy(path_to_energy, val, target=cfg.DATALOADER.ENERGY.USAGE_RANK, num_shot=min(num_shots, 4))
                data = {"train": train, "val": val}
            else:

                if os.path.exists(preprocessed):
                    print(f"Loading preprocessed few-shot data from {preprocessed}")
                    with open(preprocessed, "rb") as file:
                        data = pickle.load(file)
                        train, val = data["train"], data["val"]
                else:

                    random.seed(cfg.DATASET.SAMPLE_SEED)

                    train = self.generate_fewshot_dataset(train, num_shots=num_shots)
                    val = self.generate_fewshot_dataset(val, num_shots=min(num_shots, 4))
                    data = {"train": train, "val": val}
                    print(f"Saving preprocessed few-shot data to {preprocessed}")
                    with open(preprocessed, "wb") as file:
                        pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

                    set_random_seed(cfg.SEED)

        subsample = cfg.DATASET.SUBSAMPLE_CLASSES
        train, test = OxfordPets.subsample_classes(train, test, subsample=subsample)

        super().__init__(train_x=train, val=test, test=test)

    @staticmethod
    def read_classnames(text_file):
        """Return a dictionary containing
        key-value pairs of <folder name>: <class name>.
        """
        classnames = OrderedDict()
        with open(text_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(" ")
                folder = line[0]
                classname = " ".join(line[1:])
                classnames[folder] = classname
        return classnames

    def read_data(self, classnames, split_dir):
        split_dir = os.path.join(self.image_dir, split_dir)
        folders = sorted(f.name for f in os.scandir(split_dir) if f.is_dir())
        items = []

        for label, folder in enumerate(folders):
            imnames = listdir_nohidden(os.path.join(split_dir, folder))
            classname = classnames[folder]
            for imname in imnames:
                impath = os.path.join(split_dir, folder, imname)
                item = Datum(impath=impath, label=label, classname=classname)
                items.append(item)

        return items
