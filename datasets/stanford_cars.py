import os
import pickle
import random
from scipy.io import loadmat

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import mkdir_if_missing, set_random_seed

from .oxford_pets import OxfordPets


@DATASET_REGISTRY.register()
class StanfordCars(DatasetBase):

    dataset_dir = "stanford_cars"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.split_path = os.path.join(self.dataset_dir, "split_zhou_StanfordCars.json")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)

        if os.path.exists(self.split_path):
            train, val, test = OxfordPets.read_split(self.split_path, self.dataset_dir)
        else:
            trainval_file = os.path.join(self.dataset_dir, "devkit", "cars_train_annos.mat")
            test_file = os.path.join(self.dataset_dir, "cars_test_annos_withlabels.mat")
            meta_file = os.path.join(self.dataset_dir, "devkit", "cars_meta.mat")
            trainval = self.read_data("cars_train", trainval_file, meta_file)
            test = self.read_data("cars_test", test_file, meta_file)
            train, val = OxfordPets.split_trainval(trainval)
            OxfordPets.save_split(train, val, test, self.split_path, self.dataset_dir)

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
        train, val, test = OxfordPets.subsample_classes(train, val, test, subsample=subsample)

        super().__init__(train_x=train, val=val, test=test)

    def read_data(self, image_dir, anno_file, meta_file):
        anno_file = loadmat(anno_file)["annotations"][0]
        meta_file = loadmat(meta_file)["class_names"][0]
        items = []

        for i in range(len(anno_file)):
            imname = anno_file[i]["fname"][0]
            impath = os.path.join(self.dataset_dir, image_dir, imname)
            label = anno_file[i]["class"][0, 0]
            label = int(label) - 1  # convert to 0-based index
            classname = meta_file[label][0]
            names = classname.split(" ")
            year = names.pop(-1)
            names.insert(0, year)
            classname = " ".join(names)
            item = Datum(impath=impath, label=label, classname=classname)
            items.append(item)

        return items
