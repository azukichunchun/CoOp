import os
import pickle
import random

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import mkdir_if_missing, set_random_seed

from .oxford_pets import OxfordPets
from .dtd import DescribableTextures as DTD


@DATASET_REGISTRY.register()
class Food101(DatasetBase):

    dataset_dir = "food101"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "images")
        self.split_path = os.path.join(self.dataset_dir, "split_zhou_Food101.json")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)

        if os.path.exists(self.split_path):
            train, val, test = OxfordPets.read_split(self.split_path, self.image_dir)
        else:
            train, val, test = DTD.read_and_split_data(self.image_dir)
            OxfordPets.save_split(train, val, test, self.split_path, self.image_dir)

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
