import os
import pickle
import random
from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import mkdir_if_missing, set_random_seed

from .oxford_pets_active import OxfordPets_Active
from .oxford_pets import OxfordPets
from .dtd import DescribableTextures as DTD

NEW_CNAMES = {
    "AnnualCrop": "Annual Crop Land",
    "Forest": "Forest",
    "HerbaceousVegetation": "Herbaceous Vegetation Land",
    "Highway": "Highway or Road",
    "Industrial": "Industrial Buildings",
    "Pasture": "Pasture Land",
    "PermanentCrop": "Permanent Crop Land",
    "Residential": "Residential Buildings",
    "River": "River",
    "SeaLake": "Sea or Lake",
}


@DATASET_REGISTRY.register()
class EuroSAT_Active(DatasetBase):

    dataset_dir = "eurosat"
    dataset_dir_active = "eurosat_active"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.dataset_dir_active = os.path.join(root, self.dataset_dir_active)
        self.image_dir = os.path.join(self.dataset_dir, "2750")
        self.split_path = os.path.join(self.dataset_dir, "split_zhou_EuroSAT.json")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        self.split_fewshot_dir_active = os.path.join(self.dataset_dir_active, "split_fewshot")
        
        mkdir_if_missing(self.split_fewshot_dir)

        if os.path.exists(self.split_path):
            train, val, test = OxfordPets.read_split(self.split_path, self.image_dir)
        else:
            train, val, test = DTD.read_and_split_data(self.image_dir, new_cnames=NEW_CNAMES)
            OxfordPets.save_split(train, val, test, self.split_path, self.image_dir)

        num_shots = cfg.DATASET.NUM_SHOTS
        if num_shots >= 1:
            seed = cfg.SEED
            sample_seed = cfg.DATASET.SAMPLE_SEED
            preprocessed_active = os.path.join(self.split_fewshot_dir_active, f"shot_{num_shots}-seed_{seed}-active.pkl")
            preprocessed = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl")
            
            if cfg.DATALOADER.ENERGY.USE_ENERGY:

                # energyとpathのリストを読み込む
                with open(os.path.join(self.dataset_dir, 'energy_score_list.pkl'), "rb") as file:
                    path_to_energy = pickle.load(file)

                train = self.generate_fewshot_dataset_based_on_energy(path_to_energy, train, target=cfg.DATALOADER.ENERGY.USAGE_RANK, num_shot=num_shots)
                val = self.generate_fewshot_dataset_based_on_energy(path_to_energy, val, target=cfg.DATALOADER.ENERGY.USAGE_RANK, num_shot=min(num_shots, 4))
                data = {"train": train, "val": val}
            else:
                if os.path.exists(preprocessed):
                    print(f"Loading preprocessed few-shot data from {preprocessed_active}")
                    with open(preprocessed_active, "rb") as file:
                        data = pickle.load(file)
                        train = data
                    print(f"Loading preprocessed few-shot data from {preprocessed}")
                    with open(preprocessed, "rb") as file:
                        data = pickle.load(file)
                        val = data["val"]
                else:
                    raise ValueError("Query from AL is needed.")
                    random.seed(cfg.DATASET.SAMPLE_SEED)
                    train = self.generate_fewshot_dataset(train, num_shots=num_shots)
                    val = self.generate_fewshot_dataset(val, num_shots=min(num_shots, 4))
                    data = {"train": train, "val": val}
                    print(f"Saving preprocessed few-shot data to {preprocessed}")
                    with open(preprocessed, "wb") as file:
                        pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)
                    set_random_seed(cfg.SEED)

        num_class = cfg.DATASET.NUM_CLASS
        subsample = cfg.DATASET.SUBSAMPLE_CLASSES
        train, val, test = OxfordPets_Active.subsample_classes_for_active(train, val, test, num_class=num_class, subsample=subsample)
        
        super().__init__(train_x=train, val=val, test=test)

    def update_classname(self, dataset_old):
        dataset_new = []
        for item_old in dataset_old:
            cname_old = item_old.classname
            cname_new = NEW_CLASSNAMES[cname_old]
            item_new = Datum(impath=item_old.impath, label=item_old.label, classname=cname_new)
            dataset_new.append(item_new)
        return dataset_new
