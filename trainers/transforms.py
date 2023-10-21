import torchvision.transforms as transforms
from .data_augment import BlockShuffle, PhaseMasking

def get_train_transform(resize_size: tuple = (224, 224)):
    transform = transforms.Compose([transforms.Resize(resize_size),
                                    transforms.ToTensor(),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomVerticalFlip(),
                                    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], 
                                                         std=[0.26862954, 0.26130258, 0.27577711]),])
    return transform

def get_test_transform(resize_size: tuple = (224, 224)):
    transform = transforms.Compose([transforms.Resize(resize_size),
                                    transforms.ToTensor(), 
                                    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], 
                                                         std=[0.26862954, 0.26130258, 0.27577711]),])
    return transform


def get_tuning_transform(resize_size: tuple = (224, 224), grid: int = 3, pm_ratio: float = 0.01):
    transform = transforms.Compose([transforms.Resize(resize_size),
                                    #BlockShuffle(grid),
                                    #PhaseMasking(pm_ratio),
                                    transforms.ToTensor(), 
                                    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], 
                                                         std=[0.26862954, 0.26130258, 0.27577711]),
                                    #transforms.Resize(resize_size),
                                    ])
    return transform
