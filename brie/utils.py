from brie.models.mf_elvis import MF_ELVis
from brie.models.elvis import ELVis
from brie.models.brie import BRIE

from brie.datamodule import *


def get_model(model_name, config, nusers):
    if model_name == 'MF_ELVis':
        model = MF_ELVis(d=config['d'],
                         nusers=nusers,
                         lr=config['lr'])
    elif model_name == 'ELVis':
        model = ELVis(d=config['d'],
                      nusers=nusers,
                      lr=config['lr'])
    elif model_name == 'BRIE':
        model = BRIE(d=config['d'],
                     nusers=nusers,
                     lr=config['lr'],
                     dropout=config['dropout'])
    else:
        raise ValueError(f"Unknown model {model_name!r}")
    return model


def get_dataset_constructor(model_name):
    if model_name in ['MF_ELVis', 'ELVis']:
        dataset = TripadvisorImageAuthorshipBCEDataset
    elif model_name in ['BRIE']:
        dataset = TripadvisorImageAuthorshipBPRDataset
    else:
        raise ValueError(f"Unknown model {model_name!r}")
    return dataset
