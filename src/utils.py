from src.models.mf_elvis import MF_ELVis
from src.models.elvis import ELVis
from src.models.presley import PRESLEY
from src.models.collei import COLLEI

from src.datamodule import *


def get_model(model_name, dm):
    if model_name == 'MF_ELVis':
        model = MF_ELVis(512, dm.nusers)
    elif model_name == 'ELVis':
        model = ELVis(256, dm.nusers)
    elif model_name == 'PRESLEY':
        model = PRESLEY(64, dm.nusers)
    elif model_name == 'COLLEI':
        model = COLLEI(256, dm.nusers)
    return model


def get_dataset_constructor(model_name):
    if model_name in ['MF_ELVis', 'ELVis']:
        dataset = TripadvisorImageAuthorshipBCEDataset
    elif model_name in ['COLLEI']:
        dataset = TripadvisorImageAuthorshipCLDataset
    elif model_name in ['PRESLEY']:
        dataset = TripadvisorImageAuthorshipBPRDataset
    return dataset
