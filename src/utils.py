from src.models.mf_elvis import MF_ELVis
from src.models.elvis import ELVis
from src.models.presley import PRESLEY


def get_model(model_name, dm):
    if model_name == 'MF_ELVis':
        model = MF_ELVis(512, dm.nusers)
    elif model_name == 'ELVis':
        model = ELVis(256, dm.nusers)
    elif model_name == 'PRESLEY':
        model = PRESLEY(512, dm.nusers)
    return model
