import numpy as np

def estimate_model_size(model):
    weights = model.count_params()
    return weights * 1 # 1 byte per parameter for int8 quantization

def fits_in_ram(model, limitkb=50):
    est = estimate_model_size(model)/1024
    return est <= limitkb
