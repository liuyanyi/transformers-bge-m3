import torch


def normalize(tensor):
    return torch.nn.functional.normalize(tensor, dim=-1)


def compare_float(float1, float2, tol=1e-6):
    return abs(float1 - float2) < tol


def compare_dict(dict1, dict2, tol=1e-6):
    if len(dict1) != len(dict2):
        return False
    for key in dict1:
        if key not in dict2:
            return False
        if not compare_float(dict1[key], dict2[key], tol=tol):
            return False
    return True
