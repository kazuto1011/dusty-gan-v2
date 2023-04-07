import torch


def compute_depth_error(depth_ref, depth_gen, mask=None):
    assert depth_ref.ndim == depth_gen.ndim == mask.ndim == 4
    mask = torch.ones_like(depth_ref) if mask is None else mask
    depth_ref = depth_ref.add(1e-8)
    depth_gen = depth_gen.add(1e-8)
    # relative absolute error
    abs_rel = torch.abs(depth_ref - depth_gen) / depth_ref * mask
    abs_rel = abs_rel.sum(dim=(1, 2, 3)) / mask.sum(dim=(1, 2, 3))
    # relative squared error
    sq_rel = (depth_ref - depth_gen) ** 2 / depth_ref * mask
    sq_rel = sq_rel.sum(dim=(1, 2, 3)) / mask.sum(dim=(1, 2, 3))
    # root mean squared error
    rmse = (depth_ref - depth_gen) ** 2 * mask
    rmse = torch.sqrt(rmse.sum(dim=(1, 2, 3)) / mask.sum(dim=(1, 2, 3)))
    # root mean squared error on log depth
    rmse_log = (torch.log(depth_ref) - torch.log(depth_gen)) ** 2 * mask
    rmse_log = torch.sqrt(rmse_log.sum(dim=(1, 2, 3)) / mask.sum(dim=(1, 2, 3)))
    return {
        "abs_rel": abs_rel,
        "sq_rel": sq_rel,
        "rmse": rmse,
        "rmse_log": rmse_log,
    }


def compute_depth_accuracy(depth_ref, depth_gen, mask=None):
    assert depth_ref.ndim == depth_gen.ndim == mask.ndim == 4
    mask = torch.ones_like(depth_ref) if mask is None else mask

    # threshold accuracy
    delta = torch.max(depth_ref / depth_gen, depth_gen / depth_ref)
    a1 = (delta < 1.25**1).float() * mask
    a2 = (delta < 1.25**2).float() * mask
    a3 = (delta < 1.25**3).float() * mask
    a1 = a1.sum(dim=(1, 2, 3)) / mask.sum(dim=(1, 2, 3))
    a2 = a2.sum(dim=(1, 2, 3)) / mask.sum(dim=(1, 2, 3))
    a3 = a3.sum(dim=(1, 2, 3)) / mask.sum(dim=(1, 2, 3))
    return {
        "accuracy_1": a1,
        "accuracy_2": a2,
        "accuracy_3": a3,
    }
