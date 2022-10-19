
# [0]
# What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?
# https://arxiv.org/pdf/1703.04977.pdf
import torch

# implementation of the gaussian negative log likelihood loss
# similar to https://pytorch.org/docs/stable/generated/torch.nn.GaussianNLLLoss.html#torch.nn.GaussianNLLLoss
# the following implementation drives the model to learn log(sigma^2)
# this is more numerical stable than learning sigma^2 directly [see 0]
def gnlll_loss(y_pred, y_gt, s_pred, use_smoothed=False, beta=1.0, reduction=True, mask=None):
    s_pred = torch.clip(s_pred, -20.0, 20.0)
    if use_smoothed: return smoothed_gnlll_loss(y_pred, y_gt, s_pred, beta=beta, reduction=reduction)
    inner_term = 0.5 * torch.exp(-s_pred) * (y_gt-y_pred)**2
    regularization_term = 0.5 * s_pred
    loss = inner_term + regularization_term

    if mask is not None:
        loss *= mask

    if reduction: loss = torch.mean(loss) # per sample mean
    return loss # batch mean

def weighted_gnlll_loss(y_pred, y_gt, s_pred, weights, use_smoothed=False, beta=1.0, reduction=False):
    s_pred = torch.clip(s_pred, -20.0, 20.0)
    if use_smoothed:
        loss = smoothed_gnlll_loss(y_pred, y_gt, s_pred, beta=beta, reduction=reduction)
    else:
        inner_term = 0.5 * torch.exp(-s_pred) * ((y_gt-y_pred))**2
        regularization_term = 0.5 * s_pred
        loss = inner_term + regularization_term
    assert weights.shape == loss.shape, print(weights.shape, loss.shape)
    weighted_loss = loss * weights
    #print(loss[weights==weights.max()], weighted_loss[weights==weights.max()])
    #weighted_loss = torch.mean(weighted_loss) # per sample mean
    # apply weights and normalize, important if the total loss is a combination of multiple losses
    # this is true if different weights are used for each part. loss
    weighted_loss = weighted_loss.sum() / (weights.sum()+1e-7)
    return weighted_loss

def smoothed_gnlll_loss(y_pred, y_gt, s_pred, beta=1.0, reduction=True):
    variance_term = 0.5 * torch.exp(-s_pred)

    # smoothed mse term
    # see https://pytorch.org/docs/stable/generated/torch.nn.SmoothL1Loss.html
    y_delta = y_gt-y_pred
    inner_term = y_delta**2 / beta
    smoothed_inner_term = 2*torch.abs(y_delta) - beta
    condition = torch.where(torch.abs(y_delta)<beta,True,False)
    combined_inner_term = condition * inner_term + ~condition * smoothed_inner_term

    regularization_term = 0.5 * s_pred
    loss = variance_term*combined_inner_term + regularization_term
    if reduction: loss = torch.mean(loss) # per sample mean
    return loss # batch mean

if __name__ == "__main__":
    # test
    y=torch.tensor([2.0,3.0,4.0,1.5]).reshape(-1,1,2,2)
    pred_mean=torch.tensor([2.2,-3.0,4.1,1.0]).reshape(-1,1,2,2)
    pred_var=torch.tensor([0.1,0.2,0.01,0.5]).reshape(-1,1,2,2)
    print(gnlll_loss(pred_mean, y, pred_var))
    print(weighted_gnlll_loss(pred_mean, y, pred_var))
    print(weighted_gnlll_loss(pred_mean, y, pred_var, weight_type="max"))
    print(weighted_gnlll_loss(pred_mean, y, pred_var, weight_type="mean"))