import torch
from torch.distributions.multivariate_normal import MultivariateNormal
import numpy as np

def gaussian_pdf(x, mean, var):
    return torch.exp(-(x-mean)**2/(2*var+1e-7))
    #return 1.0/torch.sqrt(2*3.141*var) * torch.exp(-(x-mean)**2/(2*var))

def get_weights(pred_mean, pred_var, y, type="", epsilon=1e-7):
    with torch.no_grad():
        if type=="":
            weights = torch.ones_like(pred_mean)
        elif type == "max":
            weights = 0.5 * (1 + torch.erf((y - pred_mean) / (torch.sqrt(torch.exp(pred_var))+1e-7) / np.sqrt(2)))
        elif type == "max_v2":
            var = torch.exp(pred_var)
            weights = 0.5 * (1 + torch.erf((y - pred_mean) / (torch.sqrt(var)+1e-7) / np.sqrt(2) - var))
        elif type== "mean":
            weights = gaussian_pdf(y, pred_mean, torch.exp(pred_var))
        elif type== "mean_v2":
            var = torch.exp(pred_var)
            weights = torch.sqrt(2*3.141*var) * gaussian_pdf(y, pred_mean, var)
        weights += epsilon
        weights = torch.clip(weights, min=None, max=1e3) 
        return weights

if __name__ == "__main__":
    # test
    y=torch.tensor([2.0,3.0,4.0]).reshape(-1,1)
    pred_mean=torch.tensor([2.2,-3.0,4.1]).reshape(-1,1)
    pred_var=torch.tensor([0.1,0.2,0.01]).reshape(-1,1)
    print(get_weights(pred_mean, pred_var, y))
    print(get_weights(pred_mean, pred_var, y,type="max"))
    print(get_weights(pred_mean, pred_var, y,type="mean"))
    