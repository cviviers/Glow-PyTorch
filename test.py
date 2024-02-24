import torch
from tqdm import tqdm

def compute_nll(dataloader, model, hparams, device):
    dataloader = torch.utils.data.DataLoader(dataloader, batch_size=1, num_workers=6)
    
    bpds = []
    grads = []
    nlls = []

    result = {}
    
    for idx, (x, y) in enumerate(tqdm(dataloader)):
    #for idx, (x, y) in enumerate(dataloader):
        # print(idx)
        # print(x.shape)
        
        x = x.to(device)
        x.requires_grad_()

        if hparams['y_condition']:
            y = y.to(device)
        else:
            y = None
        
        _, bpd, _, nll = model(x, y_onehot=y)
        bpd.backward(retain_graph=True)

        gradient_norm = torch.flatten(x.grad, start_dim=1).norm(dim=1, p=2).mean(dim=0)
        # print(len(nll))
        # print(gradient_norm)

        nlls.append(nll.detach().cpu().numpy())
        grads.append(gradient_norm.detach().cpu().numpy())
        bpds.append(bpd.detach().cpu().numpy())

    result = {'nlls': nlls, 'bpds': bpds, 'grads': grads}
    return result