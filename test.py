import torch
from tqdm import tqdm
import time

def compute_nll(dataloader, model, hparams, device):
    dataloader = torch.utils.data.DataLoader(dataloader, batch_size=1, num_workers=6)
    model.eval()
    bpds = []
    grads = []
    nlls = []

    result = {}
    times = []
    for idx, (x, y) in enumerate(tqdm(dataloader)):
    #for idx, (x, y) in enumerate(dataloader):
        # print(idx)
        # print(x.shape)
        
        x = x.to(device)



        x.requires_grad_()
        # get time
        start = time.time()
        if hparams['y_condition']:
            y = y.to(device)
        else:
            y = None

        try:
            _, bpd, _, nll = model(x, y_onehot=y)
            bpd.backward(retain_graph=True)
        except Exception as e:
            # print exception
            print(x.shape)
            print(e)
            break

        gradient_norm = torch.flatten(x.grad, start_dim=1).norm(dim=1, p=2).mean(dim=0)
        # print(len(nll))
        # print(gradient_norm)
        end_time = time.time()
        times.append(end_time - start)

        nlls.append(nll.detach().cpu().numpy())
        grads.append(gradient_norm.detach().cpu().numpy())
        bpds.append(bpd.detach().cpu().numpy())
    print('Average time: ', sum(times) / len(times)*1000, 'ms')
    result = {'nlls': nlls, 'bpds': bpds, 'grads': grads}
    return result