import torch
import random

def PCGrad_loss(losses, optimizers, nets, device):
    
    """
    losses: List of multitask loss functions
    optimizers: List of optimizers used on network
    nets: modules which contain the parameters
    device: device on which code is running i.e. cuda or cpu
    """
    
    flattened_grads = [torch.Tensor([]).to(device) for i in range(len(losses))]
    
    random.shuffle(losses)

    # Accumulate the gradient vectors
    for i, loss in enumerate(losses):
        for optim in optimizers:
            optim.zero_grad()
        loss.backward(retain_graph = True)
        for net in nets:
            for name, param in net.named_parameters():
                flattened_grads[i] = torch.cat([flattened_grads[i],param.grad.view(-1)])

        
    # Compute inner products and projects for gradients
    
    #projected_grads = [torch.Tensor([]) for i in range(len(losses))]
    for i in range(len(flattened_grads)):
        for j in range(len(flattened_grads)):
            inner_product = torch.dot(flattened_grads[i], flattened_grads[j])
            proj_direction = inner_product/torch.dot(flattened_grads[j],flattened_grads[j])
            #print(proj_direction)
            flattened_grads[i] -= torch.clamp_max(proj_direction, 0) * flattened_grads[j]

    # Reassign the gradients to the grad varaible of parameters
    flattened_grads = torch.sum(torch.stack(flattened_grads), dim = 0)
    
    
    # Updated Gradients
    start_idx = 0
    for net in nets:
        for name, param in net.named_parameters():
            flattend_dim = int(np.prod([j for j in param.shape]))
            param.grad = flattened_grads[start_idx:start_idx+flattend_dim].reshape(param.shape).clone().detach()
            start_idx += flattend_dim
