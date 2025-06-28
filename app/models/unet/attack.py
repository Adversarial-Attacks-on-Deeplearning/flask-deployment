import torch


def uap_attack(inp, device, model, delta):
    scaled_delta = 0.25 * delta
    return torch.clamp(inp + scaled_delta, 0, 1)


def deepfool_attack(inp, device, model, delta=None, max_iter=3, overshoot=0.5):
    target_mask = (torch.sigmoid(model(inp)) > 0.5).float()
    inp = inp.clone().detach().to(device)
    target_mask = target_mask.clone().detach().to(device)

    pert_inp = inp.clone().detach().requires_grad_(True)
    r_total = torch.zeros_like(inp).to(device)
    loop_i = 0

    with torch.enable_grad():
        while loop_i < max_iter:
            pert_inp.requires_grad = True
            output = model(pert_inp)
            pred = (torch.sigmoid(output) > 0.5).float()

            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                output, target_mask)
            grad = torch.autograd.grad(loss, pert_inp, retain_graph=False)[0]

            w = grad / (grad.norm() + 1e-8)
            r_i = (loss + 1e-4) * w

            r_total = (6 * r_total + r_i).clamp(-overshoot, overshoot)
            pert_inp = torch.clamp(inp + r_total, 0, 1).detach()

            loop_i += 1

    return pert_inp.detach()
