import torch.nn as nn
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


def _ensure_batched(x: torch.Tensor) -> torch.Tensor:
    """
    Ensures input x is 4D (N,C,H,W). If x is 3D (C,H,W), unsqueeze batch dim.
    """
    if x.dim() == 3:
        return x.unsqueeze(0)
    elif x.dim() == 4:
        return x
    else:
        raise ValueError(f"Expected tensor of 3 or 4 dims, got {x.dim()}.")


def MI_fgsm_single_image(image, device, model, delta=None, epsilon=0.031, iterations=3, mu=1):
    """
    MI-FGSM attack for single image or batch of one.
    Accepts image shape (C,H,W) or (1,C,H,W).
    """
    model.eval()
    # Prepare input
    img = image.clone().detach().to(device)
    batched_img = _ensure_batched(img)

    # Generate pseudo-label
    with torch.no_grad():
        pred = model(batched_img)
        label = (torch.sigmoid(pred) > 0.5).float()

    alpha = epsilon / iterations
    adv = batched_img.clone().detach().requires_grad_(True)
    velocity = torch.zeros_like(adv)
    loss_fn = nn.BCEWithLogitsLoss()

    for _ in range(iterations):
        model.zero_grad()
        out = model(adv)
        loss = loss_fn(out, label)
        loss.backward()

        grad = adv.grad.data
        grad = grad / (grad.abs().sum(dim=(1, 2, 3), keepdim=True) + 1e-12)
        velocity = mu * velocity + grad
        adv = adv + alpha * velocity.sign()

        # Project and clamp
        perturb = torch.clamp(adv - batched_img, min=-epsilon, max=epsilon)
        adv = torch.clamp(batched_img + perturb, 0, 1)
        adv = adv.detach().requires_grad_(True)

    # Return same shape as input
    return adv.squeeze(0) if image.dim() == 3 else adv.detach()


def FGSM_single_image(image, device, model, delta=None, epsilon=0.031):
    """
    FGSM attack for single image or batch of one. No external label: uses model's own segmentation output.
    Accepts image shape (C,H,W) or (1,C,H,W).
    """
    model.eval()
    img = image.clone().detach().to(device)
    batched_img = _ensure_batched(img)

    # Generate pseudo-label for segmentation
    with torch.no_grad():
        pred = model(batched_img)
        label = (torch.sigmoid(pred) > 0.5).float()

    adv = batched_img.clone().detach().requires_grad_(True)
    out = model(adv)
    loss_fn = nn.BCEWithLogitsLoss()
    loss = loss_fn(out, label)
    loss.backward()

    grad = adv.grad.data
    adv = adv + epsilon * grad.sign()
    adv = torch.clamp(adv, 0, 1)

    return adv.squeeze(0) if image.dim() == 3 else adv.detach()


def PGD_single_image(image, device, model, delta=None, epsilon=0.031, iterations=3, random_start=True):
    """
    PGD attack for single image or batch of one. No external label: uses model's own segmentation output.
    Accepts image shape (C,H,W) or (1,C,H,W).
    """
    model.eval()
    img = image.clone().detach().to(device)
    batched_img = _ensure_batched(img)
    alpha = epsilon / iterations

    # Random start
    if random_start:
        adv = batched_img + \
            torch.empty_like(batched_img).uniform_(-epsilon, epsilon)
        adv = torch.clamp(adv, 0, 1)
    else:
        adv = batched_img.clone().detach()
    adv = adv.requires_grad_(True)

    # Pseudo-label
    with torch.no_grad():
        pred = model(batched_img)
        label = (torch.sigmoid(pred) > 0.5).float()
    loss_fn = nn.BCEWithLogitsLoss()

    for _ in range(iterations):
        model.zero_grad()
        out = model(adv)
        loss = loss_fn(out, label)
        loss.backward()

        grad = adv.grad.data
        adv = adv + alpha * grad.sign()
        perturb = torch.clamp(adv - batched_img, min=-epsilon, max=epsilon)
        adv = torch.clamp(batched_img + perturb, 0, 1)
        adv = adv.detach().requires_grad_(True)

    return adv.squeeze(0) if image.dim() == 3 else adv.detach()
