
import sys
import io
import torch
from ultralytics import YOLO
from app.models.yolo.utils import preprocess_image  # custom function
import torch.nn.functional as F
import os
import numpy as np
from typing import Optional
from PIL import Image
from torchvision import transforms


def disappearance_dag_attack(
    image_path,
    model_path='yolov8n.pt',
    num_iterations=10,
    gamma=0.03,
    conf_threshold=0.5,
    device='cpu'
):
    """
    Demonstration of a DAG-like multi-iteration attack on YOLOv8
    that pushes all predictions (with high confidence) away from
    their currently predicted class.

    Args:
        image_path (str): Path to the input image.
        raw_model_path (str): Path to the YOLOv8 weights for the raw model.
        high_level_model_path (str): Path to the YOLOv8 weights for the high-level model.
        num_iterations (int): Number of attack iterations.
        gamma (float): L_inf step size per iteration.
        conf_threshold (float): Confidence threshold for selecting rows to attack.
        device (str): 'cuda' or 'cpu'.
    """

    # ----------------------------------------------------------------
    # 1. load the model
    # ----------------------------------------------------------------
    model = YOLO(model_path).to(device)

    # ----------------------------------------------------------------
    # 2. Preprocess and load the image
    # ----------------------------------------------------------------
    # should return a torch.Tensor of shape [1, 3, H, W]
    image = preprocess_image(image_path)
    image = image.to(device)
    image.requires_grad_(True)

    # ----------------------------------------------------------------
    # 3. Iterative attack loop
    # ----------------------------------------------------------------
    for iteration in range(num_iterations):
        # ------------------------------------------------------------
        # (a) Forward pass through the raw underlying model
        # ------------------------------------------------------------
        # The raw model can return a tuple (preds, training_outputs), so unpack:
        raw_preds, _ = model.model(image)

        # raw_preds might be shape [1, 84, N], so we permute to [1, N, 84]
        raw_preds = raw_preds.permute(0, 2, 1)  # now [1, N, 84]
        # Split out the class logits: first 4 channels are box coords, next 80 are class scores
        class_logits = raw_preds[..., 4:]  # shape [1, N, 80]

        # Apply sigmoid if these are logits
        conf = class_logits.sigmoid()  # shape [1, N, 80]

        # Get max confidence per row and the corresponding predicted class
        pred_conf, pred_class = conf.max(dim=-1)  # both [1, N]

        # ------------------------------------------------------------
        # (b) Identify rows to attack
        # ------------------------------------------------------------
        # Let's see what the *high-level* model is currently predicting
        # (This is optional, but helps you track final bounding boxes, etc.)
        with torch.no_grad():
            high_level_preds = model(image, conf=conf_threshold)
        results = high_level_preds[0]  # 'Results' object
        # the classes YOLO is showing after NMS
        pred_class_ids = results.boxes.cls.tolist()

        # Now, from the raw model side, pick all rows with conf > conf_threshold
        # whose predicted class is in pred_class_ids. (We treat them as "currently predicted")
        rows_to_attack = []
        target_classes = set(int(x) for x in pred_class_ids)
        for i in range(pred_class.shape[1]):
            row_pred_class = int(pred_class[0, i].item())
            row_pred_conf = float(pred_conf[0, i].item())
            if row_pred_class in target_classes and row_pred_conf > conf_threshold:
                rows_to_attack.append(i)

        # If no rows to attack, we can break early
        if not rows_to_attack:
            print(
                f"Iteration {iteration}: no rows above threshold => stopping early.")
            break

        # ------------------------------------------------------------
        # (c) Build the loss
        # ------------------------------------------------------------
        # We'll do an untargeted approach: push the logit of the predicted class down.
        loss = torch.zeros(1, device=device)
        for i in rows_to_attack:
            c = pred_class[0, i]  # the predicted class index
            logit_c = class_logits[0, i, c]
            loss += -logit_c  # negative logit => push it down

        # ------------------------------------------------------------
        # (d) Backprop and update the image
        # ------------------------------------------------------------
        # Zero out old gradients
        if image.grad is not None:
            image.grad.zero_()

        # Backprop
        loss.backward()

        # Get the gradient on the image
        grad = image.grad.data

        # L-infinity step
        step = gamma * grad.sign()
        with torch.no_grad():
            image += step
            image.clamp_(0, 1)

        image.grad.zero_()  # reset for next iteration

        print(
            f"Iteration {iteration}: #rows_to_attack={len(rows_to_attack)}, loss={loss.item():.4f}")

    # ----------------------------------------------------------------
    # 4. Return or save the final adversarial image
    # ----------------------------------------------------------------
    return image.detach()


def targeted_dag_attack(
    image_path,
    model_path='yolov8n.pt',
    adversarial_class=2,
    num_iterations=10,
    gamma=0.03,
    conf_threshold=0.5,
    device='cpu'
):
    """
    Demonstration of a DAG-like targeted attack that:
      1) Attacks rows whose predicted class is in 'target_classes' (from the clean image).
      2) If no rows remain, we push the adversarial_class globally (so it still emerges).
      3) We stop early if the adversarial_class appears in high-level predictions
         and none of the original target classes remain.

    Args:
        image_path (str): Path to input image.
        raw_model_path (str): YOLOv8 weights for the raw model (for gradients).
        high_level_model_path (str): YOLOv8 weights for the high-level model (for post-NMS checks).
        adversarial_class (int): Class index to push everything toward.
        num_iterations (int): Max number of iterations.
        gamma (float): L_inf step size.
        conf_threshold (float): Confidence threshold for selecting rows and for post-NMS.
        device (str): 'cuda' or 'cpu'.
    """

    # ---------------------------------------------------------------
    # 1. Load the models
    # ---------------------------------------------------------------
    high_level_model = YOLO(model_path)
    underlying_model = YOLO(model_path).model
    underlying_model.eval()
    for param in underlying_model.parameters():
        param.requires_grad = False
    underlying_model.to(device)

    # ---------------------------------------------------------------
    # 2. Load and prepare the image
    # ---------------------------------------------------------------
    image = preprocess_image(image_path)  # [1, 3, H, W]
    image = image.to(device)
    image.requires_grad_(True)

    # ---------------------------------------------------------------
    # 3. Determine original target classes from the clean image
    # ---------------------------------------------------------------
    with torch.no_grad():
        clean_preds = high_level_model(image, conf=conf_threshold)
    clean_results = clean_preds[0]
    original_pred_class_ids = clean_results.boxes.cls.tolist()
    target_classes = set(int(x) for x in original_pred_class_ids)

    print(f"Original classes above conf={conf_threshold}: {target_classes}")
    # If none found, we can either return or proceed with an empty set.

    # ---------------------------------------------------------------
    # 4. Attack loop
    # ---------------------------------------------------------------
    for iteration in range(num_iterations):
        # -----------------------------------------------------------
        # (a) Raw forward pass (for gradients)
        # -----------------------------------------------------------
        raw_preds, _ = underlying_model(image)        # shape [1, 84, N]
        raw_preds = raw_preds.permute(0, 2, 1)        # [1, N, 84]
        class_logits = raw_preds[..., 4:]             # [1, N, 80]
        conf = class_logits.sigmoid()                # [1, N, 80]
        pred_conf, pred_class = conf.max(dim=-1)      # both [1, N]
        num_rows = pred_class.shape[1]

        # -----------------------------------------------------------
        # (b) Identify rows to attack (pred_class in target_classes)
        # -----------------------------------------------------------
        rows_to_attack = []
        for i in range(num_rows):
            row_pred_class = int(pred_class[0, i].item())
            row_pred_conf = float(pred_conf[0, i].item())
            if row_pred_class in target_classes and row_pred_conf > conf_threshold:
                rows_to_attack.append(i)

        # -----------------------------------------------------------
        # (c) Build the loss
        # -----------------------------------------------------------
        loss = torch.zeros(1, device=device)
        if rows_to_attack:
            # (1) Usual targeted push: (logit_adv - logit_orig)
            for i in rows_to_attack:
                c = pred_class[0, i]
                logit_orig = class_logits[0, i, c]
                logit_adv = class_logits[0, i, adversarial_class]
                loss += (logit_adv - logit_orig)
        else:
            # (2) If no rows remain, just boost the adversarial class globally
            #     so we force it to appear somewhere in the image.
            print(
                f"Iteration {iteration}: no rows above threshold => boosting adversarial_class globally.")
            for i in range(num_rows):
                logit_adv = class_logits[0, i, adversarial_class]
                loss += logit_adv

        # -----------------------------------------------------------
        # (d) Backprop + update
        # -----------------------------------------------------------
        if image.grad is not None:
            image.grad.zero_()
        loss.backward()

        grad = image.grad.data
        step = gamma * grad.sign()
        with torch.no_grad():
            image += step
            image.clamp_(0, 1)
        image.grad.zero_()

        # -----------------------------------------------------------
        # (e) Check stop condition
        #     If adversarial_class is in current predictions
        #     AND none of the original target_classes remain
        # -----------------------------------------------------------
        with torch.no_grad():
            current_preds = high_level_model(image, conf=conf_threshold)
        results = current_preds[0]
        current_class_ids = set(int(box.cls.item()) for box in results.boxes)
        # Condition: adversarial_class in current_class_ids AND disjoint from target_classes
        if (adversarial_class in current_class_ids) and current_class_ids.isdisjoint(target_classes):
            print(
                f"Iteration {iteration}: Stop condition met! {adversarial_class=} present, {target_classes=} gone.")
            break

        print(
            f"Iteration {iteration}: #rows_to_attack={len(rows_to_attack)}, loss={loss.item():.4f}")

    # ---------------------------------------------------------------
    # Final check: determine if attack was successful.
    # ---------------------------------------------------------------
    with torch.no_grad():
        final_preds = high_level_model(image, conf=conf_threshold)
    final_results = final_preds[0]
    final_class_ids = set(int(box.cls.item()) for box in final_results.boxes)
    success = (adversarial_class in final_class_ids) and final_class_ids.isdisjoint(
        target_classes)
    if success:
        print("Attack successful: adversarial_class present and original target classes are gone.")
    else:
        print("Attack failed: conditions not met.")

    return image.detach()


def fool_detectors_attack(
    image_path,
    model_path='yolov8n.pt',
    num_iterations=10,
    gamma=0.03,
    conf_threshold=0.5,
    lambda_reg=0.01,
    device='cpu'
):
    """
    Implements a DAG-like multi-iteration attack on YOLOv8 targeting specified classes,
    inspired by the paper "Adversarial Examples that Fool Detectors".

    The attack minimizes the mean detection confidence for the target classes in the image by
    iteratively updating the input image using the sign of the gradient (L_inf update). An L2 penalty 
    is added to ensure the adversarial image remains close to the original. The attack stops early if 
    the high-level model no longer detects any objects of the target classes.

    Args:
        image_path (str): Path to the input image.
        raw_model_path (str): Path to the YOLOv8 weights for the raw model (used for gradient computation).
        high_level_model_path (str): Path to the YOLOv8 weights for the high-level model (used for detection).
        num_iterations (int): Number of iterations for the attack.
        gamma (float): Step size for the L_inf gradient update.
        conf_threshold (float): Confidence threshold for selecting detections.
        lambda_reg (float): Weight for the L2 penalty to maintain similarity to the original image.
        target_classes (list of int): List of COCO class IDs to target (e.g., [11] for stop signs).
        device (str): Device to use ('cuda' or 'cpu').

    Returns:
        torch.Tensor: The final adversarial image tensor.

    Alternative Name:
        FoolDetectorsAttack
    """
    # Use target_classes as provided.

    # 1. Load the YOLO models.
    # --------------------------------------------------------------------------
    # The high-level model is used to check detection results (post-processing),
    # while the underlying (raw) model is used for gradient-based updates.
    high_level_model = YOLO(model_path)
    results = high_level_model(str(image_path))
    target_classes = results[0].boxes.cls.int().tolist()
    underlying_model = YOLO(model_path).model
    underlying_model.eval()  # Set to evaluation mode.
    # Disable gradient computations for the underlying model parameters.
    for param in underlying_model.parameters():
        param.requires_grad = False
    underlying_model.to(device)

    # 2. Preprocess and load the input image.
    # --------------------------------------------------------------------------
    # preprocess_image should return a torch.Tensor of shape [1, 3, H, W]
    image = preprocess_image(image_path)
    image = image.to(device)
    # Save a copy of the original image for the L2 (similarity) penalty.
    original_image = image.clone().detach()
    # Enable gradient computation on the input image.
    image.requires_grad_(True)

    # 3. Iterative attack loop.
    # --------------------------------------------------------------------------
    for iteration in range(num_iterations):
        # Forward pass: compute predictions using the raw model.
        raw_preds, _ = underlying_model(image)
        raw_preds = raw_preds.permute(0, 2, 1)
        class_logits = raw_preds[..., 4:]
        conf_scores = class_logits.sigmoid()

        # Extract confidence scores for the target classes.
        # This indexes dimension 2 with the list of target classes.
        # Shape: [1, N_boxes, len(target_classes)]
        target_confidences = conf_scores[..., target_classes]
        # Compute the detector loss as the mean confidence score for target classes.
        loss_det = target_confidences.mean()

        # Compute the L2 penalty to ensure the adversarial image remains similar to the original.
        loss_l2 = lambda_reg * F.mse_loss(image, original_image)

        # Total loss: detector loss plus the L2 penalty.
        loss = loss_det + loss_l2

        # Optional: Use the high-level model to check if any target objects are detected.
        with torch.no_grad():
            high_level_preds = high_level_model(image, conf=conf_threshold)
        results = high_level_preds[0]
        # Extract predicted class IDs (assuming results.boxes.cls holds these).
        detected_classes = results.boxes.cls.tolist(
        ) if hasattr(results.boxes, "cls") else []
        # Early stopping: if none of the detected classes is in target_classes, break.
        if not set(target_classes).intersection(set(detected_classes)):
            print(
                f"Iteration {iteration}: No target objects detected. Early stopping.")
            break

        # Backpropagation: Compute gradients of the loss with respect to the image.
        if image.grad is not None:
            image.grad.zero_()  # Reset gradients.
        loss.backward()

        # Get the gradient and update the image in the negative gradient direction to minimize the loss.
        grad = image.grad.data
        with torch.no_grad():
            # Subtract the sign of the gradient scaled by gamma (L_inf update).
            image -= gamma * grad.sign()
            # Ensure the pixel values remain valid (clamped to [0, 1]).
            image.clamp_(0, 1)
        image.grad.zero_()  # Reset gradient for next iteration.

        # Print the current loss values for debugging.
        print(f"Iteration {iteration}: Loss_det={loss_det.item():.4f}, "
              f"Loss_L2={loss_l2.item():.4f}, Total_loss={loss.item():.4f}")

    # Return the final adversarial image tensor.
    return image.detach()

# Adversarial patch attack

# Define the loss function for the Creation Attack


def creation_attack_loss(raw_pred, confidences, patch_location, target_id):
    """
    Compute the Creation Attack Loss.
    Args:
        raw_pred: raw output tensor from YOLO v8 to get box location.
        confidences: classes confidences to optimize for the target class.
        patch_location: Tuple (x, y) representing the grid cell where the patch is placed.
    return: 
    Loss value
    stop flag
    """
    loss = None
    grid_cell_x, grid_cell_y = patch_location
    # loop on cells to get the patch cell
    for pred, conf in zip(raw_pred[0], confidences):
        x_min, y_min = pred[0].int(), pred[1].int()
        if (x_min == int(grid_cell_x) and y_min == int(grid_cell_y)):
            # Minimize this to increase the target class probability
            loss = 1 - conf[target_id]
            break

    return loss

# Generate an initial random patch


def preprocess_init_patch(size):
    """
    preprocess initial patch.
    Args:
        size: Tuple (width, height) for the patch size.
    return: 
        initial patch as a numpy array.
    """
    patch = Image.open('ss_patch.png').convert('RGB')
    patch = patch.resize(size)
    patch = np.array(patch) / 255.0
    patch = np.transpose(patch, (2, 0, 1))
    return patch
# Apply the patch to an image at a fixed random location


def apply_patch(image, patch, patch_location, patch_size):
    """
    Apply the adversarial patch to an image at a predetermined location.
    Args:
        image: Input image (tensor).
        patch: Adversarial patch (tensor).
        patch_location: Tuple (x, y) representing the location to place the patch.
    return:
        Tensor image with the patch applied.
    """

    _, _, image_width, image_height = image.shape
    patch_w, patch_h = patch_size

    x, y = patch_location
    # Ensure the patch fits within the image boundaries
    x_start = max(0, int(x) - patch_w // 2)
    y_start = max(0, int(y) - patch_h // 2)
    x_end = min(image_width, x_start + patch_w)
    y_end = min(image_height, y_start + patch_h)

    # Adjust patch slicing if the patch extends beyond the image boundaries
    patch_x_start = 0
    patch_y_start = 0
    patch_x_end = patch_w
    patch_y_end = patch_h

    # Ensure the target region and patch have the same dimensions
    target_region = image[:, :, y_start:y_end, x_start:x_end]
    patch_slice = patch[:, patch_y_start:patch_y_end,
                        patch_x_start:patch_x_end]

    # Check if dimensions match
    if target_region.shape[2:] != patch_slice.shape[1:]:
        print(
            f"Warning: Patch dimensions {patch_slice.shape[1:]} do not match target region dimensions {target_region.shape[2:]}. Resizing patch to fit.")

        # Resize the patch to match the target region dimensions
        from torch.nn.functional import interpolate
        patch_slice = interpolate(
            patch_slice.unsqueeze(0),  # Add batch dimension
            size=target_region.shape[2:],  # Target size (height, width)
            mode='bilinear',  # Interpolation mode
            align_corners=False
        ).squeeze(0)  # Remove batch dimension

    # Apply the patch to the image
    image[:, :, y_start:y_end, x_start:x_end] = patch_slice
    return image

# Optimize the adversarial patch


def optimize_patch(
    image_path,
    patch,
    target_id,
    num_iterations=200,
    raw_model_path="yolov8n_road.pt",
    learning_rate=0.06,
    conf_th=0.5,
    device="cpu"
):
    """
    Optimize the adversarial patch using gradient descent.
    Args:
        images: List of images for optimization.
        patch: Initial adversarial patch.
        num_iterations: Number of optimization iterations.
        raw_model_path (str): YOLOv8 weights for the raw model (for gradients).
        learning_rate: Learning rate for gradient descent.
    return: 
        Optimized adversarial patch.
    """
    # Move the patch to the model device
    if device == "cuda":
        patch = torch.tensor(patch, dtype=torch.float32,
                             device="cuda", requires_grad=True)
    else:
        patch = torch.tensor(patch, dtype=torch.float32, requires_grad=True)

    # Load the yolo model
    high_level_model = YOLO(raw_model_path)
    underlying_model = YOLO(raw_model_path).model
    underlying_model.eval()  # Set to evaluation mode.
    # Disable gradient computations for the underlying model parameters.
    for param in underlying_model.parameters():
        param.requires_grad = False
    underlying_model.to(device)

    # Preprocess and load the input image.
    image = preprocess_image(image_path)
    image = image.to(device)
    # Save a copy of the original image
    original_image = image.clone().detach()

    # Forward pass: compute predictions using the raw model.
    raw_preds, _ = underlying_model(original_image)
    # Rearrange predictions from shape [1, channels, N_boxes] to [1, N_boxes, channels].
    raw_preds = raw_preds.permute(0, 2, 1)
    class_logits = raw_preds[0][:, 4:]
    conf = class_logits.sigmoid()

    # Select the box with the highest class probability for the target class
    best_pred_idx = torch.argmax(conf[:, target_id]).item()
    best_pred = raw_preds[0][best_pred_idx, :]
    # Get the bounding box coordinates (x, y) for the best prediction
    patch_x, patch_y = best_pred[0].item(), best_pred[1].item()
    patch_location = (patch_x, patch_y)
    patch_size = (best_pred[2].int().item(), best_pred[3].int().item())
    for iteration in range(num_iterations):
        torch.cuda.empty_cache()
        # Original loss calculation
        loss = torch.tensor(0.0, device=image.device, requires_grad=True)
        # apply patch on determined box
        patched_image = apply_patch(
            original_image.clone(), patch, patch_location, patch_size)

        # Forward pass: Compute predictions using the raw model on patched image.
        raw_preds, _ = underlying_model(patched_image)
        raw_preds = raw_preds.permute(0, 2, 1)
        class_logits = raw_preds[0][:, 4:]
        conf = class_logits.sigmoid()

        # Compute the loss
        loss = creation_attack_loss(raw_preds, conf, patch_location, target_id)

        # stop when target object added
        with torch.no_grad():
            current_preds = high_level_model(patched_image, conf=conf_th)
        results = current_preds[0]
        current_class_ids = set(int(box.cls.item()) for box in results.boxes)
        # Condition: adversarial_class in current_class_ids
        if (target_id in current_class_ids):
            print(f"Iteration {iteration}: Stop condition met!")
            break

        # Backpropagate and update the patch
        if loss is not None:
            loss.backward(retain_graph=True)
            # Manually update the patch using gradient descent
            with torch.no_grad():
                grad = patch.grad.data
                grad_norm = torch.norm(grad, p=float('inf'))
                if grad_norm == 0:
                    grad_norm = 1e-8
                patch -= learning_rate * (grad / grad_norm)  # Update patch
                patch.grad.zero_()  # Reset gradients
                # Clip the patch values to ensure they are valid pixel values
                patch.data = torch.clamp(patch.data, 0, 1)

            # Re-enable gradients for the next iteration
            patch.requires_grad = True

            # Print the loss for monitoring
            print(f"Iteration {iteration}, Loss: {loss.item()}")

        else:
            print(f"Iteration {iteration}: location not found")
            # change patch location
            # Select the prediction with the highest class probability for the target class
            best_pred_idx = torch.argmax(conf[:, target_id]).item()
            best_pred = raw_preds[0][best_pred_idx, :]
            # Get the bounding box coordinates (x, y) for the best prediction
            patch_x, patch_y = best_pred[0].item(), best_pred[1].item()
            patch_location = (patch_x, patch_y)

    return patched_image.detach().cpu(), patch.detach().cpu()


def fgsm_attack_detector(
    image_path: str,
    model: YOLO,
    epsilon: float = 0.05,
    conf_threshold: float = 0.5,
    device: str = 'cpu'
) -> Optional[torch.Tensor]:
    """
    Generates an adversarial example for an object detection model using the Fast Gradient Sign Method (FGSM).

    This implementation attacks high-confidence detections by perturbing the input image in the direction
    that maximizes classification loss while maintaining visual similarity to the original image.

    Args:
        image_path (str): Path to the input image file (JPG/PNG)
        model (YOLO): Initialized YOLOv8 detection model
        epsilon (float): Magnitude of perturbation (controls attack strength, typically 0.01-0.1)
        conf_threshold (float): Minimum confidence score to consider detections (0.0-1.0)
        device (str): Computation device ('cpu' or 'cuda')

    Returns:
        torch.Tensor: Adversarial image tensor of shape [1, 3, H, W] clamped to [0,1]
        None: If no high-confidence detections are found

    Raises:
        RuntimeError: If model output format is unexpected
    """

    # 1. Model and Image Preparation ===========================================
    # Move model to target device
    model = model.to(device)

    # Preprocess image (normalization, resizing) and enable gradient tracking
    # preprocess_image() should return tensor of shape [1, 3, H, W] in [0,1] range
    image = preprocess_image(image_path).to(device)
    image.requires_grad_(True)

    # 2. Raw Model Output Extraction ==========================================
    # Bypass post-processing to access raw predictions:
    # Shape: [batch_size, num_anchors, 4 + 1 + num_classes]
    # Where 4=box_coords (x_center, y_center, width, height)
    #       1=objectness score
    #       num_classes=class logits
    raw_outputs = model.model(image)

    # Verify output dimensions
    if raw_outputs[0].shape[-1] < 5:
        raise RuntimeError(
            f"Unexpected output format. Expected at least 5 channels, got {raw_outputs[0].shape[-1]}")

    # 3. Prediction Processing ================================================
    raw_preds = raw_outputs[0]  # First detection head output
    num_classes = raw_preds.shape[-1] - 5  # Calculate number of classes

    # Extract components from raw predictions
    box_coords = raw_preds[..., :4]   # Bounding box coordinates (xywh format)
    obj_scores = raw_preds[..., 4:5]  # Objectness scores (anchor quality)
    # Class prediction logits (before softmax)
    cls_logits = raw_preds[..., 5:]

    # 4. Confidence Calculation ===============================================
    # Combined confidence = objectness * max class probability
    # Convert to probability [0,1]
    obj_probs = obj_scores.sigmoid()
    cls_probs = cls_logits.softmax(
        dim=-1).max(dim=-1, keepdim=True)[0]  # Max class prob
    conf_scores = obj_probs * cls_probs             # Final detection confidence

    # 5. High-Confidence Detection Filtering ==================================
    # Create boolean mask for detections above confidence threshold
    mask = conf_scores.squeeze(-1) > conf_threshold

    if not mask.any():
        print(f"No detections found with confidence > {conf_threshold}")
        return None

    # 6. Loss Calculation =====================================================
    # Get original predicted classes (detached from computation graph)
    target_classes = cls_logits.argmax(dim=-1)[mask].detach()

    # Extract relevant logits for high-confidence detections
    current_logits = cls_logits[mask]

    # Cross-entropy loss between current predictions and original classes
    classification_loss = torch.nn.functional.cross_entropy(
        current_logits,
        target_classes
    )

    # Optional: Add bounding box regression loss
    # target_boxes = box_coords[mask].detach()
    # box_loss = torch.nn.functional.smooth_l1_loss(box_coords[mask], target_boxes)
    # total_loss = classification_loss + box_loss

    total_loss = classification_loss

    # 7. Adversarial Perturbation Generation ==================================
    # Clear previous gradients and backpropagate
    model.zero_grad()
    total_loss.backward()

    # Extract gradient from input image
    image_grad = image.grad.data

    # Apply FGSM: x_adv = x + ε * sign(∇x J(x, y))
    perturbed_image = image + epsilon * image_grad.sign()

    # Ensure valid pixel range and disable gradient tracking
    perturbed_image = torch.clamp(perturbed_image, 0, 1).detach()

    return perturbed_image


def train_universal_attack(
    model,
    final_image_paths,
    preprocess_fn,
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    num_epochs_uap=20,
    lambda_reg=0.01,
    epsilon=0.1,
    conf_threshold=0.2,
    lr=0.1,
    momentum=0.9
):
    """
    Trains a universal adversarial perturbation (delta) for the given YOLO model.

    :param model: The YOLO model (with .model as PyTorch module).
    :param final_image_paths: List of image paths used to train the universal perturbation.
    :param preprocess_fn: A function to preprocess a given image_path -> returns a torch.Tensor [1, C, H, W].
    :param device: Torch device (e.g., cuda or cpu).
    :param num_epochs_uap: Number of epochs to train.
    :param lambda_reg: Regularization coefficient for delta's L2 norm.
    :param epsilon: Clamping range for the universal delta in each pixel dimension.
    :param conf_threshold: Confidence threshold to filter detections.
    :param lr: Learning rate for SGD.
    :param momentum: Momentum for SGD.

    :return: The learned universal delta (torch.Tensor) and a list of (epoch_loss) for each epoch.
    """
    # Move model to device
    model.model.to(device)

    # Use one sample to get shape for delta initialization
    img_sample = preprocess_fn(final_image_paths[0]).to(device)
    delta = torch.zeros_like(img_sample, requires_grad=True, device=device)

    # Set up optimizer for delta
    optimizer = torch.optim.SGD([delta], lr=lr, momentum=momentum)

    # Record epoch losses if you want to track progress
    epoch_losses = []

    for epoch in range(num_epochs_uap):
        epoch_loss = 0.0

        for image_path in final_image_paths:
            image = preprocess_fn(image_path).to(device)

            # For universal perturbation, same delta for all images
            adv_image = image + delta

            # Forward pass on perturbed image
            raw_outputs = model.model(adv_image)
            if raw_outputs[0].shape[-1] < 5:
                raise RuntimeError(
                    f"Unexpected output format. Expected at least 5 channels, got {raw_outputs[0].shape[-1]}"
                )

            raw_preds = raw_outputs[0]
            num_classes = raw_preds.shape[-1] - 5

            obj_scores = raw_preds[..., 4:5]
            cls_logits = raw_preds[..., 5:]

            obj_probs = obj_scores.sigmoid()
            cls_probs = cls_logits.softmax(dim=-1).max(dim=-1, keepdim=True)[0]
            conf_scores = obj_probs * cls_probs

            mask = conf_scores.squeeze(-1) > conf_threshold
            if not mask.any():
                # No detections => skip
                continue

            target_classes = cls_logits.argmax(dim=-1)[mask].detach()
            current_logits = cls_logits[mask]

            classification_loss = torch.nn.functional.cross_entropy(
                current_logits, target_classes
            )

            # Optional L2 reg on delta
            reg_loss = lambda_reg * torch.norm(delta, p=2)
            total_loss = classification_loss + reg_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            with torch.no_grad():
                delta.data = torch.clamp(delta, -epsilon, epsilon)

            epoch_loss += total_loss.item()

        avg_loss = epoch_loss / len(final_image_paths)
        epoch_losses.append(avg_loss)
        print(f"UAP Epoch [{epoch+1}/{num_epochs_uap}] Loss: {avg_loss:.4f}")

    print("Training complete.")
    return delta, epoch_losses


def pgd_attack_detector(
    image_path: str,
    model: YOLO,
    epsilon: float = 0.05,
    num_steps: int = 7,
    step_size: float = 0.005,
    conf_threshold: float = 0.5,
    random_start: bool = True,
    device: str = 'cpu'
) -> Optional[torch.Tensor]:
    """
    Generates an adversarial example for an object detection model using the Projected Gradient Descent (PGD) attack.

    This implementation attacks high-confidence detections by iteratively perturbing the input image to maximize
    classification loss, projecting perturbations to stay within an L∞ ε-ball to maintain visual similarity.

    Args:
        image_path (str): Path to the input image file (JPG/PNG)
        model (YOLO): Initialized YOLOv8 detection model
        epsilon (float): Maximum L∞ perturbation magnitude (typically 0.01-0.1)
        num_steps (int): Number of PGD iterations (default: 7)
        step_size (float): Perturbation step size per iteration (default: 0.005, typically ε/4 or smaller)
        conf_threshold (float): Minimum confidence score to consider detections (0.0-1.0)
        random_start (bool): Whether to initialize with random noise in ε-ball (default: True)
        device (str): Computation device ('cpu' or 'cuda')

    Returns:
        torch.Tensor: Adversarial image tensor of shape [1, 3, H, W] clamped to [0,1]
        None: If no high-confidence detections are found

    Raises:
        RuntimeError: If model output format is unexpected
    """

    # 1. Model and Image Preparation ===========================================
    # Move model to target device and ensure training mode for gradient computation
    model = model.to(device)
    model.model.train()  # Explicitly set to training mode to enable gradients

    # Preprocess image (normalization, resizing) and enable gradient tracking
    # preprocess_image() should return tensor of shape [1, 3, H, W] in [0,1] range
    original_image = preprocess_image(image_path).to(device)
    image = original_image.clone().requires_grad_(True)

    # Initialize adversarial image
    adv_image = image.clone().detach()

    # Optional: Random start within ε-ball
    if random_start:
        random_noise = torch.empty_like(adv_image).uniform_(-epsilon, epsilon)
        adv_image = adv_image + random_noise
        adv_image = torch.clamp(adv_image, 0, 1).detach()

    # 2. PGD Iteration Loop ===================================================
    for _ in range(num_steps):
        # Enable gradient tracking for current iteration
        adv_image.requires_grad_(True)

        # 3. Raw Model Output Extraction ======================================
        # Run model in training mode to get raw predictions
        with torch.enable_grad():  # Ensure gradients are enabled
            raw_outputs = model.model(adv_image)

        # Verify output dimensions
        if raw_outputs[0].shape[-1] < 5:
            raise RuntimeError(
                f"Unexpected output format. Expected at least 5 channels, got {raw_outputs[0].shape[-1]}")

        # 4. Prediction Processing ============================================
        raw_preds = raw_outputs[0]  # First detection head output
        num_classes = raw_preds.shape[-1] - 5  # Calculate number of classes

        # Extract components from raw predictions
        # Bounding box coordinates (xywh format)
        box_coords = raw_preds[..., :4]
        obj_scores = raw_preds[..., 4:5]  # Objectness scores (anchor quality)
        # Class prediction logits (before softmax)
        cls_logits = raw_preds[..., 5:]

        # 5. Confidence Calculation ===========================================
        # Combined confidence = objectness * max class probability
        # Convert to probability [0,1]
        obj_probs = obj_scores.sigmoid()
        cls_probs = cls_logits.softmax(
            dim=-1).max(dim=-1, keepdim=True)[0]  # Max class prob
        conf_scores = obj_probs * cls_probs             # Final detection confidence

        # 6. High-Confidence Detection Filtering ==============================
        # Create boolean mask for detections above confidence threshold
        mask = conf_scores.squeeze(-1) > conf_threshold

        if not mask.any():
            print(
                f"No detections found with confidence > {conf_threshold} in iteration")
            return None

        # 7. Loss Calculation =================================================
        # Get original predicted classes (detached from computation graph)
        target_classes = cls_logits.argmax(dim=-1)[mask].detach()

        # Extract relevant logits for high-confidence detections
        current_logits = cls_logits[mask]

        # Cross-entropy loss between current predictions and original classes
        classification_loss = torch.nn.functional.cross_entropy(
            current_logits,
            target_classes
        )

        # Optional: Add bounding box regression loss
        # target_boxes = box_coords[mask].detach()
        # box_loss = torch.nn.functional.smooth_l1_loss(box_coords[mask], target_boxes)
        # total_loss = classification_loss + box_loss

        total_loss = classification_loss

        # 8. Gradient Computation =============================================
        # Clear previous gradients and backpropagate
        model.zero_grad()
        total_loss.backward()

        # Extract gradient from adversarial image
        image_grad = adv_image.grad.data

        # 9. Update Adversarial Image =========================================
        # PGD update: x_adv = x_adv + α * sign(∇x J(x, y))
        adv_image = adv_image + step_size * image_grad.sign()

        # 10. Project to ε-ball: Clip perturbation to [-ε, ε] around original image
        adv_image = torch.clamp(
            adv_image, original_image - epsilon, original_image + epsilon)

        # 11. Ensure valid pixel range [0,1]
        adv_image = torch.clamp(adv_image, 0, 1).detach()

    # 12. Return Final Adversarial Image ======================================
    return adv_image


def square_attack_detector(
    image_path: str,
    model_path: str = 'yolov8n.pt',
    epsilon: float = 0.05,
    patch_size: int = 16,
    num_iterations: int = 500,
    conf_threshold: float = 0.5,
    device: str = 'cpu'
) -> Optional[torch.Tensor]:
    # 1. Model and Image Preparation
    model = YOLO(model_path).to(device)
    image = preprocess_image(image_path).to(device)  # [1, 3, H, W], [0,1]
    adv_image = image.clone().detach()
    H, W = image.shape[2], image.shape[3]

    # 2. Loss based on pushing confidences below threshold
    def compute_loss(img: torch.Tensor) -> float:
        raw_outputs = model.model(img)
        raw_preds = raw_outputs[0]
        obj_scores = raw_preds[..., 4:5].sigmoid()
        cls_logits = raw_preds[..., 5:]
        cls_probs = cls_logits.softmax(dim=-1).max(dim=-1, keepdim=True)[0]
        conf_scores = (obj_scores * cls_probs).squeeze(-1)

        mask = conf_scores > conf_threshold
        if not mask.any():
            # early stop condition: no detections above threshold
            return 0.0

        target_classes = cls_logits.argmax(dim=-1)[mask].detach()
        current_logits = cls_logits[mask]
        loss = torch.nn.functional.cross_entropy(
            current_logits, target_classes, reduction='sum')
        return loss.item()

    # 2b. Check initial detections
    initial_loss = compute_loss(image)
    if initial_loss == 0.0:
        print(f"No detections ≥ {conf_threshold} to attack.")
        return None

    best_adv_image = adv_image.clone()
    best_loss = initial_loss

    # 3. Square Attack with early stopping
    for i in range(num_iterations):
        # random patch
        x = np.random.randint(0, W - patch_size + 1)
        y = np.random.randint(0, H - patch_size + 1)
        trial = adv_image.clone()
        perturb = torch.empty(1, 3, patch_size, patch_size,
                              device=device).uniform_(-epsilon, epsilon)
        trial[:, :, y:y+patch_size, x:x+patch_size] += perturb

        # project & clamp
        trial = torch.clamp(trial, image - epsilon, image + epsilon)
        trial = torch.clamp(trial, 0, 1)

        trial_loss = compute_loss(trial)

        # EARLY STOP: if we've driven all confs below threshold
        if trial_loss == 0.0:
            print(
                f"Early stopping at iteration {i}: all detections < {conf_threshold}")
            break

        if trial_loss > best_loss:
            best_loss = trial_loss
            adv_image = trial.clone()
            best_adv_image = trial.clone()

        if i % 50 == 0:
            print(
                f"Iteration {i}/{num_iterations} — best loss: {best_loss:.4f}")

    return best_adv_image.detach()


def compute_distance(x1, x2, norm="l2"):
    """
    Compute distance between two images using PyTorch tensors.
    """
    if norm == "l2":
        return torch.norm(x1.flatten() - x2.flatten()).item()
    elif norm == "linf":
        return torch.max(torch.abs(x1 - x2)).item()
    else:
        raise ValueError("Unsupported norm type. Use 'l2' or 'linf'.")


def get_top_class(output):
    """
    Extract the top class from YOLO output. Assumes output is a list of detections.
    """
    if len(output[0].boxes) == 0:
        return -1  # No detections
    conf, cls = output[0].boxes.conf.max(
    ), output[0].boxes.cls[output[0].boxes.conf.argmax()]
    return cls.item()


def get_confidence(output, label):
    """
    Get the confidence of the specified label from YOLO output.
    """
    if len(output[0].boxes) == 0:
        return 0.0
    conf = output[0].boxes.conf[output[0].boxes.cls == label].max()
    return conf.item() if conf.numel() > 0 else 0.0


# Define the SuppressPrints class


class SuppressPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = io.StringIO()

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._original_stdout

# Helper function to query the model with suppressed prints


def query_model(model, image):
    with SuppressPrints():
        output = model(image)
    return output


def init_adv_hopskip(image, model, max_attempts=20):
    """
    Initialize an adversarial example with large perturbation for YOLO.
    """
    INPUT_MIN = 0.0
    INPUT_MAX = 1.0

    original_output = query_model(model, image)
    original_label = get_top_class(original_output)
    original_conf = get_confidence(original_output, original_label)
    print(f"Original label: {original_label}, confidence: {original_conf:.4f}")

    noise_factor = 0.2
    max_noise_factor = 1.0
    noise_step = 0.1

    for attempt in range(max_attempts):
        noise = torch.rand_like(image) * 2 - 1  # Random noise in [-1, 1]
        x_adv = image + noise_factor * noise
        x_adv = torch.clamp(x_adv, INPUT_MIN, INPUT_MAX)

        adv_output = query_model(model, x_adv)
        adv_label = get_top_class(adv_output)

        if adv_label != original_label:
            print(f"Found initial adversarial example (attempt {attempt+1}):")
            print(
                f"  Adversarial label: {adv_label}, confidence: {get_confidence(adv_output, adv_label):.4f}")
            print(
                f"  Initial distance: {compute_distance(image, x_adv, 'l2'):.4f}")
            return x_adv

        if (attempt + 1) % 5 == 0 and noise_factor < max_noise_factor:
            noise_factor += noise_step
            print(f"Increasing noise factor to {noise_factor}")

    print("Warning: Failed to generate initial adversarial example")
    return image


def estimate_gradient(original_image, adv_image, model, delta, batch_size, norm="l2"):
    """
    Estimate gradient direction using finite differences for YOLO.
    """
    delta = max(delta, 0.01)
    original_label = get_top_class(query_model(model, original_image))
    adv_label = get_top_class(query_model(model, adv_image))

    if adv_label == original_label:
        print("Warning: Current example is not adversarial in estimate_gradient!")
        random_dir = torch.randn_like(original_image)
        if norm == "l2":
            random_dir /= torch.norm(random_dir.flatten()) + 1e-10
        else:
            random_dir = torch.sign(random_dir)
        return random_dir

    noise_shape = (batch_size,) + original_image.shape[1:]
    if norm == "l2":
        noise = torch.randn(noise_shape).to(original_image.device)
        noise = noise / \
            torch.norm(noise.view(batch_size, -1),
                       dim=1).view(batch_size, 1, 1, 1)
    else:
        noise = torch.sign(torch.randn(noise_shape).to(original_image.device))

    perturbed_samples = torch.clamp(adv_image + delta * noise, 0, 1)
    perturbed_outputs = [query_model(model, sample.unsqueeze(0))
                         for sample in perturbed_samples]
    f_val = torch.tensor([1.0 if get_top_class(
        output) != original_label else -1.0 for output in perturbed_outputs])

    if torch.all(f_val == -1.0):
        return -torch.mean(noise, dim=0)
    elif torch.all(f_val == 1.0):
        return torch.mean(noise, dim=0)

    f_val -= torch.mean(f_val)
    grad_estimate = torch.mean(f_val.view(batch_size, 1, 1, 1) * noise, dim=0)

    grad_norm = torch.norm(grad_estimate.flatten())
    if grad_norm > 1e-10:
        if norm == "l2":
            grad_estimate /= grad_norm
        else:
            grad_estimate = torch.sign(grad_estimate)
    else:
        grad_estimate = torch.randn_like(grad_estimate)
        if norm == "l2":
            grad_estimate /= torch.norm(grad_estimate.flatten())
        else:
            grad_estimate = torch.sign(grad_estimate)

    return grad_estimate


def geometric_step_search(x_adv, grad, x_orig, model, epsilon, t, norm="l2", max_trials=10):
    """
    Geometric step search to refine the adversarial example.
    """
    original_label = get_top_class(query_model(model, x_orig))
    adv_label = get_top_class(query_model(model, x_adv))

    if adv_label == original_label:
        print("  Warning: Current example is not adversarial in geometric_step_search!")
        return x_adv

    init_step_size = min(0.5, epsilon / (t + 1))
    step_size = init_step_size
    best_candidate = x_adv.clone()
    best_dist = compute_distance(x_adv, x_orig, norm)

    step_multipliers = [0.01, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]

    for multiplier in step_multipliers:
        step_size = init_step_size * multiplier
        x_candidate = x_adv + step_size * grad
        x_candidate = torch.clamp(x_candidate, 0, 1)

        candidate_output = query_model(model, x_candidate)
        candidate_label = get_top_class(candidate_output)

        if candidate_label != original_label:
            candidate_dist = compute_distance(x_candidate, x_orig, norm)
            print(
                f"    Step size {step_size:.4f}: Still adversarial, distance: {candidate_dist:.4f}")
            if candidate_dist < best_dist:
                best_candidate = x_candidate.clone()
                best_dist = candidate_dist
                print(
                    f"    Found better candidate at distance: {best_dist:.4f}")

    if best_dist < compute_distance(x_adv, x_orig, norm):
        print(
            f"  Geometric search improved distance from {compute_distance(x_adv, x_orig, norm):.4f} to {best_dist:.4f}")
        return best_candidate
    else:
        print("  No improvement from geometric search")
        return x_adv


def binary_search_projection(x_adv, x_orig, epsilon, model, norm="l2", max_trials=20):
    """
    Binary search to project the adversarial example closer to the original.
    """
    original_label = get_top_class(query_model(model, x_orig))
    adv_label = get_top_class(query_model(model, x_adv))

    if adv_label == original_label:
        print("  Warning: Current example is not adversarial in binary_search_projection!")
        return x_adv

    print("  Starting binary search projection")

    low = 0.0
    high = 1.0
    threshold = 0.001

    best_adv = x_adv.clone()
    best_dist = compute_distance(x_adv, x_orig, norm)

    for i in range(max_trials):
        mid = (low + high) / 2.0
        x_mid = (1 - mid) * x_orig + mid * x_adv
        x_mid = torch.clamp(x_mid, 0, 1)

        mid_output = query_model(model, x_mid)
        mid_label = get_top_class(mid_output)

        if mid_label != original_label:
            high = mid
            mid_dist = compute_distance(x_mid, x_orig, norm)
            if mid_dist < best_dist:
                best_adv = x_mid.clone()
                best_dist = mid_dist
                print(
                    f"    Trial {i+1}: Better adversarial example found with α={mid:.4f}, distance: {best_dist:.4f}")
        else:
            low = mid

        if high - low < threshold:
            print(f"    Binary search converged after {i+1} iterations")
            break

    if best_dist < compute_distance(x_adv, x_orig, norm):
        print(
            f"  Binary search improved distance from {compute_distance(x_adv, x_orig, norm):.4f} to {best_dist:.4f}")
        return best_adv
    else:
        print("  No improvement from binary search")
        return x_adv


def hop_skip_jump_attack(model, x_orig, epsilon=None, delta=0.1, batch_size=100, norm="l2", max_queries=5000, max_iters=40):
    """
    HopSkipJump attack implementation for PyTorch YOLO model with suppressed model prints.
    """
    query_count = 0
    original_output = query_model(model, x_orig)
    original_label = get_top_class(original_output)
    query_count += 1

    print(
        f"Starting attack (original label: {original_label}, confidence: {get_confidence(original_output, original_label):.4f})")

    x_adv = init_adv_hopskip(x_orig, model)
    query_count += 20  # Approximate queries for initialization

    adv_output = query_model(model, x_adv)
    adv_label = get_top_class(adv_output)
    query_count += 1

    if adv_label == original_label:
        print("Failed to initialize adversarial example.")
        return x_orig

    initial_dist = compute_distance(x_adv, x_orig, norm)
    print(f"Initial distance: {initial_dist:.4f}")

    x_best = x_adv.clone()
    best_dist = initial_dist

    no_improvement_counter = 0
    max_no_improvement = 5

    if epsilon is None:
        epsilon = initial_dist

    for t in range(max_iters):
        if query_count >= max_queries:
            print(f"Query limit reached: {query_count}/{max_queries}")
            break

        dist = compute_distance(x_adv, x_orig, norm)
        print(
            f"\nIteration {t+1}, current distance: {dist:.4f}, queries: {query_count}/{max_queries}")

        current_output = query_model(model, x_adv)
        current_label = get_top_class(current_output)
        query_count += 1

        if current_label == original_label:
            print(
                "Warning: Current example is no longer adversarial! Reverting to best known.")
            x_adv = x_best.clone()
            continue

        current_delta = min(delta, dist / 10)

        if no_improvement_counter >= 3:
            current_delta *= 2.0
            print(
                f"No recent improvement, increasing delta to {current_delta:.4f}")

        print(
            f"Estimating gradient with delta={current_delta:.4f}, batch_size={batch_size}")
        grad = estimate_gradient(x_orig, x_adv, model,
                                 current_delta, batch_size, norm)
        query_count += batch_size

        x_candidate = geometric_step_search(
            x_adv, grad, x_orig, model, epsilon, t, norm)
        query_count += 10  # Approximate

        x_refined = binary_search_projection(
            x_candidate, x_orig, epsilon, model, norm)
        query_count += 10  # Approximate

        refined_dist = compute_distance(x_refined, x_orig, norm)

        if refined_dist < dist:
            print(
                f"Iteration {t+1} improved distance: {dist:.4f} -> {refined_dist:.4f}")
            x_adv = x_refined.clone()
            if refined_dist < best_dist:
                x_best = x_refined.clone()
                best_dist = refined_dist
                print(f"New best distance: {best_dist:.4f}")
                no_improvement_counter = 0
            else:
                no_improvement_counter += 1
        else:
            print(f"No distance improvement in iteration {t+1}")
            no_improvement_counter += 1

            if no_improvement_counter >= max_no_improvement:
                print("No improvement for multiple iterations, trying random restart...")
                noise_scale = best_dist / 10
                noise = torch.randn_like(x_best) * noise_scale
                if norm == "l2":
                    noise = noise / torch.norm(noise.flatten()) * noise_scale
                else:
                    noise = torch.sign(noise) * noise_scale

                x_perturbed = torch.clamp(x_best + noise, 0, 1)
                perturb_output = query_model(model, x_perturbed)
                perturb_label = get_top_class(perturb_output)

                if perturb_label != original_label:
                    print("Random restart successful")
                    x_adv = x_perturbed.clone()
                    no_improvement_counter = 0
                else:
                    x_perturbed = torch.clamp(x_best - noise, 0, 1)
                    perturb_output = query_model(model, x_perturbed)
                    perturb_label = get_top_class(perturb_output)
                    if perturb_label != original_label:
                        print("Random restart in opposite direction successful")
                        x_adv = x_perturbed.clone()
                        no_improvement_counter = 0

    final_output = query_model(model, x_best)
    final_label = get_top_class(final_output)
    final_dist = compute_distance(x_best, x_orig, norm)

    print(f"\nAttack completed:")
    print(
        f"  Original label: {original_label}, confidence: {get_confidence(original_output, original_label):.4f}")
    print(
        f"  Final label: {final_label}, confidence: {get_confidence(final_output, final_label):.4f}")
    print(f"  Distance: {final_dist:.4f}")
    print(f"  Queries: {query_count}")

    return x_best


MODULE_DIR = os.path.dirname(__file__)
UAP_PATH = os.path.join(MODULE_DIR, "universal_delta_1_epsilon.pth")
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'
delta_1 = torch.load(UAP_PATH,
                     map_location=torch.device(device))


def uap(image_path, device=device):
    test_image = preprocess_image(image_path)
    test_image = test_image.to(device)
    adv = test_image+delta_1
    return adv.detach()
