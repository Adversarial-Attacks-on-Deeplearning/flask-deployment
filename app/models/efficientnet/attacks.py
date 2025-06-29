import pandas as pd
import numpy as np
import tensorflow as tf


def fgsm_attack_single_image(model, image, label, epsilon, normalized=True):
    """
    Perform FGSM attack on a single image without normalization.

    Args:
        model (tf.keras.Model): Trained model.
        image (tf.Tensor): Input image of shape (H, W, C), values in [0, 255].
        label (tf.Tensor): True label (integer).
        epsilon (float): Perturbation magnitude.
        normalized (bool): Normalized image.

    Returns:
        tf.Tensor: Adversarial example in [0, 255].
    """
    # Adjust epsilon for non normalized images
    if not normalized:
        epsilon = epsilon * 255.0

    # Ensure the image has a batch dimension
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    if len(image.shape) == 3:
        image = tf.expand_dims(image, axis=0)

    # Add batch dimension
    label = tf.expand_dims(label, axis=0)

    with tf.GradientTape() as tape:
        tape.watch(image)
        # Forward pass
        prediction = model(image, training=False)
        # Calculate loss
        loss = tf.keras.losses.sparse_categorical_crossentropy(
            label, prediction)

    # Calculate gradient of the loss with respect to the input image
    gradient = tape.gradient(loss, image)

    # Get the sign of the gradient
    gradient_sign = tf.sign(gradient)

    # Generate adversarial example
    adversarial_image = image + epsilon * gradient_sign

    # Clip values to stay in valid range [0, 255]
    if normalized:
        adversarial_image = tf.clip_by_value(adversarial_image, 0, 1)
    else:
        adversarial_image = tf.clip_by_value(adversarial_image, 0, 255)

    return tf.squeeze(adversarial_image)


def mi_fgsm(model, x, y, epsilon=8.0, T=10, mu=1.0):
    """
    MI-FGSM attack for models that take unnormalized inputs in [0, 255].

    Args:
        model: A tf.keras.Model that outputs logits.
        x: Input image tensor (float32) in [0, 255], shape (batch_size, H, W, C).
        y: Ground-truth label (int scalar, int vector, or one-hot),
           with shape (), (batch_size,), (num_classes,), or (batch_size, num_classes).
        epsilon: Perturbation bound (L∞ norm), e.g. 8.0 for images in [0, 255].
        T: Number of iterations.
        mu: Momentum decay factor.

    Returns:
        x_star: Adversarial image tensor in [0, 255], same shape as x.
    """
    # Scale epsilon to pixel range
    epsilon = epsilon * 255.0

    # Cast inputs
    x = tf.cast(x, tf.float32)
    x_star = tf.identity(x)
    batch_size = tf.shape(x)[0]

    # Step size per iteration
    alpha = epsilon / float(T)

    # Initialize momentum buffer
    g = tf.zeros_like(x)

    # Prepare labels
    num_classes = model.output_shape[-1]
    y = tf.cast(y, tf.int32)
    y_shape = y.shape
    # Case 1: scalar label
    if y_shape.ndims == 0:
        y = tf.expand_dims(y, 0)
        y = tf.one_hot(y, depth=num_classes)
    # Case 2: vector
    elif y_shape.ndims == 1:
        # If length equals num_classes, assume one-hot vector
        if y_shape[0] == num_classes:
            y = tf.expand_dims(tf.cast(y, tf.float32), 0)
        else:
            # integer class indices
            y = tf.one_hot(y, depth=num_classes)
    # Case 3: already (batch_size, num_classes)
    elif y_shape.ndims == 2 and y_shape[1] == num_classes:
        y = tf.cast(y, tf.float32)
    else:
        raise ValueError(f"Unsupported label shape: {y_shape}")

    loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

    # Iterative attack
    for _ in range(T):
        with tf.GradientTape() as tape:
            tape.watch(x_star)
            logits = model(x_star)
            loss = loss_object(y, logits)

        # Compute gradient
        grad = tape.gradient(loss, x_star)

        # Normalize by L1 norm
        grad_norm = tf.reduce_sum(
            tf.abs(grad),
            axis=list(range(1, len(grad.shape))),
            keepdims=True
        )
        grad_norm = tf.maximum(grad_norm, 1e-8)
        normalized_grad = grad / grad_norm

        # Momentum update
        g = mu * g + normalized_grad

        # Perturbation step
        x_star = x_star + alpha * tf.sign(g)

        # Project back into epsilon-ball
        x_star = tf.clip_by_value(x_star, x - epsilon, x + epsilon)

        # Clip to valid pixel range
        x_star = tf.clip_by_value(x_star, 0.0, 255.0)

    return x_star


def pgd_attack_single_image(model, image, label, epsilon, alpha, iterations, normalized=True):
    """
    Perform PGD attack on a single image without normalization.

    Args:
        model (tf.keras.Model): Trained model.
        image (tf.Tensor): Input image of shape (H, W, C), values in [0, 255].
        label (tf.Tensor): True label (integer).
        epsilon (float): Maximum perturbation magnitude.
        alpha (float): Step size for each iteration.
        iterations (int): Number of PGD iterations.
        normalized (bool): For normalized images, set to True.

    Returns:
        tf.Tensor: Adversarial example in [0, 255].
    """
    # Adjust epsilon for non normalized images
    if not normalized:
        epsilon = epsilon * 255.0
        alpha = alpha*255.0
    # Ensure the image has a batch dimension
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    if len(image.shape) == 3:
        image = tf.expand_dims(image, axis=0)
    label = tf.expand_dims(label, axis=0)  # Add batch dimension

    adversarial_image = image
    for _ in range(iterations):
        with tf.GradientTape() as tape:
            tape.watch(adversarial_image)
            # Forward pass
            prediction = model(adversarial_image, training=False)
            # Calculate loss
            loss = tf.keras.losses.sparse_categorical_crossentropy(
                label, prediction)

        # Calculate gradient of the loss w.r.t. the adversarial image
        gradient = tape.gradient(loss, adversarial_image)

        # Get the sign of the gradient and update the adversarial image
        adversarial_image = adversarial_image + alpha * tf.sign(gradient)

        # Project the adversarial image into the epsilon-ball and clip to [0, 255]
        adversarial_image = tf.clip_by_value(
            adversarial_image, image - epsilon, image + epsilon)
        if normalized:
            adversarial_image = tf.clip_by_value(adversarial_image, 0, 1)
        else:
            adversarial_image = tf.clip_by_value(adversarial_image, 0, 255)

    return tf.squeeze(adversarial_image)


def compute_output_grads(model, image):
    """
    Compute the element-wise grads matrix of the model output with respect to the input image.
    Args:
        input image with dimensions: [H, W, C]
        tf model
    """
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(image)
        output = model(tf.expand_dims(image, axis=0))  # Forward pass
    element_wise_grad = tape.jacobian(
        output, image, experimental_use_pfor=True)
    return tf.squeeze(element_wise_grad, axis=0)  # Remove batch dimension


def saliency_map(jacobian, target_label):
    """
    Compute the saliency map to determine which pixels to perturb.
    Args:
        the element-wise gradients matrix
        the targetted label
    """
    # Get gradient for target class [H, W, C]
    J_t = jacobian[target_label]

    # Sum gradients for all other classes [H, W, C]
    sum_J_other = tf.reduce_sum(jacobian, axis=0) - J_t

    # Compute conditions
    cond1 = J_t < 0                          # Condition 1: J_t < 0
    cond2 = sum_J_other > 0                  # Condition 2: Sum of other grads > 0
    zero_mask = tf.logical_or(cond1, cond2)  # Where to set saliency to 0

    # Compute saliency values
    saliency_values = J_t * tf.abs(sum_J_other)

    # Apply conditions
    saliency_map = tf.where(zero_mask,
                            tf.zeros_like(saliency_values),
                            saliency_values)

    return saliency_map


def jsma_attack(model, image, target_label, gamma, theta, num_pixels):
    """
    Performs a JSMA attack on multi-class classification model.

    Args:
        model: The TensorFlow/Keras model.
        image: The input image (tensor of shape [H, W, C]) (un-normalized).
        target_label: The desired class output (class id).
        theta: The perturbation step size.
        gamma: Maximum total distortion (percentage of modified pixels).
        num_pixels: Number of pixels to perturb at each iteration.

    Returns:
        The adversarial image.
    """
    adversarial = tf.identity(image)
    itr = 0
    distortion = 0.0
    class_id = 0
    # print initial prediction
    pred = model(tf.expand_dims(image, axis=0))
    class_id = np.argmax(pred)
    print(f"prediction before attack: {class_id}")

    # stop if maximum distortion reached or attack succeeded or max. num. of iterartions reached
    while distortion < gamma and class_id != target_label and itr < 15:
        # step 1: Compute the output grads with respect to input image
        grads = compute_output_grads(model, adversarial)

        # step 2: Compute saliency map to perturb most important pixels
        saliency = saliency_map(grads, target_label)

        # step 3: Get maximum pixels value to perturb.
        # Flatten the saliency map to 1D and get top N values/indices
        flat_saliency = tf.reshape(saliency, [-1])  # Shape [H*W*C]
        top_values, flat_indices = tf.math.top_k(flat_saliency, k=num_pixels)
        # Convert flat indices to 3D coordinates (h, w, c)
        h = flat_indices // (saliency.shape[1] * saliency.shape[2])
        remainder = flat_indices % (saliency.shape[1] * saliency.shape[2])
        w = remainder // saliency.shape[2]
        c = remainder % saliency.shape[2]

        # Stack into [N, 3] tensor
        top_indices = tf.stack([h, w, c], axis=1)

        # Create updates tensor (shape [N])
        updates = tf.ones([tf.shape(top_indices)[0]]) * theta

        # Apply scatter add operation
        adversarial = tf.tensor_scatter_nd_add(
            adversarial,
            top_indices,  # [N, 3] indices
            updates       # [N] values to add
        )
        # Clip to maintain constraints
        adversarial = tf.clip_by_value(adversarial, 0, 255)

        # Check if misclassification occurs
        pred = model(tf.expand_dims(adversarial, axis=0))
        class_id = np.argmax(pred)
        print(f"current prediction : {class_id}")
        if (class_id == target_label):
            print(f"Attacked successfully, predicted class: {class_id}")

        itr += 1
    if (class_id != target_label):
        print(f"Attack failed :(")
    return adversarial
