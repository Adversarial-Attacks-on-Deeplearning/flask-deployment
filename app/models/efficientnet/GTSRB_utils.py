import numpy as np
import tensorflow as tf
from PIL import Image



# GTSRB Class ID to Sign Name Mapping (43 classes)
GTSRB_CLASSES = {
    0: "Speed limit 20",
    1: "Speed limit 30",
    2: "Speed limit 50",
    3: "Speed limit 60",
    4: "Speed limit 70",
    5: "Speed limit 80",
    6: "End of speed limit 80",
    7: "Speed limit 100",
    8: "Speed limit 120",
    9: "No passing",
    10: "No passing for vehicles over 3.5 tons",
    11: "Right-of-way at next intersection",
    12: "Priority road",
    13: "Yield",
    14: "Stop",
    15: "No vehicles",
    16: "Vehicles over 3.5 tons prohibited",
    17: "No entry",
    18: "General caution",
    19: "Dangerous curve left",
    20: "Dangerous curve right",
    21: "Double curve",
    22: "Bumpy road",
    23: "Slippery road",
    24: "Road narrows on the right",
    25: "Road work",
    26: "Traffic signals",
    27: "Pedestrians",
    28: "Children crossing",
    29: "Bicycles crossing",
    30: "Beware of ice/snow",
    31: "Wild animals crossing",
    32: "End of all speed and passing limits",
    33: "Turn right ahead",
    34: "Turn left ahead",
    35: "Ahead only",
    36: "Go straight or right",
    37: "Go straight or left",
    38: "Keep right",
    39: "Keep left",
    40: "Roundabout mandatory",
    41: "End of no passing",
    42: "End of no passing for vehicles over 3.5 tons"
}




def load_ppm_image(image_path, target_size=(240, 240)):
    # Open the .ppm image using Pillow
    with Image.open(image_path) as img:
        # Convert image to RGB (if not already)
        img = img.convert("RGB")
        # Resize the image to the target size (now 240x240)
        img = img.resize(target_size)
        # Convert image to numpy array
        img_array = np.array(img)
        # Convert numpy array to a TensorFlow tensor and cast to float32
        img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
        img_tensor = tf.expand_dims(img_tensor, axis=0)
    return img_tensor



def predict_traffic_sign(img_input, model, class_mapping):
    """
    Accepts a tensor (rank 3 or 4), raw image bytes, or a file path.
    """
    # Get predictions
    prop = model.predict(img_input)
    print("Prediction shape:", prop.shape)
    class_id = np.argmax(prop)
    print("Predicted class:", class_mapping[class_id])
    return class_id



def create_subset_loader(train_generator, subset_fraction=0.2, batch_size=32):
    """
    Create a subset DataLoader from an ImageDataGenerator 
    
    Parameters:
        train_generator (tf.keras.preprocessing.image.DirectoryIterator): Original training data loader.
        subset_fraction (float): Fraction of training data to use (e.g., 0.2 for 20%).
        batch_size (int): Batch size for training.
    
    Returns:
        tf.data.Dataset: TensorFlow dataset for training UAP.
    """
    total_samples = len(train_generator.filenames)
    subset_size = int(subset_fraction * total_samples)
    
    images, labels = [], []
    count = 0
    
    for img_batch, label_batch in train_generator:
        for img, label in zip(img_batch, label_batch):
            images.append(img)
            labels.append(label)
            count += 1
            if count >= subset_size:
                break
        if count >= subset_size:
            break
        print(f'Loading subset: {count}/{subset_size}', end='\r')
    
    images = np.array(images)
    labels = np.array(labels)
    
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.batch(batch_size).shuffle(buffer_size=subset_size)

    print(f'Subset DataLoader: {subset_size} samples')
    return dataset