import os
from PIL import Image
from torchvision.transforms import Compose, RandomRotation, RandomHorizontalFlip, ColorJitter
from multiprocessing import Pool, cpu_count
from shutil import copy2
from tqdm import tqdm

# Define the augmentation transformations
augmentations = Compose([
    RandomRotation(30),
    RandomHorizontalFlip(),
    ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
])

# Worker function for multiprocessing
def augment_worker(args):
    image_subset, augmented_dir = args
    current_class = image_subset[0][1]  # Get the class of the first image in the subset
    for img_path, label_name in image_subset:
        # Load the image
        image = Image.open(img_path).convert('RGB')
        
        # Copy the original image to the Augmented_Images directory
        original_img_path = os.path.join(augmented_dir, label_name, os.path.basename(img_path))
        if not os.path.exists(os.path.dirname(original_img_path)):
            os.makedirs(os.path.dirname(original_img_path))
        copy2(img_path, original_img_path)  # <-- Copy the original image
        
        # Apply augmentations
        augmented_image = augmentations(image)
        
        # Save the augmented image with a prefix or suffix to differentiate from the original
        augmented_img_path = os.path.join(augmented_dir, label_name, "aug_" + os.path.basename(img_path))
        augmented_image.save(augmented_img_path)

    return current_class

def augment_dataset(root_dir):
    classes = os.listdir(root_dir)
    image_paths = []

    for cls in classes:
        for image_name in os.listdir(os.path.join(root_dir, cls)):
            image_paths.append((os.path.join(root_dir, cls, image_name), cls))

    # Directory to save augmented images
    global augmented_dir
    augmented_dir = 'Augmented_Images'
    if not os.path.exists(augmented_dir):
        os.makedirs(augmented_dir)

    # Split the images into subsets for multiprocessing
    num_processes = cpu_count()
    chunk_size = len(image_paths) // num_processes
    image_subsets = [image_paths[i:i + chunk_size] for i in range(0, len(image_paths), chunk_size)]

    # Use a Pool of workers to parallelize the augmentation
    args_for_workers = [(subset, augmented_dir) for subset in image_subsets]
    with Pool(num_processes) as pool:
        with tqdm(total=num_processes, desc="Processing subsets") as pbar:
            for current_class in pool.imap_unordered(augment_worker, args_for_workers):
                pbar.set_description(f"Processed class: {current_class}")  # Update the description
                pbar.update(1)


if __name__ == "__main__":
    augment_dataset('Images')
