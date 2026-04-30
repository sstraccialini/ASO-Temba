import os
import torch
import clip
from PIL import Image
import numpy as np
from natsort import natsorted  # to naturally sort frame filenames

# Check if CUDA is available and set device accordingly
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the CLIP model
model, preprocess = clip.load("ViT-L/14", device=device)


def extract_clip_features_from_images(image_folder, window_size=16, save_dir=None):
    """
    Extract CLIP features from pre-extracted video frames (stored as .jpg images) and 
    perform temporal averaging across 'window_size' frames.

    Args:
    - image_folder: str, path to the folder containing the .jpg frames of a video
    - window_size: int, number of frames to average over
    - save_dir: str, directory where extracted features are saved

    Returns:
    - aggregated_features: numpy array, temporal window averaged features
    """
    # Get all the .jpg files in the folder and sort them by natural order
    image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]
    # Sort files naturally by frame number
    image_files = natsorted(image_files)

    # Check the total number of frames
    num_frames = len(image_files)
    # print(f"Total number of frames in {image_folder}: {num_frames}")

    # Store frame features for aggregation
    frame_features = []
    aggregated_features = []

    # Process each frame
    for frame_file in image_files:
        # Load the image and preprocess it for CLIP
        image_path = os.path.join(image_folder, frame_file)
        pil_image = Image.open(image_path).convert('RGB')  # Ensure it's RGB
        image = preprocess(pil_image).unsqueeze(0).to(device)

        # Extract CLIP features with no gradient tracking
        with torch.no_grad():
            features = model.encode_image(image)

        # Save the features for temporal aggregation
        frame_features.append(features.squeeze().cpu().numpy())

        # Aggregate every 'window_size' frames
        if len(frame_features) == window_size:
            avg_features = np.mean(frame_features, axis=0)
            aggregated_features.append(avg_features)
            frame_features = []  # Reset for the next window

    # If there are leftover frames, average them as well
    if frame_features:
        avg_features = np.mean(frame_features, axis=0)
        aggregated_features.append(avg_features)

    # Convert to numpy array
    aggregated_features = np.array(aggregated_features)

    # Check if the number of extracted features matches the number of frames before averaging
    expected_num_segments = (num_frames // window_size) + \
        (1 if num_frames % window_size != 0 else 0)
    if len(aggregated_features) != expected_num_segments:
        # print(f"Warning: Mismatch in frame count! Expected {expected_num_segments}, but {len(aggregated_features)} feature windows extracted.")
        pass

    # Optionally save the features
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        video_name = os.path.basename(image_folder)
        save_path = os.path.join(save_dir, f"{video_name}.npy")
        np.save(save_path, aggregated_features)
        # print(f"Saved features to {save_path}")

    return aggregated_features


def process_all_folders(image_root_dir, save_dir, window_size=16):
    """
    Process all subfolders in the root directory and extract features from the images inside.

    Args:
    - image_root_dir: str, root directory containing folders of images
    - save_dir: str, directory where extracted features are saved
    - window_size: int, number of frames to average over
    """
    # Loop over all the subfolders in the root directory
    for folder in os.listdir(image_root_dir):
        folder_path = os.path.join(image_root_dir, folder)
        if os.path.isdir(folder_path):
            # Extract features for the current folder
            extract_clip_features_from_images(
                folder_path, window_size=window_size, save_dir=save_dir)


# Set paths for the root directory and where to save features
image_root_dir = '/path/to/input_folder/'
save_dir = '/path/to/output_folder/'

# Process all subfolders and extract features
process_all_folders(image_root_dir, save_dir, window_size=16)