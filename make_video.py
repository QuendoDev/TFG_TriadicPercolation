import cv2
import os
import glob
import re


def create_video_from_list(image_list, output_full_path, fps=5):
    """
    Creates an MP4 video from a list of image paths.

    :param image_list: List of string paths to the images.
    :param output_full_path: Full path where the video will be saved (including .mp4).
    :param fps: Frames Per Second (speed of the video).
    """

    if not image_list:
        print(f"Warning: No images found for {os.path.basename(output_full_path)}")
        return

    # 1. Natural Sort
    # We sort the files by the number contained in the filename.
    # This ensures frame 2 comes after frame 1, and frame 10 comes after frame 9.
    image_list.sort(key=lambda f: int(re.findall(r'\d+', os.path.basename(f))[-1]))

    # 2. Read the first image to determine video dimensions
    first_frame = cv2.imread(image_list[0])
    if first_frame is None:
        print(f"Error: Could not read image {image_list[0]}")
        return

    height, width, layers = first_frame.shape

    # 3. Initialize VideoWriter
    # 'mp4v' is the standard codec for MP4 files.
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    video = cv2.VideoWriter(output_full_path, fourcc, fps, (width, height))

    print(f"Processing {len(image_list)} frames for: {os.path.basename(output_full_path)}...")

    # 4. Write frames
    for img_path in image_list:
        frame = cv2.imread(img_path)

        if frame is None:
            continue

        video.write(frame)

    # 5. Release resources
    video.release()
    print(f"Done. Video saved at: {output_full_path}")


def make_video(target_folder, fps=5):
    """
    Creates two MP4 videos from PNG images in the specified folder:
    1. Physical Network (images without 'B' suffix)
    2. Regulatory Map (images with 'B' suffix)

    :param target_folder: Path to the folder containing the PNG images.
    :param fps: Frames Per Second (speed of the video).
    """

    # Folders where frame_viewer.py saves the images
    structural_folder = os.path.join(target_folder, "frames_structural")
    regulatory_folder = os.path.join(target_folder, "frames_regulatory")

    if os.path.exists(target_folder):
        print(f"\nCreating videos for: {target_folder}")

        # Get all PNG files
        files_network = glob.glob(os.path.join(structural_folder, "*.png"))
        files_regulation = glob.glob(os.path.join(regulatory_folder, "*.png"))

        # Create Video 1: Structural Network
        if files_network:
            create_video_from_list(
                files_network,
                os.path.join(structural_folder, "video_structural.mp4"),
                fps=fps
            )

        # Create Video 2: Regulatory Network
        if files_regulation:
            create_video_from_list(
                files_regulation,
                os.path.join(regulatory_folder, "video_regulatory.mp4"),
                fps=fps
            )
    else:
        print(f"Error: The folder '{target_folder}' does not exist.")