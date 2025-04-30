import os
import pycolmap
import shutil
import open3d as o3d
import numpy as np

def reconstruct_room(image_dir):
    """
    Perform 3D reconstruction of a room using COLMAP.

    Args:
        image_dir (str): Path to directory containing input images
    """
    # Set up paths
    database_path = "database.db"
    output_path = "sparse"

    # Extract features to create COLMAP Database
    print("Extracting Features")
    if not os.path.exists(database_path):
        # Extract features
        pycolmap.extract_features(database_path, image_dir)
    else:
        print("Database file already exists. Skipping feature extraction.")

    print("Matching Features")
    # Perform feature matching
    pycolmap.match_exhaustive(database_path)

    # Create a sparse reconstruction
    print("Performing Incremental Mapping")
    reconstruction = pycolmap.Reconstruction()
    pycolmap.incremental_mapping(database_path, image_dir, output_path)

    # Load the reconstructed sparse model
    sparse_model = pycolmap.Reconstruction("sparse/0")
    print(sparse_model.summary())

    # Convert to point cloud for visualization
    points = []
    colors = []
    for point in sparse_model.points3D.values():
        points.append(point.xyz)
        colors.append(point.color)

    points = np.array(points)
    colors = np.array(colors)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)  # Normalize colors to range [0, 1]

    # Visualize the point cloud
    print("Visualizing the reconstructed point cloud...")
    o3d.visualization.draw_geometries([pcd], window_name="Reconstructed Room")

    return sparse_model

def main():
    # Specify the directory containing your room images
    image_dir = "room_images"  # Change this to your image directory

    if not os.path.exists(image_dir):
        print(f"Error: Directory '{image_dir}' does not exist!")
        print("Please create a directory named 'room_images' and place your images there.")
        return

    # Check if there are any images in the directory
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        print(f"Error: No images found in '{image_dir}'!")
        print("Please add some images to the directory.")
        return

    print(f"Found {len(image_files)} images in the directory.")
    print("Starting reconstruction...")

    # Perform reconstruction
    sparse_model = reconstruct_room(image_dir)

    print("\nReconstruction completed!")
    print(f"Number of reconstructed points: {len(sparse_model.points3D)}")
    print(f"Number of images used: {len(sparse_model.images)}")

if __name__ == "__main__":
    main()