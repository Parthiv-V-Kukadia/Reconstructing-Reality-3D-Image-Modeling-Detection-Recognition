# Reconstructing Reality: 3D Image Modeling, Detection, & Recognition

This repository showcases a computer vision pipeline for reconstructing 3D scenes from 2D images and performing object detection and recognition within those scenes. The project leverages Structure from Motion (SfM) using COLMAP and deep learning-based object detection with YOLOv8, **along with image classification using a ResNet model, and incorporates comprehensive evaluation and integration strategies.** All components are intended for integration with Blender for 3D visualization. 

This project leverages the **COCO (Common Objects in Context) dataset**, a large-scale dataset widely used for object detection and recognition tasks. Specifically, we utilize the **Train2017 split** for training our deep learning models and the **Val2017 split** for validation. Our focus within the COCO dataset is on the **"chair" and "dining table" object categories** (IDs 62 and 67, respectively). The rich annotations provided by COCO, including bounding boxes, are essential for training our YOLOv8 object detection model and ResNet image classification model.

## Key Features

### 3D Surface and Mesh Reconstruction (COLMAP + Open3D)
- Sparse and dense reconstruction from RGB frames using COLMAP
- Point cloud filtering using statistical outlier removal
- Segmentation via DBSCAN clustering
- Surface reconstruction using Poisson mesh generation
- Mesh refinement with Laplacian and Taubin smoothing
- Merging segmented objects into a single scene mesh
- Visualization and export to `.ply` format

### Object Detection & Recognition (YOLOv8 + ResNet)
- Detection of "chair" and "dining table" objects from 2D images using YOLOv8
- Image classification of cropped detections using a ResNet model
- Evaluation of detection (precision, recall, mAP) and classification (accuracy, F1-score) metrics
- Planned integration of detection outputs with COLMAP's camera poses for 3D localization (PnP)

## Project Highlights

* **3D Reconstruction with COLMAP:** Developed and optimized an end-to-end pipeline for SfM-based 3D reconstruction. This involves feature extraction and keypoint matching to reconstruct real-world environments from 2D images. Multi-View Stereo (MVS) refinement is applied to enhance scene accuracy, generating high-fidelity depth maps and dense 3D models.
* **Deep Learning-based Object Detection and 3D Localization:** Engineered real-time object recognition by implementing YOLOv8. Precise 3D localization of detected objects is achieved using Perspective-n-Point (PnP) algorithms. The detection and localization results are intended for integration into Blender for interactive spatial scene understanding.
* **Image Classification with ResNet:** Implemented and trained a ResNet deep learning model for classifying detected chair and dining table objects. This provides an additional layer of recognition and verification within the pipeline.
* **Comprehensive Evaluation:** The code includes the implementation of detailed evaluation metrics for both the object detection (e.g., precision, recall, mAP) and image classification (e.g., accuracy, F1-score) models, allowing for a thorough assessment of their performance.
* **Integration of Detection and Classification:** Strategies have been explored and implemented to integrate the results from the YOLOv8 detection and ResNet classification. This may involve using the ResNet model to verify the class labels predicted by YOLOv8 or to refine the classification of detected objects.

## Building the Pipeline Locally
### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended)
- COLMAP with CUDA acceleration
- PyCOLMAP (built from source)

### Installation Steps

1. Install COLMAP with CUDA acceleration:
   - Follow the official installation guide: [COLMAP Installation Guide](https://colmap.github.io/install.html)
   - Ensure CUDA is properly installed and configured
   - Build COLMAP from source with CUDA support:

2. Install PyCOLMAP (build from source):
   - Follow the official guide: [PyCOLMAP Guide](https://colmap.github.io/pycolmap/index.html)

3. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

4. Install Python dependencies:
```bash
pip install -r requirements.txt
```

5. Install custom submodules:
```bash
pip install -e core/submodules/diff-gaussian-rasterization
pip install -e core/submodules/simple-knn
```

## How to Run Pipeline

1. Clone the repository locally

3. Prepare your input:
   - Place your video file in the `images/video` directory
   - Ensure the video is in MP4 format

4. Run the reconstruction pipeline:
```bash
python reconstructor.py
```
The script will:
- Extract frames from your video
- Run COLMAP conversion
- Train the 3D model
- Perform object detection
- Save results in the `detection_outputs` directory

5. View results:
   - Check the `detection_outputs` directory for processed images
   - The point cloud will be available in `images/output/point_cloud.ply`

## 3D Reconstruction Usage and Pipeline Steps
The `dense_reconstruction.py` script is designed to be perform the following steps:

1. **Sparse Reconstruction:**
   * The sparse reconstruction step takes the color images and performs incremental mapping using COLMAP.
   * Features are extracted from the images to create a COLMAP database.
   * Feature matching is performed to establish correspondences between images.
   * Incremental mapping is performed to estimate camera poses and generate a sparse 3D model.
   * The camera poses are also extracted from the sparse model if needed. (Can be used with Open3D to get specific viewpoints)
2. **Dense Reconstruction:**
   * The dense reconstruction step takes the sparse reconstruction data and the color images to generate a dense point cloud. (Requires CUDA compilation of COLMAP)
   * COLMAP's patch match stereo algorithm is used to estimate depth maps for each image pair.
   * The depth maps are then merged to create a dense point cloud representing the scene which provides a detailed representation of the scene geometry.
3. **Point Cloud Filtering:**
   * The dense point cloud generated from the previous step may contain noise and outliers.
   * Statistical outlier removal is applied to the point cloud to remove points that deviate significantly from their neighbors to reduce noise and improve the quality of the point cloud.
4. **Point Cloud Segmentation:**
   * The filtered point cloud is segmented into different objects using DBSCAN clustering, which is a density-based clustering algorithm that groups together points that are closely packed.
   * By adjusting the clustering parameters (`eps` and `min_points`), the script separates the point cloud into distinct objects, allowing for individual processing and reconstruction of each object.
5. **Surface Reconstruction:**
   * For each segmented object, surface reconstruction is performed using the Poisson surface reconstruction algorithm.
   * Poisson surface reconstruction estimates the surface that best fits the point cloud by solving a Poisson equation.
   * This step generates a triangular mesh representation of each object's surface.
6. **Mesh Refinement:**
   * As the reconstructed meshes usually contain some artifacts or noise, Laplacian and Taubin smoothing techniques are applied to refine the meshes.
   * Laplacian smoothing smooths the mesh by averaging the positions of neighboring vertices.
   * Taubin smoothing is a two-step smoothing process that helps to preserve mesh details while reducing noise.
   * Additionally, degenerate triangles, duplicated vertices, and non-manifold edges are removed to improve mesh quality.
7. **Visualization and Merging:**
   * The reconstructed objects are visualized using Open3D to provide a visual representation of the results.
   * The individual object meshes are then merged into a single mesh using mesh concatenation.
   * If needed, each object mesh can be individually saved as a `.ply` file to export to tools such as blender for further mesh refinement.
   * The final merged mesh represents the complete reconstructed scene.
8. **Saving Results:**
   * The final merged mesh is saved as `dense_reconstruction_mesh.ply` using the PLY file format.
  
## Running Image Detection and Recognition Model (Optional)

The `Detection_Recognition.ipynb` Notebook is designed to be run sequentially in a Python environment (preferably Google Colab due to potential GPU usage).

1.  **Mount Google Drive:** If using Google Colab, ensure you mount your Google Drive to access the COCO dataset.
2.  **Install Dependencies:** Install the necessary Python libraries. The Notebook includes pip commands for installing `ultralytics` (YOLOv8) and imports other libraries like `torchvision`, `pycocotools`, `opencv-python`, `numpy`, `scikit-learn`, `torch`, `torch.nn`, `torch.optim`, `torchvision.models`, `torchvision.transforms`, `torch.utils.data`, `PIL`.
3.  Download the COCO dataset
```bash
# Images
wget http://images.cocodataset.org/zips/train2017.zip
unzip train2017.zip -d images/train2017
wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip -d images/val2017

# Annotations
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip -d annotations
```
4.  **COCO Dataset:**
    * The code assumes the COCO dataset (specifically the **Train2017 and Val2017 splits**) is organized in a specific directory structure within Google Drive (`/content/drive/MyDrive/Colab Notebooks/ME6402 Project/coco/`). You might need to adjust these paths based on your local setup or download the COCO 2017 dataset and place it accordingly.
    * The project utilizes the annotation files (`instances_train2017.json` and `instances_val2017.json`, or filtered subsets thereof) for training and validation.
    * Ensure that the image directories (`train2017` and `val2017`) are also present within the specified COCO directory.
    * The Notebook includes steps to filter the annotations to keep only the "chair" (category ID 62) and "dining table" (category ID 67) classes, which are used for both object detection and image classification.
    * The data loading and preparation steps are designed to handle the image data for both YOLOv8 and ResNet training and evaluation.
    * The `torchvision.datasets.CocoDetection` function handles the parsing of the COCO annotations.
5.  **Filter and Subset Data (Optional):** The Notebook includes code to filter the COCO dataset to include only "chair" and "dining table" annotations and create smaller subsets for training and evaluating both models. You can run or skip these cells as needed.
6.  **Convert to YOLO Labels (Optional):** If you haven't already, run the cells that convert the COCO annotations into the format required by YOLO for object detection training.
7.  **Train YOLOv8 Model:** The core of the object detection lies in training the YOLOv8 model using the prepared COCO data and YOLO-formatted labels. You can modify the training parameters as needed. Save the best weights as best.pt.
8.  **Use best.pt**: To generate object crops and train a ResNet classifier.
9.  **Train ResNet Model:** The Notebook includes sections for training a ResNet model for image classification, including data loaders, model definition, loss function, optimizer, and training loop.
10.  **Evaluate Models:** Run the sections of the code that calculate and display the evaluation metrics for both the trained YOLOv8 and ResNet models. Save final ResNet weights as best_resnet_model.pth
11.  **Integrate Detection and Classification:** Execute the code that demonstrates the integration of the YOLOv8 detection results with the ResNet classification model. This might involve passing the cropped regions of detected objects to the ResNet model for classification.
