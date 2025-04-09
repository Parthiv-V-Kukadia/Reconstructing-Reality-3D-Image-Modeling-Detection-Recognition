# Reconstructing Reality: 3D Image Modeling, Detection, & Recognition

This repository showcases a computer vision pipeline for reconstructing 3D scenes from 2D images and performing object detection and recognition within those scenes. The project leverages Structure from Motion (SfM) using COLMAP and deep learning-based object detection with YOLOv8, **along with image classification using a ResNet model, and incorporates comprehensive evaluation and integration strategies.** All components are intended for integration with Blender for 3D visualization. 

This project leverages the **COCO (Common Objects in Context) dataset**, a large-scale dataset widely used for object detection and recognition tasks. Specifically, we utilize the **Train2017 split** for training our deep learning models and the **Val2017 split** for validation. Our focus within the COCO dataset is on the **"chair" and "dining table" object categories** (IDs 62 and 67, respectively). The rich annotations provided by COCO, including bounding boxes, are essential for training our YOLOv8 object detection model and ResNet image classification model.


## Project Highlights

* **3D Reconstruction with COLMAP:** Developed and optimized an end-to-end pipeline for SfM-based 3D reconstruction. This involves feature extraction and keypoint matching to reconstruct real-world environments from 2D images. Multi-View Stereo (MVS) refinement is applied to enhance scene accuracy, generating high-fidelity depth maps and dense 3D models.
* **Deep Learning-based Object Detection and 3D Localization:** Engineered real-time object recognition by implementing YOLOv8. Precise 3D localization of detected objects is achieved using Perspective-n-Point (PnP) algorithms. The detection and localization results are intended for integration into Blender for interactive spatial scene understanding.
* **Image Classification with ResNet:** Implemented and trained a ResNet deep learning model for classifying detected chair and dining table objects. This provides an additional layer of recognition and verification within the pipeline.
* **Comprehensive Evaluation:** The code includes the implementation of detailed evaluation metrics for both the object detection (e.g., precision, recall, mAP) and image classification (e.g., accuracy, F1-score) models, allowing for a thorough assessment of their performance.
* **Integration of Detection and Classification:** Strategies have been explored and implemented to integrate the results from the YOLOv8 detection and ResNet classification. This may involve using the ResNet model to verify the class labels predicted by YOLOv8 or to refine the classification of detected objects.

## Code Structure

The repository contains the following Python Notebook:

* **`Detection_Recognition.ipynb`:** This Jupyter/Colab Notebook contains the Python code for:
    * Loading and preparing the COCO dataset for training object detection and classification models.
    * Filtering COCO annotations to focus on "chair" and "dining table" classes.
    * Converting COCO annotations to YOLO format for object detection.
    * Implementing and training a ResNet model for image classification of chair and dining table objects. This includes defining the ResNet architecture, setting up data loaders, defining loss functions and optimizers, and running the training loop.
    * Training YOLOv8 models for detecting both "chair" and "dining table" together, and separately for each class.
    * **Implementing evaluation metrics:** This section calculates and reports metrics such as precision, recall, and mean Average Precision (mAP) for the YOLOv8 object detection model, and accuracy and F1-score for the ResNet image classification model.
    * **Integrating detection and classification:** This part of the code demonstrates how the outputs of the YOLOv8 detection (bounding boxes of potential chairs and dining tables) can be used as input for the trained ResNet model to classify the cropped object regions, potentially improving the overall accuracy and robustness of the recognition process.
    * Potentially including steps for Non-Max Suppression (NMS), voxelization, outlier removal (commented out, suggesting future work with Open3D), and model evaluation.

**Note:** The provided code focuses on object detection using YOLOv8, image classification using ResNet, **their evaluation, and initial integration strategies.** The 3D reconstruction part using COLMAP and direct automated integration with Blender, as mentioned in the project description, might involve separate scripts or manual steps.

## Setup

1.  **Clone the Repository:** Clone this repository to your local machine or Google Colab.
2.  **Install Dependencies:** Install the necessary Python libraries. The Notebook includes pip commands for installing `ultralytics` (YOLOv8) and imports other libraries like `torchvision`, `pycocotools`, `opencv-python`, `numpy`, `scikit-learn`, `torch`, `torch.nn`, `torch.optim`, `torchvision.models`, `torchvision.transforms`, `torch.utils.data`, `PIL`, and potentially `open3d` (for future 3D processing).
3.  **COCO Dataset:**
    * The code assumes the COCO dataset (specifically the **Train2017 and Val2017 splits**) is organized in a specific directory structure within Google Drive (`/content/drive/MyDrive/Colab Notebooks/ME6402 Project/coco/`). You might need to adjust these paths based on your local setup or download the COCO 2017 dataset and place it accordingly.
    * The project utilizes the annotation files (`instances_train2017.json` and `instances_val2017.json`, or filtered subsets thereof) for training and validation.
    * Ensure that the image directories (`train2017` and `val2017`) are also present within the specified COCO directory.
    * The Notebook includes steps to filter the annotations to keep only the "chair" (category ID 62) and "dining table" (category ID 67) classes, which are used for both object detection and image classification.
    * The data loading and preparation steps are designed to handle the image data for both YOLOv8 and ResNet training and evaluation.
5.  **Blender:** While the code doesn't directly interact with Blender, you'll need Blender installed to visualize the 3D scene and the localized/classified objects, likely by exporting data from the Python scripts and importing it into Blender.
6.  **COLMAP:** The 3D reconstruction using COLMAP is mentioned but not directly coded here. This step would involve using the COLMAP software separately to process 2D images and generate 3D models.

## Usage

The `Detection_Recognition.ipynb` Notebook is designed to be run sequentially in a Python environment (preferably Google Colab due to potential GPU usage).

1.  **Mount Google Drive:** If using Google Colab, ensure you mount your Google Drive to access the COCO dataset.
2.  **Install Libraries:** The Notebook starts by installing the required libraries using `pip`.
3.  **Load COCO Dataset:** The code loads the **COCO Train2017 and Val2017 datasets** using the specified image directories and annotation files. The `torchvision.datasets.CocoDetection` function handles the parsing of the COCO annotations.
4.  **Filter and Subset Data (Optional):** The Notebook includes code to filter the COCO dataset to include only "chair" and "dining table" annotations and create smaller subsets for training and evaluating both models. You can run or skip these cells as needed.
5.  **Convert to YOLO Labels (Optional):** If you haven't already, run the cells that convert the COCO annotations into the format required by YOLO for object detection training.
6.  **Train YOLOv8 Model:** The core of the object detection lies in training the YOLOv8 model using the prepared COCO data and YOLO-formatted labels. You can modify the training parameters as needed.
7.  **Train ResNet Model:** The Notebook includes sections for training a ResNet model for image classification, including data loaders, model definition, loss function, optimizer, and training loop.
8.  **Evaluate Models:** Run the sections of the code that calculate and display the evaluation metrics for both the trained YOLOv8 and ResNet models.
9.  **Integrate Detection and Classification:** Execute the code that demonstrates the integration of the YOLOv8 detection results with the ResNet classification model. This might involve passing the cropped regions of detected objects to the ResNet model for classification.
10. **3D Localization (Conceptual):** The code doesn't explicitly implement the PnP algorithm for 3D localization. This step would likely involve:
    * Obtaining camera pose information from the COLMAP reconstruction.
    * Using the 2D bounding box of detected objects and camera intrinsics to estimate the 3D pose of the objects.
11. **Blender Integration (Conceptual):** The results (3D models from COLMAP, 3D bounding boxes from YOLOv8, and class labels from ResNet) would then be imported into Blender for visualization and spatial understanding. This part might involve exporting data from Python scripts.

## Further Work (Based on Code and Description)

* **Implement PnP for 3D Localization:** Add Python code to perform Perspective-n-Point (PnP) to estimate the 3D pose of detected objects using camera parameters and 2D bounding boxes.
* **Integrate with COLMAP Data:** Load camera pose information from COLMAP's output to facilitate the PnP algorithm.
* **Blender Scripting:** Develop Python scripts to automate the import and visualization of the 3D models and localized/classified objects within Blender.
* **Voxelization and Outlier Removal:** Implement the commented-out Open3D functionalities for processing the reconstructed 3D models.

