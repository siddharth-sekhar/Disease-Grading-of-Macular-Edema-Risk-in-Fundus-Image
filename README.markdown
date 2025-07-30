# Disease Grading of Macular Edema Risk in Fundus Images

## Overview
This repository contains a Jupyter Notebook (`Disease-Grading-of-Macular-Edema-Risk-in-Fundus-Image.ipynb`) for analyzing retinal fundus images to assess the risk of diabetic macular edema (DME). The code processes fundus images to detect exudates, identify the fovea, and calculate features like the minimum distance from the fovea to the nearest exudate and the exudate count. These features are used to visualize and evaluate DME severity grades. The project leverages computer vision techniques with OpenCV and Python for image processing and visualization.

The notebook is designed for educational or research purposes, likely as part of a medical imaging or computer vision study, using datasets like IDRiD (Indian Diabetic Retinopathy Image Dataset).

## Repository Contents
- **Disease-Grading-of-Macular-Edema-Risk-in-Fundus-Image.ipynb**: The main Jupyter Notebook containing the code for:
  - Loading and preprocessing fundus images.
  - Detecting exudates and computing their centroids.
  - Calculating the minimum distance from the fovea to the nearest exudate.
  - Visualizing results with annotated images showing fovea, exudates, and distance lines.
- **Output Directories** (created by the code):
  - `segmentation_masks/exudates/`: Stores exudate segmentation masks.
  - `fovea_masks/`: Stores fovea segmentation masks.
  - `fovea_on_exudate_masks/`: Stores combined fovea and exudate masks.
  - `distance_line_images/`: Stores images with distance lines drawn between fovea and nearest exudate.
  - `example_plots/`: Stores visualization outputs (e.g., `disease_grades.png`).
- **Expected Input Data**:
  - Fundus images from the IDRiD dataset, specifically from paths like `C. Localization/1. Original Images/a. Training Set/` and `A. Segmentation/1. Original Images/b. Testing Set/`.
  - A `train_data` DataFrame (not included in the notebook) containing columns: `image_name`, `risk_of_macular_edema`, `fovea_x`, `fovea_y`, `min_distance`, and `exudate_count`.
  - Arrays `train_features`, `train_features_mean`, and `train_features_std` for feature denormalization.

## Prerequisites
To run the notebook, you need:
- **Python**: Version 3.6 or later (tested with 3.12.4 as per notebook metadata).
- **Jupyter Notebook**: For executing the `.ipynb` file.
- **Libraries**:
  - `opencv-python` (OpenCV for image processing)
  - `numpy`, `pandas` (data manipulation)
  - `matplotlib`, `seaborn` (visualization)
  - `tensorflow` (though not heavily used in the provided code)
  - `scikit-learn` (for metrics like confusion matrix and accuracy)
  - `tqdm` (progress bars)
- **Dataset**: IDRiD dataset or similar fundus image dataset with labeled fovea coordinates and DME grades. Download from [IDRiD dataset](https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid) or ensure the folder structure matches the paths in the code.
- **Hardware**: Standard CPU is sufficient; GPU is optional as no deep learning training is performed in the provided code.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/siddharth-sekhar/Disease-Grading-of-Macular-Edema-Risk-in-Fundus-Image.git
   ```
2. Navigate to the repository folder:
   ```bash
   cd Disease-Grading-of-Macular-Edema-Risk-in-Fundus-Image
   ```
3. Install dependencies:
   ```bash
   pip install opencv-python numpy pandas matplotlib seaborn tensorflow scikit-learn tqdm
   ```
4. Set up the IDRiD dataset:
   - Download the IDRiD dataset and place the images in the folder structure expected by the code (e.g., `C. Localization/1. Original Images/a. Training Set/` and `A. Segmentation/1. Original Images/b. Testing Set/`).
   - Ensure the `train_data` DataFrame and feature arrays (`train_features`, `train_features_mean`, `train_features_std`) are available. These may need to be generated or obtained from the dataset provider.

## Usage
1. **Prepare the Environment**:
   - Open the Jupyter Notebook in an environment with the required libraries.
   - Ensure the dataset is correctly placed and accessible at the paths specified in the code.
2. **Run the Notebook**:
   - Launch Jupyter Notebook:
     ```bash
     jupyter notebook Disease-Grading-of-Macular-Edema-Risk-in-Fundus-Image.ipynb
     ```
   - Execute the cells sequentially.
3. **Key Functionality**:
   - **Cell 1**: A test cell computing `1+10` (outputs `11`), likely for verifying the notebook environment.
   - **Cell 2**: Imports libraries, creates output directories, and loads a sample fundus image (`IDRiD_55.jpg`) for segmentation testing.
   - **Cell 3**: Core analysis cell:
     - Denormalizes features (`min_distance`, `exudate_count`) from `train_data` using precomputed mean and standard deviation.
     - Selects three specific images (`IDRiD_043`, `IDRiD_094`, `IDRiD_078`) from `train_data` for visualization.
     - For each image:
       - Loads the fundus image from the training set.
       - Detects exudates using a `detect_exudates` function (not shown in the provided code but assumed to exist).
       - Computes the centroids of exudates and the minimum distance from the fovea to the nearest exudate.
       - Draws annotations: fovea center (red cross), exudates (yellow circles), nearest exudate (green circle), and a line connecting the fovea to the nearest exudate (blue line).
       - Prints debug information comparing precomputed and recalculated distances and exudate counts.
       - Visualizes the annotated images in a 1x3 subplot with titles showing the DME grade, distance, and exudate count.
     - Saves the visualization as `example_plots/disease_grades.png`.
   - **Cell 4**: Empty cell, possibly for future use.
4. **Expected Output**:
   - A plot (`example_plots/disease_grades.png`) showing three fundus images with annotations for fovea, exudates, and distance lines, labeled with DME grades and feature values.
   - Console output with debug information for each image, including precomputed and recalculated feature values.

### Example Output
The notebook generates a visualization with three subplots, each showing:
- A fundus image with:
  - Red cross: Fovea center.
  - Yellow circles: Exudate centroids.
  - Green circle: Nearest exudate to the fovea.
  - Blue line: Distance from fovea to nearest exudate.
- Title format: `Grade {risk_of_macular_edema}\nDist: {min_distance:.2f}, Count: {exudate_count:.0f}`.
- Legend explaining the annotations.

## Code Details
- **Dataset Dependency**: The code assumes access to the IDRiD dataset, which includes fundus images and annotations for fovea coordinates and DME grades. The `train_data` DataFrame must contain:
  - `image_name`: e.g., `IDRiD_043`.
  - `risk_of_macular_edema`: DME severity grade (e.g., 0, 1, 2).
  - `fovea_x`, `fovea_y`: Coordinates of the fovea center.
  - `min_distance`, `exudate_count`: Precomputed features (denormalized in the code).
- **Exudate Detection**: The `detect_exudates` function (not shown) is critical for identifying exudates. It likely uses thresholding, morphological operations, or a pretrained model to segment bright lesions (exudates) in fundus images.
- **Feature Calculation**:
  - **Minimum Distance**: Computed as the Euclidean distance from the fovea (`fovea_x`, `fovea_y`) to the nearest exudate centroid, normalized by dividing by 200 (possibly to scale to a specific unit, e.g., millimeters or relative to image size).
  - **Exudate Count**: Number of detected exudate regions.
- **Visualization**: Uses Matplotlib to create a 1x3 subplot with annotated images, saved as a PNG file. The legend uses custom `Line2D` elements to explain annotations.
- **Limitations**:
  - The `detect_exudates` function is not included, so the code is incomplete without it.
  - The notebook assumes precomputed features (`train_features`, `train_features_mean`, `train_features_std`) and a `train_data` DataFrame, which are not provided.
  - Only three specific images are visualized, limiting the scope of analysis.
  - No deep learning model training or evaluation is performed in the provided code, despite importing TensorFlow and scikit-learn.

## Dataset
The code references the IDRiD dataset, which includes:
- Fundus images in `C. Localization/1. Original Images/a. Training Set/` and `A. Segmentation/1. Original Images/b. Testing Set/`.
- Annotations for fovea coordinates and DME grades (0: No DME, 1: Mild, 2: Severe).
Download the dataset from [IEEE Dataport](https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid) and organize it according to the paths in the code.

## Contributing
Contributions to improve the code, add missing functions (e.g., `detect_exudates`), or enhance documentation are welcome:
1. Fork the repository.
2. Create a new branch:
   ```bash
   git checkout -b feature-branch
   ```
3. Commit changes:
   ```bash
   git commit -m "Add feature or fix"
   ```
4. Push to the branch:
   ```bash
   git push origin feature-branch
   ```
5. Open a pull request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details. (Note: If no LICENSE file exists, contact the repository owner for clarification.)

## References
- Porwal, P., et al. (2018). "Indian Diabetic Retinopathy Image Dataset (IDRiD): A Database for Diabetic Retinopathy Screening Research." *IEEE Dataport*.
- Giancardo, L., et al. (2012). "Exudate-based diabetic macular edema detection in fundus images using publicly available datasets." *Medical Image Analysis*.

## Contact
For questions or issues, please open an issue on GitHub or contact the repository owner, [siddharth-sekhar](https://github.com/siddharth-sekhar).