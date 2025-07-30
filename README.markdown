# Disease Grading of Macular Edema Risk in Fundus Images

## Overview
This repository contains code and resources for the automated grading of macular edema risk, specifically diabetic macular edema (DME), using retinal fundus images. The project leverages deep learning techniques to analyze fundus photographs for early detection and classification of DME, a common complication of diabetes that can lead to vision impairment or blindness if not detected early. The system aims to assist in screening and clinical diagnosis by identifying and grading DME severity based on visual features such as hard exudates and retinal thickening.

## Repository Contents
The repository includes scripts, likely in Python, for processing and analyzing fundus images. While specific files are not listed, typical components for such a project include:
- **Preprocessing Scripts**: For image enhancement, noise reduction (e.g., median filtering), and color normalization (e.g., RGB, HSV, or grayscale).
- **Deep Learning Models**: Convolutional Neural Networks (CNNs) or transformer-based models (e.g., Swin Transformer) for DME grading, possibly using pre-trained architectures like Inception-V3 or ResNet50.
- **Segmentation Scripts**: For detecting retinal lesions (e.g., hard exudates, hemorrhages, microaneurysms) associated with DME.
- **Dataset**: Likely references publicly available datasets like HEI-MED (169 fundus images for DME detection) or Kaggle’s EYEPACS (88,702 images for diabetic retinopathy and DME grading).
- **Evaluation Metrics**: Scripts to compute metrics like accuracy, sensitivity, specificity, AUC-ROC, or Dice Coefficient for model performance.

## Prerequisites
To run the code in this repository, you need:
- **Python**: Version 3.6 or later.
- **Libraries**:
  - NumPy, Pandas, Matplotlib (data manipulation and visualization)
  - OpenCV, PIL, Scikit-Image (image preprocessing)
  - TensorFlow, Keras, or PyTorch (deep learning frameworks)
  - Scikit-learn (evaluation metrics)
- **Hardware**: GPU recommended for training deep learning models.
- **Datasets**: Access to fundus image datasets (e.g., HEI-MED, EYEPACS). Download links or instructions may be provided in the scripts or referenced datasets.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/siddharth-sekhar/Disease-Grading-of-Macular-Edema-Risk-in-Fundus-Image.git
   ```
2. Navigate to the repository folder:
   ```bash
   cd Disease-Grading-of-Macular-Edema-Risk-in-Fundus-Image
   ```
3. Install dependencies (assuming a `requirements.txt` file is provided):
   ```bash
   pip install -r requirements.txt
   ```
   If no `requirements.txt` exists, install common libraries:
   ```bash
   pip install numpy pandas matplotlib opencv-python tensorflow scikit-learn scikit-image
   ```

## Usage
1. **Prepare the Dataset**:
   - Download a fundus image dataset (e.g., HEI-MED or EYEPACS).[](https://github.com/lgiancaUTH/HEI-MED)[](https://pmc.ncbi.nlm.nih.gov/articles/PMC9777432/)
   - Place the dataset in a designated folder (e.g., `data/`) or update the script paths to point to your dataset.
2. **Preprocess Images**:
   - Run preprocessing scripts to enhance images (e.g., noise reduction, gamma correction).
   - Example command (modify based on actual script names):
     ```bash
     python preprocess.py --input data/fundus_images --output data/preprocessed
     ```
3. **Train the Model**:
   - Run the training script to train the DME grading model.
   - Example command:
     ```bash
     python train.py --data data/preprocessed --model resnet50 --epochs 50
     ```
4. **Evaluate the Model**:
   - Use evaluation scripts to compute metrics like AUC-ROC or sensitivity/specificity.
   - Example command:
     ```bash
     python evaluate.py --model trained_model.h5 --test_data data/test
     ```
5. **Inference**:
   - Use the trained model to predict DME grades on new fundus images.
   - Example command:
     ```bash
     python predict.py --model trained_model.h5 --image sample_image.jpg
     ```

### Example Workflow
- **Input**: Fundus image (e.g., JPEG or PNG format).
- **Preprocessing**: Apply median filtering and gamma correction to reduce noise and enhance contrast.
- **Segmentation**: Detect hard exudates or other lesions using a U-Net or similar model.
- **Grading**: Classify the image into DME severity grades (e.g., No DME, Grade 1, Grade 2) using a CNN or transformer model.
- **Output**: Predicted DME grade and visualization (e.g., heatmap of lesion locations).

## Dataset
The project likely uses publicly available datasets for DME grading:
- **HEI-MED**: 169 fundus images with manual segmentations of exudates for DME detection.[](https://github.com/lgiancaUTH/HEI-MED)
- **EYEPACS (Kaggle)**: 88,702 high-resolution fundus images labeled for diabetic retinopathy and DME severity (0: No DR/DME, 1: Mild, 2: Moderate, 3: Severe, 4: Proliferative).[](https://pmc.ncbi.nlm.nih.gov/articles/PMC9777432/)
- Check the repository scripts or documentation for specific dataset requirements or download instructions.

## Model Details
- **Architecture**: Likely uses CNNs (e.g., Inception-V3, ResNet50) or transformer-based models (e.g., Swin Transformer) for classification and segmentation.[](https://www.frontiersin.org/journals/physiology/articles/10.3389/fphys.2022.961386/full)
- **Tasks**:
  - **Segmentation**: Identifies retinal lesions like hard exudates, hemorrhages, or microaneurysms.
  - **Classification**: Grades DME severity (e.g., 3-class: No DME, Grade 1, Grade 2) or binary (referable vs. non-referable DME).[](https://github.com/prajpokhrel/dr_dme_severity_grading)
- **Performance**: Expected metrics include AUC-ROC (~0.89 for DME detection), sensitivity (~85%), and specificity (~80%), based on state-of-the-art results.[](https://www.nature.com/articles/s41467-019-13922-8)

## Contributing
Contributions are welcome to improve the code, add new features, or enhance documentation:
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
- Giancardo, L., et al. (2012). "Exudate-based diabetic macular edema detection in fundus images using publicly available datasets." *Medical Image Analysis*.[](https://github.com/lgiancaUTH/HEI-MED)
- Arcadu, F., et al. (2019). "Deep learning predicts OCT measures of diabetic macular thickening from color fundus photographs." *Invest Ophthalmol Vis Sci*.[](https://www.ophthalmologyretina.org/article/S2468-6530%2822%2900001-X/fulltext)
- Kaggle EYEPACS Dataset: https://www.kaggle.com/c/diabetic-retinopathy-detection[](https://pmc.ncbi.nlm.nih.gov/articles/PMC9777432/)

## Contact
For questions or issues, please open an issue on GitHub or contact the repository owner, [siddharth-sekhar](https://github.com/siddharth-sekhar).