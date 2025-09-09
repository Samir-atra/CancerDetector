# CancerDetector

This project is focused on the detection of brain tumors from MRI scans using deep learning. It originated as a final project and has since been updated and expanded. The project explores two different models for cancer detection: one for detecting meningioma tumors and another for classifying multiple types of brain tumors.

## Features

*   **Meningioma Tumor Detection:** A model to classify MRI scans as either positive or negative for meningioma tumors, achieving an accuracy of 93%.
*   **Multi-Class Tumor Detection:** A more advanced model that detects and classifies three types of brain tumors (Glioma, Meningioma, and Pituitary), achieving an accuracy of 95%.
*   **Transfer Learning:** Utilizes the InceptionV3 architecture with weights pre-trained on ImageNet for the multi-class detector, demonstrating the effectiveness of transfer learning for medical imaging tasks.
*   **Jupyter Notebooks:** The repository includes the Jupyter notebooks used for training, evaluation, and prediction, providing a clear view of the entire workflow.

## Models

This project includes two distinct models for brain tumor detection.


This project is focused on the detection of brain tumors from MRI scans using deep learning. It originated as a final project and has since been updated and expanded. The project explores two different models for cancer detection: one for detecting meningioma tumors and another for classifying multiple types of brain tumors.

## Features

*   **Meningioma Tumor Detection:** A model to classify MRI scans as either positive or negative for meningioma tumors, achieving an accuracy of 93%.
*   **Multi-Class Tumor Detection:** A more advanced model that detects and classifies three types of brain tumors (Glioma, Meningioma, and Pituitary), achieving an accuracy of 95%.
*   **Transfer Learning:** Utilizes the InceptionV3 architecture with weights pre-trained on ImageNet for the multi-class detector, demonstrating the effectiveness of transfer learning for medical imaging tasks.
*   **Jupyter Notebooks:** The repository includes the Jupyter notebooks used for training, evaluation, and prediction, providing a clear view of the entire workflow.

## Models

This project includes two distinct models for brain tumor detection.

### Meningioma Detector

This model is a Convolutional Neural Network (CNN) built from scratch using TensorFlow. It is designed for the binary classification of meningioma tumors.

*   **Architecture:**
    *   The model consists of three convolutional layers with 16, 32, and 64 filters, respectively.
    *   The `ELU` (Exponential Linear Unit) activation function is used in all convolutional and dense layers.
    *   `MaxPooling2D` is applied after each convolutional layer to downsample the feature maps.
    *   Two fully-connected (`Dense`) layers with 128 units each follow the convolutional layers.
    *   The final output layer uses a `softmax` activation function for classification.
*   **Regularization:** To prevent overfitting, the model employs two regularization techniques:
    *   `Dropout` with a rate of 0.2 is applied after each pooling layer and between the dense layers.
    *   `L2 regularization` is applied to the weights of all convolutional and dense layers.
*   **Performance:** This model achieves an accuracy of 93% on the test set.

### Multi-Class Cancer Detector

This model is designed to classify MRI scans into four categories: Glioma, Meningioma, Pituitary tumor, or no tumor. It leverages transfer learning to achieve high accuracy.

*   **Technique:** Transfer learning and fine-tuning.
*   **Base Model:** The `InceptionV3` model, pre-trained on the ImageNet dataset, is used as the base for feature extraction. The original classification head of InceptionV3 is removed.
*   **Custom Head:** A new classification head is added on top of the InceptionV3 base, which includes:
    *   A `GlobalAveragePooling2D` layer.
    *   A `Dense` output layer with 4 units and a `softmax` activation function.
*   **Training Process:**
    1.  **Feature Extraction:** The model is first trained with the InceptionV3 base frozen (weights are not updated). This allows the new classification head to adapt to the brain tumor dataset.
    2.  **Fine-Tuning:** After the initial training, the InceptionV3 base is unfrozen, and the entire model is trained with a very low learning rate. This fine-tunes the pre-trained weights to be more specific to the task of tumor detection. During this phase, the `BatchNormalization` layers in InceptionV3 are kept frozen to stabilize training.
*   **Data Augmentation:** To improve generalization, the training data is augmented with `RandomFlip` (horizontal) and `RandomRotation`.
*   **Performance:** This model achieves an accuracy of 95% on the test set.

## Datasets

The models were trained on datasets from Kaggle:

*   **Brain Tumor Classification (MRI):** [https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri)
*   **Brain Tumor MRI Dataset:** [https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

## Installation and Usage

To use this project, you will need to have Python, TensorFlow, and other standard data science libraries installed. The `Transfer.ipynb` and `Train&Test.ipynb` notebooks in the `Cancer_Detector` and `Meningioma_Detector` directories, respectively, contain the code for training the models.

The pre-trained model file for the multi-class detector can be found at the following link:
[https://drive.google.com/drive/folders/1o7ts623pJQxxuOs5kQBkyjEyorH8lT0X?usp=sharing](https://drive.google.com/drive/folders/1o7ts623pJQxxuOs5kQBkyjEyorH8lT0X?usp=sharing)
