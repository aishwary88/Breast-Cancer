# 🩺 Breast Cancer Classification using Deep Learning

## About the Project

Breast cancer is one of the most common cancers worldwide, and early detection can make a significant difference in treatment outcomes. The goal of this project was to build an AI system capable of analyzing mammogram images and classifying them as benign or malignant.

I developed this project using the CBIS-DDSM mammography dataset and a ConvNeXt-Tiny deep learning model. Along with classification, I also integrated Grad-CAM visualizations to better understand what regions of the image influenced the model's predictions.

To make the project easier to use, I built a Streamlit application where users can upload a mammogram image, view the prediction, and visualize the Grad-CAM heatmap.

---

## What I Learned

This project helped me gain hands-on experience with:

* Medical Image Processing
* Deep Learning with PyTorch
* Transfer Learning
* Cross Validation Strategies
* Model Evaluation
* Explainable AI (Grad-CAM)
* Building and Deploying AI Applications

---

## Dataset

The project uses the CBIS-DDSM (Curated Breast Imaging Subset of DDSM) dataset, a publicly available mammography dataset commonly used for breast cancer research.

The dataset contains mammogram images along with pathology labels indicating whether the case is benign or malignant.

---

## Model Pipeline

The workflow of the project is shown below:

Mammogram Image
→ CLAHE Preprocessing
→ ConvNeXt-Tiny Model
→ Prediction
→ Grad-CAM Visualization

To improve model performance and reduce data leakage, patient-wise splitting was performed using StratifiedGroupKFold cross-validation.

---

## Results

After training and evaluation, the model achieved the following results:

| Metric    | Score      |
| --------- | ---------- |
| ROC-AUC   | **0.9175** |
| Accuracy  | **83.92%** |
| Precision | **89%**    |
| Recall    | **81%**    |
| F1-Score  | **85%**    |

The ROC-AUC score of **0.9175** indicates strong classification performance on mammography images while maintaining a good balance between precision and recall.

---

## Explainability with Grad-CAM

Medical AI models should not be treated as black boxes. To improve interpretability, Grad-CAM was integrated into the pipeline.

Grad-CAM generates heatmaps that highlight image regions that contribute most to the model's decision, providing better insight into the prediction process.

---

## Streamlit Application

The project also includes a Streamlit-based web application that allows users to:

* Upload a mammogram image
* Generate predictions
* View prediction confidence
* Visualize Grad-CAM heatmaps

This makes the model easier to demonstrate and interact with.

---

## Technologies Used

* Python
* PyTorch
* OpenCV
* NumPy
* Pandas
* Scikit-learn
* Streamlit
* Matplotlib
* Grad-CAM

---

## Future Improvements

Some ideas for future work include:

* Ensemble inference using multiple trained folds
* Cloud deployment
* Multi-view mammogram analysis
* Automated report generation
* Integration with Large Language Models (LLMs)

---

## Disclaimer

This project was developed for educational and research purposes only. It should not be used as a substitute for professional medical diagnosis.
