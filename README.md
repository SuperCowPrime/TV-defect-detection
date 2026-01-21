# Automated TV Defect Detection Using Generative AI

This project presents a novel approach to detecting and classifying defects on TV screens in production lines. Addressing the challenge of "Data Scarcity" for broken screens, we utilize Generative AI to create a realistic synthetic dataset for training QA models.

## Project Team

* Amit Wagensberg
* Ori Zarfaty
* Yaniv Hananis

## Repository Structure

This repository is organized as follows:

* **`Automated TV Defect Detection Using Generative AI Code/`**: Main folder containing source code and presentations.
*  **`code/`**: Sub folder containing source code.
    * **`GenerationV_4.py`**: Code used to generate the synthetic images.
    * **`Mask_Creation.py`**: Code used to generate the masks.
    * **`Defect_Creation.py`**: Code used to generate the defects.
    * **`Model_Training.py`**: Training and evaluation of the YOLO11n-cls model.
    * **`Rectify.py`**: Preprocessing script that applies perspective transformation to flatten the TV screens (used for comparative analysis).
      
*  **`slides/`**: Sub folder containing presentations.
* **`Automated-TV-Defect-Detection (proposal slides).pptx`**: Project proposal.
* **`Project-Review-AI-for-Defect-Detection.pptx`**: Interim report presentation (PPT format).
* **`Project-Review-AI-for-Defect-Detection.pdf`**: Interim report presentation (PDF format).
* **`Project_Final_PresentationV2.pptx`**: Final presentation (PPT format).
* **`Project_Final_PresentationV2.pdf`**:  Final presentation (PDF format).

## The Pipeline

The project consists of a fully automated pipeline with four main stages:

1.  **Base Generation:** Generating images of healthy TVs in an industrial environment using `SDXL Lightning` (RealVisXL V4.0).
2.  **Smart Labeling & Masking:** Detecting the screen and creating accurate binary masks using a combination of `OWLv2` (Object Detection) and `SAM` (Segmentation).
3.  **Defect Injection:** Using Inpainting (`RealVisXL_V4.0_inpainting`) to "inject" defects into the masked areas.
4.  **Perspective Rectification:** Preprocessing the dataset by isolating and "flattening" the TV screens using OpenCV perspective transformation. We conduct a comparative analysis between models trained on Raw Images vs. Rectified Images to evaluate the impact on accuracy.
5.  **Classification:** Training two separate YOLO11n-cls models on the synthetic datasets (one Raw, one Rectified) to classify the defect type and establish a robust performance comparison.

## Defect Classes

The dataset utilized for training consists of **7,200 synthetic images**, balanced across 5 distinct classes:

* **Good:** Healthy screen (3600 images).
* **Spiderweb:** Web-like cracks (900 images).
* **Scratch:** Surface scratches (900 images).
* **Shattered_corner:** Structural damage in the corner (900 images).
* **Puncture:** Impact holes/crushed glass (900 images).

## Results & Comparative Analysis

We trained the YOLO11n-cls model for 10 epochs on two datasets to validate our preprocessing pipeline:
1.  **Raw Dataset:** Standard images with industrial background.
2.  **Rectified Dataset:** Images processed via `Rectify.py` to isolate the screen.

### Performance Comparison

| Metric | Raw Dataset | Rectified Dataset |
| :--- | :--- | :--- |
| **Top-1 Accuracy** | 99.09% | **99.03%** |
| **Training Time** | 2318.97 s | **1308.96 s** |

### Conclusion
The **Rectification** step proved critical. By isolating the TV screen, we reduced training time by ~43% while maintaining comparable accuracy, making the model more efficient for deployment.


### Final Metrics (Rectified Model)
* **Model:** YOLO11n-cls
* **Epochs:** 10
* **Best Accuracy:** 99.03% (Epoch 7)
* **Loss:** 0.0346 (Validation)
