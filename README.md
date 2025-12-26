# Automated TV Defect Detection Using Generative AI

This project presents a novel approach to detecting and classifying defects on TV screens in production lines. Addressing the challenge of "Data Scarcity" for broken screens, we utilize Generative AI to create a realistic synthetic dataset for training QA models.

## Project Team

* Amit Wagensberg
* Ori Zarfaty
* Yaniv Hananis

## Repository Structure

This repository is organized as follows:

* **`Automated TV Defect Detection Using Generative AI Code/`**: Main folder containing source code and data.
    * **`[GenerationV_4].py`**: Code used to generate the synthetic images.
    * **`[Mask_Creation].py`**: Code used to generate the masks.
    * **`[Defect_Creation].py`**: Code used to generate the Defects.
   * **`[Model_Training].py`**: Training and evaluation of the YOLO11n-cls model.
    * **`[TV_Dataset]/`**: Contains the generated synthetic dataset.
     * **`[Masks]/`**: Contains the generated mask dataset.
   * **`[Defected_TVs]/`**: Contains the generated inpainted dataset(and some of the none defected TVs).
* **`Automated-TV-Defect-Detection (proposal slides)`**: Project proposal (PPT format).
* **`Project-Review-AI-for-Defect-Detection.pptx`**: Interim report presentation (PPT format).
* **`Project-Review-AI-for-Defect-Detection.pdf`**: Interim report presentation (PDF format).

## The Pipeline

The project consists of a fully automated pipeline with four main stages:

1.  **Base Generation:** Generating images of healthy TVs in an industrial environment using `SDXL Lightning` (RealVisXL V4.0).
2.  **Smart Labeling & Masking:** Detecting the screen and creating accurate binary masks using a combination of `OWLv2` (Object Detection) and `SAM` (Segmentation).
3.  **Defect Injection:** Using Inpainting (`RealVisXL_V4.0_inpainting`) to "inject" defects into the masked areas.
4.  **Classification:** Training a `YOLO11n-cls` model on the synthetic dataset to classify the defect type.

## Defect Classes

The model is trained to identify 5 distinct classes:

* **Good:** Healthy screen.
* **Spiderweb:** Web-like cracks.
* **Scratch:** Surface scratches.
* **Shattered_corner:** Structural damage in the corner.
* **Puncture:** Impact holes/crushed glass.

## Results

The model was trained for 10 Epochs and achieved high performance on the Validation set:

* **Accuracy:** 97.1%
* **Recall:** 0.966
* **Model:** YOLO11n-cls
