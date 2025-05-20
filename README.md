# Car Color Detection 

## Project Description

This project implements a system using machine learning to analyze traffic images. It aims to:

1.  **Detect Objects:** Identify cars and people within an input image.
2.  **Classify Car Color:** Determine the color of detected cars.
3.  **Count Objects:** Count the total number of detected cars and people.
4.  **Apply Specific Visual Rules:**
    *   Draw a **RED** rectangle around detected cars classified as **BLUE**.
    *   Draw a **BLUE** rectangle around detected cars of **any other color**.
    *   Draw a **GREEN** rectangle around detected people (configurable).
5.  **Provide a GUI:** Offer a graphical user interface for loading images, previewing input, running analysis, and displaying the processed image with results.

**Constraint:** All core machine learning models (object detection and color classification) must be trained from scratch, without using pre-trained models or weights, as per the project guidelines.

## Features

*   **Object Detection:** Detects 'car' and 'person' classes.
*   **Color Classification:** Predicts the color of detected cars from a defined list (e.g., Blue, Red, White, Black, Silver, Other).
*   **Counting:** Displays the count of detected cars and people on the output image.
*   **Conditional Bounding Boxes:** Applies specific colors to bounding boxes based on car color rules.
*   **Graphical User Interface (GUI):** Built with Tkinter for user interaction.
    *   Load image button.
    *   Image preview area.
    *   Analyze button.
    *   Display of processed image with detections and counts.

## Technology Stack

*   **Language:** Python 3.x
*   **Machine Learning:** TensorFlow / Keras
*   **Image Processing:** OpenCV (cv2)
*   **GUI:** Tkinter (standard Python library)
*   **Libraries:** NumPy, Pillow (PIL Fork), Scikit-learn (for evaluation metrics), Matplotlib (for plotting training history)

