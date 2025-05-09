                                      DEEP LEARNING APPROACH FOR BLOOD GROUP IDENTIFICATION USING FINGERPRINTS

This work presents a solution for AI-powered non-invasive blood group detection from fingerprint images. Based on deep learning and machine learning methodologies, the aim is to provide an alternative to conventional blood sampling. ResNet101 is applied for deep feature extraction from the fingerprint image and then classifying with the best XGBoost model optimized specifically for it. The web platform is delivered using Flask as a web application interface, with fingerprint scans easy to upload by the users and displaying real-time predicted blood group information.

PROJECT OVERVIEW:

Motivation:

In emergency situations and distant healthcare environments, the availability of rapid and trustworthy blood group data can be a matter of life and death. Conventional testing demands medical staff and intrusive methods. The idea behind this project is to utilize the capabilities of biometric features (in this case, fingerprints) to deduce blood group data in a swift, contactless, and hardware-less manner.

Objectives:

1.Develop a deep learning pipeline to predict human blood groups from fingerprint images.

2.Remove the necessity for laboratory-based blood group testing.

3.Design and implement a user-friendly web application to make real-time predictions.

4.Prove integration of preprocessing, augmentation, deep learning, and web development.

PROJECT PIPELINE:

1.Preprocessing
   - Improves fingerprint image quality by eliminating noise and normalizing contrast and size.

2.Augmentation
   - Mimics dataset diversity with methods such as rotation, flipping, scaling, and brightness correction.

3.Segmentation
- Identifies the fingerprint region of interest with the Watershed algorithm to extract features precisely.

4.Feature Extraction
   - Leverages a pre-trained ResNet101 Convolutional Neural Network (CNN) to extract high-dimensional features.

5.Classification
   - Uses XGBoost, a gradient boosting framework, to classify extracted features as one of the 8 blood groups.

6.Deployment
- Web application built with Flask to handle image uploads, predictions, and display results dynamically via an HTML interface.

TECHNOLOGIES USED:

Programming Language: Python  
Deep Learning: TensorFlow, Keras (ResNet101)  
Machine Learning: XGBoost  
Image Processing: OpenCV, NumPy  
Web Development: Flask, HTML, CSS  
Model Deployment: Flask Routing.
