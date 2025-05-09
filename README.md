                                              DEEP LEARNING APPROACH FOR BLOOD GROUP IDENTIFICATION USING FINGERPRINTS

This project introduces an AI-driven solution for non-invasive blood group detection using fingerprint images. By leveraging deep learning and machine learning techniques, it aims to eliminate the need for traditional blood sampling. The application utilizes ResNet101 for robust feature extraction from fingerprint images, followed by classification using an optimized XGBoost model. The system is deployed via a web interface built with Flask, allowing users to easily upload fingerprint scans and receive real-time blood group predictions.

PROJECT OVERVIEW:
Motivation:
In emergencies and remote healthcare settings, access to quick and reliable blood group information can be life-saving. Traditional testing requires medical personnel and invasive procedures. This project explores the potential of biometric features (specifically fingerprints) to infer blood group information, providing a *fast, **contactless, and **hardware-free* alternative.

Objectives:
*Build a deep learning pipeline for predicting human blood groups from fingerprint images.
*Eliminate the need for laboratory-based blood group testing.
*Design and deploy a user-friendly web application for real-time predictions.
*Demonstrate integration of preprocessing, augmentation, deep learning, and web development.

PROJECT PIPELINE:

1.Preprocessing 
   - Enhances fingerprint image quality by removing noise and standardizing contrast and size.

2.Augmentation* 
   - Simulates dataset diversity using techniques like rotation, flipping, scaling, and brightness adjustments.

3.Segmentation  
   - Extracts the fingerprint region of interest using the Watershed algorithm for accurate feature extraction.

4.Feature Extraction 
   - Utilizes a pre-trained ResNet101 Convolutional Neural Network (CNN) to extract high-dimensional features.

5.Classification 
   - Applies XGBoost, a gradient boosting framework, to classify extracted features into one of the 8 blood groups.

6.Deployment 
   - Web application built with Flask to handle image uploads, predictions, and display results dynamically via an HTML interface.

TECHNOLOGIES USED:

Programming Language: Python  
Deep Learning: TensorFlow, Keras (ResNet101)  
Machine Learning: XGBoost  
Image Processing: OpenCV, NumPy  
Web Development: Flask, HTML, CSS  
Model Deployment: Flask Routing, Jinja Templates

