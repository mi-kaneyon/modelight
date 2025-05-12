# Modelight

## 1. Concept

Modelight is a real-time “actress light” camera effect that automatically enhances facial appearance by detecting faces with MTCNN, parsing skin regions with Segformer, and applying exposure and complexion boosts plus skin smoothing. It delivers a natural, studio-like “beauty cam” result without any external lighting hardware.

> **日本語説明**  
> このプロジェクトは、日本人の肌の特徴と好みに配慮し、人種差別的な表現を避けながら、バラエティ番組で使用される「女優ライト」効果を再現することを目的としています。

## 2. Required Modules

Before running `modelight.py`, install the following Python packages:

```bash
pip install \
  torch torchvision \
  facenet-pytorch \
  transformers[torch] \
  pillow requests opencv-python
```

- **torch**, **torchvision**: Core deep learning framework for Segformer and GPU acceleration.  
- **facenet-pytorch**: MTCNN face detector for robust bounding-box localization.  
- **transformers[torch]**: SegformerImageProcessor and SegformerForSemanticSegmentation from Hugging Face.  
- **pillow**, **requests**: Image I/O and remote image fetching.  
- **opencv-python**: Frame capture, HSV manipulation, and bilateral filtering.

## 3. modelight.py Functionality

- **Face Detection**  
  Uses MTCNN to locate all faces in each camera frame.

- **Skin Parsing**  
  Crops each detected face and runs a Segformer model (`jonathandinu/face-parsing`) to generate a precise skin-region mask.

- **Exposure & Complexion Boost**  
  Converts pixel values to HSV, then multiplies V (value) by 1.2 and adds a bias of 15 (≈20% brighter), and multiplies S (saturation) by 1.15 (≈15% more vivid skin tone).

- **Skin Smoothing**  
  Applies a `cv2.bilateralFilter` (d=9, sigmaColor=75, sigmaSpace=75) to smooth skin while preserving edges for a soft-focus look.

- **Real-Time Display**  
  Captures webcam feed, processes each frame on GPU (if available), and displays the enhanced video in a window until `q` is pressed.
