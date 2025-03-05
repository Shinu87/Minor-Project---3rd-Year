# 🖼️ Image Captioning Projects  

## 📌 Overview  
This repository contains various implementations of **Image Captioning**, a task that involves generating textual descriptions for images. The models used in these projects leverage **computer vision** and **natural language processing (NLP)** techniques to generate meaningful captions.  

---

## 📂 Project Descriptions  

### 📍 Image Captioning using Pretrained Transformers  
This project utilizes the **BLIP (Bootstrapped Language-Image Pretraining) model** from **Hugging Face's Transformers library**. The model processes input images and generates descriptive captions based on learned patterns. Key features include:  

- **Pretrained BLIP model** for caption generation  
- **BLEU Score evaluation** to measure caption accuracy  
- **Image visualization with captions**  
- **Segmented image captions** to analyze different parts of the image  

### 📍 End-to-End Transformer-Based Image Captioning  
This project implements a complete transformer-based pipeline for **image captioning from scratch**. It integrates:  

- **CNN (e.g., ResNet, EfficientNet) as a feature extractor**  
- **Transformer-based sequence model** for caption generation  
- **Tokenization and attention mechanisms** to learn image-text relationships  
- **Training on an image-caption dataset** for fine-tuning performance  
- **Evaluation metrics such as BLEU and CIDEr**  

This approach ensures a fully trainable and customizable **image captioning pipeline** rather than relying on pretrained models.  

---
