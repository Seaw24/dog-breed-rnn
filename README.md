# 🐕 Dog Breed Classification with Neural Network from Scratch

[![Python](https://img.img.shields.io/badge/NumPe](https://img.shields.io/badge/License-MIT-greenn of a **multi-class image classifier** built entirely from scratch using only NumPy. This project demonstrates the complete machine learning pipeline for classifying dog breed images using a 3-layer neural network with advanced optimization techniques.

## 🎯 Project Overview

This project showcases deep learning fundamentals by implementing a neural network **without relying on high-level frameworks** like TensorFlow or PyTorch. Every component, from forward propagation to backpropagation, is built from the ground up to provide a deep understanding of how neural networks actually work.

### ✨ Key Features

- **🔧 Pure NumPy Implementation**: Complete neural network built without deep learning frameworks
- **📊 Advanced Data Pipeline**: Automated image loading, resizing, and preprocessing
- **🧠 3-Layer Dense Architecture**: Fully connected neural network with ReLU and Softmax activations
- **⚡ Optimized Training**: Mini-batch gradient descent with L2 regularization
- **🎲 Smart Initialization**: Xavier/Glorot weight initialization for stable training
- **📈 Numerical Stability**: Robust softmax and cross-entropy implementations
- **🎯 High Performance**: Achieves 96.53% test accuracy on 9 dog breeds

## 📊 Model Performance

| Metric | Accuracy |
|--------|----------|
| **Training** | 100.00% |
| **Validation** | 95.40% |
| **Test** | **96.53%** |

*Trained for 1,500 epochs with learning rate 0.001 and L2 regularization (λ = 0.5)*

## 🐕 Dataset

The model classifies **9 popular dog breeds**:

- 🐕 **Beagle** (100 images)
- 🥊 **Boxer** (100 images)  
- 🌭 **Dachshund** (96 images)
- 🐺 **German Shepherd** (96 images)
- 🟡 **Golden Retriever** (91 images)
- 🦮 **Labrador Retriever** (95 images)
- 🐩 **Poodle** (100 images)
- 💪 **Rottweiler** (89 images)
- 🎀 **Yorkshire Terrier** (100 images)

**Total**: 867 images across 9 classes

### Dataset Structure
```
dataset/
├── Beagle/
│   ├── image1.jpg
│   └── ...
├── Boxer/
├── Dachshund/
└── ...
```

## 🛠️ Installation

### Prerequisites
- Python 3.7+
- Jupyter Notebook

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/Seaw24/dog-breed-rnn.git
cd dog-breed-rnn
```

2. **Install dependencies**
```bash
pip install numpy pandas Pillow matplotlib scikit-learn jupyter
```

3. **Prepare your dataset**
   - Create a `dataset/` folder in the project root
   - Organize images into breed-specific subfolders
   - Supported formats: `.png`, `.jpg`, `.jpeg`, `.gif`, `.bmp`, `.tiff`

4. **Launch Jupyter Notebook**
```bash
jupyter notebook
```

5. **Run the model**
   - Open `Predict-dog-breed-model-checkpoint.ipynb`
   - Execute cells sequentially from top to bottom

## 🏗️ Architecture Details

### Network Structure
- **Input Layer**: 62,400 features (160×130×3 flattened images)
- **Hidden Layer 1**: 24 neurons with ReLU activation
- **Hidden Layer 2**: 16 neurons with ReLU activation  
- **Output Layer**: 9 neurons with Softmax activation

### Training Configuration
- **Optimizer**: Mini-batch Gradient Descent
- **Batch Size**: 80
- **Learning Rate**: 0.001[1]
- **Regularization**: L2 with λ = 0.5
- **Epochs**: 1,500
- **Weight Initialization**: Xavier/Glorot

### Advanced Features
- **Numerical Stability**: Prevents overflow in softmax calculations
- **Data Standardization**: Z-score normalization for optimal convergence
- **Stratified Splitting**: Maintains class distribution across train/validation/test sets

## 📈 Training Process

The model demonstrates excellent convergence characteristics:

```
Epoch 0, Cost: 2.9915
Epoch 100, Cost: 0.5350
Epoch 200, Cost: 0.3065
...
Epoch 1400, Cost: 0.0689
```

## 🚀 Future Improvements

### 🔬 Immediate Enhancements
- **Adam Optimizer**: Implement adaptive learning rate optimization for potentially faster convergence and better performance[1][4][6]
- **Learning Rate Scheduling**: Dynamic learning rate adjustment during training[3][5]
- **Data Augmentation**: Random rotations, flips, and zooms to improve generalization

### 🏗️ Architecture Upgrades
- **Convolutional Neural Network (CNN)**: Replace dense layers with convolutional layers for better spatial pattern recognition
- **Batch Normalization**: Add normalization layers for training stability
- **Dropout Regularization**: Additional overfitting prevention

### 📊 Advanced Techniques
- **Transfer Learning**: Leverage pre-trained models for improved accuracy
- **Ensemble Methods**: Combine multiple models for robust predictions
- **Cross-Validation**: More comprehensive model evaluation

## 📁 Project Structure

```
dog-breed-classification/
├── 📓 Predict-dog-breed-model-checkpoint.ipynb  # Main notebook
├── 📁 dataset/                                  # Image data
├── 📄 README.md                                 # This file
└── 📄 requirements.txt                          # Dependencies
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Dataset contributors and the computer vision community
- NumPy development team for providing excellent mathematical computing tools
- Jupyter Project for the interactive development environment

---

⭐ **Star this repository** if you found it helpful for learning neural networks from scratch!

