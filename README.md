# 🐕 Dog Breed Classification with Neural Network from Scratch

A comprehensive implementation of a **multi-class image classifier** built entirely from scratch using only NumPy. This project demonstrates the complete machine learning pipeline for classifying dog breed images using a 3-layer neural network.

## 🎯 Project Overview

This project showcases deep learning fundamentals by implementing a neural network **without relying on high-level frameworks** like TensorFlow or PyTorch. Every component, from forward propagation to backpropagation, is built from the ground up to provide a deep understanding of how neural networks actually work.

### ✨ Key Features

- **🔧 Pure NumPy Implementation**: Complete neural network built without deep learning frameworks
- **📊 Advanced Data Pipeline**: Automated image loading, resizing, and preprocessing with robust error handling
- **🧠 3-Layer Dense Architecture**: Fully connected neural network with ReLU and Softmax activations
- **⚡ Mini-Batch Training**: Optimized gradient descent with configurable batch sizes
- **🎲 Smart Initialization**: Xavier/Glorot weight initialization for stable training
- **📈 Numerical Stability**: Robust softmax and cross-entropy implementations with overflow protection
- **🔄 L2 Regularization**: Prevents overfitting and improves generalization

## 📊 Current Model Performance

**Note**: The current implementation shows lower than expected performance and requires optimization.

| Metric | Current Accuracy | Target Accuracy |
|--------|------------------|-----------------|
| **Training** | ~12.50% | >90% |
| **Validation** | ~9.77% | >85% |
| **Test** | ~13.87% | >85% |

*Results from training with learning rate 0.01, batch size 100, and L2 regularization (λ = 0.5)*

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

### Current Training Configuration
- **Optimizer**: Mini-batch Gradient Descent
- **Batch Size**: 100
- **Learning Rate**: 0.01 *(may need adjustment)*
- **Regularization**: L2 with λ = 0.5
- **Epochs**: 1,500
- **Weight Initialization**: Xavier/Glorot

## 🚨 Known Issues & Debugging

The current model shows **low accuracy (~13%)** which indicates several potential issues:

### 🔍 Immediate Fixes Needed
1. **Learning Rate**: Current rate (0.01) may be too high - try 0.001 or 0.0001
2. **Cost Function**: Verify gradient calculations and numerical stability
3. **Data Preprocessing**: Ensure proper normalization and standardization
4. **Architecture**: Consider increasing hidden layer sizes or adding more layers

### 🧪 Debugging Steps
```python
# Check if cost is decreasing
plt.plot(cost_history)
plt.title('Training Cost Over Time')
plt.show()

# Verify gradient calculations
# Add gradient checking implementation

# Test with smaller dataset first
# Reduce complexity for initial debugging
```

## 🚀 Future Improvements

### 🔬 Immediate Enhancements
- **Adam Optimizer**: Implement adaptive learning rate optimization for better convergence[1]
- **Learning Rate Scheduling**: Dynamic learning rate adjustment during training
- **Gradient Checking**: Verify backpropagation implementation
- **Data Augmentation**: Random rotations, flips, and zooms to improve generalization

### 🏗️ Architecture Upgrades
- **Convolutional Neural Network (CNN)**: Replace dense layers with convolutional layers for better spatial pattern recognition[1][3][4]
- **Batch Normalization**: Add normalization layers for training stability
- **Dropout Regularization**: Additional overfitting prevention
- **Transfer Learning**: Leverage pre-trained models like VGG-16 or ResNet-50[4][5][6]

### 📊 Advanced Techniques
- **Ensemble Methods**: Combine multiple models for robust predictions
- **Cross-Validation**: More comprehensive model evaluation
- **Hyperparameter Tuning**: Systematic optimization of learning parameters

## 📁 Project Structure

```
dog-breed-classification/
├── 📓 Predict-dog-breed-model-checkpoint.ipynb  # Main notebook
├── 📁 dataset/                                  # Image data
├── 📄 README.md                                 # This file
└── 📄 requirements.txt                          # Dependencies
```

## 🔧 Troubleshooting

### Common Issues
- **Low Accuracy**: Check learning rate, verify gradients, ensure proper data preprocessing
- **Cost Not Decreasing**: Reduce learning rate, check for numerical instabilities
- **Memory Issues**: Reduce batch size or image resolution

### Performance Optimization
- Start with simpler architecture and gradually increase complexity
- Use learning rate scheduling
- Implement early stopping based on validation loss

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Dataset contributors and the computer vision community
- NumPy development team for providing excellent mathematical computing tools
- Jupyter Project for the interactive development environment
- Research papers on CNN architectures and transfer learning techniques[1][3][4][5]

---

⭐ **Star this repository** if you found it helpful for learning neural networks from scratch!

**Note**: This project is currently under development. The low accuracy indicates implementation issues that are being actively addressed. Check the Issues tab for current debugging efforts and solutions.

