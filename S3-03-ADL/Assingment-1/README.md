# Assignment 1: Feature Extraction via Dimensionality Reduction

## Overview
This assignment explores feature extraction using various autoencoder architectures and dimensionality reduction techniques on CIFAR10 and MNIST datasets.

## Tasks Completed

### Task 1: PCA-based Feature Extraction [2 marks]
- Standard PCA with 95% variance retention on CIFAR10 grayscale images
- Logistic regression classification (10 classes)
- ROC curves for multi-class classification
- Randomized PCA comparison with standard PCA
- Performance metrics and visualizations

### Task 2: Linear Autoencoder vs PCA [2 marks]
- Single-layer linear autoencoder with tied weights (decoder = encoder transpose)
- Unit norm constraint on weight vectors
- Comparison of PCA eigenvectors vs autoencoder weight matrices
- Visualization as grayscale images
- Theoretical and practical analysis of similarities

### Task 3: Deep Convolutional Autoencoders [4 marks]
- Deep convolutional autoencoder with latent dimension K
- Single hidden layer autoencoder (sigmoid activation)
- Three hidden layer autoencoder with distributed nodes
- Reconstruction error comparison (MSE, RMSE)
- Visual comparison of reconstructed images

### Task 4: MNIST 7-Segment LED Classifier [3 marks]
- Deep convolutional autoencoder trained on MNIST
- Feature extraction using encoder
- 7-segment LED display mapping for digits 0-9
- MLP classifier on latent features
- Confusion matrix and performance analysis

## Setup Instructions

### 1. Create Virtual Environment
```bash
# Using Python venv
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate  # Windows
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch Jupyter Notebook
```bash
jupyter notebook Assignment-1-Solution.ipynb
```

## Running the Notebook

### Option 1: Run All Cells
- In Jupyter: `Kernel > Restart & Run All`
- This will execute all tasks sequentially

### Option 2: Run Individual Tasks
- Navigate to specific task sections
- Execute cells in order within each task

## Expected Outputs

### Task 1
- PCA variance plots
- ROC curves (10 classes, one-vs-rest)
- Classification reports
- Comparison table: Standard vs Randomized PCA

### Task 2
- Training loss/MAE curves
- PCA eigenvector visualizations (16 components)
- Autoencoder weight visualizations (16 components)
- Side-by-side comparison (8 pairs)
- Commentary on similarities/differences

### Task 3
- Three autoencoder architectures trained
- Reconstruction visualizations
- MSE/RMSE comparison table
- Bar chart of reconstruction errors

### Task 4
- MNIST autoencoder training
- 7-segment LED display visualizations
- MLP classifier training curves
- Confusion matrix (10x10)
- Sample predictions with LED displays

## File Structure
```
Assingment-1/
├── Assignment-1.md              # Original problem statement
├── Assignment-1-Solution.ipynb  # Complete solution notebook
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Key Dependencies
- **TensorFlow 2.10+**: Deep learning framework for autoencoders
- **scikit-learn 1.0+**: PCA, logistic regression, metrics
- **NumPy**: Numerical computations
- **Matplotlib/Seaborn**: Visualizations
- **Pandas**: Data manipulation

## Dataset Information

### CIFAR10
- **Training**: 42,000 grayscale images (70% split)
- **Test**: 18,000 grayscale images (30% split)
- **Classes**: 10 (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
- **Image size**: 32x32 pixels (converted to grayscale)

### MNIST
- **Training**: 60,000 images
- **Test**: 10,000 images
- **Classes**: 10 digits (0-9)
- **Image size**: 28x28 pixels (grayscale)

## Training Times (Approximate)
- **PCA**: < 1 minute
- **Linear Autoencoder**: ~2-3 minutes (50 epochs)
- **Deep Conv Autoencoder (CIFAR10)**: ~10-15 minutes (30 epochs)
- **Single/3-Layer Autoencoders**: ~5-8 minutes each (30 epochs)
- **MNIST Conv Autoencoder**: ~5-7 minutes (20 epochs)
- **7-Segment Classifier**: ~3-4 minutes (20 epochs)

**Total estimated time**: 40-60 minutes on CPU, 15-25 minutes on GPU

## Export Instructions

### Generate HTML
```bash
jupyter nbconvert --to html Assignment-1-Solution.ipynb
```

### Generate PDF (requires LaTeX)
```bash
jupyter nbconvert --to pdf Assignment-1-Solution.ipynb
```

### Alternative PDF (via HTML)
1. Export to HTML first
2. Open in browser
3. Print to PDF

## Notes

- **Random Seeds**: Set to 42 for reproducibility
- **Normalization**: CIFAR10 uses StandardScaler (mean=0, std=1), MNIST uses min-max scaling [0,1]
- **GPU**: Will automatically use GPU if available (CUDA for TensorFlow)
- **Memory**: Requires ~4-8GB RAM for full execution

## Troubleshooting

### Import Errors
```bash
pip install --upgrade tensorflow scikit-learn numpy
```

### CUDA Errors (GPU)
- TensorFlow will fall back to CPU automatically
- For GPU support, install tensorflow-gpu version matching CUDA

### Kernel Crashes
- Reduce batch sizes in training
- Close other applications to free memory

## Results Summary

All tasks completed with:
- PCA analysis and classification
- Linear autoencoder comparison
- Deep convolutional architectures
- MNIST 7-segment LED classifier
- All visualizations and metrics
- Embedded outputs in notebook

## Author
Advanced Deep Learning - Session 3 Assignment

## References
- CIFAR10: https://www.cs.toronto.edu/~kriz/cifar.html
- MNIST: http://yann.lecun.com/exdb/mnist/
- TensorFlow Documentation: https://www.tensorflow.org/
- Scikit-learn Documentation: https://scikit-learn.org/
