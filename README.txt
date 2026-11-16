SPHEREx Transient Detector

An automated, lightweight deep learning pipeline and interactive web interface designed for the NASA Space Apps Challenge to ingest, difference, and classify astronomical transients in multi-epoch SPHEREx infrared survey data.

---

Overview

Analyzing large-scale infrared sky surveys manually is a major bottleneck in modern astronomy. The **SPHEREx Transient Detector** solves this by automating the pipeline from raw multi-extension FITS cutouts to real-time neural network inference. 

The system extracts temporal changes across epochs, builds 3-channel spatial-temporal tensors, processes them through a custom PyTorch convolutional neural network (`SimpleCNN`), and serves the results through an interactive Streamlit UI complete with a visual multi-epoch "blinker."

---

Architecture & Pipeline

preprocess.py: Ingests multi-extension FITS files using `astropy`, normalizes flux arrays, and outputs 3-channel tensors (`[Epoch 1, Epoch 2, Residual Difference]`).
model.py: Houses the `SimpleCNN` PyTorch architecture optimized for binary classification (Transient vs. Background Field).
train.py: Executes training loops with Binary Cross-Entropy Loss and Adam optimization, exporting optimized weights to `spherex_cnn.pth`.
app.py: A zero-friction Streamlit web application providing instant FITS upload handling, side-by-side comparison, and real-time probability scoring.

---

Installation & Setup

1. Clone the repository:
   ```bash
   git clone [https://github.com/your-username/spherex-transient-detector.git](https://github.com/your-username/spherex-transient-detector.git)
   cd spherex-transient-detector