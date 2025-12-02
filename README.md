# 🌾 Paddy Disease Classifier

A deep learning application for classifying rice leaf diseases using a fine-tuned EfficientNet-B4 model. Upload an image of a rice leaf and get instant predictions with confidence scores.

## 🎯 Supported Diseases

The model can identify the following paddy leaf conditions:

| Disease              | Description                                                           |
| -------------------- | --------------------------------------------------------------------- |
| **Bacterial Blight** | Caused by _Xanthomonas oryzae_, characterized by water-soaked lesions |
| **Brown Spot**       | Fungal disease causing brown oval spots on leaves                     |
| **Rice Blast**       | Caused by _Magnaporthe oryzae_, produces diamond-shaped lesions       |

## 🚀 Live Demo

Try the app here: **[Paddy Disease Classifier](https://paddy-doc.streamlit.app)**

## 📁 Project Structure

```
paddy-disease-classifier/
├── app/
│   ├── streamlit_app.py      # Streamlit web app
│   ├── requirements.txt      # Minimal dependencies for deployment
│   └── __init__.py
├── src/
│   ├── model.py              # EfficientNet classifier architecture
│   ├── data_loader.py        # Custom dataset class
│   ├── train.py              # Training loop
│   ├── evaluate.py           # Model evaluation
│   ├── main.py               # Training pipeline entry point
│   ├── utils/
│   │   ├── transforms.py     # Image transforms
│   │   ├── losses.py         # Focal loss implementation
│   │   └── visualization.py  # Training plots
│   └── __init__.py
├── config/
│   ├── config.py             # Model & training configuration
│   └── __init__.py
├── data/
│   └── raw/
│       ├── train/            # Training images by class
│       └── test/             # Test images by class
├── outputs/
│   ├── models/               # Trained model weights
│   ├── plots/                # Training history & confusion matrix
│   └── logs/                 # Training logs
├── notebooks/                # EDA notebooks
├── report/                   # Report
├── mlruns/                   # MLflow experiment tracking
├── requirements.txt          # Full project dependencies
├── mlflow_utils.py           # MLflow tracking utilities
├── LICENSE
└── README.md
```

## 🛠️ Model Architecture

-   **Backbone**: EfficientNet-B4 (pretrained on ImageNet)
-   **Classifier Head**: Custom fully connected layers
    -   Linear(1792 → 512) → GELU → Linear(512 → 3)
-   **Loss Function**: Focal Loss (α=1, γ=2) for handling hard edge cases
-   **Input Size**: 224×224 RGB images
-   **Normalization**: ImageNet mean/std

## 🏃 Quick Start

### Local Development

1. **Clone the repository**

    ```bash
    git clone https://github.com/hello-wald/paddy-disease-classifier.git
    cd paddy-disease-classifier
    ```

2. **Create a virtual environment**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. **Install dependencies**

    ```bash
    pip install -r requirements.txt
    ```

4. **Run the Streamlit app**

    ```bash
    streamlit run app/streamlit_app.py
    ```

5. Open http://localhost:8501 in your browser

### Using Conda

```bash
conda create -n paddy python=3.12
conda activate paddy
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

## 🎓 Training

To train the model from scratch:

1. **Prepare your dataset**

    ```
    data/raw/
    ├── train/
    │   ├── Bacterial Blight/
    │   ├── Brown Spot/
    │   └── Rice Blast/
    └── test/
        ├── Bacterial Blight/
        ├── Brown Spot/
        └── Rice Blast/
    ```

    > **Dataset Information**: This project uses a combined dataset from two sources:
    > - [Kaggle: 20k Multi-Class Crop Disease Images](https://www.kaggle.com/datasets/jawadali1045/20k-multi-class-crop-disease-images)
    > - [Paddy Doctor Dataset](https://paddydoc.github.io/dataset/)
    > 
    > The processed dataset is also available on [Hugging Face](https://huggingface.co/datasets/hello-wald/paddy-disease-classification) (2,861 images).
    > 
    > See [`data/README.md`](data/README.md) for detailed information about the dataset structure, sources, and statistics.

2. **Configure training parameters** in `config/config.py`

3. **Run training**
    ```bash
    python -m src.main
    ```

Training uses:

-   **Optimizer**: Adam with learning rate scheduling
-   **Early Stopping**: Patience of 7 epochs
-   **Data Augmentation**: Random rotation, horizontal flip, color jitter
-   **Experiment Tracking**: MLflow

## 📊 Training Configuration

| Parameter               | Value |
| ----------------------- | ----- |
| Batch Size              | 32    |
| Epochs                  | 30    |
| Learning Rate           | 0.001 |
| Early Stopping Patience | 7     |
| LR Reduce Patience      | 5     |
| Focal Loss α            | 1     |
| Focal Loss γ            | 2     |

## 🖥️ Hardware Support

The application automatically detects and uses:

-   **Apple Silicon**: MPS (Metal Performance Shaders)
-   **NVIDIA GPU**: CUDA
-   **CPU**: Fallback for deployment

## 📄 License

This project is licensed under the terms specified in the [LICENSE](LICENSE) file.
