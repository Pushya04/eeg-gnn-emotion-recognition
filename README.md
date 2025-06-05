
# EEG Emotion Recognition using Spectral Graph Neural Networks

This project focuses on enhancing EEG-based emotion classification (valence/arousal) using graph signal processing and advanced Graph Neural Networks (GNNs). It combines PLV-based brain connectivity, spectral GNNs, and deep graph embeddings to capture functional dynamics in EEG signals.

---

## 📂 Project Structure
```
.
├── preprocessing.py           # EEG preprocessing & PSD feature extraction
├── spectral_gnn.py           # Basic Spectral GNN using ChebConv layers
├── showcase_pool.py          # Pooling strategy comparison (TopK, SAG, Mean)
├── graph_embeddings.py       # Advanced embedding model with GAT + GCN + Residuals
├── accurate.py               # Aggregates accuracy across subjects
├── summarize.txt             # Project summary and documentation
├── report.pdf               # [Include if exists: detailed project report]
```

---

## 🧠 Overview
- **Dataset**: [DEAP Dataset](http://www.eecs.qmul.ac.uk/mmv/datasets/deap/)
- **Signals**: 32-channel EEG
- **Labels**: Valence and Arousal (binary classification)
- **Segments**: 2-second windows (256 samples at 128 Hz)

---

## 🧪 Pipeline Summary

### 1. Preprocessing (`preprocessing.py`)
- Bandpass Filtering: 4–48 Hz
- Artifact Removal: ICA using Fp1
- Feature Extraction: PSD (theta, alpha, beta, gamma bands)
- Output: Subject-wise CSV files

### 2. Graph Construction (`spectral_gnn.py`)
- Computes Phase Locking Value (PLV) to build EEG connectivity graphs
- Uses ChebConv-based Spectral GNN for classification
- Trains with 70/30 train/test split

### 3. Pooling Comparison (`showcase_pool.py`)
- Tests TopK, SAG, and Mean pooling
- Logs training loss, accuracy, confusion matrices
- Saves pooling-wise visualizations

### 4. Graph Embedding + Residual GNN (`graph_embeddings.py`)
- Architecture: GAT ➜ 3x GCNConv ➜ Residuals ➜ Projection ➜ Classification
- Integrates: Learning rate scheduling, early stopping, t-SNE
- Outputs: Visual embeddings + metrics

### 5. Accuracy Aggregation (`accurate.py`)
- Computes average and subject-wise accuracy
- Saves logs to file

---

## 🏗️ Model Architecture

### `AdvancedSpectralGNN` (in `graph_embeddings.py`):
- `GATConv` → `GCNConv` x3 (with residuals)
- `Pooling`: TopK/SAG/Mean
- `2x Projection Layers` → Final Classification
- Visual embedding via t-SNE

---

## 📈 Results & Observations
- PLV-based graph construction outperforms raw signal-based models
- TopK/SAG pooling shows better performance for complex emotional states
- Graph embeddings improve classification and reveal meaningful cluster patterns

---

## 🖼️ Visualizations
- Brain connectivity graphs (PLV)
- Confusion matrices
- t-SNE embeddings
- Training loss/accuracy curves

---

## 🏁 How to Run the Project

### 1. Preprocess EEG:
```bash
python preprocessing.py --subject s01 --deap_dataset_path <path> --datafiles_path <path>
```

### 2. Train Spectral GNN:
```bash
python spectral_gnn.py --subject s01 --datafiles_path <path> --label_type valence
```

### 3. Compare Pooling:
```bash
python showcase_pool.py --subject s01 --datafiles_path <path>
```

### 4. Train Advanced GNN + Embeddings:
```bash
python graph_embeddings.py --subject s01 --datafiles_path <path> --label_type arousal
```

### 5. Compute Average Accuracy:
```bash
python accurate.py
```

---

## 📎 Report
A detailed report (`report.pdf`) is included outlining:
- Literature background
- Methodology
- Results
- Conclusions

---

## 🧠 Future Work
- Replace PSD with Fractal Dimensions in preprocessing
- Include more frequency-specific channel weightings
- Explore multimodal emotion classification

---

## 🛠️ Setup Instructions

### 1. Install Dependencies
Create a `requirements.txt` with:
```
numpy
pandas
matplotlib
scipy
scikit-learn
torch
torchvision
torchaudio
torch-geometric
networkx
seaborn
mne
tqdm
```
Install them with:
```bash
pip install -r requirements.txt
```

### 2. Push to GitHub
1. Initialize Git:
```bash
git init
```

2. Add all files:
```bash
git add .
```

3. Commit:
```bash
git commit -m "Initial commit with EEG-GNN pipeline"
```

4. Connect to GitHub:
```bash
git remote add origin https://github.com/<your-username>/eeg-gnn-emotion-recognition.git
```

5. Push:
```bash
git branch -M main
git push -u origin main
```

---

## 🌐 Suggested GitHub Repo Title
**`eeg-gnn-emotion-recognition`**  
> Emotion recognition from EEG signals using advanced spectral graph neural networks

---

## 📜 License
This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

## 📚 Citation
If you use this work, please cite:
```
@misc{pushya2025eeggnn,
  author = {Pushya Mithra},
  title = {EEG Emotion Recognition using Spectral Graph Neural Networks},
  year = 2025,
  howpublished = {\url{https://github.com/<your-username>/eeg-gnn-emotion-recognition}}
}
```
