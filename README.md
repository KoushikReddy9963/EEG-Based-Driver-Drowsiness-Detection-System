# EEG-Based Driver Drowsiness Detection System

##  Overview

This repository contains the implementation of an **EEG-Based Cross-Subject Driver Drowsiness Recognition System** using an **Interpretable Convolutional Neural Network (CNN)**. The project leverages electroencephalogram (EEG) signals to detect driver drowsiness, aiming to enhance road safety by providing early warnings. The system is designed to be **calibration-free**, addressing the challenge of variability in EEG signals across different subjects and recording sessions.

---

##  Features

- **Data Preparation**: Utilizes a public EEG dataset collected from 27 subjects (aged 22–28) during a sustained-driving task in a virtual reality simulator. The dataset is pre-processed, down-sampled to 128 Hz, and includes 3-second samples with 30 channels × 384 sample points.
- **Model**: Implements a novel interpretable CNN with separable convolutions for spatial-temporal processing of multi-channel EEG signals.
- **Interpretation**: Includes a visualization technique to explain model decisions, highlighting relevant EEG signal features like theta bursts and alpha spindles.
- **Evaluation**: Achieves a mean cross-subject accuracy of **79.35%** on balanced data and using **leave-one-subject-out (LOSO)** cross-validation.

---

##  Installation

**Clone the repository**:
```bash
git clone https://github.com/KoushikReddy9963/EEG-Based-Driver-Drowsiness-Detection-System.git
```

**Navigate to the project directory**:
```bash
cd EEG-Based-Driver-Drowsiness-Detection-System
```

**Install dependencies**:
```bash
pip install -r requirements.txt
```

> ✅ Ensure you have Python 3.6.6, PyTorch, TensorFlow, SciPy, and scikit-learn installed.

---

## 📊 Usage

### 🧠 Training the Model:

**Run the training script**:
```bash
python Training.py
```

- Adjust hyperparameters (e.g., `batch_size=50`, `epochs=10`) inside the script if needed.

---

### Visualization:

**Generate visualizations of learned patterns**:
```bash
python visualization.py
```

---

## Results

- **Accuracy**:
  - 79.35% on the balanced dataset
- **Insights**:
  - Identifies **drowsiness-related features** (theta bursts, alpha spindles)
  - Detects **alertness-related artifacts** (beta waves, eye movements)

---

## Screenshots

### Figure 1: Workflow of the EEG-Based Driver Drowsiness Detection System

![Workflow](https://github.com/KoushikReddy9963/EEG-Based-Driver-Drowsiness-Detection-System/blob/main/Screenshot/workflow.jpg)

---

## 📚 References

Cui, J., Lan, Z., Sourina, O., & Müller-Wittig, W. (2023).  
**EEG-Based Cross-Subject Driver Drowsiness Recognition With an Interpretable Convolutional Neural Network**.  
*IEEE Transactions on Neural Networks and Learning Systems, 34(10), 7921–7932*.  
[DOI: 10.1109/TNNLS.2022.3226147](https://doi.org/10.1109/TNNLS.2022.3226147)

---

## Acknowledgments

Inspired by the research of **Jian Cui et al.**  
Opinions and findings are those of the authors and do not reflect the views of the **National Research Foundation, Singapore**.

---

## Contributing

Contributions are welcome! Please **fork** the repository and submit a **pull request** with your improvements.

---

## License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.

---

