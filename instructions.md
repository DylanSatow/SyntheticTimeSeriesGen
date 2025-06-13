# Project: Synthetic Time-Series Data Generation for Imbalanced Occupancy Detection

## 1. Project Title
Synthetic Time-Series Data Generation for Enhanced Occupancy Detection with Generative Models

## 2. Problem Statement
Real-world time-series datasets, particularly those related to sensor data for event detection (e.g., occupancy, anomaly detection), often suffer from significant class imbalance. For instance, in occupancy detection, "occupied" states might be far less frequent than "unoccupied" states, or specific rare events are crucial but underrepresented. This imbalance poses a significant challenge for training robust machine learning models, leading to biased predictions and poor performance on minority classes. Furthermore, sharing such sensitive raw sensor data for research or collaborative model development can raise privacy concerns. This project aims to address these issues by generating high-quality synthetic time-series data that maintains the statistical properties, temporal dependencies, and class distribution (or intentionally oversampling minority classes) of the real data, while also offering a privacy-preserving alternative to raw data.

## 3. Project Goal
To develop and evaluate a generative model pipeline capable of producing realistic and diverse synthetic time-series data for an imbalanced occupancy detection task, thereby improving the performance of downstream classification models on minority classes and demonstrating a privacy-preserving data sharing mechanism.

## 4. Key Skills to Complement
* **Generative AI:** Deep understanding and practical application of advanced generative models (GANs, VAEs, Diffusion Models) for sequential data.
* **Time-Series Analysis:** Expertise in handling temporal dependencies, feature engineering for time-series data, and evaluating time-series specific characteristics.
* **Data Imbalance Techniques:** Practical experience with oversampling/undersampling techniques in the context of synthetic data generation.
* **Privacy-Preserving AI:** Fundamental understanding of the role of synthetic data in privacy.
* **Advanced Evaluation Metrics:** Proficiency in utilizing and interpreting sophisticated metrics for synthetic data quality (fidelity, diversity, utility, privacy).
* **Model Optimization:** Techniques for optimizing complex deep learning models for sequential data.

## 5. Dataset Specification
* **Dataset Name:** PIRvision dataset (from UCI Machine Learning Repository)
* **Description:** This dataset contains occupancy detection data collected from a Synchronized Low-Energy Electronically-chopped Passive Infra-Red sensing node in residential and office environments. Each observation represents 4 seconds of recorded human activity within the sensor Field-of-View (FoV).
* **Characteristics:**
    * **Type:** Time-Series, Multivariate, Sequential, Tabular.
    * **Instances:** Approximately 15,310 observations.
    * **Features:** 54 features (sensor readings, timestamps, etc.).
    * **Target Variable:** Implied occupancy status (can be derived from sensor readings, potentially requiring labeling or pre-processing to define "occupied" vs. "unoccupied" events which are likely imbalanced).
    * **Potential Imbalance:** Significant class imbalance is expected between "occupied" and "unoccupied" states, or between "normal" and "event" states if focusing on specific rare occupancy events.
* **Source:** UCI Machine Learning Repository (PIRvision dataset) - direct download link for the coding agent to verify.

## 6. Technical Specifications & Methodologies

### 6.1. Data Preprocessing
* **Data Loading:** Load the PIRvision dataset.
* **Feature Engineering (if necessary):** Explore creating additional time-series features (e.g., rolling means, standard deviations, differences between consecutive readings) if the raw 54 features are not sufficiently informative for capturing temporal patterns.
* **Missing Data Handling:** Implement strategies for handling any missing values (e.g., imputation, interpolation).
* **Normalization/Scaling:** Scale numerical features (e.g., Min-Max Scaling, Z-score Normalization) to prepare data for neural networks.
* **Sequence Formatting:** Transform the raw tabular time-series data into fixed-length sequences (e.g., using sliding window techniques) suitable for recurrent or transformer-based generative models.
* **Class Imbalance Analysis:** Thoroughly analyze the distribution of the target variable to quantify the class imbalance. Identify the minority class(es).

### 6.2. Generative Model Selection & Implementation
The project will focus on advanced generative models designed for time-series data. The primary candidates are:
* **Time-series Generative Adversarial Networks (TimeGAN):** A robust GAN architecture specifically designed for time-series data generation, capable of learning complex temporal dynamics.
    * **Architecture:** Comprises a Generator, Discriminator, and a "Recovery" network to learn latent representations.
    * **Loss Functions:** Incorporates standard GAN loss, plus a supervised loss for temporal coherence and a reconstruction loss.
* **Transformer-based VAEs (e.g., TTVAE - Time-series Transformer Variational Autoencoder):** Explore how Transformer architectures, known for their ability to capture long-range dependencies, can be integrated into VAE frameworks for time-series generation.
    * **Architecture:** Encoder-Decoder structure with self-attention mechanisms.
    * **Loss Functions:** ELBO (Evidence Lower Bound) combined with reconstruction loss.
* **Conditional Generative Models:** Implement conditional versions of the chosen generative model (e.g., Conditional TimeGAN) to enable controlled generation of synthetic data for specific classes (e.g., oversampling the minority class).

### 6.3. Training Methodology
* **Hardware Requirements:** GPU acceleration is highly recommended due to the complexity of deep learning models and sequential data.
* **Frameworks:** TensorFlow or PyTorch.
* **Hyperparameter Tuning:** Use techniques like Grid Search, Random Search, or Bayesian Optimization to find optimal hyperparameters for the chosen generative model.
* **Training Schedule:** Define appropriate epochs, batch sizes, and learning rates.
* **Early Stopping:** Implement early stopping based on validation metrics to prevent overfitting.
* **Conditional Generation (for Imbalance):** Train the generative model to condition generation on the target class. During synthesis, oversample the minority class by generating more instances of that class.

### 6.4. Evaluation Metrics
Evaluation will be multi-faceted, covering the quality, utility, and privacy aspects of the synthetic data.

* **Fidelity (Statistical Similarity):** How well does the synthetic data capture the statistical properties of the real data?
    * **Marginal Distribution Similarity:** Compare distributions of individual features (e.g., using Wasserstein Distance, Jensen-Shannon Divergence, Kolmogorov-Smirnov test).
    * **Correlation Matrix Similarity:** Compare the correlation matrices between real and synthetic data.
    * **Principal Component Analysis (PCA) Visualization:** Visualize real vs. synthetic data in a lower-dimensional space.
    * **Time-Series Specific Metrics:** Autocorrelation function (ACF) and Partial Autocorrelation Function (PACF) similarity.
* **Diversity:** How varied is the generated synthetic data?
    * **Nearest Neighbor Distance (NND):** Calculate distances between synthetic samples and their nearest real neighbors.
    * **Coverage:** Ensure the synthetic data covers the full range of the real data's feature space.
* **Utility (Performance on Downstream Task):** How well does a model trained on synthetic data perform on real data?
    * **Downstream Classifier:** Train a separate classification model (e.g., LSTM, GRU, or Transformer-based classifier) on a dataset comprising:
        1.  Real data (baseline).
        2.  Synthetic data only.
        3.  Hybrid data (real minority class + synthetic minority class to balance).
    * **Metrics for Classifier:** F1-score (macro and weighted), Precision, Recall, AUC-ROC on the original, imbalanced real test set. *Emphasis on minority class performance.*
* **Privacy:** Does the synthetic data protect the privacy of the original data?
    * **Membership Inference Attack (MIA):** Test if an adversary can determine whether a specific record from the original dataset was used in the training of the synthetic data generator. Lower accuracy of MIA on synthetic data implies better privacy.
    * **Distance to Closest Original Record:** Ensure that no synthetic record is too close to an original record (e.g., Euclidean distance).

## 7. Expected Outcomes
* A trained generative model (TimeGAN or Transformer-based VAE) capable of generating high-fidelity, diverse synthetic time-series data based on the PIRvision dataset.
* Quantitative evaluation of the synthetic data demonstrating its statistical similarity to real data.
* Demonstrable improvement in the performance (specifically F1-score, recall) of a downstream occupancy detection classifier when trained on a dataset augmented with minority-class synthetic data, compared to training solely on the imbalanced real data.
* Insights into the privacy implications of the generated synthetic data.
* Well-documented code and a clear project report detailing the methodology, results, and conclusions.

## 8. Project Steps / Roadmap

### Phase 1: Data Acquisition and Preprocessing (15% of effort)
1.  **Download PIRvision Dataset:** Programmatically download and load the dataset.
2.  **Initial Data Exploration:** Analyze features, identify data types, check for missing values, and understand the raw structure.
3.  **Define Occupancy Target:** Based on dataset documentation or feature analysis, define the binary (or multi-class) occupancy target variable. Quantify class imbalance.
4.  **Time-Series Formatting:** Implement sliding window to create sequences of appropriate length for generative models.
5.  **Data Splitting:** Split the real data into training, validation, and a held-out test set (for downstream classifier evaluation).

### Phase 2: Generative Model Development & Training (40% of effort)
1.  **Model Selection & Setup:** Choose initial generative model (e.g., TimeGAN as a strong starting point). Set up the model architecture using TensorFlow/PyTorch.
2.  **Unconditional Training (Baseline):** Train the generative model to produce synthetic data without class conditioning.
3.  **Conditional Training (for Imbalance):** Modify the model to include class conditioning. Train the model to generate specific classes.
4.  **Hyperparameter Tuning:** Systematically tune hyperparameters for optimal generation quality.
5.  **Synthetic Data Generation:** Generate a sufficient quantity of synthetic data, specifically oversampling the minority class.

### Phase 3: Synthetic Data Evaluation (30% of effort)
1.  **Fidelity Evaluation:**
    * Implement statistical comparison metrics (Wasserstein Distance, Correlation Matrix similarity).
    * Visualize distributions and PCA plots for real vs. synthetic data.
    * Analyze time-series specific metrics (ACF/PACF).
2.  **Diversity Evaluation:**
    * Compute Nearest Neighbor Distances.
    * Assess coverage of the feature space.
3.  **Utility Evaluation (Downstream Classification):**
    * Develop a separate time-series classification model (e.g., LSTM, GRU, or simple CNN/Transformer for classification).
    * Train the classifier on:
        * Real imbalanced training data (baseline).
        * Synthetic training data only.
        * Augmented training data (real majority + synthetic minority to balance).
    * Evaluate all classifiers on the *same, original imbalanced real test set* using F1-score (macro and weighted), precision, recall, and AUC-ROC, focusing on minority class performance.
4.  **Privacy Evaluation (Optional but Recommended):**
    * Implement a basic Membership Inference Attack to assess data leakage.

### Phase 4: Analysis, Documentation & Reporting (15% of effort)
1.  **Result Analysis:** Interpret all evaluation metrics. Identify strengths and weaknesses of the generated synthetic data.
2.  **Comparative Analysis:** Compare the performance of the generative models and the impact of synthetic data on the downstream task.
3.  **Code Refinement:** Ensure code is clean, well-commented, and modular.
4.  **Report Generation:** Prepare a comprehensive report detailing the problem, methodology, results, challenges, and future work.

## 9. Tools and Libraries
* **Programming Language:** Python
* **Deep Learning Frameworks:** TensorFlow, PyTorch
* **Data Manipulation:** NumPy, Pandas
* **Scientific Computing:** SciPy
* **Time-Series Specific Libraries:** `tsfresh` (for feature extraction, if needed), `sktime` (for time-series utilities, if needed)
* **Generative Model Implementations:** Existing open-source implementations of TimeGAN (e.g., `synthcity`, `ydata-synthetic`) or custom implementations.
* **Evaluation Metrics:** `scikit-learn` (for classification metrics), `scipy.stats` (for statistical tests), `matplotlib`, `seaborn` (for visualization).
* **Privacy Evaluation (Optional):** `opacus` (for differentially private training), `privacy-evaluator` (for MIA).

## 10. Deliverables
* **Executable Python Code:** Well-structured, commented, and runnable code for data preprocessing, generative model training, synthetic data generation, and comprehensive evaluation.
* **Trained Generative Model Weights:** Saved model weights for reproducibility.
* **Generated Synthetic Datasets:** CSV or NumPy files containing the synthetic time-series data.
* **Evaluation Results:** Tables and visualizations summarizing all evaluation metrics (fidelity, diversity, utility, privacy).
* **Detailed Project Report:** A markdown document explaining the problem, methodology, results, challenges faced, and conclusions drawn, along with future work.

## 11. Success Criteria
* The generative model successfully trains and generates synthetic time-series data.
* Quantitative metrics demonstrate high fidelity and diversity of the synthetic data compared to the real data.
* A downstream classification model trained on synthetic-data-augmented datasets shows a significant improvement (e.g., >10-15% F1-score improvement) on the minority class on the real test set compared to training on the imbalanced real data alone.
* The project demonstrates a clear understanding and application of advanced generative models for time-series data and their role in addressing data imbalance and privacy concerns.
