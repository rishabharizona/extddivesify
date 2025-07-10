## SHAP-Enhanced Domain Adaptation for Time-Series Classification

### 1. Overview

This project implements SHAP based explainability and extends the paper OUT-OF-DISTRIBUTION REPRESENTATION LEARNING FOR TIME SERIES CLASSIFICATION at ICLR 2023.

Link- https://paperswithcode.com/paper/generalized-representations-learning-for-time 

This project implements a domain adaptation algorithm for time-series classification, with a strong emphasis on model explainability using SHAP (SHapley Additive exPlanations). The primary goal is to train a robust classifier that generalizes well to new, unseen target domains while providing deep insights into the model's decision-making process.

The pipeline is particularly tailored for high-dimensional data like Electromyography (EMG) signals, featuring a comprehensive suite of SHAP-based metrics and visualizations, including novel 4D SHAP analysis techniques.


![image](https://github.com/user-attachments/assets/33673366-109a-42d6-9af7-ead8718983df)

---

### 2. Core Pipelines
   
#### 2.1. Training Pipeline

The training process follows a multi-stage domain adaptation strategy. It iteratively updates the model's components to learn domain-invariant features.
 	
	
	        +--------------------------+
		|      Input Data          |
		| (Train, Valid, Target)   |
	        +--------------------------+
		             |
		             v
		+--------------------------+
		|  Initialize Algorithm    |
		|   (e.g., DANN, CDAN)     |
		+--------------------------+   

        //=============================\\
       ||        Training Loop         ||
        \\=============================//
                     |
                     v
    +--------------------------------+        +--------------------------------+
    | 1. Feature Updater (A)         | -----> | 2. Latent Domain Characterizer |
    | (Update All model params)      |        |     (Adversarial training)     |
    +--------------------------------+        +--------------------------------+
                     |                                  |
                     v                                  |
    +--------------------------------+                  |
    | 3. Set Domain Labels           | <-----------------+
    | (Assign pseudo-labels)         |
    +--------------------------------+
                     |
                     v
    +--------------------------------+
    | 4. Domain-Invariant Feature    |
    |    Learner (C)                 |
    | (Classifier update)           |
    +--------------------------------+

        //===============================\\
       ||  Validation & Model Selection  ||
        \\===============================//
                     |
                     v
    +----------------------------+
    |      Best Model Saved      |
    |  (Based on Valid Accuracy) |
    +----------------------------+


#### 2.2. SHAP Explainability & Evaluation Pipeline

After training, the best model is subjected to a rigorous explainability and performance analysis pipeline using SHAP.

<img width="421" height="631" alt="image" src="https://github.com/user-attachments/assets/ab5788c8-03ee-419e-94da-6bee2d463f4e" />



---

### 3. Key Features

Domain Adaptation: Implements a sophisticated training regimen to handle domain shift between training and testing data.

Comprehensive SHAP Analysis: Goes beyond standard summary plots to include:

Extended Metrics: Flip Rate, Area Over the Perturbation Curve (AOPC), SHAP Entropy, and Feature Coherence.

Similarity Metrics: Jaccard Index, Kendall's Tau, and Cosine Similarity for comparing SHAP explanations.

4D SHAP Visualization: Novel methods for visualizing SHAP values for spatio-temporal data, such as multi-channel EMG signals.

Ablation Studies: Validates the importance of features identified by SHAP by perturbing them and measuring the impact on model performance.

Detailed Logging & Visualization: Generates numerous plots and logs for training progress, model performance, and SHAP results.

---

### 5. File Structure (used specifically in SHAP)


		.
		├── train.py                 # Main script to run training and SHAP analysis
		├── alg/
		│   ├── alg/diversify.py
		│   ├── alg.py               # Contains the core algorithm class definitions
		│   └── opt.py               # Optimizer configurations
		├── datautil/
		│   └── getdataloader_single.py # Data loading and preprocessing logic
		├── utils/
		│   └── util.py              # Utility functions (e.g., seeding, arg parsing)
		├── shap_utils.py            # Core SHAP computation and plotting functions
		├── shap_utils_extended.py   # Advanced SHAP-based metric calculations
		└── shap4D.py                # Functions for 4D SHAP analysis and visualization

---

### 6. Dependencies

The project requires the following major Python libraries. You can install them using pip:

	pip install torch pandas numpy scikit-learn matplotlib shap plotly
---

### 7. Datasets Supported

Direct link - https://wjdcloud.blob.core.windows.net/dataset/diversity_emg.zip

EMG (electromyography)

		# Download the dataset
		wget https://wjdcloud.blob.core.windows.net/dataset/diversity_emg.zip
		unzip diversity_emg.zip && mv emg data/
		
		# Create necessary directories
		mkdir -p ./data/train_output/act/
		
		mkdir -p ./data/emg
		mv emg/* ./data/emg

Data utilities are prebuilt in datautil/actdata and dynamically loaded via getdataloader_single.py.

---

### 8. How to Run

Execute the main training and evaluation script from your terminal. You can customize the behavior using command-line arguments.

Basic Execution:

	python train.py --data_dir ./data/ --task cross_people --test_envs 0 --dataset emg --algorithm diversify --latent_domain_num 10 --alpha1 1.0 --alpha 1.0 --lam 0.0 --local_epoch 3 --max_epoch 1 --lr 0.01 --output ./data/train_output/act/cross_people-emg-Diversify-0-10-1-1-0-3-50-0.01 --enable_shap
	
	python train.py --data_dir ./data/ --task cross_people --test_envs 1 --dataset emg --algorithm diversify --latent_domain_num 2 --alpha1 0.1 --alpha 10.0 --lam 0.0 --local_epoch 10 --max_epoch 2 --lr 0.01 --output ./data/train_output/act/cross_people-emg-Diversify-1-2-0.1-10-0-10-15-0.01 --enable_shap
	
	python train.py --data_dir ./data/ --task cross_people --test_envs 2 --dataset emg --algorithm diversify --latent_domain_num 20 --alpha1 0.5 --alpha 1.0 --lam 0.0 --local_epoch 1 --max_epoch 2 --lr 0.01 --output ./data/train_output/act/cross_people-emg-Diversify-2-20-0.5-1-0-1-150-0.01 --enable_shap
	
	python train.py --data_dir ./data/ --task cross_people --test_envs 3 --dataset emg --algorithm diversify --latent_domain_num 5 --alpha1 5.0 --alpha 0.1 --lam 0.0 --local_epoch 5 --max_epoch 2 --lr 0.01 --output ./data/train_output/act/cross_people-emg-Diversify-3-5-5-0.1-0-5-30-0.01 --enable_shap

Common Arguments:


        --data_dir: Path to the directory containing the input data.
	
	--task: Specifies the domain generalization task setting (e.g., cross_people for cross-subject evaluation).
	
	--test_envs: Index of the environment (domain) to be used as the test set.
	
	--dataset: Name of the dataset to be used (e.g., emg for Electromyography).
	
	--algorithm: Specify the domain adaptation algorithm to use (e.g., DANN, CDAN, or diversify).
	
	--latent_domain_num: The number of latent domains to model.
	
	--alpha1: Weight for the loss used in the feature updater (e.g., domain alignment component).
	
	--alpha: Weight for the classifier update loss (e.g., supervised classification).
	
	--lam: Regularization strength for any auxiliary loss terms (e.g., entropy regularization); set to 0.0 to disable.

	--local_epoch: Number of local update steps within each round.
	
	--max_epoch: Total number of training rounds (global epochs).
	
	--lr: Learning rate for the optimizer (e.g., SGD or Adam).
	
	--output: Directory path where all model outputs (checkpoints, logs, plots) will be stored.
	
	--enable_shap: A crucial flag to activate the entire SHAP analysis pipeline after training. Defaults to False.
 
	--seed: Set the random seed for reproducibility.
 
 ---

### 9. Output and Artifacts

When run with --enable_shap, the script will generate several output files in the root directory:

    training_metrics_plot.png: Plots of training/validation/target accuracy and loss curves over epochs.

    confusion_matrix.png: A confusion matrix for the model's predictions on the validation set.

    shap_summary.png: SHAP summary plot showing global feature importance.

    shap_force_plot_sample_*.png: Individual force plots explaining single predictions.

    shap_overlay_sample*.png: Overlays SHAP values on the original input signal.

    shap_temporal_heatmap.png: A heatmap of SHAP values over time steps.

    shap_4d_surface.html: An interactive 3D surface plot of SHAP values (for 4D data).

    confidence_comparison_ablation.png: Compares model confidence before and after ablating SHAP-identified features.

    confidence_drop_per_class.png: Bar plot showing the average confidence drop for each class after ablation.

    shap_vs_confidence_correlation.png: Scatter plot showing the relationship between SHAP magnitude and model confidence.

---

### 10. SHAP Analysis In-Depth

This project leverages SHAP to not only explain model predictions but also to evaluate the model's robustness and coherence.

	## 📈 Evaluation Metrics & SHAP Analysis

	This framework integrates several SHAP-based evaluation techniques to assess not only **feature importance**, but also the **reliability**, **sparsity**, and **causal influence** of explanations.
	
	| **Metric / Plot**        | **Purpose**                                                                                  | **Implemented In**         |
	|--------------------------|----------------------------------------------------------------------------------------------|----------------------------|
	| **Summary / Force Plots** | Standard SHAP visualizations showing global and local feature importance.                    | `shap_utils.py`             |
	| **Flip Rate**             | Measures how often predictions flip when top SHAP features are removed from input.           | `shap_utils_extended.py`    |
	| **AOPC**                  | *(Area Over Perturbation Curve)*: Tracks change in confidence as top SHAP features are masked.| `shap_utils_extended.py`    |
	| **SHAP Entropy**          | Evaluates sparsity of SHAP explanations. Lower entropy → more focused and interpretable.    | `shap_utils_extended.py`    |
	| **Feature Coherence**     | Checks if SHAP values are consistent for semantically related features (e.g., adjacent time).| `shap_utils_extended.py`    |
	| **4D Surface / EMG Plot** | Visualizes SHAP importance across both time and spatial EMG sensor dimensions.               | `shap4D.py`                 |
	| **SHAP Ablation**         | Shuffles top SHAP-ranked time steps and measures the drop in accuracy/confidence.            | `train.py`                  |
	| **SHAP vs. Confidence Corr.** | Computes correlation between SHAP importance and model prediction confidence.             | `train.py`                  |

---

### 11. License
This project is free for academic and commercial use with attribution.

If using this extended framework, please cite:

	@misc{extdd2025,
	  title={EXT-D-DIVERSIFY: Explainability-Enhanced Domain Generalization},
	  author={Rishabh Gupta et al.},
	  year={2025},
	  note={https://github.com/rishabharizona/extddivesify}
	}

### 12. Contact

rishabhgupta8218@gmail.com
