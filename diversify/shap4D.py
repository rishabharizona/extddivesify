import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mutual_info_score
from scipy.stats import entropy as scipy_entropy, pearsonr
from scipy.spatial.distance import cosine
from sklearn.decomposition import PCA

# ==============================
# 🔍 4D SHAP + Signal Visualizer
# ==============================
def plot_emg_shap_4d(signal, shap_val, sample_id=0, title="4D EMG SHAP Visualization"):
    signal = signal[sample_id].detach().cpu().numpy()   # (8,1,200)
    shap_val = shap_val[sample_id]                      # (8,1,200,6)

    print(f"Signal shape before reshape: {signal.shape}")
    print(f"SHAP value shape before reshape: {shap_val.shape}")

    # Aggregate over last dim of SHAP (axis=-1)
    shap_val = shap_val.mean(axis=-1)  # (8,1,200)

    n_channels, n_aux, n_time = signal.shape
    signal = signal.squeeze()          # (8, 200)
    shap_val = shap_val.squeeze()      # (8, 200)

    data = {
        "Time": [], "Channel": [], "Signal": [], "SHAP": []
    }

    for c in range(n_channels):
        for t in range(n_time):
            data["Time"].append(t)
            data["Channel"].append(f"C{c}")
            data["Signal"].append(signal[c, t])
            data["SHAP"].append(shap_val[c, t])

    fig = px.scatter_3d(
        data,
        x="Time", y="Channel", z="Signal",
        color="SHAP",
        title=title,
        labels={"SHAP": "SHAP Importance"},
        color_continuous_scale="Inferno"
    )
    fig.update_traces(marker=dict(size=3))
    fig.show()



# ==============================
# 🌐 4D SHAP Surface Plot
# ==============================
def plot_4d_shap_surface(shap_values, output_path="shap_4d_surface.html", title="4D SHAP Visualization"):
    # Assumes shap_values has shape (N, Channels, 1, Time)
    if isinstance(shap_values, list):
        shap_array = shap_values[0].values
    else:
        shap_array = shap_values.values

    sample = shap_array[0]  # (Channels, 1, Time)
    sample = sample.squeeze()  # (Channels, Time)

    x = np.arange(sample.shape[1])  # Time
    y = np.arange(sample.shape[0])  # Channels
    X, Y = np.meshgrid(x, y)

    fig = go.Figure(data=[go.Surface(z=sample, x=X, y=Y, colorscale='RdBu', colorbar=dict(title='SHAP'))])
    fig.update_layout(
        title=title,
        autosize=True,
        margin=dict(l=20, r=20, t=50, b=20),
        scene=dict(
            xaxis_title='Time Steps',
            yaxis_title='Channels',
            zaxis_title='SHAP Value',
        )
    )
    fig.write_html(output_path)
    print(f"[INFO] Saved 4D SHAP surface plot to: {output_path}")

# ===========================
# 📏 Advanced SHAP Metrics
# ===========================

def compute_shap_channel_variance(shap_array):
    """Returns per-channel SHAP variance across time."""
    return shap_array.var(axis=2).mean(axis=0)  # mean variance over samples

def compute_shap_temporal_entropy(shap_array):
    """Entropy over time for each channel's SHAP distribution."""
    n_samples, n_channels, n_time = shap_array.shape
    entropies = []
    for c in range(n_channels):
        flattened = shap_array[:, c, :].flatten()
        probs, _ = np.histogram(flattened, bins=100, density=True)
        probs = probs[probs > 0]
        entropies.append(scipy_entropy(probs))
    return np.mean(entropies)

def compare_top_k_channels(shap1, shap2, k=5):
    """Compares top-k channels between two SHAP instances."""
    shap1_mean = np.abs(shap1).mean(axis=-1)
    shap2_mean = np.abs(shap2).mean(axis=-1)
    top1 = set(np.argsort(-shap1_mean)[:k])
    top2 = set(np.argsort(-shap2_mean)[:k])
    jaccard = len(top1 & top2) / len(top1 | top2)
    return jaccard

def compute_mutual_info(signal, shap_array):
    """Estimates mutual info between signal & SHAP."""
    signal_flat = signal.flatten()
    shap_flat = shap_array.flatten()
    return mutual_info_score(np.digitize(signal_flat, bins=20), np.digitize(shap_flat, bins=20))

def compute_pca_alignment(shap_array):
    """Measures alignment of SHAP with 1st PC of original signal."""
    B, C, T = shap_array.shape
    pca_scores = []
    for b in range(B):
        pca = PCA(n_components=1)
        pc1 = pca.fit_transform(shap_array[b])[:, 0]
        shap_mean = shap_array[b].mean(axis=0)
        corr, _ = pearsonr(pc1, shap_mean)
        pca_scores.append(abs(corr))
    return np.mean(pca_scores)

# ============================
# 🧪 Batch Metric Evaluation
# ============================
def evaluate_advanced_shap_metrics(shap_array, signal_array):
    metrics = {
        "channel_variance": compute_shap_channel_variance(shap_array),
        "temporal_entropy": compute_shap_temporal_entropy(shap_array),
        "mutual_info": compute_mutual_info(signal_array, shap_array),
        "pca_alignment": compute_pca_alignment(shap_array),
    }
    return metrics
