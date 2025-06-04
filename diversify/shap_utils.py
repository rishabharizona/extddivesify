import shap
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
from scipy.spatial.distance import cosine
from scipy.stats import kendalltau
from sklearn.metrics import accuracy_score
import os

# ✅ SHAP-safe wrapper
class PredictWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model.predict(x)

# ✅ Explainer setup
def get_shap_explainer(model, background_data):
    model.eval()
    wrapped = PredictWrapper(model)
    return shap.DeepExplainer(wrapped, background_data)

def compute_shap_values(explainer, inputs):
    return explainer(inputs)

def _get_shap_array(shap_values):
    if isinstance(shap_values, list):
        return shap_values[0].values
    return shap_values.values

# ✅ Summary plot
def plot_summary(shap_values, inputs, output_path="shap_summary.png", log_to_wandb=False):
    shap_array = _get_shap_array(shap_values)
    flat_inputs = inputs.reshape(inputs.shape[0], -1)
    flat_shap_values = shap_array.reshape(shap_array.shape[0], -1)

    if flat_inputs.shape[1] != flat_shap_values.shape[1]:
        print(f"[WARN] Adjusting flat_inputs from {flat_inputs.shape[1]} to {flat_shap_values.shape[1]}")
        repeat_factor = flat_shap_values.shape[1] // flat_inputs.shape[1]
        flat_inputs = np.repeat(flat_inputs, repeat_factor, axis=1)

    plt.figure()
    shap.summary_plot(flat_shap_values, flat_inputs, show=False)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    if log_to_wandb:
        wandb.log({"SHAP Summary": wandb.Image(output_path)})

# ✅ Force plot (first instance)
def plot_force(explainer, shap_values, inputs, index=0, output_path="shap_force.html", log_to_wandb=False):
    shap_array = _get_shap_array(shap_values)
    shap_for_instance = shap_array[index].reshape(-1)

    if hasattr(explainer, "expected_value"):
        ev = explainer.expected_value
        expected_value = ev[0] if isinstance(ev, (list, np.ndarray)) else ev
    else:
        expected_value = 0

    force_html = shap.plots.force(expected_value, shap_for_instance)
    shap.save_html(output_path, force_html)

    if log_to_wandb:
        wandb.log({"SHAP Force Plot": wandb.Html(open(output_path).read())})

# ✅ Overlay SHAP importance on signal
def overlay_signal_with_shap(signal, shap_val, output_path="shap_overlay.png", log_to_wandb=False):
    signal = signal.reshape(-1)
    shap_val = shap_val.reshape(-1)

    plt.figure(figsize=(12, 4))
    plt.plot(signal, label="Signal", alpha=0.7)
    plt.fill_between(np.arange(len(shap_val)), 0, shap_val, color="red", alpha=0.3, label="SHAP")
    plt.title("Signal with SHAP Overlay")
    plt.xlabel("Time")
    plt.ylabel("Signal / SHAP")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    if log_to_wandb:
        wandb.log({"SHAP Overlay": wandb.Image(output_path)})

# ✅ SHAP heatmap: channels × time
def plot_shap_heatmap(shap_values, index=0, output_path="shap_heatmap.png", log_to_wandb=False):
    shap_array = _get_shap_array(shap_values)
    instance = shap_array[index]

    if instance.ndim > 2:
        instance = instance.reshape(instance.shape[0], -1)  # e.g. (channels, time)

    plt.figure(figsize=(12, 6))
    sns.heatmap(instance, cmap="coolwarm", center=0)
    plt.title(f"SHAP Heatmap (Instance {index})")
    plt.xlabel("Time Steps")
    plt.ylabel("Channels")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    if log_to_wandb:
        wandb.log({"SHAP Heatmap": wandb.Image(output_path)})

# ✅ Mask most influential inputs and evaluate impact
def evaluate_shap_impact(model, inputs, shap_values, top_k=10):
    base_preds = model.predict(inputs).detach().cpu().numpy()
    shap_array = _get_shap_array(shap_values)
    flat_shap = np.abs(shap_array).reshape(shap_array.shape[0], -1)
    sorted_indices = np.argsort(-flat_shap, axis=1)[:, :top_k]

    masked_inputs = inputs.clone()
    for i, indices in enumerate(sorted_indices):
        flat = masked_inputs[i].flatten()
        flat[indices] = 0
        masked_inputs[i] = flat.view_as(masked_inputs[i])

    masked_preds = model.predict(masked_inputs).detach().cpu().numpy()
    accuracy_drop = np.mean(np.argmax(base_preds, axis=1) != np.argmax(masked_preds, axis=1))
    return base_preds, masked_preds, accuracy_drop

# ✅ Background data for DeepExplainer
def get_background_batch(loader, size=100):
    x_bg = []
    for batch in loader:
        x = batch[0]
        x_bg.append(x)
        if len(torch.cat(x_bg)) >= size:
            break
    return torch.cat(x_bg)[:size]

# ✅ Similarity metrics
def compute_jaccard_topk(shap1, shap2, k=10):
    top1 = set(np.argsort(-np.abs(shap1.flatten()))[:k])
    top2 = set(np.argsort(-np.abs(shap2.flatten()))[:k])
    return len(top1 & top2) / len(top1 | top2)

def compute_kendall_tau(shap1, shap2):
    return kendalltau(shap1.flatten(), shap2.flatten())[0]

def cosine_similarity_shap(shap1, shap2):
    return 1 - cosine(shap1.flatten(), shap2.flatten())

# ✅ Save SHAP values
def log_shap_numpy(shap_values, save_path="shap_values.npy"):
    shap_array = _get_shap_array(shap_values)
    np.save(save_path, shap_array)


