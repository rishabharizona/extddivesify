import shap
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.spatial.distance import cosine
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import accuracy_score

class PredictWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model.predict(x)

def get_shap_explainer(model, background_data):
    model.eval()
    wrapped = PredictWrapper(model)
    return shap.DeepExplainer(wrapped, background_data)

def compute_shap_values(explainer, inputs):
    return explainer(inputs)

def plot_summary(shap_values, inputs, output_path="shap_summary.png"):
    plt.figure()
    flat_inputs = inputs.reshape(inputs.shape[0], -1)
    flat_shap_values = shap_values.values.reshape(shap_values.shape[0], -1)

    shap.summary_plot(flat_shap_values, flat_inputs, show=False)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_force(explainer, shap_values, inputs, index=0, output_path="shap_force.html"):
    force_plot = shap.force_plot(explainer.expected_value[0], shap_values.values[index], inputs[index], matplotlib=False)
    shap.save_html(output_path, force_plot)

def evaluate_shap_impact(model, inputs, shap_values, top_k=10):
    base_preds = model.predict(inputs).detach().cpu().numpy()
    flat_shap = np.abs(shap_values.values).reshape(shap_values.shape[0], -1)
    sorted_indices = np.argsort(-flat_shap, axis=1)[:, :top_k]

    masked_inputs = inputs.clone()
    for i, indices in enumerate(sorted_indices):
        flat = masked_inputs[i].flatten()
        flat[indices] = 0
        masked_inputs[i] = flat.view_as(masked_inputs[i])

    masked_preds = model.predict(masked_inputs).detach().cpu().numpy()
    accuracy_drop = np.mean(np.argmax(base_preds, axis=1) != np.argmax(masked_preds, axis=1))
    return base_preds, masked_preds, accuracy_drop

def get_background_batch(loader, size=100):
    x_bg = []
    for batch in loader:
        x = batch[0]
        x_bg.append(x)
        if len(torch.cat(x_bg)) >= size:
            break
    return torch.cat(x_bg)[:size]

def compute_jaccard_topk(shap1, shap2, k=10):
    top1 = set(np.argsort(-np.abs(shap1.flatten()))[:k])
    top2 = set(np.argsort(-np.abs(shap2.flatten()))[:k])
    intersection = len(top1 & top2)
    union = len(top1 | top2)
    return intersection / union

def compute_kendall_tau(shap1, shap2):
    return kendalltau(shap1.flatten(), shap2.flatten())[0]

def cosine_similarity_shap(shap1, shap2):
    return 1 - cosine(shap1.flatten(), shap2.flatten())

def log_shap_numpy(shap_values, save_path="shap_values.npy"):
    np.save(save_path, shap_values.values)

def overlay_signal_with_shap(signal, shap_val, output_path="shap_overlay.png"):
    plt.figure(figsize=(10, 4))
    for i in range(signal.shape[1]):
        plt.plot(signal[:, i], label=f"Feature {i}", alpha=0.6)
    plt.imshow(np.abs(shap_val.T), aspect='auto', cmap='coolwarm', alpha=0.4)
    plt.title("Signal with SHAP Attribution")
    plt.colorbar(label="SHAP Importance")
    plt.savefig(output_path, dpi=300)
    plt.close()

def save_for_wandb(tag, shap_vals, raw_inputs):
    import wandb
    wandb.log({f"{tag}_summary": wandb.Image(plot_summary(shap_vals, raw_inputs))})
    wandb.log({f"{tag}_force": wandb.Html(plot_force(shap_vals, raw_inputs))})
    wandb.log({f"{tag}_shap_vals": wandb.Histogram(shap_vals.values)})


