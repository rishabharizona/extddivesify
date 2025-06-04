import sys
sys.path.append("./alg")  # ✅ So Python can find algs/modelopera.py
sys.path.append("./datautil")
import torch
import numpy as np
from modelopera import predict_proba
from shap_utils import (
    get_shap_explainer,
    compute_shap_values,
    plot_summary,
    plot_force,
    evaluate_shap_impact,
    compute_jaccard_topk,
    compute_kendall_tau,
    cosine_similarity_shap,
    overlay_signal_with_shap,
    log_shap_numpy,
    get_background_batch
)
from datautil.getdataloader_single import get_act_dataloader
import os


def run_shap_evaluation(model, loader, device, out_dir="shap_outputs"):
    os.makedirs(out_dir, exist_ok=True)

    print("Sampling background data...")
    background_data = get_background_batch(loader, size=64).to(device)
    sample_data = background_data[:10]

    print("Initializing SHAP explainer...")
    explainer = get_shap_explainer(model, background_data)

    print("Computing SHAP values...")
    shap_vals = compute_shap_values(explainer, sample_data)

    print("Generating plots...")
    plot_summary(shap_vals, sample_data.cpu().numpy(), output_path=os.path.join(out_dir, "summary.png"))
    plot_force(explainer, shap_vals, sample_data.cpu().numpy(), index=0, output_path=os.path.join(out_dir, "force.html"))
    overlay_signal_with_shap(sample_data[0].cpu().numpy(), shap_vals.values[0], output_path=os.path.join(out_dir, "overlay_sample0.png"))

    print("Evaluating perturbation impact...")
    base_preds, masked_preds, acc_drop = evaluate_shap_impact(model, sample_data, shap_vals, top_k=10)
    print(f"Perturbation-based accuracy drop: {acc_drop:.4f}")

    print("Saving SHAP values for meta-analysis...")
    log_shap_numpy(shap_vals, save_path=os.path.join(out_dir, "shap_values.npy"))

    print("Computing SHAP agreement metrics...")
    # Simulate pairwise stability across 2 samples
    jaccard = compute_jaccard_topk(shap_vals.values[0], shap_vals.values[1], k=10)
    tau = compute_kendall_tau(shap_vals.values[0], shap_vals.values[1])
    cos_sim = cosine_similarity_shap(shap_vals.values[0], shap_vals.values[1])

    print(f"Jaccard similarity (top-10): {jaccard:.4f}")
    print(f"Kendall’s Tau: {tau:.4f}")
    print(f"Cosine Similarity: {cos_sim:.4f}")


if __name__ == "__main__":
    import argparse
    from algs.diversify import Algorithm
    from utils.params import parse_args

    def add_shap_flag():
        parser = argparse.ArgumentParser()
        parser.add_argument('--enable_shap', action='store_true', help='Enable SHAP evaluation')
        parser.add_argument('--resume', type=str, help='Path to checkpoint file')
        args, _ = parser.parse_known_args()
        return args

    args = parse_args()
    shap_args = add_shap_flag()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not shap_args.enable_shap:
        print("SHAP evaluation not enabled. Use --enable_shap flag.")
        exit()

    print("Loading model and data loader...")
    loader, _ = get_loader(args)
    model = Algorithm(args).to(device)
    model.load_state_dict(torch.load(shap_args.resume))

    run_shap_evaluation(model, loader["val_loader"], device)
