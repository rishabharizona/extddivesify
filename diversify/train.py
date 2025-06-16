# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import time
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
from alg.opt import *
from alg import alg, modelopera
from utils.util import set_random_seed, get_args, print_row, print_args, train_valid_target_eval_names, alg_loss_dict, print_environ
from datautil.getdataloader_single import get_act_dataloader

from shap_utils import (
    get_shap_explainer,
    compute_shap_values,
    _get_shap_array,
    plot_summary,
    plot_force,
    evaluate_shap_impact,
    plot_shap_heatmap,
    get_background_batch,
    compute_jaccard_topk,
    compute_kendall_tau,
    cosine_similarity_shap,
    log_shap_numpy,
    overlay_signal_with_shap
)
from shap_utils_extended import (
    compute_flip_rate,
    compute_confidence_change,
    compute_aopc,
    compute_feature_coherence,
    compute_shap_entropy
)
from shap4D import (
    plot_emg_shap_4d,
    compute_shap_channel_variance,
    compute_shap_temporal_entropy,
    compare_top_k_channels,
    compute_mutual_info,
    compute_pca_alignment,
    plot_4d_shap_surface
)
import plotly.io as pio
pio.renderers.default = 'colab'  # Or use 'notebook' if you're not on Colab

def main(args):
    s = print_args(args, [])
    set_random_seed(args.seed)
    print_environ()
    print(s)

    if args.latent_domain_num < 6:
        args.batch_size = 32 * args.latent_domain_num
    else:
        args.batch_size = 16 * args.latent_domain_num

    train_loader, train_loader_noshuffle, valid_loader, target_loader, _, _, _ = get_act_dataloader(args)

    best_valid_acc, target_acc = 0, 0
    logs = {k: [] for k in ['epoch', 'class_loss', 'dis_loss', 'total_loss', 'train_acc', 'valid_acc', 'target_acc', 'total_cost_time']}

    algorithm_class = alg.get_algorithm_class(args.algorithm)
    algorithm = algorithm_class(args).cuda()
    algorithm.train()
    optd = get_optimizer(algorithm, args, nettype='Diversify-adv')
    opt = get_optimizer(algorithm, args, nettype='Diversify-cls')
    opta = get_optimizer(algorithm, args, nettype='Diversify-all')

    for round in range(args.max_epoch):
        print(f'\n======== ROUND {round} ========')
        print('==== Feature update ====')
        print_row(['epoch', 'class_loss'], colwidth=15)
        for step in range(args.local_epoch):
            for data in train_loader:
                loss_result_dict = algorithm.update_a(data, opta)
            print_row([step, loss_result_dict['class']], colwidth=15)

        print('==== Latent domain characterization ====')
        print_row(['epoch', 'total_loss', 'dis_loss', 'ent_loss'], colwidth=15)
        for step in range(args.local_epoch):
            for data in train_loader:
                loss_result_dict = algorithm.update_d(data, optd)
            print_row([step, loss_result_dict['total'], loss_result_dict['dis'], loss_result_dict['ent']], colwidth=15)

        algorithm.set_dlabel(train_loader)

        print('==== Domain-invariant feature learning ====')
        loss_list = alg_loss_dict(args)
        eval_dict = train_valid_target_eval_names(args)
        print_key = ['epoch'] + [f"{item}_loss" for item in loss_list] + [f"{item}_acc" for item in eval_dict] + ['total_cost_time']
        print_row(print_key, colwidth=15)

        sss = time.time()
        for step in range(args.local_epoch):
            for data in train_loader:
                step_vals = algorithm.update(data, opt)

            results = {
                'epoch': round * args.local_epoch + step,
                'train_acc': modelopera.accuracy(algorithm, train_loader_noshuffle, None),
                'valid_acc': modelopera.accuracy(algorithm, valid_loader, None),
                'target_acc': modelopera.accuracy(algorithm, target_loader, None),
                'total_cost_time': time.time() - sss
            }
            for key in loss_list:
                results[f"{key}_loss"] = step_vals[key]
            for key in logs:
                logs[key].append(results.get(key, 0))
            if results['valid_acc'] > best_valid_acc:
                best_valid_acc = results['valid_acc']
                target_acc = results['target_acc']
            print_row([results[k] for k in print_key], colwidth=15)

    print(f'\n🎯 Final Target Accuracy: {target_acc:.4f}')

    if args.enable_shap:
        print("\n📊 Running SHAP explainability...")
        background = get_background_batch(valid_loader, size=64).to('cuda')
        X_eval = background[:10]
        shap_explainer = get_shap_explainer(algorithm, background)
        shap_vals = compute_shap_values(shap_explainer, X_eval)
        shap_array = _get_shap_array(shap_vals)

        plot_summary(shap_vals, X_eval.cpu().numpy())
        plot_force(shap_explainer, shap_vals, X_eval.cpu().numpy())
        overlay_signal_with_shap(X_eval[0].cpu().numpy(), shap_array[0], output_path="shap_overlay_sample0.png")

        base_preds, masked_preds, acc_drop = evaluate_shap_impact(algorithm, X_eval, shap_vals)
        log_shap_numpy(shap_vals)

        print(f"[SHAP] Accuracy Drop: {acc_drop:.4f}")
        print(f"[SHAP] Flip Rate: {compute_flip_rate(base_preds, masked_preds):.4f}")
        print(f"[SHAP] Confidence Δ: {compute_confidence_change(base_preds, masked_preds):.4f}")
        print(f"[SHAP] AOPC: {compute_aopc(algorithm, X_eval, shap_vals, evaluate_shap_impact):.4f}")
        print(f"[SHAP] Entropy: {compute_shap_entropy(shap_array):.4f}")
        print(f"[SHAP] Coherence: {compute_feature_coherence(shap_array):.4f}")

        if len(shap_array) > 1:
            print(f"[SHAP] Jaccard: {compute_jaccard_topk(shap_array[0], shap_array[1]):.4f}")
            print(f"[SHAP] Kendall’s Tau: {compute_kendall_tau(shap_array[0], shap_array[1]):.4f}")
            print(f"[SHAP] Cosine Sim: {cosine_similarity_shap(shap_array[0], shap_array[1]):.4f}")

        # 🔬 4D-specific visualization and metrics
        # ---- Plot EMG SHAP 4D ----
        from shap4D import plot_4d_shap_surface

        try:
            plot_emg_shap_4d(X_eval, shap_array)
        except Exception as e:
            print(f"[WARNING] plot_emg_shap_4d failed to render: {e}")

        plot_4d_shap_surface(shap_vals, output_path="shap_4d_surface.html")
        
        # Reshape SHAP array from (samples, 1, time, aux) to (samples, channels, time)
        shap_array_reshaped = shap_array.reshape(shap_array.shape[0], -1, shap_array.shape[2])

        print(f"[SHAP4D] Channel Variance: {compute_shap_channel_variance(shap_array):.4f}")
        print(f"[SHAP4D] Temporal Entropy: {compute_shap_temporal_entropy(shap_array_reshaped):.4f}")
        signal_sample = X_eval[0].cpu().numpy()                     # shape: (8,1,200)
        shap_sample = shap_array[0].mean(axis=-1)                   # reduce (8,1,200,6) → (8,1,200)
        print(f"[SHAP4D] Mutual Info: {compute_mutual_info(signal_sample, shap_sample):.4f}")
        shap_array_reduced = shap_array.mean(axis=-1)
        print(f"[SHAP4D] PCA Alignment: {compute_pca_alignment(shap_array_reduced):.4f}")
        true_labels, pred_labels = [], []
        for data in valid_loader:
            x, y = data[0].cuda(), data[1]
            preds = algorithm.predict(x).cpu()
            true_labels.extend(y.cpu().numpy())
            pred_labels.extend(torch.argmax(preds, dim=1).detach().cpu().numpy())

        cm = confusion_matrix(true_labels, pred_labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap="Blues")
        plt.title("Confusion Matrix (Validation Set)")
        plt.savefig("confusion_matrix.png", dpi=300)
        plt.show()

        plot_shap_heatmap(shap_array, output_path="shap_temporal_heatmap.png")

        print("\n📊 Training baseline model for SHAP comparison...")
        baseline_model = algorithm_class(args).cuda()
        baseline_model.eval()
        baseline_shap_explainer = get_shap_explainer(baseline_model, background)
        baseline_shap_vals = compute_shap_values(baseline_shap_explainer, X_eval)
        baseline_shap_array = _get_shap_array(baseline_shap_vals)
        plot_shap_heatmap(baseline_shap_array, output_path="shap_heatmap_baseline.png")

        print("\n🔍 Running ablation: shuffling SHAP-important segments...")
        
        X_ablation = X_eval.clone()
        
        # Step 1: Extract SHAP sample
        shap_sample = shap_array[0]  # (1, 200, 6)
        
        # Step 2: Reduce to 1D SHAP importance per time step
        shap_scores = torch.from_numpy(np.abs(shap_sample).mean(axis=(0, 2)))  # shape: (200,)
        
        # Step 3: Safe top-k selection
        topk = min(100, shap_scores.numel())
        if topk == 0:
            print("[SKIP] SHAP ablation: not enough time steps for top-k selection.")
        else:
            shap_mask = shap_scores.topk(topk, largest=True).indices
        
            # Step 4: Apply time-step shuffle for selected top-k
            original = X_ablation[0, :, :, shap_mask].clone()
            perm = shap_mask[torch.randperm(len(shap_mask))]
            X_ablation[0, :, :, shap_mask] = X_ablation[0, :, :, perm]
        
            # Step 5: Evaluate accuracy after ablation
            post_preds = algorithm.predict(X_ablation)
            post_labels = torch.argmax(post_preds, dim=1).cpu().numpy()
            original_labels = torch.argmax(base_preds, axis=1)
        
            print(f"[Ablation] Accuracy post SHAP shuffle: {(post_labels == original_labels).mean():.4f}")




        print("\n🛠 Real-world Context: EMG classification can support gesture-based interfaces in prosthetics or rehabilitation systems, and insights from SHAP improve trust in deployed models.")

    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(logs['epoch'], logs['class_loss'], label="Class Loss", marker='o')
    plt.plot(logs['epoch'], logs['dis_loss'], label="Dis Loss", marker='x')
    plt.plot(logs['epoch'], logs['total_loss'], label="Total Loss", linestyle='--')
    plt.title("Losses over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(logs['epoch'], logs['train_acc'], label="Train Accuracy", marker='o')
    plt.plot(logs['epoch'], logs['valid_acc'], label="Valid Accuracy", marker='x')
    plt.plot(logs['epoch'], logs['target_acc'], label="Target Accuracy", linestyle='--')
    plt.title("Accuracy over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("training_metrics_plot.png", dpi=300)
    plt.show()

if __name__ == '__main__':
    args = get_args()
    main(args)
