"""Train or evaluate MIL with configurable cluster embedding and histogram heads.

Requires DMIN.py and clustering_pipeline.py from the same project.
Provide --feature_dir, --label_csv, --split_dir and --cluster_root.
Use identical configuration and --output_dir for training and evaluation.
"""

import argparse
import csv
import logging
import os
import random

import numpy as np


METRIC_NAMES = ("loss", "auc", "acc", "precision", "recall", "f1")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="MIL training and evaluation with configurable cluster modules",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data and output paths.
    parser.add_argument("--feature_dir", required=True, help="Slide feature directory")
    parser.add_argument("--label_csv", required=True, help="Slide label CSV")
    parser.add_argument("--split_dir", required=True, help="Train/val/test split directory")
    parser.add_argument("--cluster_root", required=True, help="Clustering input directory")
    parser.add_argument("--coord_pkl_path", default="", help="Optional coordinate override")
    parser.add_argument("--output_dir", default="./experiments", help="Experiment root")
    parser.add_argument("--dataset", default="CustomDataset")
    parser.add_argument("--pretrain", default="ResNet50_ImageNet")

    # Training and fold configuration.
    parser.add_argument("--phase", default="train", choices=["train", "val", "test"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fold", type=int, default=10, help="Total number of folds")
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--feature_dim", type=int, default=1536)
    parser.add_argument("--label_frac", type=float, default=1.0)
    parser.add_argument("--n_classes", type=int, default=2)
    parser.add_argument("--subtyping", action="store_true")
    parser.add_argument("--k_sample", type=int, default=16)
    parser.add_argument("--n_groups", type=int, default=4)
    parser.add_argument("--n_epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--slide_dropout", type=float, default=0.3)
    parser.add_argument("--max_patches_per_cluster", type=int, default=300)

    # Global clustering.
    parser.add_argument("--num_clusters", type=int, default=14)
    parser.add_argument("--global_pca_dim", type=int, default=32)
    parser.add_argument("--p_landmarks", type=int, default=128)
    parser.add_argument("--r_neighbors", type=int, default=5)
    parser.add_argument("--lsc_mode", default="random", choices=["random", "kmeans"])
    parser.add_argument(
        "--center_weight_mode", default="sqrt",
        choices=["none", "sqrt", "log", "linear", "pow"],
    )
    parser.add_argument("--center_weight_alpha", type=float, default=0.5)
    parser.add_argument("--center_weight_base_rep", type=int, default=1)
    parser.add_argument("--center_weight_max_rep", type=int, default=10)
    parser.add_argument("--center_weight_max_total", type=int, default=200000)

    # Independent module settings, passed to DMINMIL without preset overrides.
    parser.add_argument(
        "--use_cluster_emb", type=int, choices=[0, 1], default=1,
        help="Enable the cluster embedding module (1=yes, 0=no)",
    )
    parser.add_argument("--cluster_emb_dim", type=int, default=8)
    parser.add_argument(
        "--use_cluster_hist", type=int, choices=[0, 1], default=0,
        help="Enable the cluster histogram head (1=yes, 0=no)",
    )
    parser.add_argument("--cluster_hist_hidden", type=int, default=128)
    parser.add_argument("--cluster_id_dropout", type=float, default=0.0)

    args = parser.parse_args(argv)
    if not 0 <= args.k < args.fold:
        parser.error("--k must satisfy 0 <= k < fold")
    if not 0.0 <= args.cluster_id_dropout <= 1.0:
        parser.error("--cluster_id_dropout must be between 0 and 1")
    if args.cluster_emb_dim < 1 or args.cluster_hist_hidden < 1:
        parser.error("--cluster_emb_dim and --cluster_hist_hidden must be positive")

    args.use_cluster_emb = bool(args.use_cluster_emb)
    args.use_cluster_hist = bool(args.use_cluster_hist)
    return args


def seed_torch(seed):
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def make_expdir_and_logs(args):
    module_config = (
        f"emb={int(args.use_cluster_emb)}_edim={args.cluster_emb_dim}"
        f"_hist={int(args.use_cluster_hist)}_hhidden={args.cluster_hist_hidden}"
        f"_idrop={args.cluster_id_dropout}"
    )
    args.exp_dir = os.path.join(
        args.output_dir,
        f"{args.dataset}{args.fold}",
        args.pretrain,
        f"label_frac={args.label_frac}",
        f"ls={args.label_smoothing}_sd={args.slide_dropout}"
        f"_mppc={args.max_patches_per_cluster}_nc={args.num_clusters}_lr={args.lr}",
        module_config,
    )
    args.log_dir = os.path.join(args.exp_dir, "logs")
    os.makedirs(args.log_dir, exist_ok=True)

    log_path = os.path.join(args.log_dir, f"{args.phase}-stdout-fold{args.k}.txt")
    logger = logging.getLogger()
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()
    formatter = logging.Formatter("%(levelname)s - %(asctime)s - %(message)s")
    for handler in (logging.StreamHandler(), logging.FileHandler(log_path, encoding="utf-8")):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def append_csv(path, header, row):
    write_header = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if write_header:
            writer.writerow(header)
        writer.writerow([
            round(value, 6) if isinstance(value, (float, np.floating)) else value
            for value in row
        ])


def append_train_summary(args, result):
    path = os.path.join(args.log_dir, "fold_metrics.csv")
    header = (
        ["fold", "best_epoch"]
        + [f"best_val_{name}" for name in METRIC_NAMES]
        + [f"test_{name}" for name in METRIC_NAMES]
        + ["checkpoint"]
    )
    row = [
        args.k, result["best_epoch"],
        *result["best_val_metrics"], *result["test_metrics"],
        result["checkpoint"],
    ]
    append_csv(path, header, row)
    logging.info("[Fold %s] Metrics saved to %s", args.k, path)


def append_eval_summary(args, metrics):
    path = os.path.join(args.log_dir, f"{args.phase}_metrics_only.csv")
    header = ["fold"] + [f"{args.phase}_{name}" for name in METRIC_NAMES]
    append_csv(path, header, [args.k] + [float(value) for value in metrics])
    logging.info(
        "[Fold %s] %s: loss=%.6f, auc=%.6f, acc=%.6f, f1=%.6f",
        args.k, args.phase.upper(), metrics[0], metrics[1], metrics[2], metrics[5],
    )


def main(argv=None):
    args = parse_args(argv)

    import torch
    from DMIN import DMINMIL
    from clustering_pipeline import run_fold_clustering

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_torch(args.seed)
    make_expdir_and_logs(args)
    logging.info(
        "Cluster modules: embedding=%s, histogram=%s, ID dropout=%s",
        args.use_cluster_emb, args.use_cluster_hist, args.cluster_id_dropout,
    )

    # The pipeline must fit on training data and map val/test with the frozen vocabulary.
    run_fold_clustering(args)

    args.ckpt_dir = os.path.join(args.exp_dir, "ckpts", f"fold-{args.k}")
    os.makedirs(args.ckpt_dir, exist_ok=True)
    mil_runner = DMINMIL(args)

    if args.phase == "train":
        # Runner contract: select by validation AUC, then evaluate test once.
        append_train_summary(args, mil_runner.train())
    else:
        metrics = mil_runner.validate() if args.phase == "val" else mil_runner.test()
        append_eval_summary(args, metrics)


if __name__ == "__main__":
    main()
