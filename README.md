# TCAMIL

**TCAMIL (Two-stage Cluster-aware Aligned Multiple Instance Learning)** is a weakly supervised computational pathology framework for **KRAS mutation prediction** from colorectal cancer (CRC) H&E whole-slide images.

## Overview

TCAMIL is designed to model the histomorphological heterogeneity of colorectal cancer. Instead of directly aggregating patch features, it first constructs a **globally aligned morphology-aware phenotype space** through within-slide clustering and cross-slide alignment, and then performs **cluster-aware hierarchical MIL aggregation** for slide-level prediction.

<p align="center">
  <img src="TCAMIL.png" width="95%">
</p>

## Framework

The main workflow of TCAMIL includes:

1. **WSI tiling**: H&E whole-slide images are tessellated into non-overlapping image tiles.
2. **Dual feature extraction**: AE-CRC extracts morphology-oriented features for clustering, while UNI2 extracts discriminative tile-level features for prediction.
3. **Within-slide local clustering**: morphologically similar tiles within each WSI are grouped into local phenotype units.
4. **Cross-slide global alignment**: local cluster prototypes from different WSIs are aligned into a shared global morphology-aware phenotype vocabulary.
5. **Cluster-aware feature fusion**: globally aligned cluster identities are embedded and fused with tile-level discriminative features.
6. **Hierarchical MIL prediction**: intra-cluster attention aggregates tiles within each phenotype unit, and inter-cluster attention aggregates cluster-level representations for final KRAS mutation prediction.
7. **Interpretability analysis**: attention-dominant phenotypes and high-attention tissue regions are analysed to provide morphology-aware model interpretation.

## Dataset

This project uses two colorectal cancer cohorts:

- **Gansu cohort**: private cohort from Gansu Provincial People's Hospital, 349 patients.
- **SurGen cohort**: publicly available Scottish colorectal cancer cohort, 350 patients.

## Evaluation setting

- **Internal validation**: patient-level cross-validation within the Gansu cohort.
- **External validation**: model trained on the Gansu cohort and tested on the independent SurGen cohort without retraining or fine-tuning.

## Local Clustering

The within-slide local clustering step is implemented in `TCAMIL/DeepCluster-main`. The running script is:

```text
TCAMIL/DeepCluster-main/run_clustering.py
```

This script is used to configure the input patch directory, output directory, feature extractor, GPU setting, and saving options. The detailed feature processing and local clustering procedure is implemented in:

```text
TCAMIL/DeepCluster-main/Clustering.py
```

Before running local clustering, users need to modify the following parameters in `run_clustering.py` according to their own dataset:

```python
args.input_path = 'path/to/patches_x20'
args.output_path = 'path/to/local_clustering_output'
args.feature_ext = 'ae_crc'
args.gpu_ids = '0'
args.batch_size = 128
args.dim_reduce = 256
args.store_features = True
args.store_clusters = False
args.store_plots = False
```

Here, `args.input_path` indicates the folder containing WSI patch folders, `args.output_path` indicates the directory for saving local clustering outputs, and `args.feature_ext` specifies the feature extractor used for patch-level representation. The local clustering results generated in this step are used as the input for the subsequent global cluster alignment stage.

## Global Clustering and TCAMIL Training

After local clustering, TCAMIL performs global cluster alignment and MIL training using:

```text
TCAMIL/baseline.py
```

This script is the main entry for the TCAMIL training stage. It first calls:

```text
TCAMIL/clustering_pipeline.py
```

to align the within-slide local clusters into a shared global morphology-aware phenotype space. Then it calls:

```text
TCAMIL/DMIN.py
```

to train and evaluate the TCAMIL model. The hierarchical MIL model is implemented in:

```text
TCAMIL/models/hdmil.py
```

Before running this stage, users need to prepare the local clustering outputs, WSI-level feature files, label file, and cross-validation split files. The main parameters include:

```bash
--cluster_root   path/to/local_clustering_output
--feature_dir    path/to/wsi_h5_features
--label_csv      path/to/labels.csv
--split_dir      path/to/splits
--num_clusters   14
--feature_dim    1536
--k              0
--phase          train
```

An example command is:

```bash
python TCAMIL/baseline.py \
    --phase train \
    --k 0 \
    --cluster_root path/to/local_clustering_output \
    --feature_dir path/to/wsi_h5_features \
    --label_csv path/to/labels.csv \
    --split_dir path/to/splits \
    --num_clusters 14 \
    --feature_dim 1536
```

For each fold, `baseline.py` first performs fold-specific global cluster alignment based on the local clustering results stored in `--cluster_root`. The global morphology vocabulary is fitted only on the training side of the current fold. The held-out test slides are not used to fit the global clusters; instead, their local cluster prototypes are assigned to the learned global cluster space during testing. This prevents information leakage while ensuring that both training and test slides are represented using the same global phenotype vocabulary.

It should be noted that the features used for clustering and the features used for MIL prediction are stored separately. The local and global clustering stages use AE-CRC features from `--cluster_root` to obtain morphology-aware global cluster IDs. For TCAMIL training, the discriminative tile-level features are loaded from `--feature_dir`, where each WSI is stored as a `.h5` file containing `features` and `coords`. In our experiments, `--feature_dir` points to the UNI2 feature directory. To replace UNI2 with another feature extractor, users only need to replace the `.h5` feature directory and set the corresponding `--feature_dim`.

During MIL training, TCAMIL reads the UNI2 feature vector of each tile from the `.h5` file and uses the tile coordinates to retrieve its aligned global cluster ID from the generated coordinate-to-cluster mapping. Therefore, each tile is represented by two complementary inputs: its UNI2 discriminative feature and its global morphology-aware cluster identity. These two inputs are then fused in the hierarchical MIL model for slide-level KRAS mutation prediction.


