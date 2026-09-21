# clustering_pipeline.py
import os
import logging
import pickle
from collections import Counter

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from scipy.sparse import coo_matrix, diags



def read_split_cases(args):
    split_csv = os.path.join(args.split_dir, f'splits_{args.k}.csv')
    df = pd.read_csv(split_csv)

    train_cases, val_cases, test_cases = set(), set(), set()

    for _, row in df.iterrows():
        if not pd.isna(row.get('train', np.nan)):
            train_cases.add(str(int(row['train'])))
        if not pd.isna(row.get('val', np.nan)):
            val_cases.add(str(int(row['val'])))
        if not pd.isna(row.get('test', np.nan)):
            test_cases.add(str(int(row['test'])))

    return train_cases, val_cases, test_cases


def find_wsi_feature_and_cluster_files(root_dir):
    result = []

    for dirpath, dirnames, filenames in os.walk(root_dir):
        if "cluster_assignments.csv" not in filenames:
            continue

        cluster_csv = os.path.join(dirpath, "cluster_assignments.csv")
        wsi_id = os.path.basename(dirpath)

        feature_csv_candidates = [
            f for f in filenames
            if f.endswith(".csv") and f != "cluster_assignments.csv"
        ]

        feature_csv = None
        for fname in feature_csv_candidates:
            path = os.path.join(dirpath, fname)
            try:
                df_head = pd.read_csv(path, nrows=5)
            except Exception:
                continue

            if "File_Path" in df_head.columns and "Cluster" not in df_head.columns:
                if df_head.shape[1] >= 10:
                    feature_csv = path
                    break

        if feature_csv is None:
            logging.warning(
                f"[Clustering] 找到 {cluster_csv}, 但没找到 features csv, 跳过该 WSI: {wsi_id}"
            )
            continue

        result.append({
            "wsi": wsi_id,
            "feature_csv": feature_csv,
            "cluster_csv": cluster_csv
        })

    logging.info(f"[Clustering] total WSI with cluster_assignments.csv: {len(result)}")
    return result


def parse_xy_from_filepath(file_path):
    import re
    fname = os.path.basename(file_path)
    name, _ = os.path.splitext(fname)

    m = re.search(r'x(\d+)_y(\d+)', name)
    if m:
        x, y = m.group(1), m.group(2)
        return int(x), int(y)

    mx = re.search(r'x(\d+)', name)
    my = re.search(r'y(\d+)', name)
    if mx and my:
        return int(mx.group(1)), int(my.group(1))

    parts = re.split(r'[_\-]', name)
    nums = [p for p in parts if p.isdigit()]
    if len(nums) >= 2:
        return int(nums[-2]), int(nums[-1])

    logging.warning(f"[Clustering] parse_xy_from_filepath 失败: {file_path}")
    return None


def _compute_replication_factors(
    sizes: np.ndarray,
    mode: str = "sqrt",
    alpha: float = 0.5,
    base_rep: int = 1,
    max_rep: int = 10,
    min_rep: int = 1,
    max_total_rep: int = 200000,
):
    sizes = np.asarray(sizes, dtype=np.float64)
    sizes[sizes < 1] = 1.0

    if mode is None:
        mode = "sqrt"

    mode = mode.lower()
    if mode == "none":
        rep = np.ones_like(sizes, dtype=np.int64)
    else:
        if mode == "sqrt":
            w = np.sqrt(sizes)
        elif mode == "log":
            w = np.log1p(sizes)
        elif mode == "linear":
            w = np.power(sizes, alpha if alpha is not None else 1.0)
        elif mode == "pow":
            w = np.power(sizes, alpha if alpha is not None else 0.5)
        else:
            logging.warning(f"[Clustering] Unknown center_weight_mode={mode}, fallback to sqrt")
            w = np.sqrt(sizes)

        med = np.median(w)
        if not np.isfinite(med) or med <= 0:
            med = 1.0

        rep_float = (w / (med + 1e-12)) * float(base_rep)
        rep = np.rint(rep_float).astype(np.int64)
        rep = np.clip(rep, min_rep, max_rep)

    total = int(rep.sum())
    if total > max_total_rep:
        scale = max_total_rep / float(total)
        rep = np.maximum(min_rep, np.floor(rep.astype(np.float64) * scale)).astype(np.int64)
        rep = np.clip(rep, min_rep, max_rep)
        logging.info(
            f"[Clustering] Replication capped: original_total={total}, "
            f"scaled_total={int(rep.sum())}, scale={scale:.4f}"
        )

    return rep


def _majority_vote_labels(rep_labels: np.ndarray, rep_to_orig: np.ndarray, n_orig: int):
    rep_labels = np.asarray(rep_labels, dtype=np.int64)
    rep_to_orig = np.asarray(rep_to_orig, dtype=np.int64)

    out = np.zeros((n_orig,), dtype=np.int64)
    for i in range(n_orig):
        idx = np.where(rep_to_orig == i)[0]
        if idx.size == 0:
            out[i] = 0
        else:
            labs = rep_labels[idx].tolist()
            out[i] = Counter(labs).most_common(1)[0][0]
    return out


def _is_new_coord_map_format(obj) -> bool:

    if not isinstance(obj, dict):
        return False
    if len(obj) == 0:
        return True
    any_val = next(iter(obj.values()))
    return isinstance(any_val, dict)


# ========================
# 1. LSCModel：Landmark-based Spectral Clustering
# ========================

class LSCModel:
    def __init__(self,
                 k: int,
                 p: int = 128,
                 r: int = 5,
                 mode: str = "random",
                 random_state: int = 42,
                 max_iter: int = 100,
                 n_init: int = 10):
        self.k = k
        self.p = p
        self.r = r
        self.mode = mode
        self.random_state = random_state
        self.max_iter = max_iter
        self.n_init = n_init

        self.marks = None
        self.sigma = None
        self.feaSum = None
        self.inv_feaSum = None
        self.S = None
        self.Vt = None
        self.kmeans = None

        self.is_fitted = False

    def _select_landmarks(self, X: np.ndarray) -> np.ndarray:
        n_smp = X.shape[0]
        p = min(self.p, n_smp)

        if self.mode == "random":
            rng = np.random.RandomState(self.random_state)
            idx = rng.choice(n_smp, size=p, replace=False)
            marks = X[idx]
        elif self.mode == "kmeans":
            km = KMeans(
                n_clusters=p,
                random_state=self.random_state,
                n_init=5,
                max_iter=20,
                verbose=0
            )
            km.fit(X)
            marks = km.cluster_centers_
        else:
            raise ValueError(f"Unsupported LSC mode: {self.mode}")

        return marks

    def _build_Z(self,
                 X: np.ndarray,
                 marks: np.ndarray,
                 sigma: float = None,
                 r: int = 5):
        logging.info("[LSC] Step 2/5: computing pairwise distances X->landmarks ...")
        D = pairwise_distances(X, marks, metric="euclidean")

        if sigma is None or sigma <= 0:
            sigma = np.mean(D)
            if sigma <= 0:
                sigma = 1.0

        n_smp, p = D.shape
        if r > p:
            r = p

        idx_r = np.argpartition(D, kth=r - 1, axis=1)[:, :r]
        dump = D[np.arange(n_smp)[:, None], idx_r]

        weights = np.exp(-dump / (2.0 * sigma * sigma))
        sumD = np.sum(weights, axis=1, keepdims=True) + 1e-12
        Gsdx = weights / sumD

        row_idx = np.repeat(np.arange(n_smp), r)
        col_idx = idx_r.reshape(-1)
        data_vals = Gsdx.reshape(-1)

        Z = coo_matrix(
            (data_vals, (row_idx, col_idx)),
            shape=(n_smp, marks.shape[0])
        ).tocsr()

        return Z, sigma

    def _compute_embedding_from_Z(self, Z_norm):
        if self.S is None or self.Vt is None:
            raise RuntimeError("S and Vt are not set. Fit LSCModel first.")

        V = self.Vt.T

        if hasattr(Z_norm, "dot"):
            U_approx = Z_norm.dot(V)
        else:
            U_approx = Z_norm @ V

        U_approx = U_approx / (self.S.reshape(1, -1) + 1e-12)
        U_eig = U_approx[:, 1:]

        row_norm = np.linalg.norm(U_eig, axis=1, keepdims=True)
        row_norm[row_norm == 0] = 1.0
        Y = U_eig / row_norm
        return Y

    def fit(self, X: np.ndarray):
        X = np.asarray(X, dtype=np.float64)

        with tqdm(
            total=5,
            desc="[LSC] Global clustering (fit)",
            unit="step",
            ncols=120,
            bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} '
                       '[{elapsed}<{remaining}, {rate_fmt}]'
        ) as pbar:
            logging.info("[LSC] Step 1/5: selecting landmarks ...")
            self.marks = self._select_landmarks(X)
            pbar.update(1)

            logging.info("[LSC] Step 2/5: building sparse Z ...")
            Z, sigma = self._build_Z(X, self.marks, sigma=None, r=self.r)
            self.sigma = sigma
            pbar.update(1)

            logging.info("[LSC] Step 3/5: column normalization of Z ...")
            feaSum = np.sqrt(np.array(Z.sum(axis=0)).ravel())
            feaSum[feaSum < 1e-12] = 1e-12
            inv_feaSum = 1.0 / feaSum
            self.feaSum = feaSum
            self.inv_feaSum = inv_feaSum

            Z_norm = Z.dot(diags(inv_feaSum, offsets=0))
            pbar.update(1)

            logging.info("[LSC] Step 4/5: SVD on Z_norm (dense) ...")
            n_components = self.k + 1

            Z_dense = Z_norm.toarray() if hasattr(Z_norm, "toarray") else np.asarray(Z_norm)

            U_full, S_full, Vt_full = np.linalg.svd(Z_dense, full_matrices=False)
            self.S = S_full[:n_components]
            self.Vt = Vt_full[:n_components, :]
            pbar.update(1)

            logging.info("[LSC] Step 5/5: compute embedding + KMeans ...")
            Y_train = self._compute_embedding_from_Z(Z_norm)

            km = KMeans(
                n_clusters=self.k,
                random_state=self.random_state,
                n_init=self.n_init,
                max_iter=self.max_iter,
                verbose=0
            )
            km.fit(Y_train)
            self.kmeans = km
            self.is_fitted = True
            pbar.update(1)

        labels = self.kmeans.labels_
        return labels

    def transform(self, X: np.ndarray):
        if not self.is_fitted:
            raise RuntimeError("LSCModel must be fit before calling transform().")

        X = np.asarray(X, dtype=np.float64)
        Z_new, _ = self._build_Z(X, self.marks, sigma=self.sigma, r=self.r)
        Z_new_norm = Z_new.dot(diags(self.inv_feaSum, offsets=0))

        Y_new = self._compute_embedding_from_Z(Z_new_norm)
        return Y_new

    def predict(self, X: np.ndarray):
        Y_new = self.transform(X)
        labels = self.kmeans.predict(Y_new)
        return labels


# ========================
# 2. 核心入口：run_fold_clustering
# ========================

def run_fold_clustering(args):

    root_dir = args.cluster_root

    global_res_dir = os.path.join(root_dir, f"val_14/global_clustering_fold_{args.k}")
    os.makedirs(global_res_dir, exist_ok=True)


    coord_pkl_train_path = os.path.join(global_res_dir, "coord_to_global_cluster_train.pkl")
    coord_pkl_val_path = os.path.join(global_res_dir, "coord_to_global_cluster_val.pkl")
    coord_pkl_test_path = os.path.join(global_res_dir, "coord_to_global_cluster_test.pkl")
    model_pkl_path = os.path.join(global_res_dir, "global_cluster_model_train_only.pkl")

    cache_paths = [
        model_pkl_path,
        coord_pkl_train_path,
        coord_pkl_val_path,
        coord_pkl_test_path,
    ]

    # Reuse cache only when all three coord maps are the WSI-aware NEW format.
    if all(os.path.exists(p) for p in cache_paths):
        try:
            coord_maps = []
            for p in (coord_pkl_train_path, coord_pkl_val_path, coord_pkl_test_path):
                with open(p, "rb") as f:
                    coord_maps.append(pickle.load(f))

            if all(_is_new_coord_map_format(m) for m in coord_maps):
                logging.info(
                    f"[Clustering] Fold-{args.k}: strict train/val/test global clustering cache exists; "
                    "skip recomputing."
                )
                args.coord_pkl_train_path = coord_pkl_train_path
                args.coord_pkl_val_path = coord_pkl_val_path
                args.coord_pkl_test_path = coord_pkl_test_path
                if not hasattr(args, "coord_pkl_path") or not args.coord_pkl_path:
                    args.coord_pkl_path = coord_pkl_train_path
                return

            logging.warning(
                f"[Clustering] Fold-{args.k}: old/invalid coord map cache detected. Recomputing."
            )
        except Exception as e:
            logging.warning(f"[Clustering] Cache check failed ({e}), will recompute.")

    logging.info(
        f"[Clustering] Fold-{args.k}: start STRICT train-only fitting under {global_res_dir}"
    )

    train_cases, val_cases, test_cases = read_split_cases(args)
    all_cases = train_cases | val_cases | test_cases

    # Guard against accidental overlap in split CSV.
    overlap_tv = train_cases & val_cases
    overlap_tt = train_cases & test_cases
    overlap_vt = val_cases & test_cases
    if overlap_tv or overlap_tt or overlap_vt:
        raise ValueError(
            f"Fold-{args.k} split leakage: train/val/test are not disjoint. "
            f"train∩val={len(overlap_tv)}, train∩test={len(overlap_tt)}, val∩test={len(overlap_vt)}"
        )

    logging.info(
        f"[Clustering] fold-{args.k}: train_cases={len(train_cases)}, "
        f"val_cases={len(val_cases)}, test_cases={len(test_cases)}"
    )

    wsi_files = find_wsi_feature_and_cluster_files(root_dir)

    centers_train, meta_train = [], []
    centers_val, meta_val = [], []
    centers_test, meta_test = [], []

    for rec in tqdm(wsi_files, desc="[Clustering] Collect local cluster centers"):
        wsi_id = str(rec["wsi"])
        if wsi_id not in all_cases:
            continue

        feature_csv = rec["feature_csv"]
        cluster_csv = rec["cluster_csv"]

        try:
            df_feat = pd.read_csv(feature_csv)
            df_clu = pd.read_csv(cluster_csv)
        except Exception as e:
            logging.warning(f"[Clustering] 读取失败, wsi={wsi_id}, error={e}, 跳过该 WSI")
            continue

        if (
            "File_Path" not in df_feat.columns
            or "File_Path" not in df_clu.columns
            or "Cluster" not in df_clu.columns
        ):
            logging.warning(f"[Clustering] 列不匹配, wsi={wsi_id}")
            continue

        df = pd.merge(df_clu, df_feat, on="File_Path", how="inner")
        if df.shape[0] == 0:
            logging.warning(f"[Clustering] wsi={wsi_id} merge 后没有样本, 跳过")
            continue

        feature_cols = [c for c in df.columns if c not in ["File_Path", "Cluster"]]

        for local_c, g in df.groupby("Cluster"):
            local_c = int(local_c)
            g_feats = g[feature_cols].values.astype(np.float32)
            if g_feats.shape[0] == 0:
                continue

            center_vec = g_feats.mean(axis=0)
            cluster_size = int(g_feats.shape[0])
            center_idx = int(g.index[0])

            meta = {
                "wsi": wsi_id,
                "local_cluster": local_c,
                "size": cluster_size,
                "center_idx": center_idx,
            }

            if wsi_id in train_cases:
                centers_train.append(center_vec)
                meta_train.append(meta)
            elif wsi_id in val_cases:
                centers_val.append(center_vec)
                meta_val.append(meta)
            elif wsi_id in test_cases:
                centers_test.append(center_vec)
                meta_test.append(meta)

    if len(centers_train) == 0:
        raise RuntimeError("[Clustering] No TRAIN local centers collected.")
    if len(centers_val) == 0:
        raise RuntimeError("[Clustering] No VAL local centers collected.")
    if len(centers_test) == 0:
        raise RuntimeError("[Clustering] No TEST local centers collected.")

    X_train = np.vstack(centers_train)
    logging.info(
        f"[Clustering] TRAIN-only local centers: {X_train.shape[0]}, feat_dim={X_train.shape[1]}"
    )

    global_pca_dim = getattr(args, "global_pca_dim", 32)

    # ----- FIT preprocessing on TRAIN only -----
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    if X_train_scaled.shape[1] > global_pca_dim:
        logging.info(f"[Clustering] PCA(TRAIN fit): {X_train_scaled.shape[1]} -> {global_pca_dim}")
        pca = PCA(n_components=global_pca_dim, random_state=args.seed)
        X_train_pca = pca.fit_transform(X_train_scaled)
    else:
        logging.info(f"[Clustering] feat_dim <= global_pca_dim={global_pca_dim}, 跳过 PCA")
        pca = None
        X_train_pca = X_train_scaled

    # ----- Size-aware replication: TRAIN only -----
    center_weight_mode = getattr(args, "center_weight_mode", "sqrt")
    center_weight_alpha = getattr(args, "center_weight_alpha", 0.5)
    center_weight_base_rep = getattr(args, "center_weight_base_rep", 1)
    center_weight_max_rep = getattr(args, "center_weight_max_rep", 10)
    center_weight_max_total = getattr(args, "center_weight_max_total", 200000)

    sizes_train = np.array([m["size"] for m in meta_train], dtype=np.float64)
    rep = _compute_replication_factors(
        sizes_train,
        mode=center_weight_mode,
        alpha=center_weight_alpha,
        base_rep=center_weight_base_rep,
        max_rep=center_weight_max_rep,
        min_rep=1,
        max_total_rep=center_weight_max_total,
    )

    if center_weight_mode.lower() != "none":
        logging.info(
            f"[Clustering] Size-aware weighting enabled on TRAIN only: mode={center_weight_mode}, "
            f"alpha={center_weight_alpha}, base_rep={center_weight_base_rep}, "
            f"max_rep={center_weight_max_rep}, total_rep={int(rep.sum())}, orig={len(meta_train)}"
        )
    else:
        logging.info("[Clustering] Size-aware weighting disabled (mode=none).")

    rep_to_orig = np.repeat(np.arange(X_train_pca.shape[0], dtype=np.int64), rep)
    X_train_pca_rep = X_train_pca[rep_to_orig]

    k_global = args.num_clusters
    p_landmarks = getattr(args, "p_landmarks", 128)
    r_neighbors = getattr(args, "r_neighbors", 5)
    lsc_mode = getattr(args, "lsc_mode", "random")

    logging.info(
        f"[Clustering] LSC TRAIN fit: K={k_global}, p={p_landmarks}, r={r_neighbors}, mode={lsc_mode}"
    )

    lsc = LSCModel(
        k=k_global,
        p=p_landmarks,
        r=r_neighbors,
        mode=lsc_mode,
        random_state=args.seed,
        max_iter=100,
        n_init=10,
    )

    train_labels_rep = lsc.fit(X_train_pca_rep)
    train_labels = _majority_vote_labels(
        rep_labels=train_labels_rep,
        rep_to_orig=rep_to_orig,
        n_orig=len(meta_train),
    )
    for i, lab in enumerate(train_labels):
        meta_train[i]["global_cluster"] = int(lab)

    model_state = {
        "fit_scope": "train_only",
        "fold": int(args.k),
        "scaler": scaler,
        "pca": pca,
        "lsc": lsc,
        "weighting": {
            "center_weight_mode": center_weight_mode,
            "center_weight_alpha": center_weight_alpha,
            "center_weight_base_rep": center_weight_base_rep,
            "center_weight_max_rep": center_weight_max_rep,
            "center_weight_max_total": center_weight_max_total,
        },
    }
    with open(model_pkl_path, "wb") as f:
        pickle.dump(model_state, f)
    logging.info(f"[Clustering] Saved TRAIN-only global clustering model to: {model_pkl_path}")

    def assign_heldout(centers, meta, split_name):
        if len(centers) == 0:
            logging.info(f"[Clustering] No {split_name} local centers to assign.")
            return

        X = np.vstack(centers).astype(np.float64)
        X_scaled = scaler.transform(X)
        X_pca = pca.transform(X_scaled) if pca is not None else X_scaled
        labels = lsc.predict(X_pca)
        for i, lab in enumerate(labels):
            meta[i]["global_cluster"] = int(lab)
        logging.info(
            f"[Clustering] Assigned frozen TRAIN vocabulary to {len(centers)} {split_name} local centers."
        )

    # VAL and TEST are held-out mappings only; they never refit scaler/PCA/LSC.
    assign_heldout(centers_val, meta_val, "VAL")
    assign_heldout(centers_test, meta_test, "TEST")

    meta_records = meta_train + meta_val + meta_test
    df_local = pd.DataFrame(meta_records)[
        ["wsi", "local_cluster", "size", "center_idx", "global_cluster"]
    ]
    local_csv_path = os.path.join(global_res_dir, "local_cluster_to_global_train_val_test.csv")
    df_local.to_csv(local_csv_path, index=False)
    logging.info(
        f"[Clustering] Saved local_cluster_to_global to: {local_csv_path}, shape={df_local.shape}"
    )

    logging.info("[Clustering] Building patch_to_global_cluster.csv for this fold only ...")

    local2global = {}
    for _, row in df_local.iterrows():
        key = (str(row["wsi"]), int(row["local_cluster"]))
        local2global[key] = int(row["global_cluster"])

    patch_records = []
    for rec in tqdm(wsi_files, desc="[Clustering] Build patch->global mapping"):
        wsi_id = str(rec["wsi"])
        if wsi_id not in all_cases:
            continue

        cluster_csv = rec["cluster_csv"]
        try:
            df_clu = pd.read_csv(cluster_csv)
        except Exception as e:
            logging.warning(f"[Clustering] 读取失败, wsi={wsi_id}, error={e}")
            continue

        if "File_Path" not in df_clu.columns or "Cluster" not in df_clu.columns:
            logging.warning(f"[Clustering] 列不匹配 (patch-level), wsi={wsi_id}")
            continue

        for _, row in df_clu.iterrows():
            fp = row["File_Path"]
            lc = int(row["Cluster"])
            key = (wsi_id, lc)
            gc = local2global[key]

            patch_records.append({
                "wsi": wsi_id,
                "File_Path": fp,
                "local_cluster": lc,
                "global_cluster": gc,
            })

    df_patch = pd.DataFrame(patch_records)
    patch_csv_path = os.path.join(global_res_dir, "patch_to_global_cluster_train_val_test.csv")
    df_patch.to_csv(patch_csv_path, index=False)
    logging.info(
        f"[Clustering] Saved patch_to_global_cluster to: {patch_csv_path}, shape={df_patch.shape}"
    )

    logging.info("[Clustering] Building separate WSI-aware TRAIN / VAL / TEST coord maps ...")

    coord_maps = {
        "train": {},
        "val": {},
        "test": {},
    }

    for _, row in df_patch.iterrows():
        wsi = str(row["wsi"])
        fp = row["File_Path"]
        gc = int(row["global_cluster"])

        xy = parse_xy_from_filepath(fp)
        if xy is None:
            continue
        xy_key = (int(xy[0]), int(xy[1]))

        if wsi in train_cases:
            split_name = "train"
        elif wsi in val_cases:
            split_name = "val"
        elif wsi in test_cases:
            split_name = "test"
        else:
            continue

        coord_maps[split_name].setdefault(wsi, {})[xy_key] = gc

    for split_name, path in (
        ("train", coord_pkl_train_path),
        ("val", coord_pkl_val_path),
        ("test", coord_pkl_test_path),
    ):
        with open(path, "wb") as f:
            pickle.dump(coord_maps[split_name], f)
        logging.info(
            f"[Clustering] Saved {split_name.upper()} coord map to: {path}, "
            f"n_wsi={len(coord_maps[split_name])}, "
            f"total_coords={sum(len(v) for v in coord_maps[split_name].values())}"
        )

    args.coord_pkl_train_path = coord_pkl_train_path
    args.coord_pkl_val_path = coord_pkl_val_path
    args.coord_pkl_test_path = coord_pkl_test_path
    args.coord_pkl_path = coord_pkl_train_path

