# ===== DMIN_only_embedding.py =====
import os
import sys
import logging
import pickle
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import h5py 
import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler, SequentialSampler, Dataset
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

sys.path.append('..')


def _is_new_coord_map_format(obj) -> bool:
    if not isinstance(obj, dict):
        return False
    if len(obj) == 0:
        return True
    any_val = next(iter(obj.values()))
    return isinstance(any_val, dict)


class WSIDataset(Dataset):
    """
    Cluster-aware WSI bag dataset (Only Embedding uses real cluster ids).
    Supports:
      - NEW: coord_map[wsi][(x,y)] = global_cluster
      - OLD: coord_map[(x,y)] = global_cluster
    Unknown cluster id = num_clusters
    """

    def __init__(self,
                 args,
                 wsi_labels,
                 infold_cases,
                 phase=None,
                 target_cluster=None,
                 coord_pkl_path=None):
        self.args = args
        self.phase = phase
        self.target_cluster = target_cluster

        self.infold_features = []
        self.infold_labels = []
        self.infold_cluster_ids = []

        self.unknown_cluster_id = int(self.args.num_clusters)

        if coord_pkl_path is None:
            if not hasattr(self.args, "coord_pkl_path") or not self.args.coord_pkl_path:
                raise ValueError("WSIDataset needs coord_pkl_path (or args.coord_pkl_path).")
            coord_map_path = self.args.coord_pkl_path
        else:
            coord_map_path = coord_pkl_path

        print(f"[DMIN DEBUG] Open coord map: {coord_map_path}")
        logging.info(f"[WSIDataset-{phase}] coord map: {coord_map_path}")

        if not os.path.exists(coord_map_path):
            raise FileNotFoundError(f"coord_to_global_cluster.pkl not found: {coord_map_path}")

        with open(coord_map_path, 'rb') as f:
            self.coord_to_cluster = pickle.load(f)

        self.coord_map_is_new = _is_new_coord_map_format(self.coord_to_cluster)
        fmt = "NEW dict[wsi][(x,y)]" if self.coord_map_is_new else "OLD dict[(x,y)]"
        print(f"[DMIN DEBUG] coord map format: {fmt}")
        logging.info(f"[WSIDataset-{phase}] coord map format: {fmt}")

        for case_id, slide_id, label in wsi_labels:
            case_id = str(case_id)
            slide_id = str(slide_id)

            if case_id not in infold_cases:
                continue

            h5_path = os.path.join(args.feature_dir, f"{slide_id}.h5")
            if not os.path.exists(h5_path):
                continue

            with h5py.File(h5_path, 'r') as f:
                feats = f['features'][:]  # (N, D)
                coords = f['coords'][:]   # (N, 2)

            cluster_ids = np.zeros((coords.shape[0]), dtype=np.int64)

            if self.coord_map_is_new:
                wsi_map = self.coord_to_cluster.get(slide_id, None)
                if wsi_map is None:
                    wsi_map = self.coord_to_cluster.get(case_id, None)
                if wsi_map is None:
                    wsi_map = {}

                for i, coord in enumerate(coords):
                    xy = (int(coord[0]), int(coord[1]))
                    cluster_ids[i] = int(wsi_map.get(xy, self.unknown_cluster_id))
            else:
                for i, coord in enumerate(coords):
                    xy = (int(coord[0]), int(coord[1]))
                    cluster_ids[i] = int(self.coord_to_cluster.get(xy, self.unknown_cluster_id))

            # clamp
            cluster_ids[(cluster_ids < 0) | (cluster_ids > self.unknown_cluster_id)] = self.unknown_cluster_id

            # optional filter by target_cluster
            if self.target_cluster is not None:
                mask = (cluster_ids == self.target_cluster)
                if mask.sum() == 0:
                    continue
                feats = feats[mask]
                cluster_ids = cluster_ids[mask]
                if feats.shape[0] < 10:
                    continue

            feats_tensor = torch.from_numpy(feats.astype(np.float32))
            cluster_ids_tensor = torch.from_numpy(cluster_ids.astype(np.int64))

            if self.phase == 'train':
                perm = torch.randperm(feats_tensor.shape[0])
                feats_tensor = feats_tensor[perm]
                cluster_ids_tensor = cluster_ids_tensor[perm]

            self.infold_features.append(feats_tensor)
            self.infold_labels.append(int(label))
            self.infold_cluster_ids.append(cluster_ids_tensor)

        print(f"[DMIN DEBUG] Loaded {len(self.infold_features)} WSIs for phase={phase}")


    def __len__(self):
        return len(self.infold_features)

    def __getitem__(self, idx):
        feats = self.infold_features[idx]
        label = torch.tensor(self.infold_labels[idx], dtype=torch.long)
        cluster_ids = self.infold_cluster_ids[idx]
        return feats, label, cluster_ids


class DMINMIL:
    METRIC_NAMES = ["loss", "auc", "acc", "precision", "recall", "f1"]

    def __init__(self, args):
        self.args = args

        if args.dataset in ['Camelyon16', 'TCGA-NSCLC', 'TCGA-BRCA', 'TCGA-RCC', 'CustomDataset']:
            self.train_loader, self.val_loader, self.test_loader = self.init_data_wsi()
        else:
            raise NotImplementedError

        # Only Embedding: emb ON, hist OFF
        self.args.use_cluster_emb = True
        self.args.use_cluster_hist = False
        self.args.cluster_id_dropout = 0.0

        self.model = self.init_model()
        if self.model is None:
            raise RuntimeError("init_model() returned None.")

        print(self.model)
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total trainable parameters: {total_params / 1e6:.3f} M")

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=args.lr, weight_decay=args.wd
        )

        self.loss = torch.nn.CrossEntropyLoss(
            reduction='mean',
            label_smoothing=getattr(self.args, "label_smoothing", 0.0)
        )

        # Model selection is based ONLY on validation AUC.
        self.best_val_auc = -np.inf
        self.best_val_metrics = None
        self.best_epoch = None
        self.final_test_metrics = None
        self.ckpt_name = os.path.join(self.args.ckpt_dir, 'best_val_auc.pth')

        self.step = 0
        self.warmup_steps = 100

    def read_wsi_label(self):
        data = pd.read_csv(self.args.label_csv)

        wsi_labels = []
        for i in range(len(data)):
            if self.args.dataset == 'CustomDataset':
                case_id = str(data.loc[i, "ID"])
                slide_id = str(data.loc[i, "ID"])
                label = int(data.loc[i, "label"])
            else:
                case_id = str(data.loc[i, "case_id"])
                slide_id = str(data.loc[i, "slide_id"])
                label = int(data.loc[i, "label"])
            wsi_labels.append([case_id, slide_id, label])
        return wsi_labels

    def read_in_fold_cases(self, fold_csv):
        data = pd.read_csv(fold_csv)
        required = {"train", "val", "test"}
        missing = required.difference(data.columns)
        if missing:
            raise ValueError(
                f"Split CSV must contain train/val/test columns. Missing: {sorted(missing)}; file={fold_csv}"
            )

        train_cases, valid_cases, test_cases = [], [], []

        for i in range(len(data)):
            train_val = data.loc[i, 'train']
            if pd.notna(train_val):
                train_cases.append(str(int(train_val)))

            val_val = data.loc[i, 'val']
            if pd.notna(val_val):
                valid_cases.append(str(int(val_val)))

            test_val = data.loc[i, 'test']
            if pd.notna(test_val):
                test_cases.append(str(int(test_val)))

        # Preserve file order while removing duplicates.
        train_cases = list(dict.fromkeys(train_cases))
        valid_cases = list(dict.fromkeys(valid_cases))
        test_cases = list(dict.fromkeys(test_cases))

        st, sv, ss = set(train_cases), set(valid_cases), set(test_cases)
        if (st & sv) or (st & ss) or (sv & ss):
            raise ValueError(
                f"Split leakage in {fold_csv}: "
                f"train∩val={len(st & sv)}, train∩test={len(st & ss)}, val∩test={len(sv & ss)}"
            )

        if len(train_cases) == 0 or len(valid_cases) == 0 or len(test_cases) == 0:
            raise ValueError(
                f"Empty split in {fold_csv}: train={len(train_cases)}, "
                f"val={len(valid_cases)}, test={len(test_cases)}"
            )

        return train_cases, valid_cases, test_cases

    def make_weights_for_balanced_classes_split(self, data_set):
        N = float(len(data_set))
        classes = {}
        for label in data_set.infold_labels:
            classes[label] = classes.get(label, 0) + 1

        weight = [0.0] * int(N)
        for idx in range(len(data_set)):
            y = data_set.infold_labels[idx]
            weight[idx] = N / classes[y]
        return torch.DoubleTensor(weight)

    def init_data_wsi(self):
        wsi_labels = self.read_wsi_label()
        split_csv = os.path.join(self.args.split_dir, f'splits_{self.args.k}.csv')

        train_cases, valid_cases, test_cases = self.read_in_fold_cases(split_csv)

        coord_pkl_train = getattr(
            self.args, "coord_pkl_train_path", getattr(self.args, "coord_pkl_path", None)
        )
        coord_pkl_val = getattr(
            self.args, "coord_pkl_val_path", getattr(self.args, "coord_pkl_path", None)
        )
        coord_pkl_test = getattr(
            self.args, "coord_pkl_test_path", getattr(self.args, "coord_pkl_path", None)
        )

        train_set = WSIDataset(
            self.args, wsi_labels, train_cases, phase='train',
            target_cluster=None, coord_pkl_path=coord_pkl_train
        )
        val_set = WSIDataset(
            self.args, wsi_labels, valid_cases, phase='val',
            target_cluster=None, coord_pkl_path=coord_pkl_val
        )
        test_set = WSIDataset(
            self.args, wsi_labels, test_cases, phase='test',
            target_cluster=None, coord_pkl_path=coord_pkl_test
        )

        if len(train_set) == 0:
            raise ValueError("Train set is empty. Check case_id matching or feature paths.")
        if len(val_set) == 0:
            raise ValueError("Validation set is empty. Check case_id matching or feature paths.")
        if len(test_set) == 0:
            raise ValueError("Test set is empty. Check case_id matching or feature paths.")

        logging.info(
            f"[Fold {self.args.k}] loaded WSI bags: "
            f"train={len(train_set)}, val={len(val_set)}, test={len(test_set)}"
        )

        # Class balancing applies ONLY to training.
        weights = self.make_weights_for_balanced_classes_split(train_set)
        train_loader = DataLoader(
            train_set,
            batch_size=1,
            sampler=WeightedRandomSampler(weights, len(weights), replacement=True),
        )
        val_loader = DataLoader(
            val_set, batch_size=1, sampler=SequentialSampler(val_set)
        )
        test_loader = DataLoader(
            test_set, batch_size=1, sampler=SequentialSampler(test_set)
        )
        return train_loader, val_loader, test_loader

    def init_model(self):
        import inspect
        from models.hdmil import HierarchicalMILModel

        mppc = getattr(self.args, "max_patches_per_cluster", None)
        if mppc is not None and mppc <= 0:
            mppc = None

        kwargs = dict(
            I=self.args.feature_dim,
            num_clusters=self.args.num_clusters,
            n_classes=self.args.n_classes,
            dropout=True,

            cluster_emb_dim=getattr(self.args, "cluster_emb_dim", 8),
            use_cluster_emb=True,
            max_patches_per_cluster=mppc,
            slide_dropout=getattr(self.args, "slide_dropout", 0.0),

            use_cluster_hist=False,
            cluster_hist_hidden=getattr(self.args, "cluster_hist_hidden", 128),
            cluster_id_dropout=0.0,
        )

        sig = inspect.signature(HierarchicalMILModel.__init__)
        allowed = set(sig.parameters.keys())
        filtered = {k: v for k, v in kwargs.items() if k in allowed}

        if "use_cluster_emb" not in allowed:
            logging.warning(
                "[Only-Embedding] Your HierarchicalMILModel has no `use_cluster_emb`. "
                "This run will NOT actually use cluster embedding."
            )

        model = HierarchicalMILModel(**filtered).to(self.args.device)
        return model

    def _save_best_checkpoint(self, epoch, val_metrics):
        checkpoint = {
            "epoch": int(epoch),
            "best_val_auc": float(val_metrics[1]),
            "val_metrics": [float(x) for x in val_metrics],
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        torch.save(checkpoint, self.ckpt_name)
        logging.info(
            f"[Fold {self.args.k}] SAVE BEST: epoch={epoch}, "
            f"val_auc={val_metrics[1]:.6f} -> {self.ckpt_name}"
        )

    def _load_best_checkpoint(self):
        if not os.path.exists(self.ckpt_name):
            raise FileNotFoundError(f"Checkpoint not found: {self.ckpt_name}")

        checkpoint = torch.load(self.ckpt_name, map_location=self.args.device)

        # New format.
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.best_epoch = checkpoint.get("epoch", self.best_epoch)
            self.best_val_auc = checkpoint.get("best_val_auc", self.best_val_auc)
            self.best_val_metrics = checkpoint.get("val_metrics", self.best_val_metrics)
            return checkpoint

        # Backward-compatible fallback for raw state_dict checkpoints.
        self.model.load_state_dict(checkpoint)
        return {"model_state_dict": checkpoint}

    def train(self):
        history = []

        for epoch in range(1, self.args.n_epochs + 1):
            self.model.train()
            train_loss_sum = 0.0
            train_batches = 0

            for fea, label, cluster_ids in tqdm(
                self.train_loader, desc=f"Fold {self.args.k} Train E{epoch}"
            ):
                self.step += 1
                fea = fea.to(self.args.device)
                label = label.to(self.args.device)
                cluster_ids = cluster_ids.to(self.args.device)

                self.optimizer.zero_grad()

                if self.step < self.warmup_steps:
                    lr_scale = self.step / self.warmup_steps
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = lr_scale * self.args.lr
                elif self.step == self.warmup_steps:
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.args.lr

                loss = self.train_inference(fea, label, cluster_ids)
                if torch.isnan(loss):
                    logging.warning(
                        f"[Fold {self.args.k}] NaN train loss at epoch={epoch}; skip batch."
                    )
                    continue

                train_loss_sum += loss.item()
                train_batches += 1
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            if train_batches == 0:
                raise RuntimeError(f"Fold {self.args.k} epoch {epoch}: no valid training batches.")
            avg_train_loss = train_loss_sum / train_batches

            # IMPORTANT: model selection uses VALIDATION only.
            val_metrics = self.evaluate_loader(self.val_loader, split_name='val')
            val_loss, val_auc, val_acc, val_precision, val_recall, val_f1 = val_metrics

            history.append({
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "val_loss": val_loss,
                "val_auc": val_auc,
                "val_acc": val_acc,
                "val_precision": val_precision,
                "val_recall": val_recall,
                "val_f1": val_f1,
            })

            logging.info(
                f"[Fold {self.args.k}] Epoch {epoch:03d}/{self.args.n_epochs}: "
                f"train_loss={avg_train_loss:.6f}, val_loss={val_loss:.6f}, "
                f"val_auc={val_auc:.6f}, val_acc={val_acc:.6f}, val_f1={val_f1:.6f}"
            )

            if np.isfinite(val_auc) and val_auc > self.best_val_auc:
                self.best_val_auc = float(val_auc)
                self.best_val_metrics = [float(x) for x in val_metrics]
                self.best_epoch = int(epoch)
                self._save_best_checkpoint(epoch, val_metrics)

        history_path = os.path.join(
            self.args.log_dir, f"epoch_history_fold{self.args.k}.csv"
        )
        pd.DataFrame(history).to_csv(history_path, index=False)
        logging.info(f"[Fold {self.args.k}] Saved epoch history: {history_path}")

        if self.best_epoch is None:
            raise RuntimeError(
                f"Fold {self.args.k}: no checkpoint was selected because validation AUC was never finite. "
                "Check that the validation split contains both classes."
            )

        # TEST is evaluated exactly once after training, using the checkpoint chosen by VAL AUC.
        self._load_best_checkpoint()
        self.final_test_metrics = [
            float(x) for x in self.evaluate_loader(self.test_loader, split_name='test')
        ]

        logging.info(
            f"[Fold {self.args.k}] FINAL: best_epoch={self.best_epoch}, "
            f"best_val_auc={self.best_val_auc:.6f}, test_auc={self.final_test_metrics[1]:.6f}"
        )

        return {
            "best_epoch": int(self.best_epoch),
            "best_val_metrics": [float(x) for x in self.best_val_metrics],
            "test_metrics": self.final_test_metrics,
            "checkpoint": self.ckpt_name,
        }

    def evaluate_loader(self, loader, split_name='val'):
        if loader is None or len(loader) == 0:
            raise ValueError(f"{split_name} loader is empty.")

        loss_sum = 0.0
        n_batches = 0

        # Disable per-cluster patch subsampling during val/test for deterministic evaluation.
        has_mppc = hasattr(self.model, "max_patches_per_cluster")
        old_mppc = None
        if has_mppc:
            old_mppc = self.model.max_patches_per_cluster
            self.model.max_patches_per_cluster = None

        self.model.eval()
        labels, probs = [], []

        try:
            with torch.no_grad():
                for fea, label, cluster_ids in tqdm(
                    loader, desc=f"Fold {self.args.k} {split_name.upper()}"
                ):
                    fea = fea.to(self.args.device)
                    label = label.to(self.args.device)
                    cluster_ids = cluster_ids.to(self.args.device)

                    loss, y_prob = self.test_inference(fea, label, cluster_ids)
                    labels.append(label.detach().cpu().numpy())
                    probs.append(y_prob.detach().cpu().numpy())
                    loss_sum += loss.item()
                    n_batches += 1
        finally:
            if has_mppc:
                self.model.max_patches_per_cluster = old_mppc

        if n_batches == 0:
            raise RuntimeError(f"No samples evaluated for split={split_name}.")

        avg_loss = loss_sum / n_batches
        labels = np.concatenate(labels, axis=0)
        probs = np.concatenate(probs, axis=0)

        auc = self.cal_AUC(probs, labels, self.args.n_classes)
        acc, precision, recall, f1 = self.cal_ACC(probs, labels, self.args.n_classes)

        return avg_loss, auc, acc, precision, recall, f1

    # Compatibility wrappers.
    def evaluate_on_val(self, epoch=None):
        return self.evaluate_loader(self.val_loader, split_name='val')

    def evaluate_on_test(self, epoch=None):
        return self.evaluate_loader(self.test_loader, split_name='test')

    def validate(self):
        self._load_best_checkpoint()
        return self.evaluate_loader(self.val_loader, split_name='val')

    def test(self):
        self._load_best_checkpoint()
        self.model.eval()
        return self.evaluate_loader(self.test_loader, split_name='test')

    def train_inference(self, fea, label, cluster_ids):
        bag_logit = self.model(fea, cluster_ids)
        loss = self.loss(bag_logit, label)
        return loss

    def test_inference(self, fea, label, cluster_ids):
        bag_logit = self.model(fea, cluster_ids)
        loss = self.loss(bag_logit, label)
        y_prob = F.softmax(bag_logit, dim=1)
        return loss, y_prob

    def cal_AUC(self, probs, labels, nclasses):
        try:
            if nclasses == 2:
                return float(roc_auc_score(labels, probs[:, 1]))
            return float(roc_auc_score(labels, probs, multi_class='ovr'))
        except ValueError as e:
            logging.warning(f"AUC is undefined for this split: {e}")
            return float('nan')

    def cal_ACC(self, probs, labels, nclasses):
        pred_hat = np.argmax(probs, 1)
        labels = labels.astype(np.int32)

        acc = accuracy_score(labels, pred_hat)
        if nclasses == 2:
            precision = precision_score(labels, pred_hat, average='binary', zero_division=0)
            recall = recall_score(labels, pred_hat, average='binary', zero_division=0)
            f1 = f1_score(labels, pred_hat, average='binary', zero_division=0)
        else:
            precision = precision_score(labels, pred_hat, average='macro', zero_division=0)
            recall = recall_score(labels, pred_hat, average='macro', zero_division=0)
            f1 = f1_score(labels, pred_hat, average='macro', zero_division=0)

        return float(acc), float(precision), float(recall), float(f1)


