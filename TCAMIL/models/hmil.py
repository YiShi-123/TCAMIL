# models/hdmil.py
import logging
import torch
import torch.nn as nn


def initialize_weights(module: nn.Module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()

class Attn_Net_Gated(nn.Module):
    def __init__(self, L: int = 512, D: int = 256, dropout: bool = False, n_classes: int = 1):
        super().__init__()
        a = [nn.Linear(L, D), nn.Tanh()]
        b = [nn.Linear(L, D), nn.Sigmoid()]
        if dropout:
            a.append(nn.Dropout(0.25))
            b.append(nn.Dropout(0.25))
        self.attention_a = nn.Sequential(*a)
        self.attention_b = nn.Sequential(*b)
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x: torch.Tensor):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = self.attention_c(a * b)
        return A, x

class HierarchicalMILModel(nn.Module):
    """
    Cluster-aware hierarchical MIL with cluster id embedding.
    unknown_cluster_id is fixed to num_clusters, and embedding size is num_clusters + 1.
    """

    def __init__(
        self,
        I: int = 512,
        num_clusters: int = 10,
        n_classes: int = 2,
        dropout: bool = True,
        cluster_emb_dim: int = 8,
        use_cluster_emb: bool = True,
        max_patches_per_cluster=None,
        slide_dropout: float = 0.0,
    ):
        super().__init__()

        self.num_clusters = int(num_clusters)
        self.n_classes = int(n_classes)
        self.input_dim = int(I)

        self.unknown_cluster_id = int(num_clusters)
        self.max_patches_per_cluster = max_patches_per_cluster

        self.cluster_emb_dim = int(cluster_emb_dim)
        self.use_cluster_emb = bool(use_cluster_emb)

        self.cluster_embedding = nn.Embedding(
            num_embeddings=self.num_clusters + 1,
            embedding_dim=self.cluster_emb_dim,
        )
        nn.init.normal_(self.cluster_embedding.weight, mean=0.0, std=0.02)

        self.cluster_fusion = nn.Sequential(
            nn.Linear(self.input_dim + self.cluster_emb_dim, self.input_dim),
            nn.ReLU(inplace=True),
        )

        self.intra_attn = Attn_Net_Gated(L=self.input_dim, D=256, dropout=dropout, n_classes=1)
        self.intra_fc = nn.Sequential(nn.Linear(self.input_dim, 512), nn.ReLU(inplace=True))

        self.inter_attn = Attn_Net_Gated(L=512, D=256, dropout=dropout, n_classes=1)
        self.inter_fc = nn.Sequential(nn.Linear(512, 256), nn.ReLU(inplace=True))

        if slide_dropout and slide_dropout > 0:
            self.classifier = nn.Sequential(nn.Dropout(slide_dropout), nn.Linear(256, self.n_classes))
        else:
            self.classifier = nn.Linear(256, self.n_classes)

        initialize_weights(self)

    def _fuse_cluster_embedding(self, h: torch.Tensor, cluster_ids: torch.Tensor):
        if (not self.use_cluster_emb) or (cluster_ids is None):
            return h

        cluster_ids = torch.clamp(cluster_ids, 0, self.unknown_cluster_id)

        B, N, I = h.shape
        emb = self.cluster_embedding(cluster_ids)  # [B,N,E]
        x = torch.cat([h, emb], dim=-1)            # [B,N,I+E]
        x = x.view(B * N, I + self.cluster_emb_dim)
        x = self.cluster_fusion(x).view(B, N, I)
        return x

    def forward(self, h: torch.Tensor, cluster_ids: torch.Tensor, return_attn: bool = False):
        assert len(h.shape) == 3, f"Expect h as [B,N,I], got {h.shape}"
        B, N, I = h.shape
        device = h.device

        cluster_ids = torch.clamp(cluster_ids, 0, self.unknown_cluster_id)
        h = self._fuse_cluster_embedding(h, cluster_ids)

        slide_feats = []
        attn_infos = []  # list of dict per slide

        for b in range(B):
            h_b = h[b]
            clusters_b = cluster_ids[b]

            cluster_feats = []
            cids_kept = []

            unique_cids = torch.unique(clusters_b)

            for cid in unique_cids:
                mask = (clusters_b == cid)
                num_patches = mask.sum().item()
                if num_patches == 0:
                    continue

                if (self.max_patches_per_cluster is not None) and (num_patches > self.max_patches_per_cluster):
                    idx = torch.nonzero(mask, as_tuple=False).squeeze(1)
                    perm = torch.randperm(num_patches, device=device)[: self.max_patches_per_cluster]
                    idx = idx[perm]
                    sub_mask = torch.zeros_like(mask, dtype=torch.bool)
                    sub_mask[idx] = True
                    mask = sub_mask

                h_c = h_b[mask]  # [Nc,I]
                A_c, h_c = self.intra_attn(h_c)  # [Nc,1]
                A_c = torch.softmax(A_c, dim=0)
                c_feat = torch.mm(A_c.T, h_c)  # [1,I]
                c_feat = self.intra_fc(c_feat)  # [1,512]

                cluster_feats.append(c_feat)
                cids_kept.append(int(cid.item()))

            if len(cluster_feats) == 0:
                slide_feat = torch.zeros(1, 256, device=device)
                slide_feats.append(slide_feat)
                if return_attn:
                    attn_infos.append({"cids": [], "weights": []})
                continue

            cluster_feats = torch.cat(cluster_feats, dim=0)  # [K',512]

            A_slide_raw, h_slide = self.inter_attn(cluster_feats)  # [K',1]
            A_slide = torch.softmax(A_slide_raw, dim=0).squeeze(1)  # [K']

            slide_feat = torch.mm(A_slide.unsqueeze(0), h_slide)  # [1,512]
            slide_feat = self.inter_fc(slide_feat)  # [1,256]
            slide_feats.append(slide_feat)

            if return_attn:
                attn_infos.append({
                    "cids": cids_kept,  # python list[int], len=K'
                    "weights": A_slide.detach().cpu()  # tensor[K']
                })

        slide_feats = torch.cat(slide_feats, dim=0)  # [B,256]
        logits = self.classifier(slide_feats)  # [B,n_classes]

        if return_attn:
            return logits, attn_infos
        return logits

__all__ = ["HierarchicalMILModel", "Attn_Net_Gated"]
