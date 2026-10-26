import json
import os
import sys
from torch import nn
from mmfewshot.detection.models.utils.aggregation_layer import AGGREGATORS
from mmcv.runner import BaseModule

import math
import numpy as np
import torch
import torch.nn.functional as F


@AGGREGATORS.register_module()
class PrototypesDistillation(BaseModule):
    def __init__(self,
                 num_queries,
                 dim,
                 num_base_cls=15,
                 num_novel=0,
                 language_conditioned_queries=None,
                 uncertainty_aware_prototypes=None,
                 caption_guided_part_prototypes=None):
        super().__init__()
        self.num_queries = num_queries
        self.num_base_cls = num_base_cls
        self.num_novel = num_novel
        self.dim = dim

        k_dim = dim // 4
        self.k_dim = k_dim

        self.query_embed = nn.Embedding(num_queries*num_base_cls, k_dim)
        self.duplicated = False
        if self.num_novel > 0:
            self.query_embed_novel = nn.Embedding(num_queries * num_novel, k_dim)
        self.w_qk = nn.Linear(dim, k_dim, bias=False)

        # Optional Language-Conditioned Feature Queries (LC-FQ).
        cfg = language_conditioned_queries or {}
        self.lcq_enabled = bool(cfg.get('enabled', False))
        self.lcq_text_dim = int(cfg.get('text_dim', 512))
        self.lcq_residual = bool(cfg.get('residual', True))
        self.lcq_init_scale = float(cfg.get('init_scale', 1.0))
        self.lcq_novel_init = cfg.get('novel_init', 'duplicate')
        if self.lcq_novel_init not in ('duplicate', 'language', 'hybrid'):
            raise ValueError(
                f'Unsupported novel_init={self.lcq_novel_init}. '
                'Expected one of: duplicate, language, hybrid.')

        if self.lcq_enabled:
            base_text_feats = self._load_text_features(
                cfg.get('base_text_features', None), self.num_base_cls,
                'base_text_features')
            self.register_buffer('base_text_features', base_text_feats)

            if self.num_novel > 0:
                novel_text_feats = self._load_text_features(
                    cfg.get('novel_text_features', None), self.num_novel,
                    'novel_text_features')
                self.register_buffer('novel_text_features', novel_text_feats)
            else:
                self.register_buffer(
                    'novel_text_features',
                    torch.empty((0, self.lcq_text_dim), dtype=torch.float32))

            hidden_dim = int(cfg.get('hidden_dim', self.lcq_text_dim))
            self.query_generator = nn.Sequential(
                nn.Linear(self.lcq_text_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, self.num_queries * self.k_dim))

            if self.lcq_residual:
                self.lcq_residual_scale = nn.Parameter(torch.tensor(1.0))

        uncertainty_cfg = uncertainty_aware_prototypes or {}
        self.uapd_enabled = bool(uncertainty_cfg.get('enabled', False))
        self.uapd_min_variance = float(uncertainty_cfg.get('min_variance', 1e-4))
        regularization_cfg = uncertainty_cfg.get('regularization', {})
        self.uapd_regularization_enabled = bool(
            regularization_cfg.get('enabled', self.uapd_enabled))
        self.uapd_regularization_weight = float(
            regularization_cfg.get('loss_weight', 0.05))
        self.uapd_prior_variance = float(
            regularization_cfg.get('prior_variance', 1.0))
        if self.uapd_enabled:
            uncertainty_hidden_dim = int(
                uncertainty_cfg.get('hidden_dim', max(self.dim // 2, self.k_dim)))
            self.prototype_var_head = nn.Sequential(
                nn.Linear(self.dim, uncertainty_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(uncertainty_hidden_dim, self.dim))

        cgpap_cfg = caption_guided_part_prototypes or {}
        self.cgpap_enabled = bool(cgpap_cfg.get('enabled', False))
        self.cgpap_token_dim = int(cgpap_cfg.get('token_dim', 512))
        self.cgpap_query_scale = float(cgpap_cfg.get('query_scale', 1.0))
        self.cgpap_alignment_weight = float(cgpap_cfg.get('alignment_weight', 0.1))
        self.cgpap_alignment_type = cgpap_cfg.get('alignment_type', 'cosine')
        if self.cgpap_alignment_type not in ('cosine', 'smooth_l1'):
            raise ValueError(
                f'Unsupported alignment_type={self.cgpap_alignment_type}. '
                'Expected one of: cosine, smooth_l1.')

        if self.cgpap_enabled:
            base_caption_tokens = self._load_caption_tokens(
                cgpap_cfg.get('base_caption_tokens', None), self.num_base_cls,
                'base_caption_tokens')
            self.register_buffer('base_caption_tokens', base_caption_tokens)

            if self.num_novel > 0:
                novel_caption_tokens = self._load_caption_tokens(
                    cgpap_cfg.get('novel_caption_tokens', None), self.num_novel,
                    'novel_caption_tokens')
                self.register_buffer('novel_caption_tokens', novel_caption_tokens)
            else:
                self.register_buffer(
                    'novel_caption_tokens',
                    torch.empty((0, 0, self.cgpap_token_dim), dtype=torch.float32))

            hidden_dim = int(cgpap_cfg.get('hidden_dim', self.cgpap_token_dim))
            self.caption_token_to_query = nn.Sequential(
                nn.Linear(self.cgpap_token_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, self.k_dim))
            self.caption_token_to_proto = nn.Sequential(
                nn.Linear(self.cgpap_token_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, self.dim))
            self.caption_query_gate = nn.Parameter(torch.tensor(1.0))

    def _load_feature_tensor(self, data_or_path, name):
        if data_or_path is None:
            raise ValueError(f'Missing required feature source for `{name}`.')

        if isinstance(data_or_path, (list, tuple)):
            feats = torch.tensor(data_or_path, dtype=torch.float32)
        elif isinstance(data_or_path, torch.Tensor):
            feats = data_or_path.float().detach().cpu()
        elif isinstance(data_or_path, str):
            if not os.path.exists(data_or_path):
                raise FileNotFoundError(f'Cannot find feature file: {data_or_path}')
            ext = os.path.splitext(data_or_path)[-1].lower()
            if ext == '.npy':
                feats = torch.from_numpy(np.load(data_or_path)).float()
            elif ext in ('.pt', '.pth'):
                loaded = torch.load(data_or_path, map_location='cpu')
                if isinstance(loaded, dict):
                    if 'features' in loaded:
                        loaded = loaded['features']
                    else:
                        raise ValueError(
                            f'`{name}` .pt/.pth must be Tensor or contain key `features`.')
                feats = loaded.float()
            elif ext == '.json':
                with open(data_or_path, 'r', encoding='utf-8') as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict) and 'features' in loaded:
                    loaded = loaded['features']
                feats = torch.tensor(loaded, dtype=torch.float32)
            else:
                raise ValueError(
                    f'Unsupported feature format `{ext}` for `{name}`.')
        else:
            raise TypeError(f'Unsupported type for `{name}`: {type(data_or_path)}')
        return feats

    def _load_text_features(self, data_or_path, expected_rows, name):
        if data_or_path is None:
            raise ValueError(
                f'`language_conditioned_queries.enabled=True` requires `{name}`.')

        feats = self._load_feature_tensor(data_or_path, name)

        if feats.dim() != 2:
            raise ValueError(f'`{name}` must be a 2D tensor/list, got shape {tuple(feats.shape)}.')
        if feats.size(0) != expected_rows:
            raise ValueError(
                f'`{name}` first dim mismatch: expected {expected_rows}, got {feats.size(0)}.')
        if feats.size(1) != self.lcq_text_dim:
            raise ValueError(
                f'`{name}` second dim mismatch: expected {self.lcq_text_dim}, got {feats.size(1)}.')
        return F.normalize(feats, dim=-1)

    def _load_caption_tokens(self, data_or_path, expected_rows, name):
        if data_or_path is None:
            raise ValueError(
                f'`caption_guided_part_prototypes.enabled=True` requires `{name}`.')

        feats = self._load_feature_tensor(data_or_path, name)
        if feats.dim() != 3:
            raise ValueError(f'`{name}` must be a 3D tensor/list, got shape {tuple(feats.shape)}.')
        if feats.size(0) != expected_rows:
            raise ValueError(
                f'`{name}` first dim mismatch: expected {expected_rows}, got {feats.size(0)}.')
        if feats.size(-1) != self.cgpap_token_dim:
            raise ValueError(
                f'`{name}` last dim mismatch: expected {self.cgpap_token_dim}, got {feats.size(-1)}.')
        return F.normalize(feats, dim=-1)

    def _generate_queries_from_language(self, forward_novel=False):
        if not self.lcq_enabled:
            raise RuntimeError('LC-FQ is disabled but language query generation was requested.')

        if forward_novel:
            if self.num_novel == 0:
                raise RuntimeError('No novel classes configured for language-conditioned novel queries.')
            class_text = self.novel_text_features
        else:
            class_text = self.base_text_features

        generated = self.query_generator(class_text)
        generated = generated.view(class_text.size(0), self.num_queries, self.k_dim)
        return generated * self.lcq_init_scale

    def _get_class_queries(self, support_gt_labels, forward_novel=False):
        if forward_novel:
            query_emb = self.query_embed_novel.weight
            query_bank = query_emb.reshape(self.num_novel, self.num_queries, query_emb.size(-1))
        else:
            query_emb = self.query_embed.weight
            query_bank = query_emb.reshape(self.num_base_cls, self.num_queries, query_emb.size(-1))

        if self.lcq_enabled:
            generated = self._generate_queries_from_language(forward_novel=forward_novel)
            if self.lcq_residual:
                query_bank = query_bank + self.lcq_residual_scale * generated
            else:
                query_bank = generated

        return query_bank[support_gt_labels, ...]

    def _estimate_prototype_variance(self, prototypes):
        if not self.uapd_enabled:
            return None
        variance = self.prototype_var_head(prototypes)
        return F.softplus(variance) + self.uapd_min_variance

    def _get_caption_tokens(self, support_gt_labels, forward_novel=False):
        if not self.cgpap_enabled:
            return None
        caption_tokens = self.novel_caption_tokens if forward_novel else self.base_caption_tokens
        return caption_tokens[support_gt_labels, ...]

    def _caption_guided_query_refine(self, q, support_gt_labels, forward_novel=False):
        caption_tokens = self._get_caption_tokens(support_gt_labels, forward_novel=forward_novel)
        if caption_tokens is None:
            return q, None, None

        query_tokens = self.caption_token_to_query(caption_tokens)
        attn = torch.matmul(q / math.sqrt(q.size(-1)), query_tokens.transpose(-2, -1))
        attn = attn.softmax(-1)
        query_context = torch.matmul(attn, query_tokens)
        q = q + self.caption_query_gate * self.cgpap_query_scale * query_context

        proto_tokens = self.caption_token_to_proto(caption_tokens)
        semantic_prototypes = torch.matmul(attn, proto_tokens)
        return q, semantic_prototypes, caption_tokens

    def _compute_caption_alignment_loss(self, prototypes, semantic_prototypes):
        if (not self.cgpap_enabled or semantic_prototypes is None or
                self.cgpap_alignment_weight <= 0):
            return None

        if self.cgpap_alignment_type == 'smooth_l1':
            loss = F.smooth_l1_loss(prototypes, semantic_prototypes)
        else:
            prototypes_n = F.normalize(prototypes, dim=-1)
            semantic_n = F.normalize(semantic_prototypes, dim=-1)
            loss = 1.0 - (prototypes_n * semantic_n).sum(-1).mean()
        return loss * self.cgpap_alignment_weight

    def uncertainty_regularization_loss(self, prototype_variances):
        if not self.uapd_regularization_enabled or prototype_variances is None:
            return None
        normalized_variance = prototype_variances / self.uapd_prior_variance
        regularization = normalized_variance - normalized_variance.log() - 1.0
        regularization = 0.5 * regularization.mean()
        return regularization * self.uapd_regularization_weight

    def forward(self, support_feats, support_gt_labels=None, forward_novel=False, forward_novel_test=False):
        """
        Args:
            support_feats: Tensor with shape (B, C, H, W).
            support_gt_labels: Support gt labels.
            forward_novel (bool): Novel classes.
            forward_novel_test (bool): Test time.
        Returns:
            tensor with shape (15, 1024)
        """

        # at the fine-tuning stage, duplicate the most compatible feature queries for the novel classes
        # ************************************************************
        if not self.duplicated and self.num_novel > 0 and not forward_novel_test and forward_novel:
            with torch.no_grad():
                if self.lcq_enabled and self.lcq_novel_init in ('language', 'hybrid'):
                    lang_novel = self._generate_queries_from_language(
                        forward_novel=True).reshape(self.num_novel * self.num_queries, self.k_dim)
                    self.query_embed_novel.weight.copy_(lang_novel)

                do_duplicate = (self.lcq_novel_init != 'language')
                if do_duplicate:
                    support_feats_mp = F.max_pool2d(support_feats, kernel_size=2, stride=2)
                    B, C, H, W = support_feats_mp.shape
                    k = support_feats_mp.reshape(B, C, H * W).permute(0, 2, 1)
                    k = self.w_qk(k)
                    q = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1)
                    B, Nt, E = q.shape
                    attn = torch.bmm(q / math.sqrt(E), k.transpose(-2, -1))
                    weight = torch.topk(attn, 20, dim=-1)[0].mean(-1)

                    drop = 5
                    top_indices = torch.topk(
                        weight, self.num_queries + drop,
                        dim=-1)[1][:, -self.num_queries:]
                    top_emb = torch.gather(
                        self.query_embed.weight.unsqueeze(0).expand(B, -1, -1), 1,
                        top_indices.unsqueeze(-1).expand(-1, -1, self.k_dim))

                    # support_gt_labels can contain multiple shots per novel class.
                    # Aggregate selected embeddings per class to get a stable
                    # [num_novel, num_queries, k_dim] tensor before flattening.
                    sort_ids = torch.sort(support_gt_labels, dim=0)[1]
                    sorted_labels = support_gt_labels[sort_ids]
                    top_emb = top_emb[sort_ids]
                    if top_emb.size(0) != self.num_novel:
                        class_emb = torch.zeros(
                            self.num_novel,
                            self.num_queries,
                            self.k_dim,
                            device=top_emb.device,
                            dtype=top_emb.dtype)
                        fallback_emb = self.query_embed_novel.weight.reshape(
                            self.num_novel, self.num_queries, self.k_dim)
                        for cls_id in range(self.num_novel):
                            cls_mask = (sorted_labels == cls_id)
                            if cls_mask.any():
                                class_emb[cls_id] = top_emb[cls_mask].mean(dim=0)
                            else:
                                class_emb[cls_id] = fallback_emb[cls_id]
                        top_emb = class_emb
                    top_emb = top_emb.reshape(self.num_novel * self.num_queries, self.k_dim)

                    if self.lcq_enabled and self.lcq_novel_init == 'hybrid':
                        top_emb = 0.5 * (top_emb + self.query_embed_novel.weight)
                    self.query_embed_novel.weight.copy_(top_emb)
            self.duplicated = True  # set as True once duplicated
        # ************************************************************

        support_feats_mp = F.max_pool2d(support_feats, kernel_size=2, stride=2)
        B, C, H, W = support_feats_mp.shape
        k = v = support_feats_mp.reshape(B, C, H*W).permute(0, 2, 1)  # (B, 196, 1024)
        k = self.w_qk(k)  # (B, 196, 1024)

        # scaled dot-product attention
        q = self._get_class_queries(
            support_gt_labels=support_gt_labels, forward_novel=forward_novel)
        q, semantic_prototypes, _ = self._caption_guided_query_refine(
            q, support_gt_labels=support_gt_labels, forward_novel=forward_novel)
        B, Nt, E = q.shape
        attn = torch.bmm(q / math.sqrt(E), k.transpose(-2, -1))
        weight = torch.topk(attn, 20, dim=-1)[0].mean(-1)
        prototypes = torch.matmul(attn.softmax(-1), v)     # (B, 5, 1024)
        prototype_variances = self._estimate_prototype_variance(prototypes)
        caption_alignment_loss = self._compute_caption_alignment_loss(
            prototypes, semantic_prototypes)

        aux_losses = {}
        if caption_alignment_loss is not None:
            aux_losses['loss_cgpap_align'] = caption_alignment_loss

        return weight, prototypes, prototype_variances, aux_losses


@AGGREGATORS.register_module()
class PrototypesAssignment(BaseModule):
    def __init__(self,
                 dim,
                 num_bg=5,
                 uncertainty_aware_attention=None,
                 semantic_background_prototypes=None):
        super().__init__()
        k_dim = dim // 4
        self.dim = dim
        self.k_dim = k_dim
        self.w_qk = nn.Linear(dim, k_dim, bias=False)

        self.num_bg = num_bg
        if self.num_bg > 0:
            self.dummy = nn.Parameter(torch.Tensor(self.num_bg, dim))
            nn.init.normal_(self.dummy)
            self.linear = nn.Linear(dim, k_dim)
        self.gamma = nn.Parameter(torch.tensor(0.))

        uncertainty_cfg = uncertainty_aware_attention or {}
        self.uapd_enabled = bool(uncertainty_cfg.get('enabled', False))
        self.uapd_min_variance = float(uncertainty_cfg.get('min_variance', 1e-4))
        self.uapd_temperature = float(uncertainty_cfg.get('temperature', 0.5))
        self.uapd_precision_power = float(uncertainty_cfg.get('precision_power', 1.0))
        self.uapd_log_variance_bias = float(uncertainty_cfg.get('log_variance_bias', 0.25))

        bg_cfg = semantic_background_prototypes or {}
        self.semantic_bg_enabled = bool(bg_cfg.get('enabled', False))
        self.semantic_bg_token_dim = int(bg_cfg.get('token_dim', 512))
        self.semantic_bg_scale = float(bg_cfg.get('scale', 1.0))
        self.semantic_bg_residual = bool(bg_cfg.get('residual', True))
        self.semantic_bg_variance = float(bg_cfg.get('background_variance', 2.0))
        if self.semantic_bg_enabled:
            background_tokens = self._load_background_tokens(
                bg_cfg.get('background_tokens', None), 'background_tokens')
            self.register_buffer('background_tokens', background_tokens)
            hidden_dim = int(bg_cfg.get('hidden_dim', self.semantic_bg_token_dim))
            self.background_token_to_proto = nn.Sequential(
                nn.Linear(self.semantic_bg_token_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, self.dim))
            if self.semantic_bg_residual:
                self.background_gate = nn.Parameter(torch.tensor(1.0))
            self.background_log_variance = nn.Parameter(
                torch.log(torch.full((self.num_bg, self.dim), self.semantic_bg_variance)))

    def _load_feature_tensor(self, data_or_path, name):
        if data_or_path is None:
            raise ValueError(f'Missing required feature source for `{name}`.')

        if isinstance(data_or_path, (list, tuple)):
            feats = torch.tensor(data_or_path, dtype=torch.float32)
        elif isinstance(data_or_path, torch.Tensor):
            feats = data_or_path.float().detach().cpu()
        elif isinstance(data_or_path, str):
            if not os.path.exists(data_or_path):
                raise FileNotFoundError(f'Cannot find feature file: {data_or_path}')
            ext = os.path.splitext(data_or_path)[-1].lower()
            if ext == '.npy':
                feats = torch.from_numpy(np.load(data_or_path)).float()
            elif ext in ('.pt', '.pth'):
                loaded = torch.load(data_or_path, map_location='cpu')
                if isinstance(loaded, dict):
                    if 'features' in loaded:
                        loaded = loaded['features']
                    else:
                        raise ValueError(
                            f'`{name}` .pt/.pth must be Tensor or contain key `features`.')
                feats = loaded.float()
            elif ext == '.json':
                with open(data_or_path, 'r', encoding='utf-8') as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict) and 'features' in loaded:
                    loaded = loaded['features']
                feats = torch.tensor(loaded, dtype=torch.float32)
            else:
                raise ValueError(
                    f'Unsupported feature format `{ext}` for `{name}`.')
        else:
            raise TypeError(f'Unsupported type for `{name}`: {type(data_or_path)}')
        return feats

    def _load_background_tokens(self, data_or_path, name):
        feats = self._load_feature_tensor(data_or_path, name)
        if feats.dim() == 2:
            if feats.size(0) != self.num_bg or feats.size(-1) != self.semantic_bg_token_dim:
                raise ValueError(
                    f'`{name}` must have shape ({self.num_bg}, {self.semantic_bg_token_dim}) '
                    f'or ({self.num_bg}, num_tokens, {self.semantic_bg_token_dim}), got {tuple(feats.shape)}.')
            feats = feats.unsqueeze(1)
        elif feats.dim() == 3:
            if feats.size(0) != self.num_bg or feats.size(-1) != self.semantic_bg_token_dim:
                raise ValueError(
                    f'`{name}` must have shape ({self.num_bg}, num_tokens, {self.semantic_bg_token_dim}), '
                    f'got {tuple(feats.shape)}.')
        else:
            raise ValueError(f'`{name}` must be a 2D or 3D tensor/list, got shape {tuple(feats.shape)}.')
        return F.normalize(feats, dim=-1)

    def _build_background_prototypes(self, device, dtype):
        if not self.semantic_bg_enabled or self.num_bg <= 0:
            return None, None

        bg_tokens = self.background_tokens.to(device=device, dtype=dtype)
        bg_proto = self.background_token_to_proto(bg_tokens).mean(1)
        bg_proto = bg_proto * self.semantic_bg_scale
        if self.semantic_bg_residual:
            bg_proto = self.dummy + self.background_gate * bg_proto

        bg_variances = None
        if self.uapd_enabled:
            bg_variances = self.background_log_variance.exp().to(device=device, dtype=dtype)
            bg_variances = bg_variances.clamp_min(self.uapd_min_variance)
        return bg_proto, bg_variances

    def forward(self,
                query_feature,
                prototypes,
                prototype_variances=None,
                query_img_metas=None):
        """
        Args:
            query_feature: Tensor with shape (B, C, H, W)
            prototypes: Tensor with shape (num_supp, num_queries, C),
            query_img_metas: Visualization.
        Returns:
            class-specific query feature: tensor(B, C, H, W)
        """

        B, C, H, W = query_feature.shape
        num_supp, num_queries, _ = prototypes.shape
        q = query_feature.reshape(B, C, H*W).permute(0, 2, 1)   # (B, H*W, 1024)
        k = v = prototypes.reshape(num_supp * num_queries, C)

        q = self.w_qk(q)
        k = self.w_qk(k)

        precision_scale = None
        log_variance_penalty = None
        if self.uapd_enabled and prototype_variances is not None:
            prototype_variances = prototype_variances.reshape(num_supp * num_queries, C)
            prototype_variances = prototype_variances.clamp_min(self.uapd_min_variance)
            precision_scale = prototype_variances.reciprocal().mean(-1)
            precision_scale = precision_scale.pow(self.uapd_precision_power)
            log_variance_penalty = prototype_variances.log().mean(-1)

        if self.num_bg > 0:
            bg_proto, bg_variances = self._build_background_prototypes(
                device=prototypes.device, dtype=prototypes.dtype)
            if bg_proto is None:
                bg_proto = torch.zeros((self.num_bg, C), device=prototypes.device, dtype=prototypes.dtype)
                bg_key = self.linear(self.dummy)
            else:
                bg_key = self.w_qk(bg_proto)
            k = torch.cat([k, bg_key], dim=0)
            v = torch.cat([v, bg_proto], dim=0)
            if precision_scale is not None:
                if bg_variances is not None:
                    dummy_precision = bg_variances.reciprocal().mean(-1)
                    dummy_precision = dummy_precision.pow(self.uapd_precision_power)
                    dummy_penalty = bg_variances.log().mean(-1)
                else:
                    dummy_precision = torch.ones(
                        self.num_bg, device=precision_scale.device, dtype=precision_scale.dtype)
                    dummy_penalty = torch.zeros(
                        self.num_bg, device=log_variance_penalty.device, dtype=log_variance_penalty.dtype)
                precision_scale = torch.cat([precision_scale, dummy_precision], dim=0)
                log_variance_penalty = torch.cat([log_variance_penalty, dummy_penalty], dim=0)
            elif self.uapd_enabled and bg_variances is not None:
                precision_scale = bg_variances.reciprocal().mean(-1)
                precision_scale = precision_scale.pow(self.uapd_precision_power)
                log_variance_penalty = bg_variances.log().mean(-1)

        k = k.unsqueeze(0)
        B, Nt, E = q.shape
        attn = torch.bmm(q / math.sqrt(E), k.expand(B, -1, -1).transpose(-2, -1))
        if precision_scale is not None:
            attn = attn * precision_scale.view(1, 1, -1)
            attn = attn - self.uapd_log_variance_bias * log_variance_penalty.view(1, 1, -1)
        attn.div_(self.uapd_temperature)

        out = torch.matmul(attn.softmax(-1), v)  # (B, 2850, 1024)
        out = out.permute(0, 2, 1).contiguous().view(B, C, H, W)
        out = query_feature + self.gamma * out
        return out


