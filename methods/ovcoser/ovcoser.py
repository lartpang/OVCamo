import open_clip
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from loguru import logger
from open_clip import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
from torch import nn

from methods.base.base_ops import LayerNorm2d, PixelNormalizer, rescale_2x, resize_to
from methods.ovcoser.layers import ConvMlp, LNConvAct
from methods.ovcoser.loss import edge_dice_loss, l1_ssim_loss, seg_loss

from .prompts import get_prompt_template_by_name


class ConvNeXtCLIP(nn.Module):
    def __init__(
        self,
        model_name="convnext_large_d_320",
        pretrained="laion2b_s29b_b131k_ft_soup",
        template_set="camoprompts",
    ):
        super().__init__()
        self.clip_model, _, self.preprocess_val = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.mean = OPENAI_DATASET_MEAN
        self.std = OPENAI_DATASET_STD
        self.text_tokenizer = open_clip.get_tokenizer(model_name)

        self.template_set = get_prompt_template_by_name(template_set)
        logger.info(f"Create the CLIP ({model_name + '-' + pretrained}) with template_set {self.template_set}")

        model_name = model_name.lower()
        assert "convnext_" in model_name
        self.model_type = "convnext"
        if "_base" in model_name:
            self.feat_chans = [128, 128, 256, 512, 1024]
        elif "_large" in model_name:
            self.feat_chans = [192, 192, 384, 768, 1536]
        elif "_xxlarge" in model_name:
            self.feat_chans = [384, 384, 768, 1536, 3072]

        self.dim_latent = self.clip_model.text_projection.shape[-1]
        self.out_strides = {"stem": 2, "res2": 4, "res3": 8, "res4": 16, "res5": 32, "emb": -1}
        self.out_chans = {
            "stem": self.feat_chans[0],
            "res2": self.feat_chans[1],
            "res3": self.feat_chans[2],
            "res4": self.feat_chans[3],
            "res5": self.feat_chans[4],
            "emb": self.dim_latent,
        }

    def output_shape(self):
        return {
            name: dict(channels=self.out_chans[name], stride=self.out_strides[name])
            for name in ["stem", "res2", "res3", "res4", "res5", "emb"]
        }

    @property
    def device(self):
        for param in self.clip_model.parameters():
            return param.device

    @torch.no_grad()
    def get_text_embs(self, text_list, normalize=True):
        """对输入的所有类别名称都使用一套模板构建平均嵌入
        return: NumberofClasses,D
        """
        self.eval()

        # reference for templates: https://github.com/mlfoundations/open_clip/blob/91f6cce16b7bee90b3b5d38ca305b5b3b67cc200/src/training/imagenet_zeroshot_data.py
        text_tokens = self.text_tokenizer(text_list).to(self.device)

        # list -> TD
        cast_dtype = self.clip_model.transformer.get_cast_dtype()
        x = self.clip_model.token_embedding(text_tokens).to(cast_dtype)  # [num_temp, n_ctx, d_model]
        #
        x = x + self.clip_model.positional_embedding.to(cast_dtype)  # 80,77,768  77,768

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.clip_model.transformer(x, attn_mask=self.clip_model.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.clip_model.ln_final(x)  # [batch_size, n_ctx, transformer.width]

        # take feats from the eot embedding (eot_token is the highest number in each sequence)
        text_embs = x[torch.arange(x.shape[0]), text_tokens.argmax(dim=-1)] @ self.clip_model.text_projection

        if normalize:
            text_embs = F.normalize(text_embs, dim=-1)  # Nc,768
        return text_embs

    @torch.no_grad()
    def get_text_embs_by_template(self, text_list):
        """对输入的所有类别名称都使用一套模板构建平均嵌入
        return: NumberofClasses,D
        """
        self.eval()

        text_embs = []
        for text in text_list:
            # reference for templates: https://github.com/mlfoundations/open_clip/blob/91f6cce16b7bee90b3b5d38ca305b5b3b67cc200/src/training/imagenet_zeroshot_data.py
            text_tokens = self.text_tokenizer([template.format(text) for template in self.template_set]).to(
                self.device
            )

            # list -> TD
            cast_dtype = self.clip_model.transformer.get_cast_dtype()
            x = self.clip_model.token_embedding(text_tokens).to(cast_dtype)  # [num_temp, n_ctx, d_model]
            #
            x = x + self.clip_model.positional_embedding.to(cast_dtype)  # 80,77,768  77,768

            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.clip_model.transformer(x, attn_mask=self.clip_model.attn_mask)
            x = x.permute(1, 0, 2)  # LND -> NLD
            x = self.clip_model.ln_final(x)  # [batch_size, n_ctx, transformer.width]

            # take feats from the eot embedding (eot_token is the highest number in each sequence) => Nc,768
            text_emb = x[torch.arange(x.shape[0]), text_tokens.argmax(dim=-1)] @ self.clip_model.text_projection
            text_embs.append(text_emb)
        text_embs = torch.stack(text_embs, dim=0)  # Nc,Nt,768

        text_embs /= text_embs.norm(dim=-1, keepdim=True)
        text_embs = text_embs.mean(1)
        text_embs /= text_embs.norm(dim=-1, keepdim=True)
        return text_embs  # Nc,768

    def visual_feats_to_embs(self, x, normalize: bool = True):
        """
        将图像特征转换为图像嵌入向量
        """
        self.eval()

        x = self.clip_model.visual.trunk.head(x)
        x = self.clip_model.visual.head(x)
        return F.normalize(x, dim=-1) if normalize else x

    @torch.no_grad()
    def get_visual_feats(self, x):
        self.eval()

        out = {}
        x = self.clip_model.visual.trunk.stem(x)
        out["stem"] = x.contiguous()  # os4
        for i in range(4):
            x = self.clip_model.visual.trunk.stages[i](x)
            out[f"res{i + 2}"] = x.contiguous()  # res 2 (os4), 3 (os8), 4 (os16), 5 (os32)

        x = self.clip_model.visual.trunk.norm_pre(x)
        out["clip_vis_dense"] = x.contiguous()
        return out


class SemanticGuidanceBlock(nn.Module):
    def __init__(self, feat_dim, guide_dim, num_heads=4):
        super().__init__()
        self.feat_dim = feat_dim
        self.guide_dim = guide_dim

        self.guide_norm = nn.LayerNorm(guide_dim)
        self.feat_norm = LayerNorm2d(feat_dim)

        self.num_heads = num_heads
        self.attn_scale = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.g_proj = nn.Linear(guide_dim, feat_dim, bias=False)
        self.qkv_proj = nn.Conv2d(feat_dim, 3 * feat_dim, 1, bias=False)
        self.o_proj = nn.Conv2d(feat_dim, feat_dim, 1, bias=True)

        self.norm_ffn = ConvMlp(feat_dim, feat_dim)

    def masked_pooling(self, image_feat, mask):
        obj_embs = mask * resize_to(image_feat, tgt_hw=mask.shape[-2:])
        obj_embs = obj_embs.sum((2, 3)) / mask.sum((2, 3))  # b d
        return obj_embs

    def forward(self, img_feats, cls_embs, cls_logits=None, seg_logits=None, verbose=False):
        """
        img_feats: B,C,H,W
        cls_embs: Nc,D
        cls_logits: B,Nc
        seg_logits: B,1,H,W
        """
        normed_img_feats = self.feat_norm(img_feats)
        _, _, H, W = normed_img_feats.shape
        normed_cls_embs = self.guide_norm(cls_embs)

        g = self.g_proj(normed_cls_embs)  # Nc,D
        qkv = self.qkv_proj(normed_img_feats)  # B,D,H,W

        g = rearrange(g, "nc (nh hd) -> nc nh hd", nh=self.num_heads)
        qkv = rearrange(qkv, "b (ng nh hd) h w -> ng b nh hd (h w)", ng=3, nh=self.num_heads)
        q, k, v = qkv.unbind(0)

        # b nh nc hw [-1, 1]
        guide_map = torch.einsum("nhd, bhdl -> bhnl", F.normalize(g, dim=-1), F.normalize(q, dim=-2))
        #
        guide_map = guide_map - guide_map.min(dim=-1, keepdim=True).values
        guide_map = guide_map / guide_map.max(dim=-1, keepdim=True).values
        # b nh hw
        guide_map = (guide_map.softmax(dim=2) * guide_map).sum(dim=2).unsqueeze(2)

        if cls_logits is not None:
            cls_logits = repeat(cls_logits, "b nc -> b nh nc", nh=self.num_heads)
            cls_obj_embs = torch.einsum("nhd, bhn -> bhd", g, cls_logits)  #

            seg_obj_embs = self.masked_pooling(k, seg_logits.sigmoid())  # b d
            seg_obj_embs = rearrange(seg_obj_embs, "b (nh hd) -> b nh hd", nh=self.num_heads)
            obj_embs = cls_obj_embs + seg_obj_embs

            c_norm_obj_embs = F.normalize(obj_embs, dim=-1)  # b nh hd
            c_norm_k = F.normalize(k, dim=-2)  # b nh hd hw
            aux_guide_map = torch.einsum("bhd, bhdl -> bhl", c_norm_obj_embs, c_norm_k)  # b,nh,hw
            aux_guide_map = aux_guide_map.unsqueeze(2)  # b,nh,1,hw
            #
            aux_guide_map = aux_guide_map - aux_guide_map.min(dim=-1, keepdim=True).values
            aux_guide_map = aux_guide_map / aux_guide_map.max(dim=-1, keepdim=True).values
            guide_map = guide_map * aux_guide_map
        v = guide_map * v

        hw_norm_q = F.normalize(q, dim=-1)  # b nh hd1 hw
        hw_norm_k = F.normalize(k, dim=-1)  # b nh hd2 hw
        attn = hw_norm_q @ hw_norm_k.transpose(-1, -2)  # b nh hd1 hw @ b nh hw hd2 => b nh hd1 hd2
        attn = self.attn_scale * attn
        qkv = attn.softmax(dim=-1) @ v  # b nh hd1 hw
        qkv = rearrange(qkv, "b nh hd (h w) -> b (nh hd) h w", h=H, w=W)

        img_feats = img_feats + self.o_proj(qkv)
        img_feats = img_feats + self.norm_ffn(img_feats)
        return img_feats


class StructureEnhancementBlock(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.dim = dim
        self.q_norm = LayerNorm2d(dim)
        self.kv_norm = LayerNorm2d(dim)

        self.num_heads = num_heads
        self.attn_scale = nn.Parameter(torch.zeros(num_heads, 1, 1))

        self.q_proj = nn.Conv2d(dim, dim, 1, bias=False)
        self.kv_proj = nn.Conv2d(dim, 2 * dim, 1, bias=False)
        self.o_proj = nn.Conv2d(dim, dim, 1, bias=True)

        self.norm_ffn = ConvMlp(dim, dim)

    def forward(self, img_feat, aux_feats=None):
        assert isinstance(aux_feats, (list, tuple))
        # B,C,H,W
        B, C, H, W = img_feat.shape
        q = self.q_norm(img_feat)
        kv = self.kv_norm(torch.cat(aux_feats, dim=0))  # 2B,C,H,W

        q = self.q_proj(q)
        kv = self.kv_proj(kv)
        q = rearrange(q, "b (nh hd) h w -> b nh hd (h w)", nh=self.num_heads)
        kv = rearrange(kv, "b (ng nh hd) h w -> ng b nh hd (h w)", ng=2, nh=self.num_heads)
        k, v = kv.unbind(0)

        hw_norm_q = F.normalize(q, dim=-1)  # b nh hd1 hw
        hw_norm_k = F.normalize(k, dim=-1)  # 2b nh hd2 hw
        if len(aux_feats) == 1:
            attn = hw_norm_q @ hw_norm_k.transpose(-1, -2)  # b nh hd1 hw @ b nh hw hd2 => b nh hd1 hd2
            qkv = attn.softmax(dim=-1) @ v  # b nh hd1 hw
        else:
            assert len(aux_feats) == 2, len(aux_feats)
            hw_norm_tex_k, hw_norm_dep_k = hw_norm_k.chunk(2, dim=0)  # b nh hd2 hw
            tex_v, dep_v = v.chunk(2, dim=0)  # b nh hd2 hw

            tex_attn = hw_norm_q @ hw_norm_tex_k.transpose(-1, -2)  # b nh hd1 hw @ b nh hw hd2 => b nh hd1 hd2
            dep_attn = hw_norm_q @ hw_norm_dep_k.transpose(-1, -2)  # b nh hd1 hw @ b nh hw hd2 => b nh hd1 hd2
            tex_qkv = tex_attn.softmax(dim=-1) @ tex_v  # b nh hd1 hw
            dep_qkv = dep_attn.softmax(dim=-1) @ dep_v  # b nh hd1 hw
            qkv = self.attn_scale.sigmoid() * tex_qkv + (1 - self.attn_scale.sigmoid()) * dep_qkv
        qkv = rearrange(qkv, "b nh hd (h w) -> b (nh hd) h w", h=H, w=W)

        img_feat = img_feat + self.o_proj(qkv)
        img_feat = img_feat + self.norm_ffn(img_feat)
        return img_feat


class OVCoser(nn.Module):
    def __init__(self, iterations=2, template_set="camoprompts"):
        super().__init__()
        assert iterations in [1, 2, 3]
        self.iterations = iterations
        logger.info(f"Current Iteration: {iterations}")

        self.clip = ConvNeXtCLIP(template_set=template_set)
        self.normalizer = PixelNormalizer(mean=self.clip.mean, std=self.clip.std)
        self.test_class_embs = None
        self.train_class_embs = None

        self.encoder_dims = [192, 384, 768, 1536]
        self.hid_dim = 128

        # fmt: off
        self.tra1 = ConvMlp(self.encoder_dims[0], self.hid_dim)
        self.tra2 = ConvMlp(self.encoder_dims[0], self.hid_dim)
        self.tra3 = ConvMlp(self.encoder_dims[1], self.hid_dim)
        self.tra4 = ConvMlp(self.encoder_dims[2], self.hid_dim)
        self.tra5 = ConvMlp(self.encoder_dims[3], self.hid_dim)

        self.dec5 = SemanticGuidanceBlock(feat_dim=self.hid_dim, guide_dim=self.clip.dim_latent)
        self.dec4 = SemanticGuidanceBlock(feat_dim=self.hid_dim, guide_dim=self.clip.dim_latent)
        self.dec3 = SemanticGuidanceBlock(feat_dim=self.hid_dim, guide_dim=self.clip.dim_latent)
        self.dec2 = SemanticGuidanceBlock(feat_dim=self.hid_dim, guide_dim=self.clip.dim_latent)
        self.dec1 = SemanticGuidanceBlock(feat_dim=self.hid_dim, guide_dim=self.clip.dim_latent)

        self.cod_head = nn.Sequential(LNConvAct(self.hid_dim, self.hid_dim, 3, 1, 1, act_name="relu"), nn.Conv2d(self.hid_dim, 1, 3, 1, 1))

        self.mrg3 = StructureEnhancementBlock(self.hid_dim)
        self.mrg2 = StructureEnhancementBlock(self.hid_dim)
        self.mrg1 = StructureEnhancementBlock(self.hid_dim)

        self.edg_stems = nn.ModuleList([nn.Sequential(LNConvAct(self.hid_dim, self.hid_dim, 3, 1, 1, act_name="relu"), nn.Conv2d(self.hid_dim, self.hid_dim, 3, 1, 1)) for _ in range(3)])
        self.edg_heads = nn.ModuleList([LNConvAct(self.hid_dim, 1, 3, 1, 1, act_name="idy") for _ in range(3)])
        self.dep_stems = nn.ModuleList([nn.Sequential(LNConvAct(self.hid_dim, self.hid_dim, 3, 1, 1, act_name="relu"), nn.Conv2d(self.hid_dim, self.hid_dim, 3, 1, 1)) for _ in range(3)])
        self.dep_heads = nn.ModuleList([LNConvAct(self.hid_dim, 1, 3, 1, 1, act_name="idy") for _ in range(3)])
        # fmt: on

    def get_visual_feats(self, image):
        image = self.normalizer(image)
        image_feats = self.clip.get_visual_feats(image)
        image_stem = rescale_2x(image_feats["stem"])  # 192, 80, 80 -> 192, 160, 160
        image_res5 = image_feats["res5"]
        image_res4 = image_feats["res4"]
        image_res3 = image_feats["res3"]
        image_res2 = image_feats["res2"]

        image_stem = self.tra1(image_stem)
        image_res2 = self.tra2(image_res2)
        image_res3 = self.tra3(image_res3)
        image_res4 = self.tra4(image_res4)
        image_res5 = self.tra5(image_res5)

        image_deep = image_feats["clip_vis_dense"]
        return image_stem, image_res2, image_res3, image_res4, image_res5, image_deep

    def map_classifier(self, logits, image_deep, normed_class_embs):
        prob = logits.sigmoid()
        image_embs = resize_to(image_deep, tgt_hw=prob.shape[-2:])

        # image_embs (B,C)
        image_embs = (prob * image_embs).sum((-1, -2)) / prob.sum((-1, -2))
        image_embs = image_embs[..., None, None]

        # B,C => B,D
        normed_image_embs = self.clip.visual_feats_to_embs(image_embs, normalize=True)
        class_logits = normed_image_embs @ normed_class_embs.T  # B,N
        class_logits = self.clip.clip_model.logit_scale.exp() * class_logits
        return class_logits

    def body(self, image, normed_class_embs):
        image_stem, image_res2, image_res3, image_res4, image_res5, image_deep = self.get_visual_feats(image)

        x5 = self.dec5(image_res5, normed_class_embs)
        x4 = image_res4 + rescale_2x(x5)
        x4 = self.dec4(x4, normed_class_embs)
        x3 = image_res3 + rescale_2x(x4)

        logits = {}
        cls_logits = None
        seg_logits = None
        for i in range(1, self.iterations + 1):
            x3 = self.dec3(x3, normed_class_embs, cls_logits=cls_logits, seg_logits=seg_logits)
            edg_feat3 = self.edg_stems[2](x3)
            dep_feat3 = self.dep_stems[2](x3)
            x3 = self.mrg3(x3, [edg_feat3, dep_feat3])

            x2 = image_res2 + rescale_2x(x3)
            x2 = self.dec2(x2, normed_class_embs, cls_logits=cls_logits, seg_logits=seg_logits)
            edg_feat2 = self.edg_stems[1](x2)
            dep_feat2 = self.dep_stems[1](x2)
            x2 = self.mrg2(x2, [edg_feat2, dep_feat2])

            x1 = image_stem + rescale_2x(x2)
            x1 = self.dec1(x1, normed_class_embs, cls_logits=cls_logits, seg_logits=seg_logits)
            edg_feat1 = self.edg_stems[0](x1)
            dep_feat1 = self.dep_stems[0](x1)
            x1 = self.mrg1(x1, [edg_feat1, dep_feat1])

            seg_logits = self.cod_head(rescale_2x(x1))
            cls_logits = self.map_classifier(seg_logits, image_deep, normed_class_embs)
            logits[f"seg{i}"] = seg_logits
            logits[f"cls{i}"] = cls_logits

            logits[f"dep{i}-3"] = self.dep_heads[2](dep_feat3)
            logits[f"dep{i}-2"] = self.dep_heads[1](dep_feat2)
            logits[f"dep{i}-1"] = self.dep_heads[0](dep_feat1)
            logits[f"edg{i}-3"] = self.edg_heads[2](edg_feat3)
            logits[f"edg{i}-2"] = self.edg_heads[1](edg_feat2)
            logits[f"edg{i}-1"] = self.edg_heads[0](edg_feat1)
        return logits

    def train_forward(self, data, gt_classes, class_names: list, **kwargs):
        image = data["image"]

        # [N=num_classes, 768]
        if self.train_class_embs is None:
            self.train_class_embs = self.clip.get_text_embs_by_template(class_names)
        normed_class_embs = self.train_class_embs  # Nc,D

        logits_dict = self.body(image, normed_class_embs)

        losses = []
        loss_str = []

        mask = data["mask"]
        depth = data["depth"]
        with torch.no_grad():
            edge_ks = 5
            eroded_mask = -F.max_pool2d(-mask, kernel_size=edge_ks, stride=1, padding=edge_ks // 2)
            dilated_mask = F.max_pool2d(mask, kernel_size=edge_ks, stride=1, padding=edge_ks // 2)
            edge = dilated_mask - eroded_mask
            edge = edge.gt(0).float()
        logits_dict["edge"] = edge

        for i in range(1, self.iterations + 1):
            seg_logits = logits_dict[f"seg{i}"]
            segl = seg_loss(logits=seg_logits, mask=mask)
            losses.append(segl)
            loss_str.append(f"segl: {segl.item():.5f}")

            d_l1ss = l1_ssim_loss(logits=logits_dict[f"dep{i}-3"], mask=depth)
            losses.append(d_l1ss)
            loss_str.append(f"d_l1ss3: {d_l1ss.item():.5f}")
            d_l1ss = l1_ssim_loss(logits=logits_dict[f"dep{i}-2"], mask=depth)
            losses.append(d_l1ss)
            loss_str.append(f"d_l1ss2: {d_l1ss.item():.5f}")
            d_l1ss = l1_ssim_loss(logits=logits_dict[f"dep{i}-1"], mask=depth)
            losses.append(d_l1ss)
            loss_str.append(f"d_l1ss1: {d_l1ss.item():.5f}")

            e_dice = edge_dice_loss(logits=logits_dict[f"edg{i}-3"], edge=edge)
            losses.append(e_dice)
            loss_str.append(f"e_dice3: {e_dice.item():.5f}")
            e_dice = edge_dice_loss(logits=logits_dict[f"edg{i}-2"], edge=edge)
            losses.append(e_dice)
            loss_str.append(f"e_dice2: {e_dice.item():.5f}")
            e_dice = edge_dice_loss(logits=logits_dict[f"edg{i}-1"], edge=edge)
            losses.append(e_dice)
            loss_str.append(f"e_dice1: {e_dice.item():.5f}")

        return dict(
            vis={k: v if k == "edge" else v.sigmoid() for k, v in logits_dict.items() if not k.startswith("cls")},
            loss=sum(losses),
            loss_str=" ".join(loss_str),
        )

    def test_forward(self, data, gt_classes, class_names: list, **kwargs):
        image = data["image"]

        # [N=num_classes, 768]
        if self.test_class_embs is None:
            self.test_class_embs = self.clip.get_text_embs_by_template(class_names)
        normed_class_embs = self.test_class_embs

        logits_dict = self.body(image, normed_class_embs)
        seg_logits = logits_dict[f"seg{self.iterations}"]
        cls_logits = logits_dict[f"cls{self.iterations}"]

        cls_id_per_image = torch.argmax(cls_logits, dim=-1)
        pred_classes = [class_names[i] for i in cls_id_per_image]
        return dict(prob=seg_logits.sigmoid(), classes=pred_classes)

    def forward(self, *arg, **kwargs):
        if self.training:
            return self.train_forward(*arg, **kwargs)
        else:
            return self.test_forward(*arg, **kwargs)

    def get_grouped_params(self):
        param_groups = {"pretrained": [], "fixed": [], "retrained": []}
        for name, param in self.named_parameters():
            if name.startswith("clip."):
                param.requires_grad = False
                param_groups["fixed"].append(param)
            else:
                param.requires_grad = True
                param_groups["retrained"].append(param)
        logger.info(
            f"Parameter Groups:{{"
            f"Pretrained: {len(param_groups['pretrained'])}, "
            f"Fixed: {len(param_groups['fixed'])}, "
            f"ReTrained: {len(param_groups['retrained'])}}}"
        )
        return param_groups
