import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
import os


import world

class PureMF(nn.Module):
    def __init__(self, config, dataset):
        super().__init__()
        self.config = config
        self.n_users = dataset.n_users
        self.m_items = dataset.m_items
        self.embedding_user = nn.Embedding(self.n_users, config.recdim)
        self.embedding_item = nn.Embedding(self.m_items, config.recdim)
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)

    def getUsersRating(self, users):
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item.weight
        return users_emb @ items_emb.t()

    def bpr_loss(self, users, pos, neg):
        u = self.embedding_user(users)
        ip = self.embedding_item(pos)
        ineg = self.embedding_item(neg)
        x = (u*ip).sum(1) - (u*ineg).sum(1)
        loss = -F.logsigmoid(x).mean()
        reg = (u.pow(2).sum(1) + ip.pow(2).sum(1) + ineg.pow(2).sum(1)).mean()
        return loss, reg

class LightGCN(nn.Module):
    def __init__(self, config, dataset):
        super().__init__()
        self.config = config
        self.n_users = dataset.n_users
        self.m_items = dataset.m_items
        self.Graph = dataset.getSparseGraph()
        self.embedding_user = nn.Embedding(self.n_users, config.recdim)
        self.embedding_item = nn.Embedding(self.m_items, config.recdim)
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)
        self.f = nn.Sigmoid()

    def computer(self):
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])  # [U+I, d]
        embs = [all_emb]
        g = self.Graph
        for _ in range(self.config.layer):
            all_emb = torch.sparse.mm(g, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.n_users, self.m_items])
        return users, items

    def getUsersRating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        return users_emb @ all_items.t()

    def bpr_loss(self, users, pos, neg):
        (all_users, all_items) = self.computer()
        users_emb = all_users[users.long()]
        pos_emb = all_items[pos.long()]
        neg_emb = all_items[neg.long()]
        reg = (self.embedding_user(users.long()).pow(2).sum(1) \
              + self.embedding_item(pos.long()).pow(2).sum(1) \
              + self.embedding_item(neg.long()).pow(2).sum(1)).mean()
        x = (users_emb*pos_emb).sum(1) - (users_emb*neg_emb).sum(1)
        loss = -F.logsigmoid(x).mean()
        return loss, reg

import dgl.nn as dglnn

class ItemGCN(nn.Module):
    def __init__(self, in_dim, out_dim, num_layers=2, num_heads=8, p_drop=0.1):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        
        self.layers = nn.ModuleList([
            dglnn.GATConv(out_dim, out_dim // num_heads, num_heads, feat_drop=p_drop, attn_drop=p_drop)
            for _ in range(max(0, num_layers))
        ])
        
        self.act = nn.LeakyReLU(0.2)
        self.drop = nn.Dropout(p_drop)
        
    def forward(self, g: dgl.DGLGraph, feats: torch.Tensor):
        
        g = dgl.add_self_loop(g)

        x = self.proj(feats)
        
        for layer in self.layers:
            x = layer(g, x)  # [num_nodes, num_heads, head_dim]
            x = x.view(x.shape[0], -1) 
            x = self.act(x)
            x = self.drop(x)
            
        return x
    

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, num_layers=2, p_drop=0.1):
        super(MLP, self).__init__()
        
        assert num_layers >= 2, "num_layers must be at least 2"
        
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(p_drop)
        
        hidden_dim = max(in_dim // 2, 128)
        
        self.layers.append(nn.Linear(in_dim, hidden_dim))
        
        for i in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            
        self.layers.append(nn.Linear(hidden_dim, out_dim))
        
        self.batch_norms = nn.ModuleList()
        for i in range(num_layers - 1):
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
    def forward(self, x):
        for i in range(len(self.layers) - 1):
            x = self.layers[i](x)
            if x.shape[0] > 1: 
                x = self.batch_norms[i](x)
            x = F.relu(x)
            x = self.dropout(x)
        
        x = self.layers[-1](x)
        return x
    
class FusionModule(nn.Module):
    def __init__(self, dim, mode='oat', p_drop=0.1, attn_heads=4, attn_dropout=0.1, attn_temp=None, alpha=0.05, beta=0.8):
        super().__init__()
        assert mode in ['concat', 'sum', 'oat']
        self.mode = mode
        self.dim = dim
        if self.mode == 'oat':
            print("OAT init done!")
            # Sinkhorn para
            self.epsilon = 1e-2
            self.max_iter = 100  
            self.scale = 100    

            # Transport para
            self.delta_img = nn.Parameter(torch.Tensor(dim, dim), requires_grad=True)  
            self.delta_txt = nn.Parameter(torch.Tensor(dim, dim), requires_grad=True)  
            nn.init.xavier_uniform_(self.delta_img)  
            nn.init.xavier_uniform_(self.delta_txt)

            # weight
            self.alpha = nn.Parameter(torch.tensor(alpha), requires_grad=True)  
            self.beta = nn.Parameter(torch.tensor(beta), requires_grad=True)   
            self.weight_softmax = nn.Softmax(dim=0)
            print('alpha: ', self.alpha, 'beta: ', self.beta, 'gamma: ', 1 - self.alpha - self.beta)

            # post-process
            self.ln = nn.LayerNorm(dim)
            self.drop = nn.Dropout(p_drop)

        elif mode == 'concat':
            self.proj = nn.Linear(3*dim, dim)
            self.ln = nn.LayerNorm(dim)
            self.drop = nn.Dropout(p_drop)
        elif mode == 'sum':
            self.proj = nn.Linear(dim, dim)
            self.ln = nn.LayerNorm(dim)
            self.drop = nn.Dropout(p_drop)
        else:
            pass


    def sinkhorn(self, mu, nu, cost_matrix):
        device = cost_matrix.device
        n, m = cost_matrix.shape

        K = torch.exp(-cost_matrix / self.epsilon)  # [dim, dim]
        u = torch.ones(n, device=device) / n       
        v = torch.ones(m, device=device) / m        

        # Sinkhorn iteration
        for _ in range(self.max_iter):
            u = mu / (K @ v + 1e-8)  
            v = nu / (K.T @ u + 1e-8)

        # optimal T
        T = torch.diag(u) @ K @ torch.diag(v)
        return T

    def cal_ot_alignment(self, mm_emb, st_emb, delta):

        device = mm_emb.device
        I, dim = mm_emb.shape

        mu = torch.ones(dim, device=device) / dim  
        nu = torch.ones(dim, device=device) / dim  


        cost_matrix = torch.pow(mm_emb.unsqueeze(2) - st_emb.unsqueeze(1), 2).mean(dim=0) * self.scale
        # cost_matrix = torch.abs(mm_emb.unsqueeze(2) - st_emb.unsqueeze(1)).mean(dim=0) * self.scale

        with torch.no_grad():
            T = self.sinkhorn(mu, nu, cost_matrix)  # [dim, dim]

        aligned_mm = mm_emb @ (T + delta)  # [I, dim] @ [dim, dim] → [I, dim]
       
        return aligned_mm
    
    def forward(self, z_cf, z_img, z_txt):

        if self.mode == 'oat':
            # z_img → z_cf
            aligned_img = self.cal_ot_alignment(z_img, z_cf, self.delta_img)
            # z_txt → z_cf
            aligned_txt = self.cal_ot_alignment(z_txt, z_cf, self.delta_txt)

            # weights = self.weight_softmax(torch.stack([self.alpha, self.beta, 1 - self.alpha - self.beta]))
            # alpha, beta, gamma = weights[0], weights[1], weights[2]
            z_fused = (1 - self.alpha - self.beta) * z_cf + self.alpha * aligned_img + self.beta * aligned_txt
            # z_fused = z_cf + aligned_img + aligned_txt

            z_fused = self.ln(z_fused)
            z_fused = self.drop(z_fused)
            return F.normalize(z_fused, dim=-1)
          
        if self.mode == 'concat':
            z = torch.cat([z_cf, z_img, z_txt], dim=-1)
            z = self.proj(z)
            z = self.ln(z)
            z = self.drop(z)
            return F.normalize(z, dim=-1)
        
        if self.mode == 'sum':
            z = z_cf + z_img + z_txt
            z = self.proj(z)
            z = self.ln(z)
            z = self.drop(z)
            return F.normalize(z, dim=-1)
        
        return None
    
class MultiModalLightGCN(nn.Module):
    def __init__(self, config, dataset, cf_backbone: LightGCN, g_text: dgl.DGLGraph=None, g_image: dgl.DGLGraph=None):
        super().__init__()
        self.cf = cf_backbone
        self.config = config
        self.n_users = dataset.n_users
        self.m_items = dataset.m_items
        d = config.recdim
        txt_path = config.text_feat
        img_path = config.image_feat
        audio_path = config.audio_feat
        self.item_text_feat = None
        self.item_image_feat = None
        self.item_audio_feat = None
        if txt_path and os.path.exists(txt_path):
            import numpy as np
            self.item_text_feat = torch.from_numpy(np.load(txt_path)).float().to(config.dev)
        if img_path and os.path.exists(img_path):
            import numpy as np
            arr = np.load(img_path)
            # float32
            self.item_image_feat = torch.from_numpy(arr.astype('float32', copy=False)).float().to(config.dev)
        if audio_path and os.path.exists(audio_path):
            import numpy as np
            arr = np.load(audio_path)
            # float32
            self.item_audio_feat = torch.from_numpy(arr.astype('float32', copy=False)).float().to(config.dev)
            in_aud = self.item_audio_feat.size(1) if self.item_audio_feat is not None else d
            self.aud_branch = MLP(in_dim=in_aud, out_dim=d, num_layers=2, p_drop=config.fuse_drop)
        self.g_text = g_text
        self.g_image = g_image
        # branches
        in_txt = self.item_text_feat.size(1) if self.item_text_feat is not None else d
        in_img = self.item_image_feat.size(1) if self.item_image_feat is not None else d
        L = config.item_branch_layers
        self.txt_branch = ItemGCN(in_dim=in_txt, out_dim=d, num_layers=L, p_drop=config.fuse_drop)
        self.img_branch = ItemGCN(in_dim=in_img, out_dim=d, num_layers=L, p_drop=config.fuse_drop)
        self.fuse = FusionModule(dim=d, mode=config.fusion, p_drop=config.fuse_drop,
                                 attn_heads=config.attn_heads, attn_dropout=config.attn_dropout, alpha=config.alpha, beta=config.beta)

    def compute_embeddings(self):
        zu_cf, zi_cf = self.cf.computer()  # [U,d], [I,d]
        zi_txt = self.txt_branch(self.g_text, self.item_text_feat) if (self.g_text is not None and self.item_text_feat is not None) else zi_cf
        zi_img = self.img_branch(self.g_image, self.item_image_feat) if (self.g_image is not None and self.item_image_feat is not None) else zi_cf
        if self.config.audio_feat:
            zi_aud = self.aud_branch(self.item_audio_feat)
        zi_fuse = self.fuse(zi_cf, zi_img, zi_txt)
        return zu_cf, zi_fuse, zi_cf, zi_img, zi_txt

    def getUsersRating(self, users):
        z_u, z_i_fuse, _, _, _ = self.compute_embeddings()
        return z_u[users.long()] @ z_i_fuse.t()

    def calculate_loss(self, users, pos, neg):
        z_u, z_i_fuse, z_i_cf, z_i_img, z_i_txt = self.compute_embeddings()
        u = z_u[users.long()]
        ip = z_i_fuse[pos.long()]
        ineg = z_i_fuse[neg.long()]
        x = (u*ip).sum(1) - (u*ineg).sum(1)
        bpr_loss = -F.logsigmoid(x).mean()
        reg = (self.cf.embedding_user(users.long()).pow(2).sum(1) \
              + self.cf.embedding_item(pos.long()).pow(2).sum(1) \
              + self.cf.embedding_item(neg.long()).pow(2).sum(1)).mean()
        
        # CMCL loss 
        contrastive_loss = self._compute_multimodal_contrastive_loss(
            users, pos, u, ip, neg, z_i_cf, z_i_img, z_i_txt
        )
        
        total_loss = bpr_loss + self.config.decay * reg + self.config.contrastive_weight * contrastive_loss
        return total_loss
    
    def InfoNCE(self, view1, view2, temperature):
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
        pos_score = (view1 * view2).sum(dim=-1)
        pos_score = torch.exp(pos_score / temperature)
        ttl_score = torch.matmul(view1, view2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
        cl_loss = -torch.log(pos_score / ttl_score)
        return torch.mean(cl_loss)
    
    def _compute_multimodal_contrastive_loss(self, users, pos, users_emb, pos_emb, neg, 
                                           z_i_cf, z_i_img, z_i_txt):
                

        loss_ui= self.InfoNCE(users_emb, pos_emb, self.config.temperature) 
        pos_indices = pos.long()
        cf_emb = z_i_cf[pos_indices]
        img_emb = z_i_img[pos_indices] if z_i_img is not None else cf_emb
        txt_emb = z_i_txt[pos_indices] if z_i_txt is not None else cf_emb
        
        loss_modality = 0.0
        modality_count = 0
        
        # CF-Image
        if z_i_img is not None:
            cf_img_sim = self.InfoNCE(cf_emb, img_emb, self.config.temperature)
            loss_modality += (1 - cf_img_sim)
            modality_count += 1
        
        # CF-Text 
        if z_i_txt is not None:
            cf_txt_sim = self.InfoNCE(cf_emb, txt_emb, self.config.temperature)
            loss_modality += (1 - cf_txt_sim)
            modality_count += 1
        
        # Image-Text
        if z_i_img is not None and z_i_txt is not None:
            img_txt_sim = self.InfoNCE(img_emb, txt_emb, self.config.temperature)
            loss_modality += (1 - img_txt_sim)
            modality_count += 1

        neg_indices = neg.long()
        cf_emb_neg = z_i_cf[neg_indices]
        img_emb_neg = z_i_img[neg_indices] if z_i_img is not None else cf_emb
        txt_emb_neg = z_i_txt[neg_indices] if z_i_txt is not None else cf_emb
        
        if z_i_img is not None:
            cf_img_sim_neg = self.InfoNCE(cf_emb_neg, img_emb_neg, self.config.temperature)
            loss_modality += (1 - cf_img_sim_neg)
            modality_count += 1
        
        if z_i_txt is not None:
            cf_txt_sim_neg = self.InfoNCE(cf_emb_neg, txt_emb_neg, self.config.temperature)
            loss_modality += (1 - cf_txt_sim_neg)
            modality_count += 1
        
        if z_i_img is not None and z_i_txt is not None:
            img_txt_sim_neg = self.InfoNCE(img_emb_neg, txt_emb_neg, self.config.temperature)
            loss_modality += (1 - img_txt_sim_neg)
            modality_count += 1
        
        if modality_count > 0:
            loss_modality = loss_modality / modality_count
        else:
            loss_modality = torch.tensor(0.0, device=users_emb.device)
        
        total_contrastive_loss = (1 - self.config.modality_consistency_weight ) * loss_ui + self.config.modality_consistency_weight * loss_modality

        return total_contrastive_loss

from utils import build_item_knn_graph as build_item_knn_graph
from utils import load_or_build_item_graph as load_or_build_item_graph
