import time
import numpy as np
import torch
import dgl
import os

class timer:
    def __init__(self):
        self.time = time.time()
    def record(self):
        now = time.time()
        span = now - self.time
        self.time = now
        return span

class Loss:
    def __init__(self, config, recmodel):
        self.config = config
        self.model = recmodel
        self.opt = torch.optim.Adam(self.model.parameters(), lr=config.lr, weight_decay=0.0)
    
    def stageOne_v2(self, users, pos, neg):
        loss = self.model.calculate_loss(users, pos, neg)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return loss.item()

def build_item_knn_graph(feats: torch.Tensor, k: int, device) -> dgl.DGLGraph:

    import torch.nn.functional as F
    I = feats.size(0)
    x = F.normalize(feats, dim=-1)               # [I,d]
    sims = x @ x.t()                             # [I,I]
    sims.fill_diagonal_(0)
    vals, idx = torch.topk(sims, k=k, dim=1)    # idx: [I,k] (device = feats.device)

    src_gpu = torch.arange(I, device=feats.device).unsqueeze(1).expand(-1, k).reshape(-1)
    dst_gpu = idx.reshape(-1)
    src = src_gpu.to('cpu')
    dst = dst_gpu.to('cpu')


    src_sym = torch.cat([src, dst], 0)
    dst_sym = torch.cat([dst, src], 0)

    g_cpu = dgl.graph((src_sym, dst_sym), num_nodes=I, device='cpu')
    g_cpu = dgl.to_simple(g_cpu) 

    deg = g_cpu.in_degrees().clamp(min=1).to(torch.float32)
    norm = 1.0 / torch.sqrt(deg)
    g_cpu.ndata['norm'] = norm
    g_cpu.apply_edges(lambda e: {'w': e.src['norm'] * e.dst['norm']})


    g = g_cpu.to(device)
    return g


def load_or_build_item_graph(feat_path: str, graph_path: str, k: int, device, force_build=False):

    if graph_path and os.path.exists(graph_path) and not force_build:
        try:
            g = dgl.load_graphs(graph_path)[0][0]
            return g.to(device)
        except Exception:
            pass
    if feat_path and os.path.exists(feat_path):
        import numpy as np
        feats_np = np.load(feat_path)
        feats = torch.from_numpy(feats_np).float()
        print("Successful construct g!")
        return build_item_knn_graph(feats, k=k, device=device)
    return None