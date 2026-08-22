import os
from typing import Dict, List, Tuple
import numpy as np
import torch
import scipy.sparse as sp
from torch.utils.data import Dataset
import world


class BasicDataset(Dataset):
    def __init__(self):
        super().__init__()

    @property
    def n_users(self) -> int:
        raise NotImplementedError

    @property
    def m_items(self) -> int:
        raise NotImplementedError

    @property
    def trainDataSize(self) -> int:
        raise NotImplementedError

    def getSparseGraph(self) -> torch.Tensor:
        raise NotImplementedError


class Loader(BasicDataset):
    def __init__(self, config=world):
        super().__init__()
        data_dir = world.dataset_path(world.config.dataset)
        train_file = os.path.join(data_dir, "train.txt")
        test_file  = os.path.join(data_dir, "test.txt")

        self.user_item_dict: Dict[int, List[int]] = {}
        train_users: List[int] = []
        train_items: List[int] = []
        train_unique_users: List[int] = []

        with open(train_file, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                parts = list(map(int, line.strip().split()))
                u, items = parts[0], parts[1:]
                if not items:
                    continue
                items = list(set(items))
                train_unique_users.append(u)
                train_users.extend([u] * len(items))
                train_items.extend(items)
                self.user_item_dict[u] = items

        self.trainUser = np.asarray(train_users, dtype=np.int64)
        self.trainItem = np.asarray(train_items, dtype=np.int64)

        self.testDict: Dict[int, List[int]] = {}
        with open(test_file, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                parts = list(map(int, line.strip().split()))
                u, items = parts[0], list(set(parts[1:]))  
                self.testDict[u] = items

        max_user_id = -1
        if train_unique_users:
            max_user_id = max(max_user_id, max(train_unique_users))
        if self.testDict:
            max_user_id = max(max_user_id, max(self.testDict.keys()))
        self._n_users = int(max_user_id + 1)

        max_item_id = -1
        if self.trainItem.size > 0:
            max_item_id = max(max_item_id, int(self.trainItem.max()))
        if self.testDict:
            for itms in self.testDict.values():
                if itms:
                    max_item_id = max(max_item_id, max(itms))
        self._m_items = int(max_item_id + 1)

        self._trainDataSize = int(self.trainItem.size)

        self.Graph: torch.Tensor = None
        self.R_norm = None

    @property
    def n_users(self) -> int:
        return self._n_users

    @property
    def m_items(self) -> int:
        return self._m_items

    @property
    def trainDataSize(self) -> int:
        return self._trainDataSize

    def __len__(self):
        return self.n_users

    def getUserItemFeedback(self, users, items):
        return np.zeros((len(users), len(items)), dtype=np.float32)

    def getSparseGraph(self) -> torch.Tensor:
        if self.Graph is not None:
            return self.Graph

        print("Building normalized adjacency (A_hat) from train edges ...")
        device = world.config.dev
        n_nodes = self.n_users + self.m_items

        users = self.trainUser
        items = self.trainItem

        if users.size == 0:
            indices = torch.empty((2, 0), dtype=torch.long, device=device)
            values = torch.empty((0,), dtype=torch.float32, device=device)
            self.Graph = torch.sparse_coo_tensor(indices, values, (n_nodes, n_nodes)).coalesce()
            return self.Graph

        pairs = np.unique(np.stack([users, items], axis=1), axis=0)
        u = pairs[:, 0]
        i = pairs[:, 1]

        row = np.concatenate([u, i + self.n_users])
        col = np.concatenate([i + self.n_users, u])
        data = np.ones_like(row, dtype=np.float32)

        adj = sp.coo_matrix((data, (row, col)), shape=(n_nodes, n_nodes), dtype=np.float32)
        rowsum = np.array(adj.sum(1)).squeeze() 

        d_inv_sqrt = np.power(rowsum, -0.5, where=rowsum > 0)
        D_inv_sqrt = sp.diags(d_inv_sqrt)
        norm_adj = D_inv_sqrt @ adj @ D_inv_sqrt 

        coo = norm_adj.tocoo()
        indices = torch.tensor(np.vstack([coo.row, coo.col]), dtype=torch.long, device=device)
        values = torch.tensor(coo.data, dtype=torch.float32, device=device)
        self.Graph = torch.sparse_coo_tensor(indices, values, (n_nodes, n_nodes)).coalesce()

        return self.Graph

    def getBipartiteRNorm(self) -> torch.Tensor:
        if self.R_norm is not None:
            return self.R_norm
        device = world.config.dev
        if self.trainDataSize == 0:
            self.R_norm = torch.sparse_coo_tensor(
                torch.empty((2,0), dtype=torch.long, device=device),
                torch.empty((0,), dtype=torch.float32, device=device),
                (self.n_users, self.m_items), device=device
            ).coalesce()
            return self.R_norm

        pairs = np.unique(np.stack([self.trainUser, self.trainItem], axis=1), axis=0)
        u = pairs[:, 0].astype(np.int64)
        i = pairs[:, 1].astype(np.int64)

        deg_u = np.bincount(u, minlength=self.n_users).astype(np.float32)
        deg_i = np.bincount(i, minlength=self.m_items).astype(np.float32)
        deg_u[deg_u == 0] = 1.0
        deg_i[deg_i == 0] = 1.0

        w = 1.0 / np.sqrt(deg_u[u] * deg_i[i])

        indices = torch.tensor(np.vstack([u, i]), dtype=torch.long, device=device)
        values = torch.tensor(w, dtype=torch.float32, device=device)
        R = torch.sparse_coo_tensor(indices, values, (self.n_users, self.m_items), device=device).coalesce()
        self.R_norm = R
        return self.R_norm
    
    def sample_cl(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:

        if self.trainDataSize == 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.int64), np.array([], dtype=np.int64)
        
        idx = batch_size
        users = self.trainUser[idx]
        posItems = self.trainItem[idx]
        negItems = np.random.randint(0, self.m_items, size=len(batch_size), dtype=np.int64)
        
        return users, posItems, negItems

    def _build_user_positive_mapping(self):
        self.user_positive_items = {}
        for user, item in zip(self.trainUser, self.trainItem):
            if user not in self.user_positive_items:
                self.user_positive_items[user] = set()
            self.user_positive_items[user].add(item)