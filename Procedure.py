import multiprocessing
import numpy as np
import torch
from utils import timer
import world
import random

@torch.no_grad()
def test_one_batch(X):
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    r = []
    for i in range(len(sorted_items)):
        r.append(np.isin(sorted_items[i], groundTrue[i]))
    return np.array(r).astype('float')


def getLabel(test_data, pred_data):
    labels = []
    for i in range(len(test_data)):
        try:
            pos = test_data[i]
        except Exception:
            pos = []
        labels.append(np.isin(pred_data[i], pos))
    return np.array(labels).astype('float')


@torch.no_grad()
def Test(dataset, Recmodel, epoch, w=None, multicore=0):
    u_batch_size = world.config.TEST_U_BATCH_SIZE
    testDict = dataset.testDict

    users = list(testDict.keys())
    try:
        assert u_batch_size <= len(users)
    except:
        u_batch_size = len(users)
    n_users = len(users)
    n_items = dataset.m_items
    results = {'recall': np.zeros(len(world.config.topks)),
               'ndcg': np.zeros(len(world.config.topks))}

    with torch.no_grad():
        users_list = []
        rating_list = []
        groundTrue_list = []
        for batch_users in np.array_split(users, max(1, n_users // u_batch_size)):
            batch_users = torch.tensor(batch_users).long().to(world.config.dev)
            rating = Recmodel.getUsersRating(batch_users)
            # filter items seen in train
            for u_i, u in enumerate(batch_users.tolist()):
                seen = set(dataset.user_item_dict.get(u, []))
                if seen:
                    rating[u_i, list(seen)] = -(1<<10)
            # get top-K
            _, topK = torch.topk(rating, k=max(world.config.topks), dim=1)
            rating = rating.cpu()
            users_list.append(batch_users.cpu())
            rating_list.append(topK.cpu())
            groundTrue_list.append([testDict[u] for u in batch_users.cpu().numpy()])

    X = zip(rating_list, groundTrue_list)
    R = []
    for x in X:
        R.append(test_one_batch(x))
    R = np.concatenate(R, axis=0)

    for i, K in enumerate(world.config.topks):
        recall = np.mean([ (R[idx][:K].sum()/len(testDict[users[idx]])) if len(testDict[users[idx]])>0 else 0 for idx in range(len(users)) ])
        # ndcg
        ndcg = 0.0
        for idx in range(len(users)):
            pos = set(testDict[users[idx]])
            gain = 0.0
            for rank, hit in enumerate(R[idx][:K]):
                if hit:
                    gain += 1.0/np.log2(rank+2)
            ideal = min(len(pos), K)
            idcg = sum(1.0/np.log2(r+2) for r in range(ideal)) if ideal>0 else 0.0
            ndcg += (0.0 if idcg==0.0 else gain/idcg)
        ndcg /= len(users)
        results['recall'][i] = recall
        results['ndcg'][i] = ndcg

    print(f"[Test@{epoch}]", {k: v for k,v in results.items()})
    return results


def train_v2(dataset, recommend_model, loss_class, epoch, neg_k=1000, w=None):
    Recmodel = recommend_model
    opt = loss_class
    CONFIG = world.config
    S = timer()
    total_loss = 0.0
    
    n_batch = int(dataset.trainDataSize / CONFIG.bpr_batch) + 1
    indices = list(range(dataset.trainDataSize))
    random.shuffle(indices)
    
    for batch_idx in range(n_batch):
        start_idx = batch_idx * CONFIG.bpr_batch
        end_idx = min((batch_idx + 1) * CONFIG.bpr_batch, dataset.trainDataSize)
        
        if start_idx >= end_idx:
            continue
            
        batch_indices = indices[start_idx:end_idx]
        
        users, posItems, negItems = dataset.sample_cl(batch_indices)
        
        users = torch.tensor(users).long().to(CONFIG.dev)
        posItems = torch.tensor(posItems).long().to(CONFIG.dev)
        negItems = torch.tensor(negItems).long().to(CONFIG.dev)
        
        l = opt.stageOne_v2(users, posItems, negItems)
        total_loss += l
    
    avg_loss = total_loss / n_batch
    return f"loss{avg_loss:.3f}-{S.record():.3f}s"
