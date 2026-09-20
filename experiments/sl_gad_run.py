"""DGL-free SL-GAD runner.

The official SL-GAD code (KimMeen/SL-GAD) needs dgl==0.4.1 / torch==1.8.1 only
for one call: `dgl.contrib.sampling.random_walk_with_restart` inside its
`generate_rwr_subgraph`.  Everything else (model.py) is pure PyTorch.  We keep
the network (sl_gad_model.py, copied verbatim) and the exact train / inference /
scoring tensor logic from run.py, and replace only the RWR sampler with a NumPy
implementation and the .mat loader with an adapter for our PyGOD datasets.

The original sampler runs restart_prob=1.0 (first pass) which collects the
target's immediate (1-hop) neighbours, falling back to a wider walk for
low-degree nodes and repeat-padding isolated ones.  We reproduce that regime.

Validation protocol: run first on the repo's own cora.mat and confirm the AUC
matches SL-GAD's published Cora number (~0.913).  Only then trust the numbers on
our datasets.

Usage:
  python sl_gad_run.py --dataset cora            # SL-GAD's own cora.mat (validation)
  python sl_gad_run.py --ours enron              # our PyGOD dataset via GraphStore
"""
import os, sys, json, time, random, argparse, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import scipy.sparse as sp
import scipy.io as sio
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import MinMaxScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sl_gad_model import Model

# .mat datasets from the official SL-GAD release; override with SLGAD_MAT_DIR.
MAT_DIR = os.environ.get("SLGAD_MAT_DIR", os.path.join(HERE, "..", "data", "SL-GAD", "dataset"))


# ----------------------------- preprocessing (from utils.py) ----------------
def preprocess_features(features):
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    return r_mat_inv.dot(features)


def normalize_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


# ----------------------------- data loaders --------------------------------
def load_mat(dataset):
    data = sio.loadmat(os.path.join(MAT_DIR, "{}.mat".format(dataset)))
    label = data['Label'] if ('Label' in data) else data['gnd']
    attr = data['Attributes'] if ('Attributes' in data) else data['X']
    network = data['Network'] if ('Network' in data) else data['A']
    adj = sp.csr_matrix(network)
    feat = sp.lil_matrix(attr)
    ano_labels = np.squeeze(np.array(label))
    return adj, feat, ano_labels


def load_ours(key):
    """Adapt one of our PyGOD datasets to (adj_csr, feat_lil, ano_label)."""
    from torch_geometric.data import Data
    from torch_geometric.data.storage import GlobalStorage, NodeStorage, EdgeStorage
    torch.serialization.add_safe_globals([GlobalStorage, NodeStorage, EdgeStorage, Data])
    from arcade.data.loader import GraphStore
    store = GraphStore(); store.load_dataset(key)
    d = store.data
    n = d.num_nodes
    ei = d.edge_index.cpu().numpy()
    adj = sp.coo_matrix((np.ones(ei.shape[1]), (ei[0], ei[1])), shape=(n, n))
    adj = adj + adj.T
    adj.data[:] = 1.0
    adj = adj.tocsr(); adj.setdiag(0); adj.eliminate_zeros()
    x = d.x.cpu().numpy().astype(np.float64)
    feat = sp.lil_matrix(x)
    ano = (np.asarray(store.labels) > 0).astype(int)
    return adj, feat, ano


# ----------------------------- NumPy RWR sampler ---------------------------
def rwr_subgraphs(indptr, indices, nb_nodes, subgraph_size, rng):
    """Return, for every node, a subgraph of `subgraph_size` node ids with the
    target node placed last.  Mirrors SL-GAD's restart_prob=1.0 regime: sample
    reduced_size distinct 1-hop neighbours; widen to 2-hop for low-degree nodes;
    repeat-pad if still short; self-fill isolated nodes."""
    reduced = subgraph_size - 1
    subv = []
    for i in range(nb_nodes):
        neigh = indices[indptr[i]:indptr[i + 1]]
        neigh = neigh[neigh != i]
        if len(neigh) >= reduced:
            sub = rng.choice(neigh, size=reduced, replace=False).tolist()
        else:
            pool = set(int(x) for x in neigh)
            for nb in neigh:                          # widen to 2-hop
                nn2 = indices[indptr[nb]:indptr[nb + 1]]
                pool.update(int(x) for x in nn2)
            pool.discard(i)
            pool = list(pool)
            if len(pool) >= reduced:
                sub = rng.choice(pool, size=reduced, replace=False).tolist()
            elif len(pool) > 0:
                sub = (pool * reduced)[:reduced]
            else:
                sub = [i] * reduced                   # isolated node
        sub = [int(x) for x in sub]
        sub.append(int(i))
        subv.append(sub)
    return subv


# ----------------------------- main run ------------------------------------
def run(adj_csr, feat_lil, ano_label, args, device):
    raw_features = np.asarray(feat_lil.todense(), dtype=np.float64)
    features = np.asarray(preprocess_features(feat_lil.tocsr()).todense(), dtype=np.float64)

    nb_nodes, ft_size = features.shape
    indptr, indices = adj_csr.indptr, adj_csr.indices

    adj_n = normalize_adj(adj_csr)
    adj_n = (adj_n + sp.eye(adj_n.shape[0])).todense()

    features = torch.FloatTensor(features[np.newaxis]).to(device)
    raw_features = torch.FloatTensor(raw_features[np.newaxis]).to(device)
    adj = torch.FloatTensor(np.asarray(adj_n)[np.newaxis]).to(device)

    subgraph_size = args.subgraph_size
    batch_size = args.batch_size

    seed = args.seed
    np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed); random.seed(seed)
    rng = np.random.RandomState(seed)

    model = Model(ft_size, args.embedding_dim, 'prelu', args.negsamp_ratio, args.readout).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    b_xent = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor([args.negsamp_ratio]).to(device))
    mse_loss = nn.MSELoss(reduction='mean')

    batch_num = nb_nodes // batch_size + 1

    def assemble(idx, subgraphs_1, subgraphs_2):
        cur_bs = len(idx)
        adj_zero_row = torch.zeros((cur_bs, 1, subgraph_size)).to(device)
        adj_zero_col = torch.zeros((cur_bs, subgraph_size + 1, 1)).to(device)
        adj_zero_col[:, -1, :] = 1.
        feat_zero_row = torch.zeros((cur_bs, 1, ft_size)).to(device)
        ba1, ba2, bf1, bf2, rbf1, rbf2 = [], [], [], [], [], []
        for i in idx:
            s1, s2 = subgraphs_1[i], subgraphs_2[i]
            ba1.append(adj[:, s1, :][:, :, s1]); ba2.append(adj[:, s2, :][:, :, s2])
            bf1.append(features[:, s1, :]);      bf2.append(features[:, s2, :])
            rbf1.append(raw_features[:, s1, :]); rbf2.append(raw_features[:, s2, :])
        ba1 = torch.cat(ba1); ba1 = torch.cat((ba1, adj_zero_row), 1); ba1 = torch.cat((ba1, adj_zero_col), 2)
        ba2 = torch.cat(ba2); ba2 = torch.cat((ba2, adj_zero_row), 1); ba2 = torch.cat((ba2, adj_zero_col), 2)
        bf1 = torch.cat(bf1); bf1 = torch.cat((bf1[:, :-1, :], feat_zero_row, bf1[:, -1:, :]), 1)
        bf2 = torch.cat(bf2); bf2 = torch.cat((bf2[:, :-1, :], feat_zero_row, bf2[:, -1:, :]), 1)
        rbf1 = torch.cat(rbf1); rbf1 = torch.cat((rbf1[:, :-1, :], feat_zero_row, rbf1[:, -1:, :]), 1)
        rbf2 = torch.cat(rbf2); rbf2 = torch.cat((rbf2[:, :-1, :], feat_zero_row, rbf2[:, -1:, :]), 1)
        return ba1, ba2, bf1, bf2, rbf1, rbf2

    # ---- training ----
    best = 1e9; best_state = None; cnt_wait = 0
    for epoch in range(args.num_epoch):
        model.train()
        all_idx = list(range(nb_nodes)); random.shuffle(all_idx)
        total_loss = 0.
        subgraphs_1 = rwr_subgraphs(indptr, indices, nb_nodes, subgraph_size, rng)
        subgraphs_2 = rwr_subgraphs(indptr, indices, nb_nodes, subgraph_size, rng)
        for batch_idx in range(batch_num):
            optimiser.zero_grad()
            is_final = (batch_idx == batch_num - 1)
            idx = all_idx[batch_idx * batch_size:] if is_final else all_idx[batch_idx * batch_size:(batch_idx + 1) * batch_size]
            cur_bs = len(idx)
            if cur_bs == 0:
                continue
            lbl = torch.unsqueeze(torch.cat((torch.ones(cur_bs),
                                             torch.zeros(cur_bs * args.negsamp_ratio))), 1).to(device)
            ba1, ba2, bf1, bf2, rbf1, rbf2 = assemble(idx, subgraphs_1, subgraphs_2)
            logits, f_1, f_2 = model(bf1, bf2, rbf1, rbf2, ba1, ba2)
            loss1 = torch.mean(b_xent(logits, lbl))
            loss2 = 0.5 * (mse_loss(f_1[:, -2, :], rbf1[:, -1, :]) + mse_loss(f_2[:, -2, :], rbf2[:, -1, :]))
            loss = args.alpha * loss1 + args.beta * loss2
            loss.backward(); optimiser.step()
            loss_v = loss.detach().cpu().numpy()
            if not is_final:
                total_loss += loss_v
        mean_loss = (total_loss * batch_size + loss_v * cur_bs) / nb_nodes
        if mean_loss < best:
            best = mean_loss; best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}; cnt_wait = 0
        else:
            cnt_wait += 1
        if cnt_wait == args.patience:
            print('Early stopping at epoch', epoch, flush=True); break
        if epoch % 20 == 0 or epoch == args.num_epoch - 1:
            print('  epoch {:4d}  loss {:.5f}'.format(epoch, float(mean_loss)), flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)

    # ---- testing ----
    model.eval()
    multi = np.zeros((args.auc_test_rounds, nb_nodes))
    multi_c = np.zeros((args.auc_test_rounds, nb_nodes))
    multi_g = np.zeros((args.auc_test_rounds, nb_nodes))
    for r in range(args.auc_test_rounds):
        all_idx = list(range(nb_nodes)); random.shuffle(all_idx)
        subgraphs_1 = rwr_subgraphs(indptr, indices, nb_nodes, subgraph_size, rng)
        subgraphs_2 = rwr_subgraphs(indptr, indices, nb_nodes, subgraph_size, rng)
        for batch_idx in range(batch_num):
            is_final = (batch_idx == batch_num - 1)
            idx = all_idx[batch_idx * batch_size:] if is_final else all_idx[batch_idx * batch_size:(batch_idx + 1) * batch_size]
            cur_bs = len(idx)
            if cur_bs == 0:
                continue
            ba1, ba2, bf1, bf2, rbf1, rbf2 = assemble(idx, subgraphs_1, subgraphs_2)
            with torch.no_grad():
                logits, dist = model.inference(bf1, bf2, rbf1, rbf2, ba1, ba2)
                logits = torch.sigmoid(torch.squeeze(logits))
            a1 = -(logits[:cur_bs] - logits[cur_bs:]).cpu().numpy()   # contrastive
            a2 = dist.cpu().numpy()                                    # generative
            a1n = MinMaxScaler().fit_transform(a1.reshape(-1, 1)).reshape(-1)
            a2n = MinMaxScaler().fit_transform(a2.reshape(-1, 1)).reshape(-1)
            multi_c[r, idx] = a1n
            multi_g[r, idx] = a2n
            multi[r, idx] = args.alpha * a1n + args.beta * a2n
        if r % 50 == 0:
            print('  test round', r, flush=True)
    final = np.mean(multi, axis=0)
    final_c = np.mean(multi_c, axis=0)
    final_g = np.mean(multi_g, axis=0)
    auc = roc_auc_score(ano_label, final) * 100
    ap = average_precision_score(ano_label, final) * 100
    if getattr(args, 'return_scores', False):
        return auc, ap, final, final_c, final_g
    return auc, ap


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', type=str, default=None, help="SL-GAD .mat dataset (validation)")
    p.add_argument('--ours', type=str, default=None, help="our PyGOD dataset key")
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight_decay', type=float, default=0.0)
    p.add_argument('--embedding_dim', type=int, default=64)
    p.add_argument('--num_epoch', type=int, default=100)
    p.add_argument('--patience', type=int, default=400)
    p.add_argument('--batch_size', type=int, default=300)
    p.add_argument('--subgraph_size', type=int, default=4)
    p.add_argument('--readout', type=str, default='avg')
    p.add_argument('--auc_test_rounds', type=int, default=256)
    p.add_argument('--negsamp_ratio', type=int, default=1)
    p.add_argument('--alpha', type=float, default=1.0)
    p.add_argument('--beta', type=float, default=0.6)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--out', type=str, default=None)
    args = p.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if args.ours:
        adj, feat, ano = load_ours(args.ours); name = args.ours
    else:
        adj, feat, ano = load_mat(args.dataset); name = args.dataset
    print("[{}] nodes={} feat={} anomalies={} (rate {:.3f})".format(
        name, adj.shape[0], feat.shape[1], int(ano.sum()), ano.mean()), flush=True)
    t = time.time()
    auc, ap = run(adj, feat, ano, args, device)
    print("[{}] AUROC={:.2f} AUPRC={:.2f}  ({:.0f}s)".format(name, auc, ap, time.time() - t), flush=True)
    if args.out:
        prev = {}
        if os.path.exists(args.out):
            prev = json.load(open(args.out))
        prev[name] = {"auroc": round(auc, 1), "auprc": round(ap, 1),
                      "epochs": args.num_epoch, "rounds": args.auc_test_rounds, "lr": args.lr}
        json.dump(prev, open(args.out, "w"), indent=2)


if __name__ == '__main__':
    main()
