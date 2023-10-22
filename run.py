import torch
from torch_geometric.utils import degree
from src.models.lightgcn import LightGCN
from src.models.ngcf import NGCF
from torch_geometric.datasets import MovieLens100K, AmazonBook
from tqdm import tqdm


def train(model, data, train_loader, train_edge_label_index, num_users, num_movie):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    total_loss = total_examples = 0

    for index in tqdm(train_loader):
        # Sample positive and negative labels.
        pos_edge_label_index = train_edge_label_index[:, index]
        neg_edge_label_index = torch.stack([
            pos_edge_label_index[0],
            torch.randint(num_users, num_users + num_movie, (index.numel(), ))
        ], dim=0)
        edge_label_index = torch.cat([pos_edge_label_index, neg_edge_label_index], dim=1)

        optimizer.zero_grad()
        pos_rank, neg_rank = model(data.edge_index, edge_label_index).chunk(2)

        loss = model.recommendation_loss(
            pos_rank,
            neg_rank,
            node_id=edge_label_index.unique(),
        )
        loss.backward()
        optimizer.step()

        total_loss += float(loss) * pos_rank.numel()
        total_examples += pos_rank.numel()

    return total_loss / total_examples


@torch.no_grad()
def test(model, data, train_edge_label_index, num_users, k: int, batch_size: int):
    emb = model.get_embedding(data.edge_index)
    user_emb, movie_emb = emb[:num_users], emb[num_users:]

    precision = recall = total_examples = 0
    for start in range(0, num_users, batch_size):
        end = start + batch_size
        logits = user_emb[start:end] @ movie_emb.t()

        # Exclude training edges:
        mask = ((train_edge_label_index[0] >= start) &
                (train_edge_label_index[0] < end))
        logits[train_edge_label_index[0, mask] - start,
               train_edge_label_index[1, mask] - num_users] = float('-inf')

        # Computing precision and recall:
        ground_truth = torch.zeros_like(logits, dtype=torch.bool)
        mask = ((data.edge_label_index[0] >= start) &
                (data.edge_label_index[0] < end))
        ground_truth[data.edge_label_index[0, mask] - start,
                     data.edge_label_index[1, mask] - num_users] = True
        node_count = degree(data.edge_label_index[0, mask] - start,
                            num_nodes=logits.size(0))

        topk_index = logits.topk(k, dim=-1).indices
        isin_mat = ground_truth.gather(1, topk_index)

        precision += float((isin_mat.sum(dim=-1) / k).sum())
        recall += float((isin_mat.sum(dim=-1) / node_count.clamp(1e-6)).sum())
        total_examples += int((node_count > 0).sum())

    return precision / total_examples, recall / total_examples


def main():
    path = '../../data/raw/ml-100k'
    dataset = MovieLens100K(path)
    data = dataset[0]
    # to make user index before movie index
    data.rename('movie', 'Movie')
    num_users, num_items = data['user'].num_nodes, data['Movie'].num_nodes
    data = data.to_homogeneous()
    # Use all message passing edges as training labels:
    batch_size = 81920
    # remove duplicate edges
    mask = data.edge_index[0] < data.edge_index[1]
    train_edge_label_index = data.edge_index[:, mask]
    train_loader = torch.utils.data.DataLoader(
        range(train_edge_label_index.size(1)),
        shuffle=True,
        batch_size=batch_size,
    )
    model = NGCF(
        num_nodes=data.num_nodes,
        embedding_dim=64,
        num_layers=2,
    )
    for epoch in range(1, 10):
        loss = train(model, data, train_loader, train_edge_label_index, num_users, num_items)
        precision, recall = test(model, data, train_edge_label_index, num_users, 20, batch_size)
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Precision@20: '
            f'{precision:.4f}, Recall@20: {recall:.4f}')
    


if __name__ == '__main__':
    main()