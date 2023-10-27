from torch_geometric.utils import degree
import torch

@torch.no_grad()
def test(model, data, train_edge_label_index, num_users, k: int, batch_size: int, use_node_features):
    if use_node_features:
        input_x = data.x
    else:
        input_x = None
    emb = model.get_embedding(data.edge_index, input_x)
    user_emb, item_emb = emb[:num_users], emb[num_users:]

    precision = recall = hits = total_examples = 0
    for start in range(0, num_users, batch_size):
        end = start + batch_size
        logits = user_emb[start:end] @ item_emb.t()

        # Exclude training edges:
        mask = ((train_edge_label_index[0] >= start) &
                (train_edge_label_index[0] < end))
        logits[train_edge_label_index[0, mask] - start,
               train_edge_label_index[1, mask] - num_users] = float('-inf')

        # Computing precision, recall, and hits@k:
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
        hits += float((isin_mat.sum(dim=-1) > 0).sum())
        total_examples += int((node_count > 0).sum())

    return precision / total_examples, recall / total_examples, hits / num_users