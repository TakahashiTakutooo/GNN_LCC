from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.nn import Embedding
from torch.utils.data import DataLoader
from torch import nn

from torch_geometric.index import index2ptr
from torch_geometric.typing import WITH_PYG_LIB, WITH_TORCH_CLUSTER
from torch_geometric.utils import sort_edge_index
from torch_geometric.utils.num_nodes import maybe_num_nodes

import torch.nn.functional as F

class LabelContextEmbedding(torch.nn.Module):
    def __init__(
        self,
        edge_index: Tensor,
        embedding_dim: int,
        walk_length: int,
        labels: Tensor,
        test_mask: Tensor,
        num_walks: int = 1,
        p: float = 100.0,
        q: float = 0.01,
        num_negative_samples: int = 1,
        num_nodes: Optional[int] = None,
        sparse: bool = False
    ):
        super().__init__()

        if WITH_PYG_LIB and p == 1.0 and q == 1.0:
            self.random_walk_fn = torch.ops.pyg.random_walk
        elif WITH_TORCH_CLUSTER:
            self.random_walk_fn = torch.ops.torch_cluster.random_walk
        else:
            if p == 1.0 and q == 1.0:
                raise ImportError(f"'{self.__class__.__name__}' "
                                  f"requires either the 'pyg-lib' or "
                                  f"'torch-cluster' package")
            else:
                raise ImportError(f"'{self.__class__.__name__}' "
                                  f"requires the 'torch-cluster' package")

        self.num_nodes = maybe_num_nodes(edge_index, num_nodes)

        row, col = sort_edge_index(edge_index, num_nodes=self.num_nodes).cpu()
        self.rowptr, self.col = index2ptr(row, self.num_nodes), col

        edge_index_reversed = edge_index[[1, 0], :]
        row_back, col_back = sort_edge_index(edge_index_reversed, num_nodes=self.num_nodes).cpu()
        self.rowptr_back, self.col_back = index2ptr(row_back, self.num_nodes), col_back

        self.EPS = 1e-15
        self.embedding_dim = embedding_dim
        self.walk_length = walk_length
        self.context_size = walk_length + 1
        self.num_walks = num_walks
        self.p = p
        self.q = q
        self.num_negative_samples = num_negative_samples
        self.labels = labels
        self.test_mask = test_mask
        self.num_classes = len(torch.unique(labels))
        self.embedding = Embedding(self.num_nodes, embedding_dim, sparse=sparse)
        self.fc = nn.Linear(embedding_dim, self.num_classes)
        self.reset_parameters()

    def reset_parameters(self):
        self.embedding.reset_parameters()
        self.fc.reset_parameters()

    def forward(self, batch: Optional[Tensor] = None) -> Tensor:
        emb = self.embedding.weight
        return emb if batch is None else emb[batch]

    # forward walk loader
    def forwardloader(self, **kwargs) -> DataLoader:
        return DataLoader(range(self.num_nodes), collate_fn=self.sample_forward_walks,
                          **kwargs)

    # backward walk loader
    def backwardloader(self, **kwargs) -> DataLoader:
        return DataLoader(range(self.num_nodes), collate_fn=self.sample_backward_walks, 
                            **kwargs)

    # sibling walk loader
    def siblingloader(self, **kwargs) -> DataLoader:
        return DataLoader(range(self.num_nodes), collate_fn=self.sample_sibling_walks,
                          **kwargs)

    # guardian walk loader
    def guardianloader(self, **kwargs) -> DataLoader:
        return DataLoader(range(self.num_nodes), collate_fn=self.sample_guardian_walks,
                            **kwargs)

    # forward walk sample
    @torch.jit.export
    def pos_forward_sample(self, batch: Tensor) -> Tensor:
        batch = batch.repeat(self.num_walks)
        rw = self.random_walk_fn(self.rowptr, self.col, batch, self.walk_length, self.p, self.q)
        if not isinstance(rw, Tensor):
            rw = rw[0]
        walks = []
        num_walks_per_rw = (self.walk_length + 1) - self.context_size + 1
        for j in range(num_walks_per_rw):
            walks.append(rw[:, j:j + self.context_size])
        return torch.cat(walks, dim=0)

    # backward walk sample
    @torch.jit.export
    def pos_back_sample(self, batch: Tensor) -> Tensor:
        batch = batch.repeat(self.num_walks)
        rw = self.random_walk_fn(self.rowptr_back, self.col_back, batch, self.walk_length, self.p, self.q)
        if not isinstance(rw, Tensor):
            rw = rw[0]
        walks = []
        num_walks_per_rw = (self.walk_length + 1) - self.context_size + 1
        for j in range(num_walks_per_rw):
            walks.append(rw[:, j:j + self.context_size])
        return torch.cat(walks, dim=0)

    # sibling walk sample
    @torch.compile(mode="reduce-overhead")
    def pos_sibling_sample(self, batch: Tensor) -> Tensor:
        walks = []
        for node_val in batch:
            node = node_val.item()
            incoming_edges = slice(self.rowptr_back[node], self.rowptr_back[node + 1])
            incoming_nodes = self.col_back[incoming_edges]
            if len(incoming_nodes) > 0:
                source_node_idx = torch.randint(len(incoming_nodes), (1,)).item()
                source_node = incoming_nodes[source_node_idx].item()

                neighbors = self.col[self.rowptr[source_node]:self.rowptr[source_node + 1]]
                neighbors = neighbors[neighbors != node] # Exclude the node itself

                sampled_neighbors_list = []
                if len(neighbors) > 0:
                    num_to_sample = min(len(neighbors), self.context_size -1)
                    perm = torch.randperm(len(neighbors))
                    sampled_neighbors_list = neighbors[perm[:num_to_sample]].tolist()

                current_sampled_count = len(sampled_neighbors_list)
                if current_sampled_count < self.context_size - 1:
                    padding_needed = self.context_size - 1 - current_sampled_count
                    sampled_neighbors_list.extend([node] * padding_needed) # Pad with the node itself
                
                sampled_neighbors_tensor = torch.tensor(sampled_neighbors_list[:self.context_size-1], dtype=torch.long, device=batch.device)
                walk = torch.cat([torch.tensor([node], dtype=torch.long, device=batch.device), sampled_neighbors_tensor])

            else:
                walk = torch.full((self.context_size,), node, dtype=torch.long, device=batch.device)

            walks.append(walk)
        return torch.stack(walks)

    # guardian walk sample
    @torch.compile(mode="reduce-overhead")
    def pos_guardian_sample(self, batch: Tensor) -> Tensor: 
        if not isinstance(batch, Tensor): 
            batch = torch.tensor(batch, device=self.rowptr.device)
        walks = []
        for node_val in batch:
            node = node_val.item()
            child_edges = slice(self.rowptr[node], self.rowptr[node + 1])
            child_nodes = self.col[child_edges]
            if len(child_nodes) > 0:
                chosen_child_idx = torch.randint(len(child_nodes), (1,)).item()
                chosen_child = child_nodes[chosen_child_idx].item()
                
                parents = self.col_back[self.rowptr_back[chosen_child]: self.rowptr_back[chosen_child + 1]]
                parents = parents[parents != node]

                sampled_neighbors_list = []
                if len(parents) > 0:
                    num_to_sample = min(len(parents), self.context_size - 1)
                    perm = torch.randperm(len(parents))
                    sampled_neighbors_list = parents[perm[:num_to_sample]].tolist()
                
                current_sampled_count = len(sampled_neighbors_list)
                if current_sampled_count < self.context_size -1:
                    padding_needed = self.context_size -1 - current_sampled_count
                    sampled_neighbors_list.extend([node] * padding_needed)

                sampled_neighbors_tensor = torch.tensor(sampled_neighbors_list[:self.context_size-1], dtype=torch.long, device=batch.device)
                walk = torch.cat([torch.tensor([node], dtype=torch.long, device=batch.device), sampled_neighbors_tensor])
            else:
                walk = torch.full((self.context_size,), node, dtype=torch.long, device=batch.device)
            
            walks.append(walk)
        return torch.stack(walks)

    @torch.jit.export
    def neg_sample_forward_backward(self, batch: Tensor) -> Tensor:
        batch = batch.repeat(self.num_walks * self.num_negative_samples)
        rw = torch.randint(self.num_nodes, (batch.size(0), self.walk_length), dtype=batch.dtype, device=batch.device)
        rw = torch.cat([batch.view(-1, 1), rw], dim=-1)

        walks = []
        num_walks_per_rw = (self.walk_length + 1) - self.context_size + 1
        for j in range(num_walks_per_rw):
            walks.append(rw[:, j:j + self.context_size])
        return torch.cat(walks, dim=0)

    @torch.jit.export
    def neg_sample_sibling_guardian(self, batch: Tensor) -> Tensor:
        batch_repeated = batch.repeat(self.num_negative_samples)
        if self.context_size <= 1:
            rw_context = torch.empty((batch_repeated.size(0), 0), dtype=batch_repeated.dtype, device=batch_repeated.device)
        else:
            rw_context = torch.randint(self.num_nodes, (batch_repeated.size(0), self.context_size - 1), dtype=batch_repeated.dtype, device=batch_repeated.device)
        
        walks = torch.cat([batch_repeated.view(-1, 1), rw_context], dim=-1)
        return walks

    @torch.jit.export
    def sample_forward_walks(self, batch: Union[List[int], Tensor]) -> Tuple[Tensor, Tensor]: 
        if not isinstance(batch, Tensor):
            batch = torch.tensor(batch, device=self.rowptr.device)
        return self.pos_forward_sample(batch), self.neg_sample_forward_backward(batch)

    @torch.jit.export
    def sample_backward_walks(self, batch: Union[List[int], Tensor]) -> Tuple[Tensor, Tensor]: 
        if not isinstance(batch, Tensor):
            batch = torch.tensor(batch, device=self.rowptr.device)
        return self.pos_back_sample(batch), self.neg_sample_forward_backward(batch)

    @torch.jit.export
    def sample_sibling_walks(self, batch: Union[List[int], Tensor]) -> Tuple[Tensor, Tensor]:
        if not isinstance(batch, Tensor):
            batch = torch.tensor(batch, device=self.rowptr.device)
        return self.pos_sibling_sample(batch), self.neg_sample_sibling_guardian(batch)

    @torch.jit.export
    def sample_guardian_walks(self, batch: Union[List[int], Tensor]) -> Tuple[Tensor, Tensor]: 
        if not isinstance(batch, Tensor):
            batch = torch.tensor(batch, device=self.rowptr.device)
        return self.pos_guardian_sample(batch), self.neg_sample_sibling_guardian(batch)


    @torch.jit.export
    def loss(self, pos_rw: Tensor, neg_rw: Tensor) -> Tensor:
        pos_loss = torch.tensor(0.0, device=pos_rw.device)
        neg_loss = torch.tensor(0.0, device=neg_rw.device)
        # positive loss
        if pos_rw.size(0) > 0 and pos_rw.size(1) > 1: 
            start_pos, rest_pos = pos_rw[:, 0], pos_rw[:, 1:].contiguous()
            h_start_pos = self.embedding(start_pos).unsqueeze(1) 
            output_pos = self.fc(h_start_pos) 

            context_labels_pos = self.labels[rest_pos] 
            one_hot_labels_pos = F.one_hot(context_labels_pos, num_classes=self.num_classes).float()
            
            logits_pos = (output_pos * one_hot_labels_pos).sum(dim=-1) 

            test_mask_rest_pos = self.test_mask[rest_pos] 
            start_pos_expanded = start_pos.unsqueeze(1).expand_as(rest_pos)
            mask_same_node_pos = (rest_pos == start_pos_expanded) 
            total_mask_pos = test_mask_rest_pos | mask_same_node_pos

            loss_terms_pos = -torch.log(torch.sigmoid(logits_pos) + self.EPS) 
            loss_terms_pos_masked = torch.where(total_mask_pos, torch.zeros_like(loss_terms_pos), loss_terms_pos)
            
            num_valid_pos = (~total_mask_pos).sum().float()
            if num_valid_pos > 0:
                current_pos_loss = loss_terms_pos_masked.sum() / num_valid_pos
                if not (torch.isnan(current_pos_loss) or torch.isinf(current_pos_loss)):
                    pos_loss = current_pos_loss

        # negative loss
        if neg_rw.size(0) > 0 and neg_rw.size(1) > 1:
            start_neg, rest_neg = neg_rw[:, 0], neg_rw[:, 1:].contiguous()
            h_start_neg = self.embedding(start_neg).unsqueeze(1)
            output_neg = self.fc(h_start_neg)

            context_labels_neg = self.labels[rest_neg]
            one_hot_labels_neg = F.one_hot(context_labels_neg, num_classes=self.num_classes).float()

            logits_neg = (output_neg * one_hot_labels_neg).sum(dim=-1)

            test_mask_rest_neg = self.test_mask[rest_neg]
            start_neg_expanded = start_neg.unsqueeze(1).expand_as(rest_neg)
            mask_same_node_neg = (rest_neg == start_neg_expanded)
            total_mask_neg = test_mask_rest_neg | mask_same_node_neg
            
            loss_terms_neg = -torch.log(1 - torch.sigmoid(logits_neg) + self.EPS)
            loss_terms_neg_masked = torch.where(total_mask_neg, torch.zeros_like(loss_terms_neg), loss_terms_neg)
            
            num_valid_neg = (~total_mask_neg).sum().float()
            if num_valid_neg > 0:
                current_neg_loss = loss_terms_neg_masked.sum() / num_valid_neg
                if not (torch.isnan(current_neg_loss) or torch.isinf(current_neg_loss)):
                    neg_loss = current_neg_loss
            
        return pos_loss + neg_loss

    def test(
        self,
        train_z: Tensor,
        train_y: Tensor,
        test_z: Tensor,
        test_y: Tensor,
        solver: str = 'lbfgs',
        *args,
        **kwargs,
    ) -> float:
        """Evaluates latent space quality via a logistic regression downstream task."""
        from sklearn.linear_model import LogisticRegression

        clf = LogisticRegression(solver=solver, *args,
                                 **kwargs).fit(train_z.detach().cpu().numpy(),
                                               train_y.detach().cpu().numpy())
        return clf.score(test_z.detach().cpu().numpy(),
                         test_y.detach().cpu().numpy())

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.embedding.weight.size(0)}, '
                f'{self.embedding.weight.size(1)})')