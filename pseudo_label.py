import copy
import os

import torch
from torch_geometric.data import Batch

def generate_pseudo_labels(trainer, test_graphs, confidence_threshold, added_indices, batch_size=32):
    trainer.model.eval()
    trainer.log.eval()
    trainer.prompt.eval()
    trainer.topo_gnn.eval()
    temp_test = copy.deepcopy(test_graphs)
    for graph in temp_test:
        graph.node_features = trainer.prompt.add(graph.node_features)
    pseudo_graphs = []
    new_added_indices = []
    with torch.no_grad():
        for i in range(0, len(temp_test), batch_size):
            batch_graphs = temp_test[i:i + batch_size]
            valid_batch_indices = [idx for idx in range(i, min(i + batch_size, len(temp_test))) if idx not in added_indices]
            valid_batch_graphs = [batch_graphs[j - i] for j in valid_batch_indices]
            if len(valid_batch_graphs) <=1:
                continue



            batch_pyg_data = [trainer.custom_to_pyg(graph) for graph in valid_batch_graphs]
            pooled_h,_,node_embeds = trainer.model(valid_batch_graphs)
            batch_pyg_data = Batch.from_data_list(batch_pyg_data).to(trainer.device)
            ph_embedding = trainer.topo_gnn(batch_pyg_data, node_embeds)
            combined = torch.cat([pooled_h, ph_embedding], dim=1)
            logits = trainer.log(combined)
            probs = torch.softmax(logits/0.6, dim=1)
            max_probs, predicted_labels = torch.max(probs, dim=1)
            print("max_probs:",max_probs)
            for j, max_prob in enumerate(max_probs):
                if max_prob.item() >= confidence_threshold:

                    new_graph = copy.copy(valid_batch_graphs[j])
                    new_graph.label = predicted_labels[j].item()
                    pseudo_graphs.append(new_graph)
                    new_added_indices.append(valid_batch_indices[j])

    added_indices.extend(new_added_indices)
    return pseudo_graphs, added_indices