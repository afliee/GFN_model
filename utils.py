import os
import sys
import torch
import networkx as nx
from torch_geometric.utils import to_networkx

def print_weights(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.shape)
    sys.stdout.flush()


def logger(info):
    fold, epoch = info['fold'], info['epoch']
    if epoch == 1 or epoch % 10 == 0:
        train_acc, test_acc = info['train_acc'], info['test_acc']
        print('{:02d}/{:03d}: Train Acc: {:.3f}, Test Accuracy: {:.3f}'.format(
            fold, epoch, train_acc, test_acc))
    sys.stdout.flush()

def visualize(model, data_loader, dataset_name, device):
    import matplotlib.pyplot as plt
    model.eval()  # Set the model to evaluation mode
    
    # Create output directory
    output_dir = f'output/{dataset_name}'
    os.makedirs(output_dir, exist_ok=True)
    
    for batch in data_loader:
        batch = batch.to(device)  # Move batch to device

        # Get predictions
        with torch.no_grad():
            out = model(batch)  # Output shape depends on model type (node-level or graph-level)
        
        for idx, data in enumerate(batch.to_data_list()):
            true_label = data.y.item()
            print(f"Graph Label: {true_label}")
            print(f"X size: {data.x.size()}")
            
            # Ensure the graph has enough features for visualization
            if data.x.size(1) >= 2:
                x_coords = data.x[:, 0].cpu().numpy()  # First feature as x-axis
                y_coords = data.x[:, 1].cpu().numpy()  # Second feature as y-axis

                # Determine whether we have node-level or graph-level predictions
                if out.size(0) == data.x.size(0):  # Node-level prediction
                    predicted_labels = out.argmax(dim=-1).cpu().numpy()
                else:  # Graph-level prediction (one label for the whole graph)
                    graph_prediction = out[idx].argmax(dim=-1).item()  # Index out the prediction for this graph
                    predicted_labels = [graph_prediction] * data.x.size(0)

                # Plot node features
                plt.figure(figsize=(8, 8))
                plt.scatter(x_coords, y_coords, c=predicted_labels, cmap='coolwarm', s=100, alpha=0.6, label=f'Predicted Label: {graph_prediction if isinstance(predicted_labels, list) else predicted_labels[0]}')
                plt.title(f"Graph Visualization | True Label: {true_label} | Predicted Label: {predicted_labels[0]}")
                plt.xlabel("Feature 1")
                plt.ylabel("Feature 2")
                plt.legend()
                plt.colorbar(label="Node Class")
                plt.grid(True)
                
                # Save figure
                plt.savefig(os.path.join(output_dir, f'graph_{idx}_true_{true_label}_pred_{predicted_labels[0]}.png'))
                
                # Show the plot
                plt.show()
            else:
                print("Node features do not have enough dimensions for 2D visualization.")
            
        # Optional: break after one batch for testing visualization on a small sample
        break
