import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, TopKPooling, SAGPooling
from scipy.signal import hilbert
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import MinMaxScaler
from torch_geometric.utils import dense_to_sparse
import matplotlib.pyplot as plt
import networkx as nx
import argparse
import sys

# Import functions from spectral_gnn.py
# Make sure spectral_gnn.py is in the same directory
try:
    from spectral_gnn import load_data, compute_plv, prepare_data, channels
except ImportError:
    print("Error: spectral_gnn.py not found in the current directory.")
    print("Please make sure spectral_gnn.py is in the same directory as showcase_pool.py")
    sys.exit(1)

# Define pooling-specific SpectralGNN with different pooling mechanisms
class SpectralGNNWithPooling(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, pooling_type='topk', pooling_ratio=0.5):
        super(SpectralGNNWithPooling, self).__init__()
        self.gat1 = GATConv(input_dim, hidden_dim // 4, heads=4, dropout=0.6)
        self.gcn1 = GCNConv(hidden_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.6)
        
        # Pooling layer based on type
        self.pooling_type = pooling_type
        if pooling_type == 'topk':
            self.pool = TopKPooling(hidden_dim, ratio=pooling_ratio)
        elif pooling_type == 'sag':
            self.pool = SAGPooling(hidden_dim, ratio=pooling_ratio)
        else:  # 'mean' pooling is handled in forward pass
            self.pool = None
    
    def forward(self, x, edge_index, edge_attr, batch):
        x = self.gat1(x, edge_index, edge_attr=edge_attr)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.gcn1(x, edge_index)
        x = F.relu(x)
        x = self.gcn2(x, edge_index)
        
        # Apply pooling based on type
        if self.pooling_type == 'topk':
            x, edge_index, edge_attr, batch, _, _ = self.pool(x, edge_index, edge_attr, batch)
        elif self.pooling_type == 'sag':
            x, edge_index, edge_attr, batch, _, _ = self.pool(x, edge_index, edge_attr, batch)
        
        # Global pooling to get graph-level representation
        x = global_mean_pool(x, batch)
        
        # Classification
        x = self.fc(x)
        return x

# Visualize pooling-specific graph
def save_pooling_specific_graph(adjacency_matrix, subject_path, pooling_type, label_type):
    output_dir = os.path.join(subject_path, f"{label_type}_{pooling_type}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create graph visualization with pooling info
    G = nx.from_numpy_array(adjacency_matrix)
    pos = nx.spring_layout(G, seed=42)
    
    plt.figure(figsize=(10, 8))
    
    # Draw nodes with colors based on brain regions
    frontal_nodes = [i for i, ch in enumerate(channels) if any(x in ch for x in ['F', 'Fp'])]
    central_nodes = [i for i, ch in enumerate(channels) if any(x in ch for x in ['C', 'T'])]
    parietal_nodes = [i for i, ch in enumerate(channels) if any(x in ch for x in ['P', 'CP'])]
    occipital_nodes = [i for i, ch in enumerate(channels) if any(x in ch for x in ['O', 'PO'])]
    
    # Draw nodes by region
    nx.draw_networkx_nodes(G, pos, nodelist=frontal_nodes, node_color='lightblue', node_size=300, alpha=0.8)
    nx.draw_networkx_nodes(G, pos, nodelist=central_nodes, node_color='lightgreen', node_size=300, alpha=0.8)
    nx.draw_networkx_nodes(G, pos, nodelist=parietal_nodes, node_color='salmon', node_size=300, alpha=0.8)
    nx.draw_networkx_nodes(G, pos, nodelist=occipital_nodes, node_color='violet', node_size=300, alpha=0.8)
    
    # Draw edges with varying thickness based on weight
    edge_weights = [adjacency_matrix[u, v] * 2 for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.6, edge_color='gray')
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, labels={i: channels[i] for i in range(len(channels))}, font_size=8)
    
    plt.title(f"EEG Connectivity - {pooling_type.upper()} Pooling - {label_type.capitalize()}", fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    
    graph_path = os.path.join(output_dir, 'graph_visualization.png')
    plt.savefig(graph_path, dpi=300)
    plt.close()
    
    return output_dir

# Train function
def train(model, loader, criterion, optimizer, num_epochs=100, subject_path=None, label_type=None, pooling_type=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.train()
    
    train_losses = []
    train_accs = []
    
    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for data in loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.edge_attr, data.batch)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            pred = out.argmax(dim=1)
            correct += (pred == data.y).sum().item()
            total += data.y.size(0)
        
        # Calculate epoch metrics
        avg_loss = total_loss / len(loader) if len(loader) > 0 else total_loss
        accuracy = correct / total if total > 0 else 0
        
        train_losses.append(avg_loss)
        train_accs.append(accuracy)
        
        # Print progress occasionally
        if (epoch + 1) % 20 == 0 or epoch == num_epochs - 1:
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}')
    
    # If directories are provided, save training curves
    if subject_path and label_type and pooling_type:
        output_dir = os.path.join(subject_path, f"{label_type}_{pooling_type}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Plot and save loss curve
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses)
        plt.title(f'Training Loss - {pooling_type.upper()} Pooling - {label_type.capitalize()}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'training_loss.png'))
        plt.close()
        
        # Plot and save accuracy curve
        plt.figure(figsize=(10, 5))
        plt.plot(train_accs)
        plt.title(f'Training Accuracy - {pooling_type.upper()} Pooling - {label_type.capitalize()}')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'training_accuracy.png'))
        plt.close()
    
    return model

# Evaluation function
def evaluate(model, loader, output_dir, subject, label_type, pooling_type):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.edge_attr, data.batch)
            pred = out.argmax(dim=1)
            y_true.extend(data.y.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())
    
    # Calculate metrics
    if len(y_true) > 0:
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted')
        
        # Save predictions
        pred_df = pd.DataFrame({'True': y_true, 'Predicted': y_pred})
        pred_df.to_csv(os.path.join(output_dir, f'{subject}_{label_type}_{pooling_type}_predictions.csv'), index=False)
        
        # Create confusion matrix
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
        
        conf_matrix = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Low', 'High'], yticklabels=['Low', 'High'])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix - {pooling_type.upper()} Pooling - {label_type.capitalize()}')
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
        plt.close()
    else:
        accuracy = 0
        f1 = 0
        print(f"Warning: No test data available for evaluation.")
    
    return accuracy, f1

# Run a single experiment with specified pooling
def run_experiment(pooling_type, label_type, subject_path, subject, num_epochs=150):
    print(f"\nRunning experiment with {pooling_type} pooling for {label_type}...")
    
    # Load data using imported function from spectral_gnn.py
    features, labels = load_data(subject_path, subject, label_type)
    scaler = MinMaxScaler()
    features = scaler.fit_transform(features)
    
    # Look for cached PLV matrix first
    plv_cache_file = os.path.join(subject_path, f"{subject}_{label_type}_plv_matrix.npy")
    
    if os.path.exists(plv_cache_file):
        print(f"Loading cached PLV matrix from {plv_cache_file}")
        adjacency_matrix = np.load(plv_cache_file)
    else:
        # Compute PLV using imported function from spectral_gnn.py
        print("Computing PLV matrix (this may take a while)...")
        adjacency_matrix = compute_plv(features)
        # Cache the PLV matrix for future use
        np.save(plv_cache_file, adjacency_matrix)
    
    # Apply threshold
    adjacency_matrix_thresh = adjacency_matrix.copy()
    adjacency_matrix_thresh[adjacency_matrix_thresh < 0.5] = 0
    
    # Create pooling-specific directory and visualization
    output_dir = save_pooling_specific_graph(adjacency_matrix_thresh, subject_path, pooling_type, label_type)
    
    # Prepare data using imported function from spectral_gnn.py
    data_list = prepare_data(features, labels, adjacency_matrix_thresh)

    # Handle the case when there's not enough data to split
    if len(data_list) > 1:
        train_data, test_data = train_test_split(data_list, test_size=0.3, random_state=42, 
                                                stratify=[d.y.item() for d in data_list])
    else:
        print(f"Warning: Not enough data to split (n_samples={len(data_list)}). Using all data for training.")
        train_data, test_data = data_list, []  # Use all data for training, none for testing

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False) if test_data else None
    
    # Initialize model with the specified pooling type
    model = SpectralGNNWithPooling(
        input_dim=4,  # 4 frequency bands per channel
        hidden_dim=64,
        output_dim=2,  # Binary classification
        pooling_type=pooling_type,
        pooling_ratio=0.5
    )
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-4)
    
    # Train model
    model = train(
        model, train_loader, criterion, optimizer, 
        num_epochs=num_epochs, 
        subject_path=subject_path, 
        label_type=label_type,
        pooling_type=pooling_type
    )
    
    # Save model
    torch.save(model.state_dict(), os.path.join(output_dir, f'{subject}_{label_type}_{pooling_type}_model.pth'))
    
    # Evaluate model if test data is available
    accuracy, f1 = 0, 0
    if test_loader:
        accuracy, f1 = evaluate(
            model, test_loader, output_dir, 
            subject, label_type, pooling_type
        )
        print(f"{pooling_type.upper()} Pooling - {label_type.capitalize()} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
    else:
        print("Skipping evaluation: No test data available.")
    
    # Save metrics
    with open(os.path.join(output_dir, 'metrics.txt'), 'w') as f:
        f.write(f"Subject: {subject}\n")
        f.write(f"Label Type: {label_type}\n")
        f.write(f"Pooling Type: {pooling_type}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
    
    return accuracy, f1

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Compare different pooling mechanisms for Spectral GNN')
    parser.add_argument('--subject', type=str, default='s01', help='Subject ID')
    parser.add_argument('--datafiles_path', type=str, required=True, help='Path to datafiles')
    parser.add_argument('--label_types', type=str, default='valence,arousal', 
                        help='Label types for classification (comma-separated)')
    parser.add_argument('--pooling_types', type=str, default='topk,sag,mean',
                        help='Pooling types to compare (comma-separated)')
    parser.add_argument('--num_epochs', type=int, default=150, help='Number of training epochs')
    args = parser.parse_args()
    
    # Convert comma-separated strings to lists
    label_types = args.label_types.split(',')
    pooling_types = args.pooling_types.split(',')
    
    # Path to subject directory
    subject_path = os.path.join(args.datafiles_path, args.subject)
    os.makedirs(subject_path, exist_ok=True)
    
    # Dictionary to store results
    results = {}
    
    # Run experiments for each label type and pooling mechanism
    for label_type in label_types:
        results[label_type] = {}
        for pooling_type in pooling_types:
            accuracy, f1 = run_experiment(
                pooling_type, label_type, subject_path, args.subject, 
                num_epochs=args.num_epochs
            )
            results[label_type][pooling_type] = {'accuracy': accuracy, 'f1': f1}
    
    # Print results summary
    print("\n=== Results Summary ===")
    for label_type in label_types:
        print(f"\n{label_type.capitalize()}:")
        for pooling_type in pooling_types:
            acc = results[label_type][pooling_type]['accuracy']
            f1 = results[label_type][pooling_type]['f1']
            print(f"  {pooling_type} - Accuracy: {acc:.4f}, F1: {f1:.4f}")
    
    # Save overall results to CSV
    results_df = pd.DataFrame(columns=['label_type', 'pooling_type', 'accuracy', 'f1'])
    row = 0
    for label_type in results:
        for pooling_type in results[label_type]:
            results_df.loc[row] = [
                label_type,
                pooling_type,
                results[label_type][pooling_type]['accuracy'],
                results[label_type][pooling_type]['f1']
            ]
            row += 1
    
    results_df.to_csv(os.path.join(subject_path, f'{args.subject}_pooling_comparison.csv'), index=False)
    
    # Create bar charts for visual comparison
    for label_type in label_types:
        # Accuracy chart
        plt.figure(figsize=(10, 6))
        accs = [results[label_type][pt]['accuracy'] for pt in pooling_types]
        plt.bar(pooling_types, accs, color=['blue', 'green', 'orange'])
        plt.ylim(0, 1.0)
        plt.title(f'Accuracy Comparison - {label_type.capitalize()}')
        plt.xlabel('Pooling Type')
        plt.ylabel('Accuracy')
        for i, v in enumerate(accs):
            plt.text(i, v + 0.02, f'{v:.4f}', ha='center')
        plt.tight_layout()
        plt.savefig(os.path.join(subject_path, f'{label_type}_accuracy_comparison.png'))
        plt.close()
        
        # F1 Score chart
        plt.figure(figsize=(10, 6))
        f1s = [results[label_type][pt]['f1'] for pt in pooling_types]
        plt.bar(pooling_types, f1s, color=['blue', 'green', 'orange'])
        plt.ylim(0, 1.0)
        plt.title(f'F1 Score Comparison - {label_type.capitalize()}')
        plt.xlabel('Pooling Type')
        plt.ylabel('F1 Score')
        for i, v in enumerate(f1s):
            plt.text(i, v + 0.02, f'{v:.4f}', ha='center')
        plt.tight_layout()
        plt.savefig(os.path.join(subject_path, f'{label_type}_f1_comparison.png'))
        plt.close()
    
    print(f"\nResults saved to {subject_path}")

if __name__ == "__main__":
    main()