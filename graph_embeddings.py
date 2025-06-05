import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.loader import DataLoader  # Fixed import
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, TopKPooling, SAGPooling
from scipy.signal import hilbert
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch_geometric.utils import dense_to_sparse
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import argparse
import sys
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", message=".*'data.DataLoader' is deprecated.*")
warnings.filterwarnings("ignore", message=".*Could not find the number of physical cores.*")
warnings.filterwarnings("ignore", category=UserWarning)

# Disable joblib CPU count warning
os.environ["LOKY_MAX_CPU_COUNT"] = "4"  # Set a fixed number of CPUs

# Import functions from spectral_gnn.py to avoid duplication
try:
    from spectral_gnn import channels, prepare_data
    # We'll redefine load_data and compute_plv with enhancements
except ImportError:
    print("Error: spectral_gnn.py not found in the current directory.")
    print("Please make sure spectral_gnn.py is in the same directory.")
    sys.exit(1)

# Enhanced data loading with feature selection
def load_data(subject_path, subject, label_type='valence'):
    """Enhanced data loading with additional preprocessing options"""
    file_path = os.path.join(subject_path, f'{subject}_features.csv')
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found at: {file_path}")
    
    df = pd.read_csv(file_path)
    
    # Get all features except labels
    features = df.drop(columns=['valence', 'arousal', 'four_class']).values
    labels = df[label_type].values
    
    # Balance dataset if needed (optional)
    if len(np.unique(labels)) > 1:
        # Count samples per class
        class_counts = np.bincount(labels)
        min_count = np.min(class_counts)
        
        # Check class imbalance
        class_ratio = max(class_counts) / min(class_counts)
        if class_ratio > 1.5:  # If imbalanced
            print(f"Detected class imbalance (ratio: {class_ratio:.2f}), applying balancing...")
            # Get indices for each class
            class_0_idx = np.where(labels == 0)[0]
            class_1_idx = np.where(labels == 1)[0]
            
            # Downsample the majority class
            if len(class_0_idx) > len(class_1_idx):
                majority_idx = class_0_idx
                minority_idx = class_1_idx
            else:
                majority_idx = class_1_idx
                minority_idx = class_0_idx
            
            # Randomly select same number of samples as minority class
            np.random.seed(42)
            sampled_majority_idx = np.random.choice(majority_idx, len(minority_idx), replace=False)
            
            # Combine indices
            balanced_idx = np.concatenate([sampled_majority_idx, minority_idx])
            
            # Get balanced dataset
            features = features[balanced_idx]
            labels = labels[balanced_idx]
            
            print(f"Balanced dataset: {len(features)} samples per class")
    
    return features, labels

# Enhanced PLV computation with frequency band weighting
def compute_plv(features, weighting=True):
    """Enhanced PLV computation with frequency band weighting"""
    n_samples, n_features = features.shape
    n_channels = n_features // 4  # Each channel has 4 frequency bands
    plv_matrix = np.zeros((n_channels, n_channels))
    
    # Weight factors for different frequency bands (theta, alpha, beta, gamma)
    # Higher weights for alpha and beta which are more relevant for emotion
    if weighting:
        weights = np.array([0.8, 1.2, 1.5, 0.9])  # theta, alpha, beta, gamma
    else:
        weights = np.array([1.0, 1.0, 1.0, 1.0])  # Equal weights
    
    # Extract and weight frequency bands for each channel
    band_features = []
    for i in range(0, n_features, 4):
        channel_features = features[:, i:i+4]
        # Apply weighting to emphasize important frequency bands
        weighted_features = np.average(channel_features, axis=1, weights=weights)
        band_features.append(weighted_features)
    band_features = np.array(band_features)
    
    # Compute PLV for each pair of channels
    phases = np.angle(hilbert(band_features))
    for i in range(n_channels):
        for j in range(n_channels):
            if i != j:
                phase_diff = phases[i, :] - phases[j, :]
                plv_matrix[i, j] = np.abs(np.mean(np.exp(1j * phase_diff)))
            else:
                plv_matrix[i, j] = 1
    
    return plv_matrix

# Improved graph construction with attention to important connections
def create_enhanced_graph(adjacency_matrix, threshold_percentile=70):
    """Create enhanced graph with focus on strongest connections"""
    # Calculate threshold based on percentile instead of fixed value
    threshold = np.percentile(adjacency_matrix, threshold_percentile)
    
    # Apply threshold
    adj_thresholded = adjacency_matrix.copy()
    adj_thresholded[adj_thresholded < threshold] = 0
    
    # Ensure graph is not too sparse (at least 3 connections per node on average)
    mean_connections = np.sum(adj_thresholded > 0) / len(adj_thresholded)
    if mean_connections < 3:
        # Lower threshold until we have enough connections
        while mean_connections < 3 and threshold_percentile > 40:
            threshold_percentile -= 10
            threshold = np.percentile(adjacency_matrix, threshold_percentile)
            adj_thresholded = adjacency_matrix.copy()
            adj_thresholded[adj_thresholded < threshold] = 0
            mean_connections = np.sum(adj_thresholded > 0) / len(adj_thresholded)
    
    print(f"Graph created with threshold at {threshold_percentile}th percentile")
    print(f"Average connections per node: {mean_connections:.2f}")
    
    return adj_thresholded

# Visualize the constructed graph
def visualize_graph(adjacency_matrix, output_dir):
    """Visualize the constructed graph with region-based coloring"""
    # Create graph from adjacency matrix
    G = nx.from_numpy_array(adjacency_matrix)
    
    # Set node labels to channel names
    node_labels = {i: channels[i] for i in range(len(channels))}
    
    # Get position layout (force-directed with seed for reproducibility)
    pos = nx.spring_layout(G, seed=42, k=0.3)  # k controls spacing
    
    # Calculate edge weights for line thickness
    edge_weights = [adjacency_matrix[u, v] * 3 for u, v in G.edges()]
    
    # Set up figure with higher resolution
    plt.figure(figsize=(12, 10), dpi=150)
    
    # Draw nodes with different colors based on brain regions
    frontal_nodes = [i for i, ch in enumerate(channels) if any(x in ch for x in ['F', 'Fp'])]
    central_nodes = [i for i, ch in enumerate(channels) if any(x in ch for x in ['C', 'T'])]
    parietal_nodes = [i for i, ch in enumerate(channels) if any(x in ch for x in ['P', 'CP'])]
    occipital_nodes = [i for i, ch in enumerate(channels) if any(x in ch for x in ['O', 'PO'])]
    
    # Draw nodes by region with better colors and larger size
    nx.draw_networkx_nodes(G, pos, nodelist=frontal_nodes, node_color='#4285F4', node_size=600, alpha=0.8)
    nx.draw_networkx_nodes(G, pos, nodelist=central_nodes, node_color='#34A853', node_size=600, alpha=0.8)
    nx.draw_networkx_nodes(G, pos, nodelist=parietal_nodes, node_color='#FBBC05', node_size=600, alpha=0.8)
    nx.draw_networkx_nodes(G, pos, nodelist=occipital_nodes, node_color='#EA4335', node_size=600, alpha=0.8)
    
    # Draw edges with varying thickness based on weight and better alpha
    nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.5, edge_color='gray')
    
    # Draw labels with better font
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10, font_weight='bold', font_color='black')
    
    # Add title
    plt.title('EEG Functional Connectivity Graph (PLV)', fontsize=16, fontweight='bold')
    
    # Add custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#4285F4', markersize=12, label='Frontal'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#34A853', markersize=12, label='Central/Temporal'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#FBBC05', markersize=12, label='Parietal'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#EA4335', markersize=12, label='Occipital')
    ]
    plt.legend(handles=legend_elements, loc='lower left', frameon=True, facecolor='white', edgecolor='black', framealpha=0.9)
    
    plt.axis('off')
    plt.tight_layout()
    
    # Save with high quality
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'plv_brain_connectivity.png'), dpi=300, bbox_inches='tight')
    plt.close()

# Advanced Graph Embedding model with residual connections
class AdvancedSpectralGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, pooling_type='mean', num_heads=4, dropout_rate=0.4):
        super(AdvancedSpectralGNN, self).__init__()
        
        # Multi-head graph attention layer for initial embedding
        self.gat = GATConv(input_dim, hidden_dim // num_heads, heads=num_heads, dropout=dropout_rate)
        
        # Spectral convolution layers
        self.gcn1 = GCNConv(hidden_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.gcn3 = GCNConv(hidden_dim, hidden_dim)
        
        # Batch normalization for stability
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        
        # Add pooling based on type
        self.pooling_type = pooling_type
        if pooling_type == 'topk':
            self.pool = TopKPooling(hidden_dim, ratio=0.5)
        elif pooling_type == 'sag':
            self.pool = SAGPooling(hidden_dim, ratio=0.5)
        else:  # Default to mean pooling
            self.pool = None
        
        # Projection layers for graph-level embedding with residual connection
        self.projection1 = nn.Linear(hidden_dim, hidden_dim)
        self.projection2 = nn.Linear(hidden_dim, hidden_dim // 2)
        
        # Classification layer
        self.classifier = nn.Linear(hidden_dim // 2, output_dim)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x, edge_index, edge_attr, batch):
        # Initial embedding with multi-head attention
        x = self.gat(x, edge_index, edge_attr=edge_attr)
        x = F.elu(x)
        x = self.dropout(x)
        
        # First spectral convolution with residual connection
        identity = x
        x = self.gcn1(x, edge_index)
        x = self.bn1(x)
        x = F.elu(x)
        x = x + identity  # Residual connection
        x = self.dropout(x)
        
        # Second spectral convolution with residual connection
        identity = x
        x = self.gcn2(x, edge_index)
        x = self.bn2(x)
        x = F.elu(x)
        x = x + identity  # Residual connection
        x = self.dropout(x)
        
        # Third spectral convolution
        x = self.gcn3(x, edge_index)
        x = self.bn3(x)
        x = F.elu(x)
        
        # Apply pooling based on type
        if self.pooling_type == 'topk':
            x, edge_index, edge_attr, batch, _, _ = self.pool(x, edge_index, edge_attr, batch)
        elif self.pooling_type == 'sag':
            x, edge_index, edge_attr, batch, _, _ = self.pool(x, edge_index, edge_attr, batch)
        
        # Global pooling to get graph-level embedding
        x = global_mean_pool(x, batch)
        
        # Projection for embedding space with residual connection
        identity = x
        embedding = self.projection1(x)
        embedding = F.elu(embedding)
        embedding = self.dropout(embedding)
        embedding = self.projection2(embedding)
        
        # Classification
        out = self.classifier(embedding)
        
        return out, embedding

# Train with learning rate scheduling and early stopping
def train_and_embed(model, train_loader, test_loader, criterion, optimizer, 
                   num_epochs=150, output_dir=None, label_type='valence'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Learning rate scheduler with silent verbose
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=15, verbose=False
    )
    
    # Track metrics
    train_losses = []
    train_accs = []
    val_accs = []
    best_acc = 0
    best_epoch = 0
    best_model_state = None
    embeddings_list = []
    labels_list = []
    patience = 25  # Early stopping patience
    no_improve = 0
    
    print(f"\nTraining {label_type} classifier on {device}...")
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        correct = 0
        total = 0
        
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            logits, _ = model(data.x, data.edge_index, data.edge_attr, data.batch)
            loss = criterion(logits, data.y)
            loss.backward()
            optimizer.step()
            
            # Track metrics
            epoch_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += (pred == data.y).sum().item()
            total += data.y.size(0)
        
        # Calculate epoch metrics
        avg_loss = epoch_loss / len(train_loader)
        accuracy = correct / total
        train_losses.append(avg_loss)
        train_accs.append(accuracy)
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}')
        
        # Evaluate on test set
        if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
            test_acc, test_f1, _ = evaluate(model, test_loader, device)
            val_accs.append(test_acc)
            
            # Update learning rate based on validation accuracy
            scheduler.step(test_acc)
            
            # Check for improvement
            if test_acc > best_acc:
                best_acc = test_acc
                best_epoch = epoch
                best_model_state = model.state_dict()
                no_improve = 0
            else:
                no_improve += 1
            
            # Early stopping
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}. Best accuracy: {best_acc:.4f} at epoch {best_epoch+1}")
                break
    
    # Report best performance
    print(f"Best validation accuracy: {best_acc:.4f} at epoch {best_epoch+1}")
    
    # Load best model for final evaluation and embedding generation
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Generate embeddings for visualization
    model.eval()
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            _, embedding = model(data.x, data.edge_index, data.edge_attr, data.batch)
            embeddings_list.append(embedding.cpu().numpy())
            labels_list.append(data.y.cpu().numpy())
    
    # Concatenate all embeddings and labels
    all_embeddings = np.vstack(embeddings_list)
    all_labels = np.concatenate(labels_list)
    
    # Save training curves
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model
        torch.save(model.state_dict(), os.path.join(output_dir, 'spectral_embedding_model.pth'))
        
        # Plot and save training curves
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, 'b-')
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(train_accs, 'r-')
        plt.title('Training Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'training_curves.png'))
        plt.close()
        
        # Visualize embeddings using t-SNE
        visualize_embeddings(all_embeddings, all_labels, output_dir, label_type)
    
    return model, all_embeddings, all_labels

# Evaluate the model
def evaluate(model, loader, device):
    model.eval()
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            logits, _ = model(data.x, data.edge_index, data.edge_attr, data.batch)
            pred = logits.argmax(dim=1)
            y_true.extend(data.y.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    return accuracy, f1, conf_matrix

# Visualize embeddings using t-SNE
def visualize_embeddings(embeddings, labels, output_dir, label_type):
    from sklearn.manifold import TSNE
    
    # Skip if too few samples
    if len(embeddings) < 5:
        print("Too few samples for t-SNE visualization, skipping...")
        return
    
    # Apply t-SNE with safer parameters
    try:
        perplexity = min(30, len(embeddings)//2)
        perplexity = max(5, perplexity)  # Ensure perplexity >= 5
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        embeddings_2d = tsne.fit_transform(embeddings)
        
        # Plot with enhanced styling
        plt.figure(figsize=(10, 8))
        
        # Use better colors for classes
        colors = ['#4285F4', '#EA4335']
        
        # Plot each class with a different color
        for i, label in enumerate(['Low', 'High']):
            idx = labels == i
            if np.any(idx):  # Only plot if class exists
                plt.scatter(
                    embeddings_2d[idx, 0], 
                    embeddings_2d[idx, 1], 
                    c=colors[i], 
                    label=label,
                    alpha=0.8,
                    edgecolors='white',
                    linewidths=0.5,
                    s=80
                )
        
        # Add styling
        plt.title(f't-SNE Visualization of Graph Embeddings for {label_type.capitalize()}', 
                fontsize=16, fontweight='bold')
        plt.xlabel('t-SNE Dimension 1', fontsize=12)
        plt.ylabel('t-SNE Dimension 2', fontsize=12)
        plt.legend(title=f"{label_type.capitalize()} Class", fontsize=12, title_fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save with high quality
        plt.savefig(os.path.join(output_dir, 'embedding_visualization.png'), dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Error in t-SNE visualization: {e}, skipping...")

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Generate graph embeddings for EEG emotion recognition')
    parser.add_argument('--subject', type=str, default='s01', help='Subject ID')
    parser.add_argument('--datafiles_path', type=str, required=True, help='Path to datafiles')
    parser.add_argument('--label_type', type=str, choices=['valence', 'arousal'], 
                       required=True, help='Label type for classification')
    parser.add_argument('--threshold_percentile', type=float, default=70, 
                       help='Percentile threshold for graph connectivity')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension')
    parser.add_argument('--num_epochs', type=int, default=150, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--test_size', type=float, default=0.3, help='Test set ratio')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed')
    parser.add_argument('--use_best_pooling', action='store_true', 
                       help='Use best pooling mechanism based on showcase_pool.py results')
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    
    # Create subject directory
    subject_path = os.path.join(args.datafiles_path, args.subject)
    os.makedirs(subject_path, exist_ok=True)
    
    # Create output directory
    output_dir = os.path.join(subject_path, f"{args.label_type}_spectral_embedding")
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine pooling type
    pooling_type = 'mean'  # Default
    
    if args.use_best_pooling:
        # Check if showcase_pool results exist to find the best pooling type
        pool_comparison_file = os.path.join(subject_path, f'{args.subject}_pooling_comparison.csv')
        if os.path.exists(pool_comparison_file):
            print(f"Loading pooling comparison results from {pool_comparison_file}")
            pool_df = pd.read_csv(pool_comparison_file)
            # Filter for current label type
            label_pools = pool_df[pool_df['label_type'] == args.label_type]
            if not label_pools.empty:
                # Find best pooling type by accuracy
                best_idx = label_pools['accuracy'].idxmax()
                pooling_type = label_pools.loc[best_idx]['pooling_type']
                print(f"Best pooling type for {args.label_type}: {pooling_type}")
            else:
                print(f"No pooling comparison results found for {args.label_type}, using default (mean)")
    
    # Load and normalize data with enhanced preprocessing
    print(f"Loading data for subject {args.subject}, label type {args.label_type}...")
    features, labels = load_data(subject_path, args.subject, args.label_type)
    
    # Apply both normalization methods and combine them
    scaler1 = MinMaxScaler()
    scaler2 = StandardScaler()
    features_minmax = scaler1.fit_transform(features)
    features_standard = scaler2.fit_transform(features)
    # Use MinMax scaling by default (better for frequency data)
    features = features_minmax
    
    # Look for cached PLV matrix first
    plv_cache_file = os.path.join(subject_path, f"{args.subject}_{args.label_type}_plv_matrix.npy")
    
    if os.path.exists(plv_cache_file):
        print(f"Loading cached PLV matrix from {plv_cache_file}")
        adjacency_matrix = np.load(plv_cache_file)
    else:
        # Compute PLV with frequency band weighting
        print("Computing weighted PLV matrix (this may take a while)...")
        adjacency_matrix = compute_plv(features, weighting=True)
        # Cache the PLV matrix for future use
        np.save(plv_cache_file, adjacency_matrix)
    
    # Create enhanced graph with adaptive thresholding
    print("Creating enhanced brain connectivity graph...")
    adj_thresholded = create_enhanced_graph(adjacency_matrix, args.threshold_percentile)
    
    # Visualize the graph with improved aesthetics
    print("Generating brain connectivity visualization...")
    visualize_graph(adj_thresholded, output_dir)
    
    # Prepare data for PyTorch Geometric
    print("Preparing data for graph neural network...")
    data_list = prepare_data(features, labels, adj_thresholded)
    
    # Split data into train/test sets
    train_data, test_data = train_test_split(
        data_list, test_size=args.test_size, 
        random_state=args.random_seed, 
        stratify=[d.y.item() for d in data_list]
    )
    
    # Create data loaders with specified batch size
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
    
    # Initialize advanced model with selected pooling type
    input_dim = 4  # Four frequency bands per channel
    output_dim = 2  # Binary classification (valence/arousal)
    model = AdvancedSpectralGNN(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        output_dim=output_dim,
        pooling_type=pooling_type,
        dropout_rate=0.4
    )
    
    # Use Adam optimizer with weight decay for regularization
    optimizer = optim.Adam(
        model.parameters(), 
        lr=args.learning_rate, 
        weight_decay=args.weight_decay
    )
    
    # Use weighted cross-entropy loss if classes are imbalanced
    if len(np.unique(labels)) > 1:
        class_counts = np.bincount(labels)
        class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
        class_weights = class_weights / class_weights.sum()
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Train model with improved training function
    print(f"Training model with {args.num_epochs} epochs...")
    model, embeddings, labels = train_and_embed(
        model, train_loader, test_loader, criterion, optimizer,
        num_epochs=args.num_epochs, output_dir=output_dir, label_type=args.label_type
    )
    
    # Final evaluation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    accuracy, f1, conf_matrix = evaluate(model, test_loader, device)
    
    print(f"\nFinal {args.label_type.upper()} Classification Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Save confusion matrix with better styling
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        conf_matrix, 
        annot=True, 
        fmt='d', 
        cmap='Blues', 
        xticklabels=['Low', 'High'], 
        yticklabels=['Low', 'High'],
        annot_kws={"size": 16},
        cbar=True
    )
    plt.xlabel('Predicted', fontsize=14)
    plt.ylabel('True', fontsize=14)
    plt.title(f'Confusion Matrix for {args.label_type.capitalize()}', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300)
    plt.close()
    
    # Save detailed results
    with open(os.path.join(output_dir, 'results.txt'), 'w') as f:
        f.write(f"Subject: {args.subject}\n")
        f.write(f"Label Type: {args.label_type}\n")
        f.write(f"Pooling Type: {pooling_type}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
        f.write(f"Hidden Dimension: {args.hidden_dim}\n")
        f.write(f"Learning Rate: {args.learning_rate}\n")
        f.write(f"Weight Decay: {args.weight_decay}\n")
        f.write(f"Threshold Percentile: {args.threshold_percentile}\n")
        f.write("\nConfusion Matrix:\n")
        f.write(str(conf_matrix))
    
    # Save embeddings for further classification
    np.savez(os.path.join(output_dir, f"{args.subject}_{args.label_type}_embeddings.npz"),
            embeddings=embeddings,
            labels=labels)
    
    # Try to compare with previous results
    try:
        # Look for previous results from spectral_gnn.py
        spectral_gnn_metrics_file = os.path.join(subject_path, args.label_type, 'metrics.txt')
        
        if os.path.exists(spectral_gnn_metrics_file):
            spectral_metrics = {}
            with open(spectral_gnn_metrics_file, 'r') as f:
                for line in f:
                    if ':' in line:
                        key, value = line.strip().split(':', 1)
                        spectral_metrics[key.strip()] = value.strip()
            
            # Create comparison
            comparison = {
                'Method': ['Spectral GNN', 'Enhanced Spectral GNN'],
                'Accuracy': [
                    float(spectral_metrics.get('accuracy', 0)),
                    float(accuracy)
                ],
                'F1 Score': [
                    float(spectral_metrics.get('f1_score', 0)),
                    float(f1)
                ]
            }
            
            # Calculate improvement
            acc_improvement = (accuracy - float(spectral_metrics.get('accuracy', 0))) * 100
            f1_improvement = (f1 - float(spectral_metrics.get('f1_score', 0))) * 100
            
            # Print improvement
            print("\nPerformance Comparison:")
            print(f"Original Spectral GNN: Accuracy = {float(spectral_metrics.get('accuracy', 0)):.4f}, F1 = {float(spectral_metrics.get('f1_score', 0)):.4f}")
            print(f"Enhanced Spectral GNN: Accuracy = {accuracy:.4f}, F1 = {f1:.4f}")
            print(f"Improvement: Accuracy +{acc_improvement:.2f}%, F1 +{f1_improvement:.2f}%")
            
            # Save comparison
            with open(os.path.join(output_dir, 'improvement.txt'), 'w') as f:
                f.write(f"Original Spectral GNN: Accuracy = {float(spectral_metrics.get('accuracy', 0)):.4f}, F1 = {float(spectral_metrics.get('f1_score', 0)):.4f}\n")
                f.write(f"Enhanced Spectral GNN: Accuracy = {accuracy:.4f}, F1 = {f1:.4f}\n")
                f.write(f"Improvement: Accuracy +{acc_improvement:.2f}%, F1 +{f1_improvement:.2f}%\n")
    except Exception as e:
        print(f"Could not create comparison: {e}")
    
    print(f"Results and embeddings saved to {output_dir}")

if __name__ == "__main__":
    main()