import os
import numpy as np

def fetch_and_average_accuracies(datafiles_path, label_type):
    """
    Fetch accuracies from results.txt files for all subjects (s01 to s32) and compute the average.
    
    Parameters:
    - datafiles_path: Directory containing subject folders (e.g., 'C:\\Users\\ahlad\\Desktop\\Pushya\\output_files')
    - label_type: Label type to process (e.g., 'valence' or 'arousal')
    
    Returns:
    - overall_accuracy: Average accuracy across all subjects with data
    - subject_accuracies: List of tuples (subject, accuracy) for subjects with data
    """
    subject_accuracies = []
    subjects = [f"s{str(i).zfill(2)}" for i in range(1, 33)]  # Generate s01 to s32

    for subject in subjects:
        # Path to the results.txt file from graph_embeddings.py
        results_file = os.path.join(datafiles_path, subject, f"{label_type}_spectral_embedding", "results.txt")
        
        if not os.path.exists(results_file):
            print(f"Warning: Results file for {subject} not found at {results_file}. Skipping.")
            continue
        
        # Read accuracy from results.txt
        with open(results_file, 'r') as f:
            for line in f:
                if 'Accuracy:' in line:
                    accuracy = float(line.split('Accuracy:')[1].strip())
                    subject_accuracies.append((subject, accuracy))
                    break

    if not subject_accuracies:
        raise ValueError("No accuracy data found for any subject. Check the datafiles_path and files.")

    accuracies = [acc for _, acc in subject_accuracies]
    overall_accuracy = np.mean(accuracies)
    return overall_accuracy, subject_accuracies

def main():
    # Hardcoded datafiles_path
    datafiles_path = r'C:\Users\ahlad\Desktop\Pushya\output_files'
    
    # Prompt for label type
    label_type = input("Enter label type (valence/arousal) [default: valence]: ").lower() or 'valence'
    while label_type not in ['valence', 'arousal']:
        print("Invalid label type. Please enter 'valence' or 'arousal'.")
        label_type = input("Enter label type (valence/arousal) [default: valence]: ").lower() or 'valence'

    # Fetch accuracies and compute average
    overall_accuracy, subject_accuracies = fetch_and_average_accuracies(datafiles_path, label_type)

    # Display results
    print(f"\nOverall Accuracy for {label_type} across subjects with data: {overall_accuracy:.4f}")
    print("\nSubject-wise Accuracies:")
    for subject, accuracy in subject_accuracies:
        print(f"{subject}: {accuracy:.4f}")

    # Save results to a file
    output_file = os.path.join(datafiles_path, f"overall_accuracy_{label_type}.txt")
    with open(output_file, 'w') as f:
        f.write(f"Overall Accuracy ({label_type}): {overall_accuracy:.4f}\n")
        f.write("Subject-wise Accuracies:\n")
        for subject, accuracy in subject_accuracies:
            f.write(f"{subject}: {accuracy:.4f}\n")
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()