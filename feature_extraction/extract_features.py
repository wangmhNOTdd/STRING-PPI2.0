import torch
import pandas as pd
from transformers import T5Tokenizer, T5EncoderModel
import numpy as np
import os
import re

def get_protein_embeddings(sequences, model, tokenizer, device):
    """
    Computes embeddings for a list of protein sequences.
    """
    # Replace rare amino acids and pad/truncate sequences
    processed_sequences = [re.sub(r"[UZOB]", "X", seq) for seq in sequences]
    
    # Tokenize sequences
    inputs = tokenizer.batch_encode_plus(
        processed_sequences, 
        add_special_tokens=True, 
        padding="longest",
        return_tensors="pt"
    )
    
    # Move inputs to the selected device
    ids = inputs['input_ids'].to(device)
    mask = inputs['attention_mask'].to(device)

    # Generate embeddings
    with torch.no_grad():
        outputs = model(input_ids=ids, attention_mask=mask)
    
    # Extract last hidden states
    embeddings = outputs.last_hidden_state

    # Mask out padding tokens and compute the mean of the remaining tokens
    mask = mask.unsqueeze(-1).expand(embeddings.size()).float()
    masked_embeddings = embeddings * mask
    summed = torch.sum(masked_embeddings, 1)
    counts = torch.clamp(mask.sum(1), min=1e-9)
    mean_pooled = summed / counts
    
    # Move embeddings to CPU and convert to numpy
    return mean_pooled.cpu().numpy()

def main():
    """
    Main function to load data, model, and extract features.
    """
    # --- Configuration ---
    # Model name from Hugging Face
    model_name = "genbio-ai/AIDO.Protein-RAG-3B"
    
    # Input data directory
    data_dir = "../processed_data"
    
    # Output directory for features
    output_dir = "."
    os.makedirs(output_dir, exist_ok=True)

    # Datasets to process
    datasets = ["SHS27k", "SHS148k", "STRING"]

    # --- Model and Tokenizer Loading ---
    print(f"Loading model: {model_name}")
    
    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load tokenizer and model
    tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(model_name).to(device)
    model.eval() # Set model to evaluation mode

    # --- Feature Extraction Loop ---
    for dataset_name in datasets:
        print(f"\nProcessing dataset: {dataset_name}")
        
        # Define file paths
        input_csv = os.path.join(data_dir, f"protein.{dataset_name}.sequences.dictionary.csv")
        output_file = os.path.join(output_dir, f"{dataset_name}_protein_embeddings.npy")
        
        # Check if features already exist
        if os.path.exists(output_file):
            print(f"Embeddings for {dataset_name} already exist. Skipping.")
            continue

        # Load protein sequences
        print(f"Loading sequences from: {input_csv}")
        try:
            df = pd.read_csv(input_csv, header=None)
            # Assuming the first column is the protein ID and the second is the sequence
            sequences = df[1].tolist()
            protein_ids = df[0].tolist()
            print(f"Found {len(sequences)} sequences.")
        except FileNotFoundError:
            print(f"Error: File not found at {input_csv}. Skipping this dataset.")
            continue
        except Exception as e:
            print(f"An error occurred while reading the file: {e}")
            continue

        # --- Generate Embeddings ---
        # Process in batches to manage memory, especially for large datasets
        batch_size = 32 # Adjust this based on your GPU memory
        all_embeddings = []
        
        print(f"Generating embeddings in batches of {batch_size}...")
        for i in range(0, len(sequences), batch_size):
            batch_sequences = sequences[i:i+batch_size]
            batch_embeddings = get_protein_embeddings(batch_sequences, model, tokenizer, device)
            all_embeddings.append(batch_embeddings)
            print(f"  - Processed batch {i//batch_size + 1}/{(len(sequences)-1)//batch_size + 1}")

        # Concatenate embeddings from all batches
        final_embeddings = np.vstack(all_embeddings)

        # --- Save Embeddings ---
        print(f"Saving embeddings to: {output_file}")
        np.save(output_file, final_embeddings)
        
        # Optional: Save protein IDs to a separate file for mapping
        id_output_file = os.path.join(output_dir, f"{dataset_name}_protein_ids.txt")
        with open(id_output_file, 'w') as f:
            for pid in protein_ids:
                f.write(f"{pid}\n")
        print(f"Protein IDs saved to: {id_output_file}")

    print("\nFeature extraction complete for all datasets.")

if __name__ == "__main__":
    main()
