import pandas as pd
import pdb
from itertools import groupby
from collections import OrderedDict,defaultdict
from collections import Counter
import json
import argparse
import os
import numpy as np
from helper_functions import sentence_embedding,convert_data_pldl_experiments,generate_data_bert,save_to_json,create_folder,metadata_to_sentence
from tqdm import tqdm
import random
from sklearn.model_selection import train_test_split

def main():
    # Updated for JSON input
    raw_input_file = "datasets/pp/PP_test.json"
    col_tweet_ID = "comment_id"
    colLabels = ["-5","-4","-3","-2","-1","0","1","2","3","4","5"]
    _id = "pp"
    foldername1 = "datasets/pp/processed/modeling_annotator"
    foldername2 = "datasets/pp/processed/disco"
    foldername3 = "datasets/pp/processed/pldl"
    worker_id_mt = "annotator_id"
    
    # Create folders
    create_folder(foldername1)
    create_folder(foldername2)
    create_folder(foldername3)
    
    # Extract split name from filename
    split_name = extract_split_from_filename(raw_input_file)
    
    # Read and process JSON data
    dfs_combine = read_json_data(raw_input_file, split_name)
    dfs_combine.drop_duplicates(inplace=True)
    
    # Process the data with majority labeling
    dfs_combine = label_majority(dfs_combine)
    
    # Create label mappings
    label_dict = {index : colLabels[index] for index in range(0,len(colLabels))}
    print(f"Label mapping: {label_dict}")
    
    dfs_combine['label'] = dfs_combine['sarcasm_score'].astype(str)
    dfs_combine['label'] = dfs_combine['label'].astype(pd.CategoricalDtype(categories=colLabels, ordered=True))
    dfs_combine['label_vector'] = dfs_combine['label'].cat.codes
    
    # Convert soft label distributions to numpy arrays
    dfs_combine['soft_label_vector'] = dfs_combine['soft_label_dist'].apply(lambda x: np.array(x))
    
    # Debug: Check label distribution
    print(f"Label distribution:")
    print(dfs_combine['sarcasm_score'].value_counts().sort_index())
    print(f"Label vector distribution:")
    print(dfs_combine['label_vector'].value_counts().sort_index())
    
    # Debug: Show soft label examples
    print(f"\nSample soft label distributions:")
    sample_soft = dfs_combine[['sarcasm_score', 'soft_label_dist']].head(5)
    for _, row in sample_soft.iterrows():
        print(f"  Sarcasm score: {row['sarcasm_score']} -> Soft dist: {row['soft_label_dist']}")
    
    cats = dfs_combine.label.astype('category')
    list_of_cats = dict(enumerate(cats.cat.categories))
    annotators = pd.unique(dfs_combine[worker_id_mt])
    
    # Create message from context and response
    dfs_combine["message"] = dfs_combine["post"] + ". " + dfs_combine["reply"]
    
    # Create annotator indices
    annotators_parsed = pd.DataFrame(annotators)
    annotators_parsed = annotators_parsed.rename(columns={0:'id'})
    annotators_parsed['Aindex'] = annotators_parsed.index
    dfs_combine = dfs_combine.join(annotators_parsed.set_index('id'), on=worker_id_mt)
    
    # Use original JSON keys as Mindex (item indices)
    # The original_id column contains the JSON keys like "1248", "1280"
    dfs_combine['Mindex'] = dfs_combine['original_id'].astype(int)

    if split_name == 'test':
        # Keep individual annotator labels but remove aggregated labels
        dfs_combine['label'] = None
        dfs_combine['label_vector'] = None

    
    # Prepare dataset for different formats
    ds_df = dfs_combine.copy()
    ds_df = ds_df.drop([col_tweet_ID, worker_id_mt], axis=1)
    ds_df = ds_df.rename(columns={"Aindex": worker_id_mt, "Mindex": col_tweet_ID})
    ds_df = ds_df[['comment_id', worker_id_mt, 'label', 'message', 'label_vector']]
    
    # Save full annotations
    path = foldername1 + "/" + _id + "_annotations.json"
    ds_df.to_json(path, orient='split')
    
    # Process the split (now using filename-based split)
    print(f"Processing split: {split_name}")
    
    # Prepare annotators array
    annotators_array = np.full(len(annotators_parsed), -1)
    
    # Save split data for modeling_annotator
    path = foldername1 + "/" + _id + f"_{split_name}.json"
    dfs_combine.to_json(path, orient='split', index=False)
    
    # Save split data for pldl
    path = foldername3 + "/" + _id + f"_{split_name}.json"
    convert_data_pldl_experiments(dfs_combine, colLabels, 'Mindex', path)
    
    # Generate data for BERT/disco format
    X_split = pd.unique(dfs_combine['message'])
    generate_data_bert(dfs_combine, foldername2, split_name, label_dict, _id, X_split, annotators_array)
    
    # --- Load and Process Metadata ---
    metadata_path = "datasets/pp/pp_metadata.json"
    annotator_metadata_embeddings = load_and_process_metadata(metadata_path, annotators_parsed)
    
    # Generate CSV files for neural network format with integrated metadata
    generate_data_nn_with_metadata(dfs_combine, foldername2, split_name, label_dict, _id, annotator_metadata_embeddings)
    
    print(f"Completed processing for {split_name} split with {len(dfs_combine)} items")


def load_and_process_metadata(metadata_path, annotators_parsed):
    """
    Load metadata and generate embeddings for annotators
    Returns a dictionary mapping annotator index to metadata embedding
    """
    with open(metadata_path, 'r') as f:
        metadata_dict = json.load(f)
    
    # Map annotator ID to metadata sentence
    annotator_id_to_sentence = {}
    for ann_id, meta in metadata_dict.items():
        annotator_id_to_sentence[ann_id] = metadata_to_sentence(meta)
    
    # Create metadata sentences for each annotator in order of their index
    annotator_metadata_sentences = []
    annotator_indices = []
    
    for idx, row in annotators_parsed.iterrows():
        ann_id = row['id']
        sentence = annotator_id_to_sentence.get(ann_id, "Unknown annotator.")
        annotator_metadata_sentences.append(sentence)
        annotator_indices.append(idx)  # This is the Aindex
    
    # Generate sentence embeddings for metadata
    meta_df = pd.DataFrame({
        'Mindex': annotator_indices, 
        'message': annotator_metadata_sentences
    })
    
    _, meta_embeddings = sentence_embedding(meta_df, annotator_indices)
    
    # Create mapping from annotator index to metadata embedding
    annotator_metadata_map = {}
    for i, embedding in enumerate(meta_embeddings):
        annotator_metadata_map[annotator_indices[i]] = embedding
    
    return annotator_metadata_map


def generate_data_nn_with_metadata(data_items, foldername, split_name, label_dict, _id, annotator_metadata_embeddings):
    """Generate neural network format CSV files for each split with integrated metadata"""
    
    # Save original JSON format
    path = foldername + "/" + _id + "_" + split_name + ".json"
    data_items.to_json(path, orient='split', index=False)
    
    # Generate AIL (Annotator-Item-Label) CSV with metadata
    path_ail_csv = foldername + "/" + _id + "_" + split_name + "_AIL.csv"
    path_ail_json = foldername + "/" + _id + "_" + split_name + "_AIL.json"
    data_items_parsed = convert_labels_hotencoding_with_metadata(data_items, len(label_dict), annotator_metadata_embeddings)
    
    # Save AIL CSV and JSON with metadata
    data_items_parsed.to_csv(path_ail_csv, index=False, header=False)
    data_items_parsed.to_json(path_ail_json, orient='split', index=False)
    
    # Prepare data for item and annotator distributions
    data_items_subset = data_items[['Mindex', 'Aindex', 'label_vector']]
    
    # Generate IL (Item-Label) CSV
    data_items_item_dist = convert_labels_per_group(data_items_subset, len(label_dict), 'Mindex')
    if not data_items_item_dist.empty:
        data_items_item_dist.columns = ["item", "label"]
        path = foldername + "/" + _id + "_" + split_name + "_IL.csv"
        data_items_item_dist.to_csv(path, index=False, header=False)
        
        # Generate Y and Yi numpy arrays
        Y = data_items_item_dist['label'].to_numpy()
        Y_final = []
        Yi_values = []
        
        for row in Y:
            row_values = []
            yi_row = []
            total = sum(row)
            for value in row:
                row_values.append(value)
                yi_row.append(value/total if total > 0 else 0)
            Y_final.append(row_values)
            Yi_values.append(yi_row)
        
        Y = np.asarray(Y_final)
        path = foldername + "/" + "Y_" + split_name + ".npy"
        np.save(path, Y)
        
        Yi_values = np.asarray(Yi_values)
        path = foldername + "/" + "Yi_" + split_name + ".npy"
        np.save(path, Yi_values)
        
        # Generate Ii (Item indices) numpy array
        Ii = data_items_item_dist['item'].to_numpy()
        Ii = np.expand_dims(np.asarray(Ii), axis=1)
        path = foldername + "/" + "Ii_" + split_name + ".npy"
        np.save(path, Ii)
    
    # Generate AL (Annotator-Label) CSV with integrated metadata
    data_items_annotator_dist = convert_labels_per_group_with_metadata(data_items_subset, len(label_dict), 'Aindex', annotator_metadata_embeddings)
    if not data_items_annotator_dist.empty:
        # Save AL CSV and JSON with metadata
        path_al_csv = foldername + "/" + _id + "_" + split_name + "_AL.csv"
        path_al_json = foldername + "/" + _id + "_" + split_name + "_AL.json"
        
        np.set_printoptions(linewidth=100000)
        data_items_annotator_dist.to_csv(path_al_csv, index=False, header=False)
        data_items_annotator_dist.to_json(path_al_json, orient='split', index=False)
        
        # Generate Ai (Annotator indices) numpy array
        Ai = data_items_annotator_dist['annotator'].to_numpy()
        Ai = np.expand_dims(np.asarray(Ai), axis=1)
        path = foldername + "/" + "Ai_" + split_name + ".npy"
        np.save(path, Ai)
        
        # Generate Ya (Annotator label distribution) numpy array
        Ya_values = []
        Ya_rows = data_items_annotator_dist['label'].to_numpy()
        for row in Ya_rows:
            ya_row = []
            total = sum(row)
            for value in row:
                ya_row.append(value/total if total > 0 else 0)
            Ya_values.append(ya_row)
        
        Ya = np.asarray(Ya_values)
        path = foldername + "/" + "Ya_" + split_name + ".npy"
        np.save(path, Ya)
        
        # Generate Xa (annotator metadata embeddings) numpy array
        metadata_embeddings_array = np.array(data_items_annotator_dist['metadata_embedding'].tolist())
        path = foldername + "/" + "Xa_" + split_name + ".npy"
        np.save(path, metadata_embeddings_array)
    
    # Generate embeddings and related files (for items)
    data_items_index = pd.unique(data_items['Mindex'])
    if len(data_items_index) > 0:
        data_items_embed, embeddings = sentence_embedding(data_items, data_items_index)
        path = foldername + "/" + _id + "_" + split_name + "_IE.csv"
        data_items_embed.to_csv(path, index=False)
        
        # Generate X (embeddings) numpy array
        X = np.asarray(embeddings)
        path = foldername + "/" + "X_" + split_name + ".npy"
        np.save(path, X)
        
        # Generate Xi (item embeddings) numpy array
        path = foldername + "/" + "Xi_" + split_name + ".npy"
        data_items_embed_Xi = data_items_embed.to_numpy()
        np.save(path, data_items_embed_Xi)


def convert_labels_hotencoding_with_metadata(data_items, no_classes, annotator_metadata_embeddings):
    """Convert labels to hot encoding format with metadata embeddings integrated"""
    hotencoded = []
    
    for index, row in data_items.iterrows():
        # Use soft label distribution if available, otherwise use hard label
        if 'soft_label_vector' in row and row['soft_label_vector'] is not None:
            labels = row['soft_label_vector']  # Already normalized soft distribution
        else:
            # Fallback to hard label encoding
            labels = np.zeros(no_classes)
            labels[row['label_vector']] = 1
        
        # Get metadata embedding for this annotator
        annotator_idx = row['Aindex']
        if annotator_idx in annotator_metadata_embeddings:
            metadata_embedding = annotator_metadata_embeddings[annotator_idx]
        else:
            # Fallback: create zero embedding if metadata not found
            embedding_dim = len(list(annotator_metadata_embeddings.values())[0]) if annotator_metadata_embeddings else 768
            metadata_embedding = np.zeros(embedding_dim)
        
        parsed_row = {}
        parsed_row['item'] = row['Mindex']
        parsed_row['annotator'] = row['Aindex']
        parsed_row['label'] = labels.astype(float)  # Keep as float for soft labels
        parsed_row['metadata_embedding'] = metadata_embedding
        hotencoded.append(parsed_row)
    
    return pd.DataFrame(hotencoded)


def convert_labels_per_group_with_metadata(data_items, no_classes, grouping_category, annotator_metadata_embeddings):
    """Convert labels per group (annotator) with metadata embeddings integrated"""
    encoded = []
    unique_data_items = pd.unique(data_items[grouping_category])
    
    for row in unique_data_items:
        encoded_row = {}
        labels = np.zeros(no_classes)
        items = data_items.loc[data_items[grouping_category] == row]
        for index, item in items.iterrows():
            labels[item['label_vector']] += 1
        
        # Get metadata embedding for this annotator (when grouping by annotator)
        if grouping_category == 'Aindex':
            annotator_idx = row
            if annotator_idx in annotator_metadata_embeddings:
                metadata_embedding = annotator_metadata_embeddings[annotator_idx]
            else:
                # Fallback: create zero embedding if metadata not found
                embedding_dim = len(list(annotator_metadata_embeddings.values())[0]) if annotator_metadata_embeddings else 768
                metadata_embedding = np.zeros(embedding_dim)
            encoded_row['metadata_embedding'] = metadata_embedding
        
        encoded_row['annotator'] = row
        encoded_row['label'] = labels.astype(int)
        encoded.append(encoded_row)
    
    return pd.DataFrame(encoded)


# Keep all your existing helper functions unchanged
def extract_split_from_filename(filename):
    """
    Extract split name from filename
    Examples:
    - dataset_train.json -> train
    - csc_dev.json -> dev
    - data_test.json -> test
    """
    # Get the base filename without extension
    base_name = os.path.splitext(os.path.basename(filename))[0]
    
    # Common split indicators
    split_indicators = ['train', 'dev', 'test', 'val', 'validation']
    
    # Check if any split indicator is in the filename
    for split_type in split_indicators:
        if split_type in base_name.lower():
            return split_type
    
    # If no standard split found, return the last part after underscore
    parts = base_name.split('_')
    if len(parts) > 1:
        return parts[-1].lower()
    
    # Default fallback
    return 'unknown'


def read_json_data(json_file_path, split_name):
    """
    Read JSON data and convert to DataFrame format expected by the rest of the code
    """
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    processed_rows = []
    
    # Process each item in the JSON
    for comment_id, item in data.items():
        # Extract basic information
        text_data = item['text']
        post = text_data['Question1']
        reply = text_data['Question2']
        annotators_list = item['annotators'].split(',')
        annotations = item['annotations']
        soft_labels = item.get('soft_label', {})
        
        # Use the split name extracted from filename
        split = split_name
        lang = item.get('lang', 'en')        
        
        # Create a row for each annotator
        for annotator in annotators_list:
            annotator = annotator.strip()  # Remove any whitespace
            if annotator in annotations:
                sarcasm_score = annotations[annotator]
                
                # Get soft label distribution for this item
                # Paraphrase format: soft_label has keys for each class (-5 to +5)
                soft_label_dist = []
                for class_idx in range(-5, 6):  # -5 to +5 (11 classes)
                    class_key = str(class_idx)
                    if class_key in soft_labels:
                        prob = soft_labels[class_key]
                    else:
                        prob = 0.0
                    soft_label_dist.append(prob)
                
                # Normalize soft distribution
                total = sum(soft_label_dist)
                if total > 0:
                    soft_label_dist = [p/total for p in soft_label_dist]
                
                row = {
                    'comment_id': f"csc_item_{comment_id}",
                    'original_id': comment_id,  # Store original JSON key
                    'annotator_id': annotator,
                    'post': post,
                    'reply': reply,
                    'sarcasm_score': str(sarcasm_score),  # Keep as string to match colLabels
                    'Q_overall': str(sarcasm_score),  # Keep for compatibility
                    'soft_label_dist': soft_label_dist,  # Add soft label distribution
                    'split': split,
                    'lang': lang,
                }
                processed_rows.append(row)
    
    return pd.DataFrame(processed_rows)


def row_values_counter(colname, data_rows_pp):
    """Count most common value in a column"""
    label_counts_sub = Counter(data_rows_pp[colname])
    most_common, num_most_common = label_counts_sub.most_common(1)[0]
    return most_common


def label_majority(comments_dataset):
    """Process majority labeling for each comment and annotator"""
    data_items = pd.unique(comments_dataset['comment_id'])
    processed_rows = pd.DataFrame()
    
    for data_item in tqdm(data_items, desc="Processing majority labels"):
        data_rows = comments_dataset.loc[comments_dataset['comment_id'] == data_item]
        annotators = pd.unique(data_rows['annotator_id'])
        
        for annotator in annotators:
            data_rows_annotator = data_rows.loc[data_rows['annotator_id'] == annotator]
            data_row_df = data_rows_annotator.head(1)
            if data_rows_annotator.empty:
                pdb.set_trace()
            data_row = data_row_df.to_dict()
            data_row['Q_overall'] = row_values_counter('Q_overall', data_rows_annotator)
            data_row['sarcasm_score'] = row_values_counter('sarcasm_score', data_rows_annotator)
            data_row = pd.DataFrame(data_row)
            processed_rows = pd.concat([processed_rows, data_row], ignore_index=True)
    
    return processed_rows


def convert_labels_per_group(data_items, no_classes, grouping_category):
    """Convert labels per group (item or annotator)"""
    encoded = []
    unique_data_items = pd.unique(data_items[grouping_category])
    
    for row in unique_data_items:
        encoded_row = {}
        labels = np.zeros(no_classes)
        items = data_items.loc[data_items[grouping_category] == row]
        for index, item in items.iterrows():
            labels[item['label_vector']] += 1
        encoded_row[grouping_category] = row
        encoded_row['label'] = labels.astype(int)
        encoded.append(encoded_row)
    
    return pd.DataFrame(encoded)


if __name__ == "__main__":
    main()
