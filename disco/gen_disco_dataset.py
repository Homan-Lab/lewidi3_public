import os
import sys, getopt
import pickle
import numpy as np
import tensorflow as tf
import pandas as pd
from utils.utils import scale_feat, gen_data_plot

def create_folder(folderpath):
    if not os.path.exists(folderpath):
        os.makedirs(folderpath)

def strToVec(str, delim=" "):
    vec_s = str.split(delim)
    vec = []
    for vs in vec_s:
        try:
            vec.append(int(vs))
        except:
            continue
    vec = np.expand_dims(np.asarray(vec), axis=0)
    return vec

def fileToMap(fname):
    print(" > Reading file: ", fname)
    map = {}
    fd = open(fname, 'r')
    count = 0
    line = fd.readline()
    while line:
        count += 1
        tok = line.split(",")
        if len(tok) > 1:
            idx = int(tok[0])
            vec_s = tok[1].replace('[', '').replace(']', '')
            map.update({idx: vec_s})
        line = fd.readline()
        print("\r {0} lines read".format(count), end="")
    print()
    return map

def parse_annotator_metadata_file(fname):
    """
    Reads a CSV file with 3 columns:
        - Column 0: stringified list of np.float32() values (embedding)
        - Column 1: annotator_id
        - Column 2: space-separated label array like: [7 5 3 0 ...]
    
    Returns:
        metadata_map: annotator_id -> np.array(embedding)
        label_map: annotator_id -> np.array(label vector)
    """
    print(" > Reading annotator metadata file: ", fname)
    df = pd.read_csv(fname, header=None)
    df.columns = ['embedding', 'annotator_id', 'label']

    def parse_embedding(emb_str):
        # Remove np.float32 and strip brackets
        emb_str = emb_str.strip().replace('np.float32', '').replace('[', '').replace(']', '')
        values = emb_str.split(',')
        cleaned_values = [x.strip().replace('(', '').replace(')', '') for x in values]
        return np.array([float(x) for x in cleaned_values if x], dtype=np.float32)


    def parse_label(label_str):
        label_str = label_str.strip().replace('[', '').replace(']', '')
        return np.array([int(x) for x in label_str.split()], dtype=np.int32)

    metadata_map = {}
    label_map = {}

    for _, row in df.iterrows():
        annot_id = int(row['annotator_id'])
        embedding = parse_embedding(row['embedding'])
        label = parse_label(row['label'])
        metadata_map[annot_id] = embedding
        label_map[annot_id] = label

    return metadata_map, label_map


################################################################################
# Parse command line arguments
options, remainder = getopt.getopt(sys.argv[1:], '', [
    "inp_dir=",
    "out_dir=",
    "annotator_item_fname=",
    "item_lab_fname=",
    "annotator_lab_fname=",
    "embeddings=",
    "split_name="
])

inp_dir = None
out_dir = None
annotator_item_fname = None
item_lab_fname = None
annotator_lab_fname = None

for opt, arg in options:
    if opt == "--inp_dir":
        inp_dir = arg.strip()
    elif opt == "--out_dir":
        out_dir = arg.strip()
    elif opt == "--annotator_item_fname":
        annotator_item_fname = arg.strip()
    elif opt == "--item_lab_fname":
        item_lab_fname = arg.strip()
    elif opt == "--annotator_lab_fname":
        annotator_lab_fname = arg.strip()
    elif opt == "--embeddings":
        embeddings_fname = arg.strip()
    elif opt == "--split_name":
        split_name = arg.strip()

################################################################################
# Load data
yi_map = fileToMap(os.path.join(inp_dir, item_lab_fname))
annotator_metadata, label_map = parse_annotator_metadata_file(os.path.join(inp_dir, annotator_lab_fname))

embeddings = np.load(os.path.join(inp_dir, embeddings_fname), allow_pickle=True)
embeddings = pd.DataFrame(embeddings)
embeddings.columns = ['data_i', 'embedding']

Yi, Ya, Y, X = None, None, None, None
Ii, Ai, Am = [], [], []  # item ids, annotator ids, annotator metadata embeddings

print(" > Reading data file: ", os.path.join(inp_dir, annotator_item_fname))
fd = open(os.path.join(inp_dir, annotator_item_fname), 'r')
count = 0
line = fd.readline()

while line:
    count += 1
    tok = line.strip().split(",")
    if len(tok) > 1:
        item_id = int(tok[0])
        annot_id = int(tok[1])

        # Item embedding
        embed = embeddings.loc[embeddings['data_i'] == item_id, 'embedding'].values
        embed = np.asarray(embed[0], dtype=np.float32)
        embed = np.expand_dims(embed, axis=0)

        # Labels
        y_lab = strToVec(tok[2].replace('[', '').replace(']', ''))
        yi_lab = strToVec(yi_map.get(item_id))
        ya_lab = strToVec(" ".join(map(str, label_map.get(annot_id))))

        yi_lab = yi_lab / (np.sum(yi_lab) + 1e-8)
        ya_lab = ya_lab / (np.sum(ya_lab) + 1e-8)

        # Annotator metadata
        annot_meta = annotator_metadata.get(annot_id)
        if annot_meta is None:
            raise ValueError(f"No metadata found for annotator {annot_id}")
        annot_meta = np.expand_dims(annot_meta, axis=0)

        # Append
        if count > 1:
            Y = np.concatenate((Y, y_lab), axis=0)
            Yi = np.concatenate((Yi, yi_lab), axis=0)
            Ya = np.concatenate((Ya, ya_lab), axis=0)
            X = np.concatenate((X, embed), axis=0)
            Am = np.concatenate((Am, annot_meta), axis=0)
        else:
            Y = y_lab
            Yi = yi_lab
            Ya = ya_lab
            X = embed
            Am = annot_meta

        Ii.append(item_id)
        Ai.append(annot_id)

    print("\r {0} data lines read".format(count), end="")
    line = fd.readline()

fd.close()

# Process ID lists
Ii = np.expand_dims(np.asarray(Ii), axis=1)
Ai = np.expand_dims(np.asarray(Ai), axis=1)
Am = np.asarray(Am)
A = np.concatenate((Ai, Am), axis=1)  # final annotator matrix with ID + metadata

print("\n >> Saving distribution design matrices to dir: {0}".format(out_dir))
create_folder(out_dir)
np.save(f"{out_dir}Xi_{split_name}.npy", X)
np.save(f"{out_dir}Yi_{split_name}.npy", Yi)
np.save(f"{out_dir}Ya_{split_name}.npy", Ya)
np.save(f"{out_dir}Y_{split_name}.npy", Y)
np.save(f"{out_dir}I_{split_name}.npy", Ii)
np.save(f"{out_dir}A_{split_name}.npy", A)  # Contains both annotator ID and metadata embedding

gen_data_plot(X, tf.cast(Y, dtype=tf.float32), use_tsne=False, fname=split_name, out_dir=out_dir)
