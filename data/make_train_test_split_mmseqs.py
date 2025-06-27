import os
import pandas as pd
import subprocess
from tqdm import tqdm
import tempfile
import shutil

MMSEQS_PATH = "mmseqs"  # Assumed in PATH
SEQUENCE_COLUMN = "Sequences"
OUTPUT_DIR = "mmseqs_split"

def write_fasta(df, fasta_path):
    with open(fasta_path, "w") as f:
        for i, row in df.iterrows():
            f.write(f">{i}\n{row[SEQUENCE_COLUMN]}\n")

def run_mmseqs_clustering(input_fasta, tmp_dir, min_seq_id=0.3):
    db = os.path.join(tmp_dir, "inputDB")
    cluster_db = os.path.join(tmp_dir, "clusterDB")
    cluster_tsv = os.path.join(tmp_dir, "clusters.tsv")

    subprocess.run([MMSEQS_PATH, "createdb", input_fasta, db], check=True)
    subprocess.run([MMSEQS_PATH, "cluster", db, cluster_db, tmp_dir,
                    "--min-seq-id", str(min_seq_id),
                    "-c", "0.8"], check=True)
    subprocess.run([MMSEQS_PATH, "createtsv", db, db, cluster_db, cluster_tsv], check=True)
    
    return cluster_tsv

def build_cluster_dict(tsv_path):
    cluster_map = {}
    with open(tsv_path) as f:
        for line in f:
            cluster, member = line.strip().split('\t')
            cluster_map.setdefault(cluster, []).append(int(member))
    return list(cluster_map.values())

def split_clusters(clusters, test_fraction=0.2):
    clusters_sorted = sorted(clusters, key=len, reverse=True)
    total = sum(len(c) for c in clusters_sorted)
    
    test_clusters, test_size = [], 0
    for c in reversed(clusters_sorted):
        if test_size / total < test_fraction:
            test_clusters.append(c)
            test_size += len(c)
        else:
            break

    test_indices = set(i for c in test_clusters for i in c)
    train_indices = set(i for c in clusters_sorted for i in c) - test_indices
    return sorted(train_indices), sorted(test_indices)

def main(input_csv):
    df = pd.read_csv(input_csv)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        fasta_path = os.path.join(tmp, "input.fasta")
        write_fasta(df, fasta_path)

        print("🔍 Running MMseqs2 clustering at 30% identity...")
        cluster_tsv = run_mmseqs_clustering(fasta_path, tmp, min_seq_id=0.3)

        print("📦 Building clusters...")
        clusters = build_cluster_dict(cluster_tsv)

        print("✂️ Splitting clusters into train/test...")
        train_idx, test_idx = split_clusters(clusters, test_fraction=0.2)

        df_train = df.iloc[train_idx].reset_index(drop=True)
        df_test = df.iloc[test_idx].reset_index(drop=True)

        train_path = os.path.join(OUTPUT_DIR, "train.csv")
        test_path = os.path.join(OUTPUT_DIR, "test.csv")
        df_train.to_csv(train_path, index=False)
        df_test.to_csv(test_path, index=False)

        print(f"✅ Train: {len(df_train)} sequences")
        print(f"✅ Test:  {len(df_test)} sequences (all <30% ID to training)")
        print(f"\n📁 Files saved to: {OUTPUT_DIR}/train.csv and test.csv")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python make_train_test_split_mmseqs.py <input_dataframe.csv>")
    else:
        main(sys.argv[1])
