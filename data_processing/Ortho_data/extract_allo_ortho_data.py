import os
import pandas as pd
from Bio.PDB import PDBParser, is_aa, Polypeptide
from collections import defaultdict
import requests
import warnings
import csv
from Bio import pairwise2

warnings.filterwarnings("ignore", category=UserWarning)


# --- Settings ---
orthosteric_dir = "~/PLM_Allosteric_Classification/data/Orthosteric_PDBs"
allosteric_dir = "~/PLM_Allosteric_Classification/data/Allosteric_PDBs"
pocket_base_dir = "~/PLM_Allosteric_Classification/data/Allo_Ortho_Pockets"

# --- Download PDB ---
def download_pdb(pdb_id, save_dir):
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    save_path = os.path.join(save_dir, f"{pdb_id}.pdb")
    if not os.path.exists(save_path):
        os.makedirs(save_dir, exist_ok=True)
        r = requests.get(url)
        if r.status_code == 200:
            with open(save_path, "w") as f:
                f.write(r.text)
        else:
            raise ValueError(f"Failed to download PDB {pdb_id}")
    return save_path

# --- Extract sequence and residue mapping ---
non_standard = {'MSE', 'SEP', 'TPO', 'PTR', 'MLY', 'CSO', 'PCA', 'HYP', 'CIR', 'F2F'}
three_to_one = lambda x: Polypeptide.three_to_one(x) if x in Polypeptide.standard_aa_names else "X"

def extract_sequences_from_pdb(pdb_path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("", pdb_path)
    sequences = {}
    residue_info = {}
    for model in structure:
        for chain in model:
            seq = []
            res_list = []
            for res in chain:
                if is_aa(res, standard=True) or res.get_resname() in non_standard:
                    resname = res.get_resname()
                    one_letter = three_to_one(resname)
                    resnum = res.get_id()[1]
                    seq.append(one_letter)
                    res_list.append((resnum, resname))
            if seq:
                sequences[chain.id] = "".join(seq)
                residue_info[chain.id] = res_list  # list of (resnum, resname)
    return sequences, residue_info

# --- Extract chain, resnum, and resname from pocket ---
def extract_residues_from_pocket(pocket_path):
    residues = set()
    with open(pocket_path) as f:
        for line in f:
            if line.startswith("ATOM"):
                chain = line[21]
                resnum = int(line[22:26])
                resname = line[17:20].strip()
                residues.add((chain, resnum, resname))
    return residues

# --- Match residue by number and amino acid ---
def label_sequence(res_list, chain_id, pocket_residues):
    labels = []
    for resnum, resname in res_list:
        match = (chain_id, resnum, resname)
        labels.append(1 if match in pocket_residues else 0)
    return labels

def map_orthosteric_pocket_to_allosteric(
    ortho_seq_chain, ortho_resinfo_chain,
    allo_seq_chain, allo_resinfo_chain,
    ortho_pocket_chain_residues, ortho_chain, allo_chain
):
    
    """
    Map residues from one orthosteric chain to one allosteric chain.
    Inputs are sequences and residue info lists **for single chains only**.
    ortho_pocket_chain_residues is a set of residue identifiers (chain, resnum, resname)
    only for this orthosteric chain.
    """
    # Extract just the (resnum, resname) tuples for the correct chain from ortho_pocket_chain_residues
    #ortho_pocket_chain_simple = set((resnum, resname) for (chain, resnum, resname) in ortho_pocket_chain_residues if chain == ortho_chain)
    #print(f"ortho pocket chain simple: {ortho_pocket_chain_simple}")
    alignment = pairwise2.align.globalxx(ortho_seq_chain, allo_seq_chain, one_alignment_only=True)[0]
    aligned_ortho = alignment.seqA
    aligned_allo = alignment.seqB

    ortho_pointer = allo_pointer = 0
    mapped_positions = set()

    for i in range(len(aligned_ortho)):
        res_o = aligned_ortho[i]
        res_a = aligned_allo[i]
        #print(f"res_o: {res_o}")
        #print(f"res_a: {res_a}")
        # Both residues present (not gaps)
        if res_o != "-" and res_a != "-":
            if ortho_pointer < len(ortho_resinfo_chain) and allo_pointer < len(allo_resinfo_chain):
                resnum, resname = ortho_resinfo_chain[ortho_pointer]
                ortho_id = (ortho_chain, resnum, resname)
                #print(ortho_id)
                allo_resnum, allo_resname = allo_resinfo_chain[allo_pointer]
                allo_id = (allo_chain, allo_resnum, allo_resname)
                #print(allo_id)
                #print(f"ortho id: {ortho_id}")
                #print(f"allo id: {allo_id}")
                if ortho_id in ortho_pocket_chain_residues:
                    mapped_positions.add(allo_id)

            ortho_pointer += 1
            allo_pointer += 1
        elif res_o != "-":
            ortho_pointer += 1
        elif res_a != "-":
            allo_pointer += 1

    return mapped_positions


def find_best_chain_match(o_chain_seq, allo_seqs):
    best_score = -float("inf")
    best_chain = None
    for a_chain, a_seq in allo_seqs.items():
        alignment = pairwise2.align.globalxx(o_chain_seq, a_seq, one_alignment_only=True, score_only=True)
        if alignment > best_score:
            best_score = alignment
            best_chain = a_chain
    return best_chain


def map_residues_from_alignment(o_seq, a_seq, o_residues):

    alignments = pairwise2.align.globalxx(o_seq, a_seq, one_alignment_only=True)
    aligned_o, aligned_a = alignments[0][:2]

    o_to_a_index = {}
    o_idx = a_idx = 0
    for i in range(len(aligned_o)):
        if aligned_o[i] != '-':
            o_pos = o_idx
            o_idx += 1
        else:
            continue

        if aligned_a[i] != '-':
            a_pos = a_idx
            a_idx += 1
        else:
            continue

        o_to_a_index[o_pos] = a_pos

    # Convert residue numbers based on mapping
    mapped_residues = []
    for chain, resnum, resname in o_residues:
        try:
            o_chain_seq = o_seq  # for clarity
            o_chain_residues = list(range(len(o_chain_seq)))  # pseudo-residue index
            o_pos = o_chain_residues.index(resnum)  # you may need a mapping from resnum -> index
            a_pos = o_to_a_index[o_pos]
            mapped_residues.append((chain, a_pos, resname))  # Use actual mapped chain name if changed
        except Exception as e:
            print(f"Could not map residue {resnum} in chain {chain}: {e}")
            continue

    return mapped_residues

# --- Main processing ---
from Bio import pairwise2

def process_dataset(df):
    all_results = []
    for i, row in df.iterrows():
        ortho_id = row["substrate_pdb"]
        allo_id = row["allosteric_pdb"]
        ortho_site = row["substrate_site"]
        allo_site = row["allosteric_site"]
        
        if not ortho_id or not ortho_site:
            print(f"Skipping row {i} because orthosteric ID or site is missing")
            continue

        ortho_path = os.path.join(orthosteric_dir, f"{ortho_id}.pdb")
        if not os.path.exists(ortho_path):
            try:
                ortho_path = download_pdb(ortho_id, orthosteric_dir)
            except Exception as e:
                print(f"Failed to download orthosteric PDB {ortho_id} for row {i}: {e}")
                continue
        else:
            print(f"Using cached orthosteric PDB: {ortho_path}")

        allo_path = os.path.join(allosteric_dir, f"{allo_id}.pdb")
        if not os.path.exists(allo_path):
            try:
                allo_path = download_pdb(allo_id, allosteric_dir)
            except Exception as e:
                print(f"Failed to download allosteric PDB {allo_id} for row {i}: {e}")
                continue
        else:
            print(f"Using cached allosteric PDB: {allo_path}")

        try:
            ortho_seqs, ortho_resinfo = extract_sequences_from_pdb(ortho_path)
            allo_seqs, allo_resinfo = extract_sequences_from_pdb(allo_path)
            
        except Exception as e:
            print(f"Failed to parse PDBs for row {i}: {e}")
            continue

        try:
            allo_pocket = extract_residues_from_pocket(os.path.join(pocket_base_dir, f"{allo_site}.pdb"))
            ortho_pocket = extract_residues_from_pocket(os.path.join(pocket_base_dir, f"{ortho_site}.pdb"))
            
           
        except FileNotFoundError:
            print(f"Missing pocket file in row {i}, skipping")
            continue

        total_chains = {t[0] for t in allo_pocket}.union({t[0] for t in ortho_pocket})
        # --- New logic: find best matching chain for each orthosteric pocket chain ---
        chain_mapping = {}  # ortho_chain -> best matching allo_chain
        for ortho_chain in {t[0] for t in ortho_pocket}:
            if ortho_chain not in ortho_seqs:
                print(f"Orthosteric chain {ortho_chain} missing in sequences")
                continue

            best_match = None
            best_score = -1
            ortho_seq = ortho_seqs[ortho_chain]
            for allo_chain in allo_seqs:
                try:
                    score = pairwise2.align.globalxx(ortho_seq, allo_seqs[allo_chain], score_only=True)
                    if score > best_score:
                        best_score = score
                        best_match = allo_chain
                except Exception:
                    continue

            if best_match:
                chain_mapping[ortho_chain] = best_match
            else:
                print(f"No suitable match found for orthosteric chain {ortho_chain}, skipping row {i}")
                continue

        if not chain_mapping:
            print(f"No chain mappings found for row {i}, skipping.")
            continue
        #print(f"chain mapping: {chain_mapping}")
        # Concatenate sequences and labels
        allo_seq_parts = []
        ortho_labels = []
        allo_labels = []

        for allo_chain in {t[0] for t in allo_pocket}:
            if allo_chain not in allo_resinfo:
                print(f"Chain {allo_chain} missing in residue info, skipping")
                continue

            allo_seq = allo_seqs[allo_chain]
            allo_res = allo_resinfo[allo_chain]

            allo_label_chain = label_sequence(allo_res, allo_chain, allo_pocket)
            allo_seq_parts.append(allo_seq)
            allo_labels.extend(allo_label_chain)

            # If this allosteric chain is mapped to from any orthosteric chain:
            matching_ortho_chains = [o for o, a in chain_mapping.items() if a == allo_chain]
            #print(f"allo chain: {allo_chain}")
            #print(f"matching ortho chains: {matching_ortho_chains}")
            if matching_ortho_chains:
                combined_mapped_positions = set()
                for ortho_chain in matching_ortho_chains:
                    if ortho_chain not in ortho_seqs or ortho_chain not in ortho_resinfo:
                        continue
                    ortho_seq = ortho_seqs[ortho_chain]
                    ortho_res = ortho_resinfo[ortho_chain]
                    ortho_pocket_chain = set(res for res in ortho_pocket if res[0] == ortho_chain)

                    for ortho_chain in matching_ortho_chains:
                        if ortho_chain not in ortho_seqs or ortho_chain not in ortho_resinfo:
                            continue

                        allo_chain = chain_mapping[ortho_chain]

                        ortho_seq_chain = ortho_seqs[ortho_chain]
                        ortho_res_chain = ortho_resinfo[ortho_chain]

                        allo_seq_chain = allo_seqs[allo_chain]
                        allo_res_chain = allo_resinfo[allo_chain]

                        ortho_pocket_chain_residues = set(res for res in ortho_pocket if res[0] == ortho_chain)

                        mapped_positions = map_orthosteric_pocket_to_allosteric(
                            ortho_seq_chain, ortho_res_chain,
                            allo_seq_chain, allo_res_chain,
                            ortho_pocket_chain_residues, ortho_chain, allo_chain
                        )

                    #print(f"mapped positions: {mapped_positions}")
                    combined_mapped_positions.update(mapped_positions)
                    print(f"combined_mapped_positions: {combined_mapped_positions}")
                ortho_label_chain = label_sequence(allo_res, allo_chain, combined_mapped_positions)
            else:
                ortho_label_chain = [0] * len(allo_label_chain)

            ortho_labels.extend(ortho_label_chain)

        allo_full_seq = "".join(allo_seq_parts)

        all_results.append({
            "pdb_id_allosteric": allo_id,
            "pdb_id_orthosteric": ortho_id,
            "sequence": allo_full_seq,
            "labels_allosteric": allo_labels,
            "labels_orthosteric": ortho_labels,
            "chains_included": list(total_chains),
        })

    return pd.DataFrame(all_results)


# --- Example usage ---
input_csv = "~/PLM_Allosteric_Classification/data/allo_ortho_table.csv"  # Replace with your actual dataset
df = pd.read_csv(input_csv)
output_df = process_dataset(df)

# Save result
output_df.to_json("labeled_sequences_with_verified_pockets.jsonl", orient="records", lines=True)
output_df.to_csv("final_allosteric_orthosteric_dataset.csv", index=False)
