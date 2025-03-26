from Bio.PDB import PDBParser, NeighborSearch, is_aa
from Bio.PDB.Polypeptide import PPBuilder, three_to_one
from Bio.PDB.PDBExceptions import PDBConstructionWarning
import warnings
import os
from Bio import PDB
import pandas as pd
from biopandas.pdb import PandasPdb
import json
import re
import pandas as pd
import os
import ast
import requests
from Bio import PDB
from Bio.PDB import PDBParser, NeighborSearch, is_aa
from Bio.PDB.Polypeptide import three_to_one, one_to_three
from Bio.SeqUtils import seq1

# Suppress warnings from Bio.PDB
warnings.simplefilter('ignore', PDBConstructionWarning)


def find_contact_residues(protein_pdb_file, ligand_pdb_file, distance_cutoff):
    # Suppress warnings from Bio.PDB
    warnings.simplefilter('ignore', PDBConstructionWarning)

    # Parse protein and ligand PDB files
    parser = PDBParser(QUIET=True)
    protein_structure = parser.get_structure('protein', protein_pdb_file)
    ligand_structure = parser.get_structure('ligand', ligand_pdb_file)

    # Extract atoms from the ligand
    ligand_atoms = list(ligand_structure.get_atoms())


    # Create a NeighborSearch object for protein atoms
    ns = NeighborSearch(list(protein_structure.get_atoms()))

    # Store residues in contact with the ligand
    contact_residues = set()

   # Iterate through ligand atoms
    for ligand_atom in ligand_atoms:
        ligand_coord = ligand_atom.get_coord()
        #print(f"Ligand Atom: {ligand_atom}, Coordinates: {ligand_coord}")
        # Find nearby atoms in the protein structure
        close_atoms = ns.search(ligand_coord, distance_cutoff, level='A')
        #print(f"Close Atoms: {close_atoms}")
        # Extract residue information from nearby protein atoms
        for protein_atom in close_atoms:
        # Extract residue information
            residue = protein_atom.get_parent()
            residue_id = residue.id
            if isinstance(residue_id, tuple):
                residue_number = residue_id[1]  # Extracting residue number
            else:
                residue_number = residue_id
            residue_name = residue.resname
            # Check if residue has a chain identifier and is not a water molecule
            if str(residue_number).strip() and residue_name != "HOH":
                chain_id = residue.get_parent().id
                if chain_id.strip():
                    contact_residues.add((chain_id, residue_number, residue_name))

    return contact_residues

def extract_sequences_from_pdb(structure):
   

    # Initialize a dictionary to store sequences for each chain
    sequences = {}
    residue_numbers_by_chain = {}
    residue_names_by_chain = {}
    # Iterate through each model in the structure
    for model in structure:
        # Iterate through each chain in the model
        for chain in model:
            # Initialize an empty string to store the sequence
            sequence = ''
            residue_numbers = []
            residue_names = []

            # Iterate through each residue in the chain
            for residue in chain:
                residue_name = residue.get_resname()
                #if residue.get_id()[0] == ' ' and residue.get_id()[0] != 'H':
                    #print(residue.get_id())
                if residue_name.strip() != 'HOH':
                    residue_numbers.append(residue.get_id()[1])
                    residue_names.append(residue_name.strip())
                        # Check if the residue is a standard amino acid
                    if PDB.is_aa(residue.get_resname(), standard=True):
                            # Append the one-letter code of the amino acid to the sequence
                        sequence += PDB.Polypeptide.three_to_one(residue.get_resname())
                    else:
                        sequence += 'X'

            # Store the sequence for the chain in the dictionary
            sequences[chain.get_id()] = sequence
            residue_numbers_by_chain[chain.get_id()] = residue_numbers
            residue_names_by_chain[chain.get_id()] = residue_names
    return sequences, residue_numbers_by_chain, residue_names_by_chain

def count_chains(structure):
    chain_set = set()

    # Iterate through each model in the structure
    for model in structure:
        for chain in model:
            # Check if the chain identifier is a letter (not an empty space)
            if chain.id.strip() and chain.id.isalpha():
                chain_set.add(chain.id)
            elif chain.id.isdigit():
                chain_set.add(chain.id)

    # Count the number of unique chains
    num_chains = len(chain_set)
    chain_list = list(chain_set)

    return num_chains, chain_list

def make_labels(pocket_residues, sequences, protein_residues, pocket_chains, folder, data):
   # data =[]
    labels = {}
    for chain in pocket_chains:
        labels[chain] = [0]*len(sequences[chain])
        pocket_filtered = [entry for entry in pocket_residues if entry[0] == chain]
        protein_filtered = [entry for entry in protein_residues if entry[0] == chain]


        for contact in pocket_filtered:
            if contact in protein_filtered:
                index = protein_filtered.index(contact)
                 #   print(contact, protein_filtered[index])
                 #   print(index, len(labels), len(sequence), len(protein_filtered))                    
                aa3 = contact[2]
                aa1 = sequences[chain][index]
                  #  print(aa3, aa1, sequence[-1])
                    #if three_to_one(aa3)!=aa1:
                     #   print("no match")
                labels[chain][index]=1
                   # print("found")

            else:
                print(contact, "is not in the list")
                break
                    
    return labels

def get_structure_info(structure):
    # Create a PDB parser object
  #  parser = PDBParser(QUIET=True)

    # Parse the PDB file
   # structure = parser.get_structure('structure', pdb_file)

    # Initialize a list to store structure information
    structure_info = []

    # Iterate over all chains in the structure
    for chain in structure.get_chains():
        chain_id = chain.id

        # Iterate over all residues in the chain
        for residue in chain.get_residues():
            residue_name = residue.get_resname()
            if residue_name.strip() != 'HOH':
                residue_number = residue.id[1]
                # Append chain ID, residue number, and residue name to the structure info list
                structure_info.append((chain_id, residue_number, residue_name))

    return structure_info

def find_file_within_directory(file_list, file):
    """
    Find strings in a list of strings that contain a specific substring.
    
    Parameters:
    - file_list (list): List of files to search through.
    - file(str): File to search for.
    
    Returns:
    - list: List of files containing the specified filename.
    """
    # Using list comprehension to filter strings containing the substring
    matching_file = [string for string in file_list if file in string]
    
    return matching_file[0]

def write_list_of_lists_to_file(data, output_file):
    """
    Write a list of lists to a file where each inner list represents a line.
    """
    with open(output_file, "w") as f:
        for inner_list in data:
            line = ",".join(map(str, inner_list)) + "\n"
            f.write(line)

def main():
    parent_folder = 'PDBbind_v2020_other_PL/v2020-other-PL'
    folders = os.listdir(parent_folder)
    #labels = []
    #chains = []
    data = []
    #all_contacts = []
    # Create an empty DataFrame with column names
    columns = ['pdb_id','chains', 'sequence', 'label']
    df = pd.DataFrame(columns=columns)
    contact_columns =['pdb_id','chains', 'contacts']
    contact_df = pd.DataFrame(columns=contact_columns)
    for folder in folders:
        if folder not in ['readme', 'index', 'convert_sdf.sh']:
            print(folder)
            files = os.listdir(parent_folder+'/'+folder)
            ligand_pdb_file = find_file_within_directory(files, 'ligand.pdb')
            protein_pdb_file = find_file_within_directory(files, 'protein.pdb')
            pocket_pdb_file = find_file_within_directory(files, 'pocket.pdb')

            proteinfilepath = parent_folder+'/'+folder+'/'+protein_pdb_file
            pockfilepath = parent_folder+'/'+folder+'/'+pocket_pdb_file
            ligfilepath = parent_folder+'/'+folder+'/'+ligand_pdb_file
            #contact_residues = find_contact_residues(proteinfilepath, ligfilepath, 7)
            #all_contacts.append(folder, contact_residues)
            parser=PDBParser(QUIET=True)
            pocket_struct= parser.get_structure('pocket',pockfilepath)
            num_chains, chains = count_chains(pocket_struct)
            print(chains)
            pocket_sequence, pocket_res_numbers, _ = extract_sequences_from_pdb(pocket_struct)
            pocket_residues = get_structure_info(pocket_struct)
            
            proteinfilepath = parent_folder+'/'+folder+'/'+protein_pdb_file
            protein_struct = parser.get_structure('protein',proteinfilepath)
            
            sequences, res_numbers, _ = extract_sequences_from_pdb(protein_struct)
            protein_residues = get_structure_info(protein_struct)
            #print("protein residues: ", protein_residues)
            #print(contact_residues)
            labels = make_labels(pocket_residues, sequences, protein_residues, chains, folder, data)
            full_sequence = []
            full_labels = []
            for chain in sorted(chains):
                full_sequence += sequences[chain]
                full_labels += labels[chain]

            contact_row = {'pdb_id': folder, 'chains': sorted(chains), 'contacts': pocket_residues}
            #print(contact_row)
            contact_df = contact_df.append(contact_row, ignore_index=True)
            row = {'pdb_id': folder, 'chains': sorted(chains), 'sequence': full_sequence, 'label': full_labels}
            df = df.append(row, ignore_index=True)
    print(df.head())
    df.to_csv('PDB_bind_all_chains_pocket.csv', index=False)
    
    contact_df.to_csv('all_contacts_PDB_pocket.txt', index=False)       
    #output_file = "all_contacts_PDB.txt"
    #write_list_of_lists_to_file(all_contacts, output_file)



if __name__ == "__main__":
    main()