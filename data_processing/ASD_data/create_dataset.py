''' This script process the ASD data and extracts allosteric binding sites by finding all residues with 7A of the allosteric modulator'''

import os
import re
import xml.etree.ElementTree as ET
import pandas as pd
from Bio import PDB
from Bio.PDB import PDBParser, NeighborSearch
from Bio.PDB.Polypeptide import is_aa

def extract_sequences_from_pdb(structure):
    NS_AA = {'MSE', 'CME', 'PTR', 'SEP', 'SNN'}
    sequences = {}
    residue_numbers_by_chain = {}

    for model in structure:
        for chain in model:
            sequence = ''
            residue_numbers = []

            for residue in chain:
                residue_name = residue.get_resname()
                if residue_name.strip() != 'HOH':
                    if PDB.is_aa(residue.get_resname(), standard=True):
                        sequence += PDB.Polypeptide.three_to_one(residue.get_resname())
                        residue_numbers.append(residue.get_id()[1])
                    elif residue_name.strip() in NS_AA:
                        sequence += 'X'
                        residue_numbers.append(residue.get_id()[1])

            sequences[chain.get_id()] = sequence
            residue_numbers_by_chain[chain.get_id()] = residue_numbers

    return sequences, residue_numbers_by_chain

def create_boolean_list(sequence, res_numbers, residues):
    boolean_list = [0] * len(sequence)
    for site in residues:
        if int(site) in res_numbers:
            index = res_numbers.index(int(site))
            boolean_list[index] = 1
        else:
            print(f"Residue {site} not found in residue numbers {res_numbers}")
    return boolean_list

def get_chain_residues(contact_residues):
    chain_residues = {}
    for chain, res in contact_residues:
        if chain not in chain_residues:
            chain_residues[chain] = []
        chain_residues[chain].append(res)
    return chain_residues

def get_chains(contact_residues):
    chain_residues = get_chain_residues(contact_residues)
    return list(chain_residues.keys())

def find_modulator_contacts(pdb_file, modulator_chains, modulator_residues, modulator_class, distance_threshold=7.0):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(os.path.basename(pdb_file), pdb_file)
    sequences, res_numbers = extract_sequences_from_pdb(structure)
    if not sequences:
        print("sequences empty")

    all_atoms = [atom for atom in structure.get_atoms() if is_aa(atom.get_parent())]
    ns = NeighborSearch(all_atoms)

    modulator_atoms = []
    if modulator_class is not None and modulator_class.lower()=='pep':
        modulator_residues=[]
        for chain in structure.get_chains():
            if chain.id in modulator_chains:
                for res in chain.get_residues():
                    modulator_atoms.extend(list(res.get_atoms()))
                    residue_name = res.get_resname()
                    if residue_name.strip() != 'HOH':
                        if PDB.is_aa(res.get_resname(), standard=True):
                            modulator_residues.append(res.get_id()[1])
                      
    else:
        for chain in structure.get_chains():
            if chain.id in modulator_chains:
                for res in chain.get_residues():
                    if res.id[1] in modulator_residues:
                        modulator_atoms.extend(list(res.get_atoms()))
    if not modulator_atoms:
        print("modulator not found", modulator_chains, modulator_residues)
    #print(modulator_class, modulator_residues)
    mod_res = []
    for res in modulator_residues:
        mod_res.append([modulator_chains[0], res])
    #print(mod_res)

    contact_residues = set()
    for atom in modulator_atoms:
        neighbors = ns.search(atom.coord, distance_threshold)
        for neighbor in neighbors:
            parent_residue = neighbor.get_parent()
            if is_aa(parent_residue) and ([parent_residue.get_parent().id, parent_residue.id[1]]) not in mod_res:
               # print((parent_residue.get_parent().id, parent_residue.id[1]), mod_res)
                contact_residues.add((parent_residue.get_parent().id, parent_residue.id[1]))
    
    if not contact_residues:
        print("contact residues empty")

    chain_residues = get_chain_residues(contact_residues)
    labels = {}
    full_sequence = []
    full_labels = []

    for chain, residues in chain_residues.items():
        labels[chain] = create_boolean_list(sequences[chain], res_numbers[chain], residues)
    chains = get_chains(contact_residues)
    if not chains:
        print("chains empty")
    for chain in sorted(chains):
        full_sequence += sequences[chain]
        full_labels += labels[chain]

    return list(contact_residues), ''.join(full_sequence), full_labels

def extract_allosteric_site_data(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()

    extracted_data = {
        'Modulator_Alias': [],
        'Modulator_Chain': [],
        'Modulator_Residue': [],
        'Allosteric_Site_Residue': [],
        'Allosteric_PDB': [],
        'Modulator_Class': [],
        'filename': []
    }

    for allosteric_site in root.findall('.//Allosteric_Site_List/Allosteric_Site'):
        modulator_alias = allosteric_site.find('Modulator_Alias').text if allosteric_site.find('Modulator_Alias') is not None else None
        modulator_chain = allosteric_site.find('Modulator_Chain').text if allosteric_site.find('Modulator_Chain') is not None else None
        modulator_residue = allosteric_site.find('Modulator_Residue').text if allosteric_site.find('Modulator_Residue') is not None else None
        allosteric_site_residue = allosteric_site.find('Allosteric_Site_Residue').text if allosteric_site.find('Allosteric_Site_Residue') is not None else None
        allosteric_pdb = allosteric_site.find('Allosteric_PDB').text if allosteric_site.find('Allosteric_PDB') is not None else None
        modulator_class = allosteric_site.find('Modulator_Class').text if allosteric_site.find('Modulator_Class') is not None else None

        extracted_data['Modulator_Alias'].append(modulator_alias)
        extracted_data['Modulator_Chain'].append(modulator_chain)
        extracted_data['Modulator_Residue'].append(modulator_residue)
        extracted_data['Allosteric_Site_Residue'].append(allosteric_site_residue)
        extracted_data['Allosteric_PDB'].append(allosteric_pdb)
        extracted_data['Modulator_Class'].append(modulator_class)
        extracted_data['filename'].append(os.path.basename(file_path))

    return extracted_data

def process_xml_files_in_directory(directory_path):
    all_data_list = []

    for filename in os.listdir(directory_path):
        if filename.endswith('.xml') and filename != 'ASD06390000_1.xml':
            file_path = os.path.join(directory_path, filename)
            file_data = extract_allosteric_site_data(file_path)

            num_entries = len(file_data['Modulator_Alias'])
            for i in range(num_entries):
                row_data = {key: file_data[key][i] for key in file_data}
                all_data_list.append(row_data)

    df = pd.DataFrame(all_data_list)
    return df

def download_pdb(pdb_id, output_dir):
    pdbl = PDB.PDBList()
    pdb_file = pdbl.retrieve_pdb_file(pdb_id, pdir=output_dir, file_format="pdb")
    return pdb_file

def parse_modulator_residues(residue_string):
    residue_list = []
    chains_list = []

    if residue_string is not None:
        for part in re.split('[,;]', residue_string):
            if '/' in part: 
                chain_id, residue_id = part.split('/')
                residue_list.append(int(residue_id))
                chains_list.append(chain_id)
            elif '-' in part: 
                start, end = map(int, part.split('-'))
                for residue_id in range(start, end + 1):
                    residue_list.append(residue_id)
                    chains_list.append(None)
            else:
                match = re.match(r"(\d+)([A-Za-z]*)", part)
                if match:
                    residue = int(match.group(1))
                    chain = match.group(2) if match.group(2) else None
                    residue_list.append(residue)
                    chains_list.append(chain)

    return residue_list, chains_list

def split_string(string):
    delimiter_pattern = r'[,;\/]'  
    split_list = re.split(delimiter_pattern, string)
    return split_list

def find_contacts(structure, modulator_chain, modulator_residue, thr=7.0):
    atom_list = [atom for atom in structure.get_atoms() if is_aa(atom.get_parent())]
    ns = NeighborSearch(atom_list)

    modulator_atoms = []
    for chain in structure.get_chains():
        if chain.id == modulator_chain:
            for res in chain.get_residues():
                if res.id[1] == modulator_residue:
                    modulator_atoms.extend(list(res.get_atoms()))
    contacts = ns.search_all(thr, level='A')  
    contact_list = []
    for contact in contacts:
        if contact[0] in modulator_atoms or contact[1] in modulator_atoms:
            contact_list.append((contact[0].get_parent().id, contact[1].get_parent().id))

    return contact_list

def generate_contacts_dataframe(dataframe, output_dir, threshold):
    contact_data = {
        'Filename': [],
        'Allosteric_PDB': [],
        'Modulator_Chain': [],
        'Modulator_Residue': [],
        'Contact_Residues': [],
        'Modulator_Class': [],
        'Sequences': [],
        'Labels': []
    }

    for index, row in dataframe.iterrows():
        filename = row['filename']
        pdb_id = row['Allosteric_PDB']
        modulator_chain = row['Modulator_Chain']
        modulator_residue_string = row['Modulator_Residue']
        modulator_class = row['Modulator_Class']
        print(filename, pdb_id)
        if modulator_class is not None:
            if modulator_class.lower()!='pep':
                modulator_residues, chains_list = parse_modulator_residues(modulator_residue_string)
                print(modulator_residues)
            else:
                modulator_residues = [0]
        
        # Check if modulator_chain is a string
        if isinstance(modulator_chain, str):
            modulator_chains = split_string(modulator_chain)
        else:
            modulator_chains = [modulator_chain]

        try:
            pdb_file = download_pdb(pdb_id, output_dir)

            for chain, residues in zip(modulator_chains, modulator_residues):
                contacts, sequences, labels = find_modulator_contacts(pdb_file, [chain], [residues], modulator_class, threshold)
                contact_data['Filename'].append(filename)
                contact_data['Allosteric_PDB'].append(pdb_id)
                contact_data['Modulator_Chain'].append(chain)
                contact_data['Modulator_Residue'].append(residues)
                contact_data['Contact_Residues'].append(contacts)
                contact_data['Modulator_Class'].append(modulator_class)
                contact_data['Sequences'].append(sequences)
                contact_data['Labels'].append(labels)

        except Exception as e:
            print(f"Error processing PDB ID {pdb_id}: {e}")

    contact_df = pd.DataFrame(contact_data)
    return contact_df


# Directory paths
xml_directory = "~/PLM_Allosteric_Classification/data/ASD_Release_202306_XF"
pdb_directory = '~/PLM_Allosteric_Classification/data/ASD_Release_202306_XF_Allosteric_PDB'

# Process XML files and generate DataFrame
allosteric_site_df = process_xml_files_in_directory(xml_directory)

# Generate contacts DataFrame
threshold=7.0
contacts_df = generate_contacts_dataframe(allosteric_site_df, pdb_directory, threshold)

# Print DataFrame to verify
print(contacts_df.head())


allosteric_site_df.to_csv('ASD_Release_202306_XF_dataframe.csv', index=False)
contacts_df.to_csv('ASD_Release_202306_XF_Allosteric_Contacts.csv')
