import gzip
import shutil
import os

folder = '~/PLM_Allosteric_Classification/data/Allo_Ortho_Pockets/'

for filename in os.listdir(folder):
    if filename.endswith('.pdb.gz'):
        gz_path = os.path.join(folder, filename)
        pdb_path = os.path.join(folder, filename[:-3])  # remove .gz

        with gzip.open(gz_path, 'rb') as f_in:
            with open(pdb_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        # Delete the original .gz file
        os.remove(gz_path)
