import os
import requests
import pandas as pd

from tqdm import tqdm

PAGE_URL = "https://mdl.shsmu.edu.cn//ASD/BrowseSite?_dc=1749937224168&page=1&start=0&limit=5000"
ALLO_URL = "https://mdl.shsmu.edu.cn/ASD2023Common/static_file//site/allo_site_pdb_gz/"
ORTHO_URL = "https://mdl.shsmu.edu.cn/ASD2023Common/static_file//site/orth_site_pdb_gz/"
OUTDIR = "/home/lshre1/Documents/PredictingAllostericSites/Allo_Ortho_Pockets"

def make_download_url(site_name, base_url):
    return base_url + site_name + ".pdb.gz"

def download_site(url, outdir="./"):
    outname = os.path.join(outdir, url.split("/")[-1])

    response = requests.get(url, stream=True)
    if response.status_code != 200:
        print(f"request to {url} failed with code {response.status_code}")
        return

    with open(outname, "wb") as dlfile:
        for chunk in response.raw.stream(1024, decode_content=False):
            if chunk:
                dlfile.write(chunk)

if __name__ == "__main__":
   
    response = requests.get(PAGE_URL)
    response.raise_for_status()

    data = response.json()['data']
    pd.DataFrame(data).to_csv(os.path.join(OUTDIR, "allo_ortho_table.csv"), index=False)

    for site in tqdm(data, desc="downloading pdb files"):
        name = site["allosteric_site"]
        if name + ".pdb.gz" not in os.listdir(OUTDIR):
            url = make_download_url(name, ALLO_URL)
            download_site(url, OUTDIR)

        name = site["substrate_site"]
        if (name != "null") and (name + ".pdb.gz" not in os.listdir(OUTDIR)):
            url = make_download_url(name, ORTHO_URL)
            download_site(url, OUTDIR)

    