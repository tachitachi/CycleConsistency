import os
import requests
from tqdm import tqdm
import math

def download(url, data_dir='data'):

    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)

    dest = os.path.join(data_dir, os.path.basename(url))

    if os.path.isfile(dest):
        return dest

    response = requests.get(url, stream=True)

    total_size = int(response.headers.get('content-length', 0))
    chunk_size=1024
    num_chunks = math.ceil(total_size // chunk_size)

    with open(dest, 'wb') as f:
        for chunk in tqdm(response.iter_content(chunk_size=chunk_size), total=num_chunks, unit='KB', unit_scale=True):
            if chunk:
                f.write(chunk)

    return dest