import os
from tqdm import tqdm
import requests, gdown

def download_base_tokenizer_vae(overwrite=False):
    # VAE codebase is inspired from https://github.com/CompVis/latent-diffusion/blob/main/ldm/models/autoencoder.py
    # VAE checkpoint (Imagenet trained) is taken from https://github.com/LTH14/mar
    download_path = "base_tokenizers/pretrained_models/vae.ckpt"
    if not os.path.exists(download_path) or overwrite:
        headers = {'user-agent': 'Wget/1.16 (linux-gnu)'}
        r = requests.get("https://www.dropbox.com/scl/fi/hhmuvaiacrarfg28qxhwz/kl16.ckpt?rlkey=l44xipsezc8atcffdp4q7mwmh&dl=0", stream=True, headers=headers)
        print("Downloading KL-16 VAE...")
        with open(download_path, 'wb') as f:
            for chunk in tqdm(r.iter_content(chunk_size=1024*1024), unit="MB", total=254):
                if chunk:
                    f.write(chunk)


def download_base_tokenizer_vqgan(overwrite=False):
    # # VQGAN codebase is inspired from MaskGIT.
    # # VQGAN checkpoint (Imagenet trained) is taken from https://github.com/LTH14/mage
    print("Downloading VQGAN...")
    file_id = "13S_unB87n6KKuuMdyMnyExW0G1kplTbP"
    download_path = "base_tokenizers/pretrained_models/vqgan.ckpt"
    gdown.download(f"https://drive.google.com/uc?id={file_id}", download_path, quiet=False)


if __name__ == "__main__":
    download_base_tokenizer_vae()
    download_base_tokenizer_vqgan(overwrite=True)