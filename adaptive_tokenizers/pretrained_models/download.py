import os
from tqdm import tqdm
import requests

pretrained_checkpoints = {
    'imagenet100': {
        "alit_small_vqgan_quantized_latents.pth": "https://www.dropbox.com/scl/fi/iszluhiop09z3afo2gw5f/alit_small_vqgan_quantized_latents.pth?rlkey=klt0zgeunb60l1snuzhjdzdq2&st=zgwzgc2r&dl=0",
        "alit_small_vqgan_continuous_latents.pth": "https://www.dropbox.com/scl/fi/twbxjch2hutxjsy3sd85y/alit_small_vqgan_continuous_latents.pth?rlkey=68ycbjp6upkrxt22w5rm58cjv&st=5nyk6uvu&dl=0",
        "alit_small_vae_quantized_latents.pth": "https://www.dropbox.com/scl/fi/svykasoyyoapfzlghjuz2/alit_small_vae_quantized_latents.pth?rlkey=dtvyj74zq593zc7c63miwkhs3&st=dwm9w2sp&dl=0",
        "alit_small_vae_continuous_latents.pth": "https://www.dropbox.com/scl/fi/bq2yzr7dufvrpnycb856b/alit_small_vae_continuous_latents.pth?rlkey=hjx6jpr6vvfzhtgkp1cc1rnrs&st=4aa1vyh9&dl=0",
        "alit_base_vqgan_quantized_latents.pth": "https://www.dropbox.com/scl/fi/6cygifz37knpqtkxgfj81/alit_base_vqgan_quantized_latents.pth?rlkey=r8hn0d4d8j8eg2wjorzl9c67s&st=c5iv7wor&dl=0",
        "alit_semilarge_vqgan_quantized_latents.pth": "https://www.dropbox.com/scl/fi/wcp7s6w86slh1yy4m3egc/alit_semilarge_vqgan_quantized_latents.pth?rlkey=cf2s13c6ah3ru4ly9tb911gi7&st=d11uoq0i&dl=0"
    },
    'imagenet':{
        "alit_small_vqgan_quantized_latents.pth": "https://www.dropbox.com/scl/fi/u8jyro5wysttp5mvs6gir/alit_vqgan_small_quantized_latent.pth?rlkey=1py1s9755gjhdyw5ifd9h2wff&st=swdbevjk&dl=0",
    }
    
}

def download_all(overwrite=False):
    base_download_path = "adaptive_tokenizers/pretrained_models/"
    for dataset in pretrained_checkpoints.keys():
        if not os.path.exists(os.path.join(base_download_path, dataset)):
            os.system('mkdir -p ' + os.path.join(base_download_path, dataset))
        for ckpt in pretrained_checkpoints[dataset].keys():
            download_path = os.path.join(base_download_path, dataset, ckpt)
            if not os.path.exists(download_path) or overwrite:
                headers = {'user-agent': 'Wget/1.16 (linux-gnu)'}
                r = requests.get(pretrained_checkpoints[dataset][ckpt], stream=True, headers=headers)
                print("Downloading {} | {} ...".format(dataset, ckpt))
                with open(download_path, 'wb') as f:
                    for chunk in tqdm(r.iter_content(chunk_size=1024*1024), unit="MB", total=254):
                        if chunk:
                            f.write(chunk)


if __name__ == "__main__":
    download_all()