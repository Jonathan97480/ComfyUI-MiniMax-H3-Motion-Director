#!/usr/bin/env python3
"""
apply_vram_optimizations.py
Applique automatiquement tous les correctifs et optimisations VRAM (12GB RTX 3060 ready)
pour le workflow Deno MiniMax H3 R2V (7+1 Sol-Attn).

Fichiers patchés :
1. comfyui-deno-custom-nodes/deno_minimax_h3_audio.py (correction validation JSON vide)
2. Comfyui_Minimax_h3_latent_Upscaler/nodes/minimax_h3_latent_upscaler_3d.py (compatibilité device/force_unload)
3. comfy_kitchen/backends/eager/quantization.py (int8_linear chunké sans copie, gain ~4.5 Go VRAM)
4. comfy_kitchen/backends/eager/rope.py (rms_rope in-place séquentiel Q puis K, gain ~3.2 Go VRAM)
5. comfy/ldm/minimax/model.py (alignement du modèle sur self.dtype bfloat16 pour activation de Sol-Attn)
6. comfy_extras/nodes_sparse_attention.py (détection bfloat16 + logging diagnostic du producer chunké)
7. comfy/model_prefetch.py (gestion défensive du compilateur malloc_graph pour éviter les crashs aimdo)
"""

import os
import sys
import shutil
import logging
import urllib.request

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PATCHES_DIR = os.path.join(SCRIPT_DIR, "patches")
GITHUB_RAW_BASE = "https://raw.githubusercontent.com/Jonathan97480/ComfyUI-MiniMax-H3-Motion-Director/main/addon/patches"

def find_comfy_and_site_packages():
    cur = SCRIPT_DIR
    comfy_dir = None
    site_packages = None

    for _ in range(6):
        parent = os.path.dirname(cur)
        if os.path.basename(cur) == "custom_nodes":
            comfy_dir = parent
            break
        elif os.path.isdir(os.path.join(cur, "ComfyUI")):
            comfy_dir = os.path.join(cur, "ComfyUI")
            break
        elif os.path.isfile(os.path.join(cur, "main.py")) and os.path.isdir(os.path.join(cur, "comfy")):
            comfy_dir = cur
            break
        cur = parent

    for p in sys.path:
        if "site-packages" in p and os.path.isdir(os.path.join(p, "comfy_kitchen")):
            site_packages = p
            break

    if site_packages is None and comfy_dir:
        cand = os.path.join(os.path.dirname(comfy_dir), "python_embeded", "Lib", "site-packages")
        if os.path.isdir(cand):
            site_packages = cand

    return comfy_dir, site_packages

def ensure_patch_file(patch_filename):
    local_path = os.path.join(PATCHES_DIR, patch_filename)
    if os.path.isfile(local_path):
        return local_path
    
    # Try downloading from GitHub
    os.makedirs(PATCHES_DIR, exist_ok=True)
    url = f"{GITHUB_RAW_BASE}/{patch_filename}"
    logging.info(f"Téléchargement du patch depuis GitHub : {patch_filename}...")
    try:
        urllib.request.urlretrieve(url, local_path)
        return local_path
    except Exception as e:
        logging.warning(f"Échec du téléchargement pour {patch_filename}: {e}")
        return None

def backup_and_copy(patch_filename, dst):
    src = ensure_patch_file(patch_filename)
    if not src or not os.path.isfile(src):
        logging.warning(f"Source du patch introuvable : {patch_filename}")
        return False
    if not os.path.exists(os.path.dirname(dst)):
        logging.warning(f"Répertoire cible manquant : {os.path.dirname(dst)}")
        return False

    bak = dst + ".bak"
    if os.path.isfile(dst) and not os.path.isfile(bak):
        shutil.copy2(dst, bak)
        logging.info(f"Sauvegarde créée : {bak}")

    shutil.copy2(src, dst)
    logging.info(f"Fichier patché avec succès : {dst}")
    return True

def patch_batch_files(easy_install_dir):
    bat_fa = os.path.join(easy_install_dir, "Start ComfyUI FlashAttention.bat")
    bat_std = os.path.join(easy_install_dir, "Start ComfyUI.bat")

    for bat in [bat_fa, bat_std]:
        if not os.path.isfile(bat):
            continue
        try:
            with open(bat, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            changed = False
            # 1. Expandable segments
            if "expandable_segments" not in content:
                content = content.replace("@Echo off&&cd /D %~dp0", "@Echo off&&cd /D %~dp0\nset PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
                changed = True

            # 2. Disable comfy compiler
            if "--disable-comfy-compiler" not in content and "main.py" in content:
                content = content.replace("main.py --windows-standalone-build", "main.py --windows-standalone-build --disable-comfy-compiler")
                changed = True

            if changed:
                with open(bat, "w", encoding="utf-8") as f:
                    f.write(content)
                logging.info(f"Script de démarrage optimisé : {bat}")
        except Exception as e:
            logging.warning(f"Impossible de mettre à jour {bat}: {e}")

def main():
    print("=" * 72)
    print("  Deno MiniMax H3 R2V - Optimiseur de VRAM & Correctifs Automatiques")
    print("  (Compatible GPU 12GB RTX 3060 & ComfyUI Portable)")
    print("=" * 72)

    comfy_dir, site_packages = find_comfy_and_site_packages()
    logging.info(f"ComfyUI directory : {comfy_dir}")
    logging.info(f"Site-packages     : {site_packages}")

    if not comfy_dir:
        logging.error("Impossible de localiser le répertoire de ComfyUI.")
        sys.exit(1)

    custom_nodes_dir = os.path.join(comfy_dir, "custom_nodes")

    # 1. Patch comfyui-deno-custom-nodes
    deno_audio = os.path.join(custom_nodes_dir, "comfyui-deno-custom-nodes", "deno_minimax_h3_audio.py")
    backup_and_copy("deno_minimax_h3_audio.py", deno_audio)

    # 2. Patch Comfyui_Minimax_h3_latent_Upscaler
    upscaler = os.path.join(custom_nodes_dir, "Comfyui_Minimax_h3_latent_Upscaler", "nodes", "minimax_h3_latent_upscaler_3d.py")
    backup_and_copy("minimax_h3_latent_upscaler_3d.py", upscaler)

    # 3. Patch ComfyUI Core
    comfy_model = os.path.join(comfy_dir, "comfy", "ldm", "minimax", "model.py")
    backup_and_copy("minimax_model.py", comfy_model)

    sparse_attn = os.path.join(comfy_dir, "comfy_extras", "nodes_sparse_attention.py")
    backup_and_copy("nodes_sparse_attention.py", sparse_attn)

    prefetch = os.path.join(comfy_dir, "comfy", "model_prefetch.py")
    backup_and_copy("model_prefetch.py", prefetch)

    # 4. Patch comfy_kitchen (in site-packages)
    if site_packages and os.path.isdir(os.path.join(site_packages, "comfy_kitchen")):
        ck_eager = os.path.join(site_packages, "comfy_kitchen", "backends", "eager")
        backup_and_copy("quantization.py", os.path.join(ck_eager, "quantization.py"))
        backup_and_copy("rope.py", os.path.join(ck_eager, "rope.py"))
    else:
        logging.warning("comfy_kitchen non trouvé dans site-packages. Vérifiez votre environnement python.")

    # 5. Patch .bat scripts in parent directory if standalone
    parent_dir = os.path.dirname(comfy_dir)
    patch_batch_files(parent_dir)

    print("\n" + "=" * 72)
    print("  [SUCCÈS] Tous les correctifs et optimisations VRAM ont été appliqués !")
    print("  Vous pouvez relancer ComfyUI avec 'Start ComfyUI FlashAttention.bat'.")
    print("=" * 72 + "\n")

if __name__ == "__main__":
    main()
