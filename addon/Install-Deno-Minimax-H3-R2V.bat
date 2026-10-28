@echo off
setlocal enabledelayedexpansion
title Deno MiniMax H3 R2V - ComfyUI Workflow Installer
chcp 65001 >nul

rem ==========================================================================
rem  Deno MiniMax H3 R2V (7+1 Memory-Optimized Sol) Installer for ComfyUI
rem  Installs workflow, custom nodes, python dependencies, and model weights.
rem
rem  Workflow:
rem    - Deno_minimax_final_R2V_7+1_memopt_Sol.json
rem
rem  Nodes:
rem    - Deno2026/comfyui-deno-custom-nodes (Resize Box, Ref Loader, Text Unload)
rem    - Zironic/H3-Optimizations (H3 Memory Optimization, QKV stream)
rem    - LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler (3D Latent Upscaler)
rem    - Kosinkadink/ComfyUI-VideoHelperSuite (Video Combine, Prune Outputs)
rem    - kijai/comfyui-kjnodes (VRAM Debug)
rem    - Jonathan97480/ComfyUI-MiniMax-H3-Motion-Director (MM Motion Director)
rem
rem  Models:
rem    - Singularity Ref2VA INT8 (WarmBloodAban/Minimax-h3_Singularity ~21 GB)
rem    - Qwen3-VL 32B INT8 ConvRot (Comfy-Org/MiniMax-H3 ~27 GB)
rem    - DMD Ref2VA 8-Step Turbo LoRA (drbaph/MiniMax-H3-Turbo-Lora-ComfyUI ~1.4 GB)
rem    - Minimax H3 3D Latent Upscaler (LBH-123-AI/Minimax_h3_latent_Upscaler ~690 MB)
rem    - MiniMax H3 Video VAE FP16 (Comfy-Org/MiniMax-H3 ~168 MB)
rem    - MiniMax H3 Audio VAE FP32 (Comfy-Org/MiniMax-H3 ~84 MB)
rem ==========================================================================

set "DENO_REPO=https://github.com/Deno2026/comfyui-deno-custom-nodes"
set "H3OPT_REPO=https://github.com/Zironic/H3-Optimizations"
set "LATENTUP_REPO=https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler"
set "VHS_REPO=https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite"
set "KJ_REPO=https://github.com/kijai/comfyui-kjnodes"
set "DIRECTOR_REPO=https://github.com/Jonathan97480/ComfyUI-MiniMax-H3-Motion-Director"

rem --- Direct HuggingFace Download URLs ---
set "URL_DIFF_SINGULARITY=https://huggingface.co/WarmBloodAban/Minimax-h3_Singularity/resolve/main/Minimax-h3_Singularity_ref2va_Pruned_v1.3_int8.safetensors"
set "URL_DIFF_REF2VA=https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors"
set "URL_TEXT_ENC_INT8=https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors"
set "URL_LORA_TURBO_R2V=https://huggingface.co/drbaph/MiniMax-H3-Turbo-Lora-ComfyUI/resolve/main/experimental/minimax_h3_dmd_ref2va_8step_turbo_pruned.safetensors"
set "URL_LATENT_UP=https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler/resolve/main/minimax_h3_latent_upscaler_3d_bf16.safetensors"
set "URL_VAE_VID=https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_video_vae_fp16.safetensors"
set "URL_VAE_AUD=https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_audio_vae_fp32.safetensors"

echo.
echo  ========================================================================
echo    Deno MiniMax H3 R2V (7+1 MemOpt Sol) - Complete Workflow Installer
echo  ========================================================================
echo.

rem ---------------------------------------------------------------- 1. Find ComfyUI and Python
set "COMFY="
set "PY="

rem --- Check Portable layouts ---
if exist "%~dp0..\python_embeded\python.exe" if exist "%~dp0..\ComfyUI\main.py" (
    set "COMFY=%~dp0..\ComfyUI"
    set "PY=%~dp0..\python_embeded\python.exe"
)
if not defined COMFY if exist "%~dp0python_embeded\python.exe" if exist "%~dp0ComfyUI\main.py" (
    set "COMFY=%~dp0ComfyUI"
    set "PY=%~dp0python_embeded\python.exe"
)
if not defined COMFY if exist "%~dp0..\..\python_embeded\python.exe" if exist "%~dp0..\..\ComfyUI\main.py" (
    set "COMFY=%~dp0..\..\ComfyUI"
    set "PY=%~dp0..\..\python_embeded\python.exe"
)
if not defined COMFY if exist "%~dp0..\..\..\python_embeded\python.exe" if exist "%~dp0..\..\..\ComfyUI\main.py" (
    set "COMFY=%~dp0..\..\..\ComfyUI"
    set "PY=%~dp0..\..\..\python_embeded\python.exe"
)
if not defined COMFY if exist "%~dp0..\python_embeded\python.exe" if exist "%~dp0main.py" (
    set "COMFY=%~dp0."
    set "PY=%~dp0..\python_embeded\python.exe"
)

rem --- Check Standard layout (git clone / non-portable) ---
if not defined COMFY (
    for %%D in ("%~dp0." "%~dp0.." "%~dp0..\.." "%~dp0..\..\.." "%~dp0ComfyUI" "%~dp0..\ComfyUI") do (
        if not defined COMFY if exist "%%~fD\main.py" if exist "%%~fD\comfy" (
            set "COMFY=%%~fD"
        )
    )
)

rem --- Prompt user if not detected ---
if not defined COMFY (
    echo.
    echo  [PROMPT] Le dossier ComfyUI n'a pas pu etre detecte automatiquement.
    set /p "USER_INPUT_PATH= Entrez le chemin complet de votre dossier ComfyUI: "
    if defined USER_INPUT_PATH (
        set "USER_INPUT_PATH=!USER_INPUT_PATH:"=!"
        if exist "!USER_INPUT_PATH!\main.py" set "COMFY=!USER_INPUT_PATH!"
        if exist "!USER_INPUT_PATH!\ComfyUI\main.py" set "COMFY=!USER_INPUT_PATH!\ComfyUI"
    )
)

if not defined COMFY (
    echo  [ERROR] Impossible de localiser ComfyUI - main.py introuvable.
    echo          Veuillez placer ce script dans le dossier Add-Ons de ComfyUI.
    goto :fail
)

for %%I in ("%COMFY%") do set "COMFY=%%~fI"
set "ROOT=%COMFY%\.."
for %%I in ("%ROOT%") do set "ROOT=%%~fI"

rem --- Find Python ---
if not defined PY (
    if exist "%COMFY%\.venv\Scripts\python.exe" (
        set "PY=%COMFY%\.venv\Scripts\python.exe"
    ) else if exist "%COMFY%\venv\Scripts\python.exe" (
        set "PY=%COMFY%\venv\Scripts\python.exe"
    ) else if exist "%ROOT%\.venv\Scripts\python.exe" (
        set "PY=%ROOT%\.venv\Scripts\python.exe"
    ) else if exist "%ROOT%\venv\Scripts\python.exe" (
        set "PY=%ROOT%\venv\Scripts\python.exe"
    ) else if exist "%ROOT%\python_embeded\python.exe" (
        set "PY=%ROOT%\python_embeded\python.exe"
    ) else (
        where python >nul 2>&1
        if not errorlevel 1 (
            set "PY=python"
        ) else (
            echo  [ERROR] Environnement Python introuvable.
            goto :fail
        )
    )
)

set "CN=%COMFY%\custom_nodes"
set "MODELS=%COMFY%\models"
set "WARN=%TEMP%\deno_minimax_install_warnings.txt"
if exist "%WARN%" del "%WARN%"

echo  [OK] ComfyUI detecte : %COMFY%
echo  [OK] Python detecte  : %PY%

rem --- Check if ComfyUI is running ---
set "PORT="
if exist "%TEMP%\deno_port.txt" del "%TEMP%\deno_port.txt"
if exist "%ROOT%\Start ComfyUI.bat" "%PY%" -I -c "import sys,re;t=open(sys.argv[1],errors='ignore').read();m=re.findall(r'^(?!\s*(?:rem|::)).*?--port[= ]+(\d+)',t,re.M|re.I);print(m[-1] if m else '')" "%ROOT%\Start ComfyUI.bat" > "%TEMP%\deno_port.txt" 2>nul
if exist "%TEMP%\deno_port.txt" set /p PORT=<"%TEMP%\deno_port.txt"
if not defined PORT set "PORT=8188"

if /i not "%~1"=="NoPause" (
    netstat -ano | findstr /r /c:":%PORT% .*LISTENING" >nul 2>&1
    if not errorlevel 1 (
        echo.
        echo  [AVERTISSEMENT] ComfyUI est actuellement en cours d'execution sur le port %PORT%.
        echo                Veuillez fermer ComfyUI pour eviter les erreurs de fichiers verrouilles.
        choice /c YN /n /m "                Continuer quand meme ? [Y=Oui / N=Non] "
        if errorlevel 2 goto :fail
    )
)

rem ---------------------------------------------------------------- 2. Check Custom Nodes
echo.
echo  [1/5] Installation et mise a jour des Custom Nodes requis...

call :install_git_node "%DENO_REPO%"     "%CN%\comfyui-deno-custom-nodes"          "comfyui-deno-custom-nodes-main"
call :install_git_node "%H3OPT_REPO%"    "%CN%\H3-Optimizations"                   "H3-Optimizations-main"
call :install_git_node "%LATENTUP_REPO%" "%CN%\Comfyui_Minimax_h3_latent_Upscaler" "Comfyui_Minimax_h3_latent_Upscaler-main"
call :install_git_node "%VHS_REPO%"      "%CN%\ComfyUI-VideoHelperSuite"           "ComfyUI-VideoHelperSuite-main"
call :install_git_node "%KJ_REPO%"       "%CN%\comfyui-kjnodes"                    "comfyui-kjnodes-main"
call :install_git_node "%DIRECTOR_REPO%" "%CN%\ComfyUI-MiniMax-H3-Motion-Director" "ComfyUI-MiniMax-H3-Motion-Director-main"

rem ---------------------------------------------------------------- 3. Python Requirements
echo.
echo  [2/5] Verification des dependances Python...

for %%D in (comfyui-deno-custom-nodes H3-Optimizations Comfyui_Minimax_h3_latent_Upscaler ComfyUI-VideoHelperSuite comfyui-kjnodes ComfyUI-MiniMax-H3-Motion-Director) do (
    if exist "%CN%\%%D\requirements.txt" (
        echo        Verification des dependances pour %%D...
        "%PY%" -I -m pip install --quiet --disable-pip-version-check --no-warn-script-location -r "%CN%\%%D\requirements.txt" >nul 2>&1
    )
)

rem Assurer la presence de nvidia-ml-py pour le monitoring VRAM Deno
"%PY%" -c "import pynvml" >nul 2>&1
if errorlevel 1 (
    echo        Installation de nvidia-ml-py pour Deno Resource Monitor...
    "%PY%" -I -m pip install --quiet --disable-pip-version-check --no-warn-script-location "nvidia-ml-py>=12" >nul 2>&1
)

echo        OK - Dependances Python pretes.

rem ---------------------------------------------------------------- 3.5 Apply VRAM Optimizations & Patches
echo.
echo  [2.5/5] Application des correctifs de noeuds et optimisations VRAM [12GB RTX 3060 ready]...

set "OPT_PY="
if exist "%~dp0apply_vram_optimizations.py" set "OPT_PY=%~dp0apply_vram_optimizations.py"
if not defined OPT_PY if exist "%CN%\ComfyUI-MiniMax-H3-Motion-Director\addon\apply_vram_optimizations.py" set "OPT_PY=%CN%\ComfyUI-MiniMax-H3-Motion-Director\addon\apply_vram_optimizations.py"

if defined OPT_PY (
    "%PY%" "%OPT_PY%"
) else (
    echo        Telechargement de apply_vram_optimizations.py depuis GitHub...
    set "OPT_URL=https://raw.githubusercontent.com/Jonathan97480/ComfyUI-MiniMax-H3-Motion-Director/main/addon/apply_vram_optimizations.py"
    call :download "!OPT_URL!" "%TEMP%\apply_vram_optimizations.py"
    if exist "%TEMP%\apply_vram_optimizations.py" (
        "%PY%" "%TEMP%\apply_vram_optimizations.py"
    )
)

rem ---------------------------------------------------------------- 4. Scan & Download Models
echo.
echo  [3/5] Analyse des modeles H3 et Upscaler requis...

if exist "%TEMP%\deno_models.txt" del "%TEMP%\deno_models.txt"
"%PY%" -c "import sys;exec(open(sys.argv[1],encoding='utf-8').read().split('#PYMODELS#\n',1)[1])" "%~f0" "%COMFY%" "%TEMP%\deno_models.txt"

if not exist "%TEMP%\deno_models.txt" (
    echo  [ERROR] Impossible d'analyser les dossiers de modeles.
    goto :fail
)

for /f "usebackq tokens=1,2,* delims=|" %%a in ("%TEMP%\deno_models.txt") do (
    set "S_%%a=%%b"
    set "T_%%a=%%c"
)

call :link_lora_and_subfolders

if not defined S_NEED (
    echo.
    echo        Tous les modeles necessaires au flux sont deja presents sur le disque.
    goto :deploy_workflow
)

echo.
if /i "%~1"=="NoPause" (
    echo        [NoPause] Telechargement automatique des modeles manquants desactive.
    goto :deploy_workflow
)

echo  Certains modeles requis pour ce flux de travail sont manquants.
echo  - Minimax-h3_Singularity_ref2va_Pruned_v1.3_int8.safetensors [~21 GB]
echo  - qwen3vl_32b_minimax_h3_int8_convrot.safetensors          [~27 GB]
echo  - minimax_h3_dmd_ref2va_8step_turbo_pruned.safetensors      [~1.4 GB]
echo  - minimax_h3_latent_upscaler_3d_bf16.safetensors            [~690 MB]
echo.
choice /c YN /n /m "  Telecharger les modeles manquants maintenant ? [Y=Oui / N=Non] "
if errorlevel 2 (
    echo        Telechargement ignore. Vous pouvez relancer ce script plus tard pour les recuperer.
    >>"%WARN%" echo  - Modeles non telecharges. Relancez l'installateur pour les telecharger.
    goto :deploy_workflow
)

echo.
if "%S_SINGULARITY%"=="MISSING" (
    echo        [1/6] Telechargement de Singularity Ref2VA INT8 [~21 GB]...
    call :download "%URL_DIFF_SINGULARITY%" "%T_SINGULARITY%"
    if errorlevel 1 >>"%WARN%" echo  - Echec du telechargement Singularity Ref2VA.
)

if "%S_TEXTENC%"=="MISSING" (
    echo        [2/6] Telechargement de Qwen3-VL 32B INT8 ConvRot Text Encoder [~27 GB]...
    call :download "%URL_TEXT_ENC_INT8%" "%T_TEXTENC%"
    if errorlevel 1 >>"%WARN%" echo  - Echec du telechargement Text Encoder INT8 ConvRot.
)

if "%S_LORA_TURBO%"=="MISSING" (
    echo        [3/6] Telechargement de DMD Ref2VA 8-Step Turbo LoRA [~1.4 GB]...
    call :download "%URL_LORA_TURBO_R2V%" "%T_LORA_TURBO%"
    if errorlevel 1 >>"%WARN%" echo  - Echec du telechargement Turbo LoRA Ref2VA.
)

if "%S_LATENTUP%"=="MISSING" (
    echo        [4/6] Telechargement de Minimax H3 Latent Upscaler 3D BF16 [~690 MB]...
    call :download "%URL_LATENT_UP%" "%T_LATENTUP%"
    if errorlevel 1 >>"%WARN%" echo  - Echec du telechargement Latent Upscaler 3D.
)

if "%S_VAEVID%"=="MISSING" (
    echo        [5/6] Telechargement de MiniMax H3 Video VAE FP16 [~168 MB]...
    call :download "%URL_VAE_VID%" "%T_VAEVID%"
    if errorlevel 1 >>"%WARN%" echo  - Echec du telechargement Video VAE.
)

if "%S_VAEAUD%"=="MISSING" (
    echo        [6/6] Telechargement de MiniMax H3 Audio VAE FP32 [~84 MB]...
    call :download "%URL_VAE_AUD%" "%T_VAEAUD%"
    if errorlevel 1 >>"%WARN%" echo  - Echec du telechargement Audio VAE.
)

call :link_lora_and_subfolders

rem ---------------------------------------------------------------- 5. Deploy Workflow
:deploy_workflow
echo.
echo  [4/5] Deploiement du flux de travail Deno MiniMax H3...

set "WF_NAME=Deno_minimax_final_R2V_7+1_memopt_Sol.json"
set "USER_WF_DIR=%COMFY%\user\default\workflows\MiniMax-H3"
set "ROOT_WF_DIR=%COMFY%\user\default\workflows"
set "DENO_WF_DIR=%CN%\comfyui-deno-custom-nodes\workflows"
set "DIR_WF_DIR=%CN%\ComfyUI-MiniMax-H3-Motion-Director\example_workflows"

if not exist "%USER_WF_DIR%" mkdir "%USER_WF_DIR%"
if not exist "%DENO_WF_DIR%" mkdir "%DENO_WF_DIR%"
if not exist "%DIR_WF_DIR%" mkdir "%DIR_WF_DIR%"

set "SRC_WF="
if exist "d:\download\%WF_NAME%" set "SRC_WF=d:\download\%WF_NAME%"
if not defined SRC_WF if exist "%~dp0%WF_NAME%" set "SRC_WF=%~dp0%WF_NAME%"
if not defined SRC_WF if exist "%DIR_WF_DIR%\%WF_NAME%" set "SRC_WF=%DIR_WF_DIR%\%WF_NAME%"
if not defined SRC_WF if exist "%USER_WF_DIR%\%WF_NAME%" set "SRC_WF=%USER_WF_DIR%\%WF_NAME%"

if defined SRC_WF (
    copy /y "%SRC_WF%" "%USER_WF_DIR%\%WF_NAME%" >nul 2>&1
    copy /y "%SRC_WF%" "%ROOT_WF_DIR%\%WF_NAME%" >nul 2>&1
    copy /y "%SRC_WF%" "%DENO_WF_DIR%\%WF_NAME%" >nul 2>&1
    copy /y "%SRC_WF%" "%DIR_WF_DIR%\%WF_NAME%" >nul 2>&1
    echo        Flux de travail installe avec succes dans :
    echo        - %USER_WF_DIR%\%WF_NAME%
    echo        - %ROOT_WF_DIR%\%WF_NAME%
    echo        - %DENO_WF_DIR%\%WF_NAME%
) else (
    echo        [AVERTISSEMENT] Fichier source %WF_NAME% non trouve pour la copie.
    >>"%WARN%" echo  - Flux %WF_NAME% non trouve dans d:\download ou a cote de l'installateur.
)

rem ---------------------------------------------------------------- Summary
echo.
echo  [5/5] Finalisation de l'installation...
echo.
echo  ========================================================================
if exist "%WARN%" (
    echo   Installation terminee avec quelques avertissements :
    type "%WARN%"
) else (
    echo   Installation terminee avec succes.
    echo   Tous les noeuds, dependances et modeles sont prets a l'emploi.
)
echo  ========================================================================
echo.
echo   Etapes suivantes :
echo   1. Lancez ComfyUI via "Start ComfyUI.bat" (ou "Start ComfyUI FlashAttention.bat").
echo   2. Ouvrez le panneau lateral des Workflows dans ComfyUI.
echo   3. Chargez : "MiniMax-H3 / Deno_minimax_final_R2V_7+1_memopt_Sol.json".
echo.
if /i not "%~1"=="NoPause" pause
exit /b 0

:fail
echo.
echo  Installation interrompue.
echo.
if /i not "%~1"=="NoPause" pause
exit /b 1

rem ==========================================================================
rem  Subroutines
rem ==========================================================================

:install_git_node
rem %1 = repo url, %2 = target directory, %3 = zip folder name
if exist "%~2\.git" (
    echo        Mise a jour de %~nx2...
    git -C "%~2" remote set-url origin "%~1" >nul 2>&1
    git -C "%~2" pull --quiet
) else if exist "%~2" (
    echo        %~nx2 deja present.
) else (
    echo        Installation de %~nx2...
    where git >nul 2>&1
    if not errorlevel 1 (
        git clone --quiet "%~1" "%~2"
    ) else (
        call :download_zip "%~1/archive/refs/heads/main.zip" "%~2" "%~3"
    )
)
exit /b 0

:link_lora_and_subfolders
rem Le workflow demande explicitement H3_R2V_AB\minimax_h3_dmd_ref2va_8step_turbo_pruned.safetensors
set "LORA_DIR=%MODELS%\loras"
if exist "%LORA_DIR%" (
    if not exist "%LORA_DIR%\H3_R2V_AB" mkdir "%LORA_DIR%\H3_R2V_AB"
    set "LORA_F=minimax_h3_dmd_ref2va_8step_turbo_pruned.safetensors"
    if exist "%LORA_DIR%\!LORA_F!" if not exist "%LORA_DIR%\H3_R2V_AB\!LORA_F!" (
        mklink /h "%LORA_DIR%\H3_R2V_AB\!LORA_F!" "%LORA_DIR%\!LORA_F!" >nul 2>&1 || copy /y "%LORA_DIR%\!LORA_F!" "%LORA_DIR%\H3_R2V_AB\" >nul 2>&1
    )
    if exist "%LORA_DIR%\H3_R2V_AB\!LORA_F!" if not exist "%LORA_DIR%\!LORA_F!" (
        mklink /h "%LORA_DIR%\!LORA_F!" "%LORA_DIR%\H3_R2V_AB\!LORA_F!" >nul 2>&1 || copy /y "%LORA_DIR%\H3_R2V_AB\!LORA_F!" "%LORA_DIR%\" >nul 2>&1
    )
)

rem Assurer l'acces aux modeles de diffusion sous unet et diffusion_models
if exist "%MODELS%\diffusion_models" if exist "%MODELS%\unet" (
    for %%F in (Minimax-h3_Singularity_ref2va_Pruned_v1.3_int8.safetensors minimax_h3_ref2va_pruned_int8_convrot.safetensors) do (
        if exist "%MODELS%\diffusion_models\%%F" if not exist "%MODELS%\unet\%%F" (
            mklink /h "%MODELS%\unet\%%F" "%MODELS%\diffusion_models\%%F" >nul 2>&1 || copy /y "%MODELS%\diffusion_models\%%F" "%MODELS%\unet\" >nul 2>&1
        )
        if exist "%MODELS%\unet\%%F" if not exist "%MODELS%\diffusion_models\%%F" (
            mklink /h "%MODELS%\diffusion_models\%%F" "%MODELS%\unet\%%F" >nul 2>&1 || copy /y "%MODELS%\diffusion_models\%%F" "%MODELS%\diffusion_models\" >nul 2>&1
        )
    )
)

rem Assurer l'acces aux modeles de texte sous text_encoders et clip
if exist "%MODELS%\text_encoders" if exist "%MODELS%\clip" (
    for %%F in (qwen3vl_32b_minimax_h3_int8_convrot.safetensors) do (
        if exist "%MODELS%\text_encoders\%%F" if not exist "%MODELS%\clip\%%F" (
            mklink /h "%MODELS%\clip\%%F" "%MODELS%\text_encoders\%%F" >nul 2>&1 || copy /y "%MODELS%\clip\%%F" "%MODELS%\clip\" >nul 2>&1
        )
        if exist "%MODELS%\clip\%%F" if not exist "%MODELS%\text_encoders\%%F" (
            mklink /h "%MODELS%\text_encoders\%%F" "%MODELS%\clip\%%F" >nul 2>&1 || copy /y "%MODELS%\clip\%%F" "%MODELS%\text_encoders\" >nul 2>&1
        )
    )
)

rem Assurer l'acces aux modeles d'upscale sous latent_upscale_models, latent_upscaler et upscale_models
if not exist "%MODELS%\latent_upscale_models" mkdir "%MODELS%\latent_upscale_models"
if not exist "%MODELS%\latent_upscaler" mkdir "%MODELS%\latent_upscaler"
for %%F in (minimax_h3_latent_upscaler_3d_bf16.safetensors) do (
    if exist "%MODELS%\latent_upscale_models\%%F" if not exist "%MODELS%\latent_upscaler\%%F" (
        mklink /h "%MODELS%\latent_upscaler\%%F" "%MODELS%\latent_upscale_models\%%F" >nul 2>&1 || copy /y "%MODELS%\latent_upscale_models\%%F" "%MODELS%\latent_upscaler\" >nul 2>&1
    )
    if exist "%MODELS%\latent_upscaler\%%F" if not exist "%MODELS%\latent_upscale_models\%%F" (
        mklink /h "%MODELS%\latent_upscale_models\%%F" "%MODELS%\latent_upscaler\%%F" >nul 2>&1 || copy /y "%MODELS%\latent_upscaler\%%F" "%MODELS%\latent_upscale_models\" >nul 2>&1
    )
)
exit /b 0

:download
rem %1 = URL, %2 = Target path
if not exist "%~dp2" mkdir "%~dp2"
curl -L --fail -C - --retry 3 -o "%~2.part" "%~1"
if errorlevel 1 (
    if exist "%~2.part" (
        echo        Telechargement interrompu - reprise en cours...
        curl -L --fail -C - --retry 3 -o "%~2.part" "%~1"
    )
)
if errorlevel 1 (
    echo        [ERROR] Echec du telechargement : %~1
    exit /b 1
)
if exist "%~2.part" move /y "%~2.part" "%~2" >nul
if not exist "%~2" (
    echo        [ERROR] Fichier cible introuvable apres telechargement : %~2
    exit /b 1
)
echo        Enregistre : %~2
exit /b 0

:download_zip
rem %1 = zip url, %2 = target folder, %3 = unpacked subfolder name
set "TMP_ZIP=%TEMP%\deno_zip_%RANDOM%.zip"
set "TMP_EXT=%TEMP%\deno_ext_%RANDOM%"
curl -L --fail -s -o "%TMP_ZIP%" "%~1"
if errorlevel 1 exit /b 1
mkdir "%TMP_EXT%"
tar -xf "%TMP_ZIP%" -C "%TMP_EXT%"
if exist "%~2" rmdir /s /q "%~2"
move "%TMP_EXT%\%~3" "%~2" >nul
rmdir /s /q "%TMP_EXT%" 2>nul
del "%TMP_ZIP%" 2>nul
exit /b 0

rem ==========================================================================
rem  Python: Models Scanner (Read ComfyUI folder_paths & extra_model_paths.yaml)
rem ==========================================================================
#PYMODELS#
import os, sys, logging

comfy, out = sys.argv[2], sys.argv[3]
sys.path.insert(0, comfy)
os.chdir(comfy)
logging.disable(logging.CRITICAL)
import folder_paths

folder_paths.add_model_folder_path("latent_upscale_models", os.path.join(folder_paths.models_dir, "latent_upscale_models"))
folder_paths.add_model_folder_path("latent_upscaler", os.path.join(folder_paths.models_dir, "latent_upscaler"))

yaml_path = os.path.join(comfy, "extra_model_paths.yaml")
if os.path.isfile(yaml_path):
    try:
        from utils.extra_config import load_extra_path_config
        load_extra_path_config(yaml_path)
    except Exception:
        pass

MODELS = [
    ("SINGULARITY", "Minimax-h3_Singularity_ref2va_Pruned_v1.3_int8.safetensors", ["diffusion_models", "unet"], "Singularity Ref2VA INT8 (~21 GB)"),
    ("TEXTENC",     "qwen3vl_32b_minimax_h3_int8_convrot.safetensors",          ["text_encoders", "clip"],    "Qwen3-VL 32B INT8 ConvRot (~27 GB)"),
    ("LORA_TURBO",  "minimax_h3_dmd_ref2va_8step_turbo_pruned.safetensors",    ["loras"],                    "DMD Ref2VA 8-Step Turbo LoRA (~1.4 GB)"),
    ("LATENTUP",    "minimax_h3_latent_upscaler_3d_bf16.safetensors",           ["latent_upscale_models", "latent_upscaler", "upscale_models"], "Minimax H3 Latent Upscaler 3D (~690 MB)"),
    ("VAEVID",      "minimax_h3_video_vae_fp16.safetensors",                    ["vae"],                      "MiniMax H3 Video VAE FP16 (~168 MB)"),
    ("VAEAUD",      "minimax_h3_audio_vae_fp32.safetensors",                    ["vae"],                      "MiniMax H3 Audio VAE FP32 (~84 MB)"),
]

def find(name, kinds):
    name = name.lower()
    for kind in kinds:
        for d in folder_paths.get_folder_paths(kind):
            if not os.path.isdir(d):
                continue
            for root, _dirs, files in os.walk(d, followlinks=True):
                for f in files:
                    if f.lower() == name:
                        return os.path.join(root, f)
    return None

lines, missing = [], 0
print()
for key, name, kinds, label in MODELS:
    hit = find(name, kinds)
    if hit:
        print("       [TROUVE]   %-45s %s" % (label, hit))
        lines.append("%s|FOUND|%s" % (key, hit))
    else:
        paths = folder_paths.get_folder_paths(kinds[0])
        default_dir = paths[0] if paths else os.path.join(folder_paths.models_dir, kinds[0])
        if key == "LORA_TURBO":
            target = os.path.join(default_dir, "H3_R2V_AB", name)
        else:
            target = os.path.join(default_dir, name)

        for d in paths:
            if os.path.isfile(os.path.join(d, name + ".part")):
                target = os.path.join(d, name)
                break
            if os.path.isfile(os.path.join(d, "H3_R2V_AB", name + ".part")):
                target = os.path.join(d, "H3_R2V_AB", name)
                break

        print("       [MANQUANT] %-45s -> %s" % (label, target))
        lines.append("%s|MISSING|%s" % (key, target))
        missing += 1

if missing:
    lines.append("NEED|1|")

with open(out, "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines) + "\n")
