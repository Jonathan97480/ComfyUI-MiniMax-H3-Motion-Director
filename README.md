# MiniMax H3 Motion Director.  [Français](README.md) | [English](README_en.md) | [简体中文](README_zh.md)

![Version](https://img.shields.io/badge/version-v1.2.0-2ea44f)
![License](https://img.shields.io/badge/license-GPL--3.0-blue)
![ComfyUI](https://img.shields.io/badge/ComfyUI-custom%20node-6f42c1)
![Languages](https://img.shields.io/badge/languages-FR%20%7C%20EN%20%7C%20ZH-orange)

**Un seul Director. D'un simple plan MiniMax H3 à un projet vidéo multi-segments complet.**

Voici le tutoriel ou souhaitez-vous d'abord lire la présentation ?

[English](docs/USER_GUIDE.md) | [简体中文](docs/USER_GUIDE_zh.md)

Créez des plans `T2V / I2V / FL2V / R2V / V2V / RV2V` dans une interface de production unifiée, combinez les méthodes de génération segment par segment, conservez le contexte visuel et audio généré à travers les plans, relancez uniquement les segments nécessitant des retouches, gérez vos ressources réutilisables, visualisez l'avancement en direct, affinez le rendu et exportez la vidéo finale sans transformer votre flux ComfyUI en un mur infini de nœuds.

> Version actuelle : **v1.2.0** · Package Registry : **1.2.0**

![MiniMax H3 Motion Director — Mode Mixte](docs/images/hero-mixed-selective-run.png)

La capture d'écran ci-dessus illustre la chronologie native du mode **Mixte** : cinq segments utilisant des méthodes de génération différentes, des contrôles de continuité visuelle/audio par transition, et l'**Exécution sélective** activée pour recalculer uniquement les segments choisis.

---

## 🌍 Langues supportées (Multilingual Support)

Motion Director intègre un système complet de localisation multilingue pour l'ensemble de l'interface utilisateur (UI), des boîtes de dialogue, de la médiathèque et des retours de progression en direct :

| Langue | Statut | Description |
|---|---|---|
| 🇫🇷 **Français** (`fr`) | **Complet** | Interface 100% traduite en français : chronologie, contrôles, post-traitement, aperçu en direct, médiathèque et messages d'état. |
| 🇬🇧 **English** (`en`) | **Complet** | Traduction intégrale en anglais de tous les contrôles et flux. |
| 🇨🇳 **简体中文** (`zh`) | **Complet** | Version originale en chinois simplifié. |

### Comment changer de langue ?
Dans la barre d'outils supérieure de la fenêtre Director, cliquez simplement sur le bouton **🌐 Langue** pour alterner instantanément :
$$\text{Chinois (ZH)} \longrightarrow \text{Anglais (EN)} \longrightarrow \text{Français (FR)} \longrightarrow \text{Chinois (ZH)}$$
Le choix de langue est mémorisé automatiquement par votre navigateur.

---

## Fonctionnalités principales

| Domaine | Ce que Motion Director apporte |
|---|---|
| **Génération autonome** | Modes `T2V / I2V / FL2V / R2V / V2V / RV2V` |
| **Mode Mixte (Mixed Mode)** | Choisissez librement `T2V / I2V / FL2V / R2V / Vidéo Source` pour chaque segment |
| **Exécution sélective (Selective Run)** | Régénérez uniquement les segments modifiés sans recalculer toute la séquence |
| **Continuité inter-segments** | Contexte de mouvement (Motion Context), verrouillage d'échelle latente, continuité audio générée, ré-ancrage couleur |
| **Réutilisation de résultat** | Réutilisez une image décodée d'un segment précédent comme entrée I2V ou FL2V |
| **Flux vidéo source** | Prise en charge dédiée V2V / RV2V et pont source (Source Bridge) aux transitions |
| **Gestion des ressources** | Références communes et Médiathèque persistante (images, audios, vidéos, prompts) |
| **Échantillonnage** | Échantillonneur interne paramétrable ou chaîne externe ComfyUI `SAMPLER + SIGMAS` |
| **Post-traitement** | Raffinement global (Global Refine), agrandissement (Upscale), NVIDIA RTX VSR / Deblur, Raffinement visage (Face Refine) |
| **Aperçu et export** | Aperçu en direct (Director Live Preview), vues Segment / Multi-segments / Résultat final, enregistrement vidéo |
| **Intégration ComfyUI** | Nœuds externes d'entrées et ressources, sorties `images / audio / fps` |

Motion Director est un `OUTPUT_NODE` : il peut donc terminer un flux tout en fournissant ses images finales, son audio et sa fréquence (FPS) aux nœuds ComfyUI en aval.

---

## Aperçu en direct (Live Preview)

Motion Director dispose de son propre système d'aperçu dynamique au lieu de dépendre uniquement de l'aperçu standard de l'échantillonneur. Il affiche l'étape active de génération pendant l'exécution, y compris les phases ultérieures de post-traitement.

![MiniMax H3 Motion Director — Aperçu en direct](docs/images/live-preview.gif)

---

## Mode Mixte : plusieurs méthodes sur une seule chronologie

Un flux H3 classique traite généralement chaque génération comme un extrait isolé. Le Mode Mixte structure votre projet comme un véritable montage vidéo.

Exemple :

```text
S1  T2V (Texte vers vidéo)
S2  I2V (Image vers vidéo)
S3  R2V (Sujet de référence)
S4  Vidéo Source (V2V / RV2V)
S5  T2V (Texte vers vidéo)
```

Chaque segment dispose de son propre mode, de son invite (prompt), de sa durée ou plage source, et de ses entrées médias adaptées.

### Modes des segments mixtes

| Mode de segment | Chemin d'exécution | Utilisation principale |
|---|---|---|
| `T2V` | T2V | Plan généré à partir d'un prompt texte |
| `I2V` | I2V | Départ depuis une image importée ou le résultat d'un plan précédent |
| `FL2V` | FL2V | Contrôle précis de la première image, de la dernière, ou des deux |
| `R2V` | R2V | Références d'identité, de décor, de mouvement ou de voix |
| `Vidéo Source` | V2V ou RV2V | Réutilise le mouvement source, avec ou sans images d'identité |

Le mode `Vidéo Source` s'adapte automatiquement :
```text
Vidéo Source + 0 image d'identité  -> V2V
Vidéo Source + images d'identité    -> RV2V
```

### Continuité par transition

La continuité peut être activée ou coupée individuellement entre les cartes de segments :

```text
[S1]  S1 -> S2  [S2]  S2 -> S3  [S3]
        visuel          visuel
        audio           audio
```

Les paramètres au niveau du nœud restent les maîtres généraux, ce qui permet d'enchaîner certains plans de manière fluide tout en provoquant une coupure nette sur d'autres.

### Réutilisation du résultat de segment (Segment Result)

Le Mode Mixte permet de réutiliser une image statique décodée issue d'un segment antérieur :
- Segment précédent ➔ dernière image
- Segment précédent ➔ index d'image précis

Cas d'usage classiques :
- Segment antérieur ➔ Image de départ I2V
- Segment antérieur ➔ Première image FL2V
- Segment antérieur ➔ Dernière image FL2V

### Exécution sélective (Selective Run)

Sur un long projet, il est rare de devoir recalculer tous les plans. Activez **« Sélectionner à exécuter »**, cochez uniquement les segments à ajuster, et conservez le reste de la séquence intacte grâce au cache. Vous gagnez un temps précieux sans recalculer les plans déjà réussis.

---

## Modes H3 autonomes

Le Director prend également en charge six modes MiniMax H3 autonomes :

| Mode | Entrée principale | Utilisation type |
|---|---|---|
| `T2V` | Prompt | Génération multi-plans à partir de texte |
| `I2V` | Prompt + image de départ | Animation d'un personnage ou d'un décor |
| `FL2V` | Prompt + première / dernière image | Contrôle visuel strict du début et de la fin |
| `R2V` | Prompt + références multimodales | Références de personnage, style, mouvement, décor ou voix |
| `V2V` | Vidéo Source + Prompt | Régénération visuelle en préservant le mouvement source |
| `RV2V` | Vidéo Source + Prompt + références | Mouvement source associé à des références d'identité et d'audio |

Pour le mode R2V, chaque groupe de ressources peut contenir jusqu'à :
```text
Images 1 à 9
Vidéos 1 à 3
Audios 1 à 3
```

---

## Références communes (Common References)

Les Références communes sont des médias définis au niveau du projet et partagés entre plusieurs segments. Elles sont idéales pour maintenir des personnages récurrents, des décors, des accessoires ou des voix sans avoir à les réimporter manuellement dans chaque segment.

![MiniMax H3 Motion Director — Références communes](docs/images/common-references.png)

---

## Médiathèque (Material Library)

La **Médiathèque** persistante permet de stocker et de réutiliser vos ressources entre différents plans ou projets futurs :
- Images
- Fichiers audio
- Vidéos
- Invites / Prompts

Les images peuvent être classées par catégories (Personnages, Décors, Accessoires, Autres). La recherche et l'assignation se font directement au sein de l'interface du Director.

![MiniMax H3 Motion Director — Médiathèque](docs/images/material-library.png)

---

## Post-traitement intégré

Le Director intègre des étapes de finition sans nécessiter de graphe de post-traitement séparé.

![MiniMax H3 Motion Director — Post-traitement](docs/images/postprocess.png)

### Raffinement global (Global Refine)
Permet une seconde passe d'échantillonnage et un agrandissement (upscale) optionnel avant raffinement :
- Redimensionnement standard / upscale
- Modèles d'upscale ComfyUI
- NVIDIA RTX Video Super Resolution (VSR)
- NVIDIA RTX Deblur
- Second échantillonnage H3

Si le raffinement global échoue, le Director conserve automatiquement le résultat de la première passe.

### Raffinement visage (Face Refine)
Restauration avancée des visages incluant :
- Détection et suivi des visages
- Régénération ciblée H3 par recadrage
- Intensité adaptative
- Contrôle des masques et de la fusion
- Harmonisation colorimétrique

---

## Gestion des résultats et sorties

La page des résultats offre trois niveaux d'inspection :
- **Segment** : examiner un segment généré individuellement.
- **Multi-segments** : prévisualiser et exporter une plage continue de segments.
- **Résultat final** : visualiser et enregistrer l'assemblage complet du pipeline avec son rapport détaillé.

### Sorties du nœud ComfyUI

| Sortie | Type | Description |
|---|---|---|
| `images` | Liste `IMAGE` | Images vidéo finales générées |
| `audio` | Liste `AUDIO` | Audio synchronisé correspondant |
| `fps` | `FLOAT` | Fréquence d'images finale |

---

## Entrées et ressources externes

Le Director s'intègre parfaitement avec le reste de votre flux ComfyUI grâce à ses nœuds d'entrées externes :

```text
MiniMax H3 Motion Director Assets
        ↓
MiniMax H3 Motion Director Inputs
        ↓
MiniMax H3 Motion Director
```

| Nœud | Rôle |
|---|---|
| `MiniMax H3 Motion Director` | Interface principale, exécution, prévisualisation, post-traitement et résultats |
| `MiniMax H3 Motion Director Inputs` | Entrées dynamiques de prompts, images et groupes d'assets |
| `MiniMax H3 Motion Director Assets` | Regroupe les médias requis par mode pour un groupe d'entrée |

---

## Installation

### Installation manuelle

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Jonathan97480/ComfyUI-MiniMax-H3-Motion-Director.git
cd ComfyUI-MiniMax-H3-Motion-Director
python -m pip install -r requirements.txt
```

*Sur une version autonome (portable) de ComfyUI sous Windows, utilisez l'exécutable Python inclus (`python_embeded\python.exe -m pip install -r requirements.txt`).*

Redémarrez complètement ComfyUI après l'installation, puis effectuez un rafraîchissement complet du cache du navigateur (<kbd>Ctrl</kbd> + <kbd>F5</kbd>).

---

## Compatibilité et prérequis

- Nécessite une version récente de ComfyUI intégrant le support officiel de **MiniMax H3**.
- Dépendances déclarées dans `requirements.txt`.
- Les fonctions NVIDIA RTX VSR / Deblur requièrent les cartes NVIDIA RTX compatibles et le package RTX VFX adéquat.
- Ne chargez pas l'extension obsolète `ComfyUI-H3-Motion-Context` en parallèle (sa logique est directement intégrée dans ce nœud).

---

## Crédits et remerciements

Ce projet adapte et intègre avec respect les travaux de plusieurs projets open-source :
- [AIMixer / ComfyUI_MiniMaxH3_Director](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director) — Apache-2.0
- [NikoDemon80 / ComfyUI-H3-Motion-Context](https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context) — GPL-3.0
- [Carasibana / ComfyUI-H3-FaceRefine](https://github.com/Carasibana/ComfyUI-H3-FaceRefine) — MIT
- [Kijai / ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes) — GPL-3.0

Consultez les fichiers [`NOTICE`](NOTICE), [`LICENSE`](LICENSE) et le dossier [`LICENSES`](LICENSES) pour les attributions détaillées.

## Licence

Ce projet est distribué sous licence **GNU GPL v3.0**.


---

## 🎬 Scénariste et Storyboard IA (`MiniMax H3 AI Story Director`)

Le nœud **MiniMax H3 AI Story Director** permet de créer automatiquement une vidéo complète à partir d'une simple **histoire textuelle** :

1. **Récit utilisateur & Durée** : Décrivez l'histoire ou le script de votre choix et spécifiez la durée globale souhaitée (ex: 15s, 30s, 60s).
2. **Découpage cinématographique intelligent** :
   - Calcule automatiquement le nombre de plans et leur durée optimale (plans de 5 à 6 secondes recommandés pour MiniMax H3).
   - Détermine la progression narrative et les valeurs de plans : *Plan d'ensemble d'exposition, plan moyen d'action, gros plan dramatique, plan en contre-plongée, plan de révélation*.
   - Attribue les trajectoires de caméra cinématiques (*Traveling avant fluide, panoramique horizontal, grue descendante, arc orbital*).
   - Formule les éclairages volumétriques et l'ambiance visuelle adaptés au style de l'histoire.
3. **Moteurs d'IA pris en charge** :
   - **Built-in AI Story Engine (Local / 100% autonome)** : Fonctionne instantanément sans nécessiter aucun serveur ni configuration externe.
   - **Ollama (LLM Local)** : Connectez directement votre instance locale Ollama (`http://127.0.0.1:11434`) avec vos modèles préférés (`qwen2.5:7b`, `gemma2:2b`, `mistral`, etc.).
   - **OpenAI-Compatible** : Compatible avec LM Studio, Jan, vLLM, LocalAI (`http://127.0.0.1:1234/v1`).
4. **Génération d'images de référence pour chaque scène** :
   - Si vous connectez un modèle de diffusion (`model`, `clip`, `vae`), le nœud génère de véritables images de référence IA pour chaque scène.
   - Si aucun modèle de diffusion n'est connecté, le nœud produit automatiquement des **cartes de storyboard cinématographiques** stylisées avec repères visuels, cadrage et résumé de scène.
   - Les images sont injectées directement dans la chronologie sous forme de vignettes et exportées en lot via la sortie `reference_images`.
5. **Connexion directe au Motion Director** :
   - Reliez simplement la sortie `storyboard` du nouveau nœud à la nouvelle entrée `storyboard` du **MiniMax H3 Motion Director**.
   - Ouvrez la fenêtre du Director pour voir instantanément l'ensemble des scènes, durées, prompts et images de référence configurés automatiquement !
