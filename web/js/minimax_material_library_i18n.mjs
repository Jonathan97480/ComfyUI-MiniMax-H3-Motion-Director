// MiniMax H3 Motion Director — Material Library localized copy.

import { getLocale, onLocaleChange } from "./minimax_i18n.js";

const ZH = {
    button: "素材库", title: "素材库", close: "关闭", cancel: "关闭", apply: "应用", applying: "应用中…",
    image: "图片", audio: "音频", video: "视频", prompt: "Prompt", add: "新增素材", search: "搜索标题…",
    allCategories: "全部", empty: "这里还没有素材", emptySearch: "没有符合条件的素材",
    selected: "已选择", preview: "分配预览", previewEmpty: "尚未选择素材", target: "应用到", common: "公共素材",
    targetRequired: "请先选择应用目标", promptCommonBlocked: "Prompt 只能应用到 Segment，不能应用到公共素材",
    append: "追加", replace: "替换", promptMode: "Prompt 应用方式", first: "首帧", last: "尾帧", fl2vRole: "图片用途",
    leftRightHint: "左键添加一次 · 右键撤销最后一次", edit: "编辑", delete: "删除", rename: "标题",
    category: "分类", content: "Prompt 正文", save: "保存", confirmDelete: "确定删除「{title}」？",
    clearPage: "清除当页选取", clearAll: "清除所有页选取", addCategory: "新增小分类", renameCategory: "重命名小分类", categoryName: "小分类名称", zoom: "放大预览",
    upload: "从电脑加入素材库", uploading: "正在保存 {name}…", saved: "已保存", applied: "素材已应用到 Director",
    error: "错误：{message}", sourceLocalOnly: "RV2V 的 Source Video 仍只能从电脑本地上传；当前 RV2V 素材库仅提供图片、音频和 Prompt。",
    promptNoShot: "FL2V 只有 Prompt、没有任何镜头/首尾帧，无法自动创建空镜头。",
    v2vNoVideo: "V2V 没有 Source Video，只有 Prompt 时无法创建可运行片段。",
    noFreeSlot: "{type} 已达到当前目标的素材上限，后续同类素材未加入。",
    createSegment: "将自动创建 S{n}", itemTo: "{type} {order} → {target}", promptJoined: "多个 Prompt 会按编号顺序组合后应用。",
    firstFrame: "首帧", lastFrame: "尾帧", sourceVideo: "Source Video", refPicture: "Reference Picture",
    refAudio: "Reference Audio", refVideo: "Reference Video", picture: "Picture", segment: "S{n}",
    categories_image: ["Characters", "Scenes", "Props", "Other"],
    categories_audio: ["Voice", "Lines", "Sound FX", "Music", "Other"],
    categories_video: ["Characters", "Scenes", "Action", "Camera", "Other"],
    categories_prompt: ["Characters", "Scenes", "Action", "Camera", "Style", "Dialogue", "Other"],
};

const EN = {
    button: "Library", title: "Material Library", close: "Close", cancel: "Close", apply: "Apply", applying: "Applying…",
    image: "Images", audio: "Audio", video: "Video", prompt: "Prompt", add: "Add material", search: "Search titles…",
    allCategories: "All", empty: "No materials yet", emptySearch: "No matching materials",
    selected: "Selected", preview: "Allocation preview", previewEmpty: "Nothing selected", target: "Apply to", common: "Common",
    targetRequired: "Choose an apply target first", promptCommonBlocked: "Prompts can only be applied to a Segment, not Common References.",
    append: "Append", replace: "Replace", promptMode: "Prompt mode", first: "First frame", last: "Last frame", fl2vRole: "Image role",
    leftRightHint: "Left-click adds · right-click removes the last occurrence", edit: "Edit", delete: "Delete", rename: "Title",
    category: "Category", content: "Prompt content", save: "Save", confirmDelete: "Delete “{title}”?",
    clearPage: "Clear current tab", clearAll: "Clear all selections", addCategory: "Add subcategory", renameCategory: "Rename subcategory", categoryName: "Subcategory name", zoom: "Zoom preview",
    upload: "Add from computer", uploading: "Saving {name}…", saved: "Saved", applied: "Materials applied to Director",
    error: "Error: {message}", sourceLocalOnly: "RV2V Source Video stays local-upload only; the current RV2V library offers Images, Audio and Prompt only.",
    promptNoShot: "FL2V cannot create an anchorless shot from Prompt alone.",
    v2vNoVideo: "V2V cannot create runnable segments from Prompt without Source Video.",
    noFreeSlot: "{type} reached the current target limit; later items were skipped.",
    createSegment: "Will create S{n}", itemTo: "{type} {order} → {target}", promptJoined: "Multiple Prompts are joined in queue order before applying.",
    firstFrame: "First Frame", lastFrame: "Last Frame", sourceVideo: "Source Video", refPicture: "Reference Picture",
    refAudio: "Reference Audio", refVideo: "Reference Video", picture: "Picture", segment: "S{n}",
    categories_image: ["Characters", "Scenes", "Props", "Other"],
    categories_audio: ["Voice", "Lines", "Sound FX", "Music", "Other"],
    categories_video: ["Characters", "Scenes", "Action", "Camera", "Other"],
    categories_prompt: ["Characters", "Scenes", "Action", "Camera", "Style", "Dialogue", "Other"],
};

const FR = {
    button: "Médiathèque", title: "Bibliothèque de médias", close: "Fermer", cancel: "Fermer", apply: "Appliquer", applying: "Application…",
    image: "Images", audio: "Audio", video: "Vidéo", prompt: "Prompt", add: "Ajouter un média", search: "Rechercher…",
    allCategories: "Tous", empty: "Aucun média pour l'instant", emptySearch: "Aucun média correspondant",
    selected: "Sélectionné", preview: "Aperçu de l'affectation", previewEmpty: "Aucun média sélectionné", target: "Appliquer à", common: "Médias communs",
    targetRequired: "Veuillez d'abord choisir une cible", promptCommonBlocked: "Les prompts s'appliquent uniquement aux segments, pas aux médias communs.",
    append: "Ajouter", replace: "Remplacer", promptMode: "Mode d'application du prompt", first: "Première image", last: "Dernière image", fl2vRole: "Rôle de l'image",
    leftRightHint: "Clic gauche : ajouter · Clic droit : annuler la dernière sélection", edit: "Modifier", delete: "Supprimer", rename: "Titre",
    category: "Catégorie", content: "Texte du prompt", save: "Enregistrer", confirmDelete: "Supprimer « {title} » ?",
    clearPage: "Effacer l'onglet actuel", clearAll: "Effacer toutes les sélections", addCategory: "Nouvelle sous-catégorie", renameCategory: "Renommer sous-catégorie", categoryName: "Nom de la sous-catégorie", zoom: "Agrandir",
    upload: "Importer depuis l'ordinateur", uploading: "Enregistrement de {name}…", saved: "Enregistré", applied: "Médias appliqués au Director",
    error: "Erreur : {message}", sourceLocalOnly: "Pour RV2V, la vidéo source doit être importée localement ; la médiathèque ne fournit que des images, de l'audio et des prompts.",
    promptNoShot: "FL2V ne peut pas créer un plan sans image clé avec seulement un prompt.",
    v2vNoVideo: "V2V ne peut pas créer de segment exécutable sans vidéo source.",
    noFreeSlot: "La limite de médias pour {type} est atteinte pour cette cible ; les éléments suivants ont été ignorés.",
    createSegment: "Créera S{n}", itemTo: "{type} {order} → {target}", promptJoined: "Les prompts multiples sont combinés dans l'ordre de la file avant application.",
    firstFrame: "Première image", lastFrame: "Dernière image", sourceVideo: "Vidéo source", refPicture: "Image de référence",
    refAudio: "Audio de référence", refVideo: "Vidéo de référence", picture: "Image", segment: "S{n}",
    categories_image: ["Personnages", "Décors", "Accessoires", "Autre"],
    categories_audio: ["Voix", "Dialogue", "Effets sonores", "Musique", "Autre"],
    categories_video: ["Personnages", "Décors", "Action", "Plan / Caméra", "Autre"],
    categories_prompt: ["Personnages", "Décors", "Action", "Mvt caméra", "Style", "Dialogue", "Autre"],
};

function dict() {
    const loc = String(getLocale?.() || "zh").toLowerCase();
    if (loc.startsWith("fr")) return FR;
    if (loc.startsWith("en")) return EN;
    return ZH;
}

export function mlT(key, vars = {}) {
    let value = dict()[key];
    if (value == null) value = EN[key] ?? ZH[key] ?? key;
    if (Array.isArray(value)) return [...value];
    return String(value).replace(/\{(\w+)\}/g, (_, name) => String(vars[name] ?? ""));
}

export function materialCategories(kind) {
    return mlT(`categories_${kind}`);
}

export function onMaterialLocaleChange(callback) {
    return onLocaleChange(callback);
}
