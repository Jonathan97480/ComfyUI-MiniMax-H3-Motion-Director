// MiniMax H3 AI Story Director — Frontend Extension
// Author: Jonathan97480 & Contributors
// Licensed under GNU GPL v3.0

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { getLocale } from "./minimax_i18n.js";
import { normalizeMixedTimeline } from "./minimax_mixed_state.mjs?boot=mixed_native_v6";
import { getDirectorMode, resolveTaskKey } from "./minimax_gen_timeline.js";
import { ensureImageBatchTimeline } from "./minimax_image_batch.js";

const STORY_DIRECTOR_CLASS = "MiniMaxH3StoryDirector";
const MOTION_DIRECTOR_CLASS = "MiniMaxH3MotionDirector";

const I18N = {
    fr: {
        generateSyncButton: "🎬 Générer & Synchroniser le Scénario",
        generateSyncBusy: "⏳ Génération des scènes & images...",
        generateSyncSuccess: "✨ Scénario & images synchronisés !",
        openDirectorButton: "👁️ Consulter / Modifier dans le Motion Director",
        syncNoTarget: "⚠️ Aucun Motion Director détecté sur le canevas. Ajoutez un nœud 'MiniMax H3 Motion Director'.",
        autoConnected: "🔗 Motion Director automatiquement connecté et synchronisé !",
        noDirectorOpened: "⚠️ Impossible d'ouvrir le panneau du Motion Director.",
        scenesCount: "scènes",
        duration: "durée",
        statusReady: "Prêt à générer le scénario",
        timelineUpdated: "✨ Timeline générée et injectée dans le Motion Director !",
        errorGenerating: "❌ Erreur lors de la génération du scénario : ",
    },
    en: {
        generateSyncButton: "🎬 Generate & Sync Storyboard",
        generateSyncBusy: "⏳ Generating scenes & images...",
        generateSyncSuccess: "✨ Storyboard & images synced!",
        openDirectorButton: "👁️ Inspect / Edit in Motion Director",
        syncNoTarget: "⚠️ No Motion Director detected on canvas. Add a 'MiniMax H3 Motion Director' node.",
        autoConnected: "🔗 Motion Director automatically connected and synced!",
        noDirectorOpened: "⚠️ Unable to open Motion Director panel.",
        scenesCount: "scenes",
        duration: "duration",
        statusReady: "Ready to generate storyboard",
        timelineUpdated: "✨ Storyboard generated and applied to Motion Director!",
        errorGenerating: "❌ Error generating storyboard: ",
    },
    zh: {
        generateSyncButton: "🎬 生成并同步分镜剧本",
        generateSyncBusy: "⏳ 正在生成分镜与图像...",
        generateSyncSuccess: "✨ 分镜与图像已成功同步到 Motion Director！",
        openDirectorButton: "👁️ 在 Motion Director 中查看 / 编辑",
        syncNoTarget: "⚠️ 画布上未找到 Motion Director 节点。请添加 'MiniMax H3 Motion Director' 节点。",
        autoConnected: "🔗 已自动连接 Motion Director 并完成同步！",
        noDirectorOpened: "⚠️ 无法打开 Motion Director 窗口。",
        scenesCount: "镜头",
        duration: "时长",
        statusReady: "分镜剧本生成器就绪",
        timelineUpdated: "✨ 分镜时间轴已成功生成并注入到 Motion Director！",
        errorGenerating: "❌ 生成分镜失败：",
    },
};

function t(key) {
    const loc = getLocale ? getLocale() : "fr";
    const dict = I18N[loc] || I18N.fr;
    return dict[key] || I18N.en[key] || key;
}

function findNodeInGraph(graph, id) {
    if (!graph || id == null) return null;
    let n = graph.getNodeById?.(id);
    if (!n && typeof id === "string" && !isNaN(Number(id))) {
        n = graph.getNodeById?.(Number(id));
    }
    if (!n && typeof id === "number") {
        n = graph.getNodeById?.(String(id));
    }
    if (!n && Array.isArray(graph._nodes)) {
        n = graph._nodes.find((item) => String(item.id) === String(id));
    }
    return n || null;
}

function findConnectedDirectors(storyNode, autoConnect = true) {
    const graph = storyNode?.graph || app.graph;
    if (!graph) return [];

    const directors = [];

    // 1. Direct links from any output of storyNode
    for (const output of storyNode.outputs || []) {
        if (!output?.links) continue;
        for (const linkId of output.links) {
            if (linkId == null) continue;
            let link = graph.links?.[linkId];
            if (!link && typeof graph.links?.find === "function") {
                link = graph.links.find((l) => l && (l.id === linkId || l[0] === linkId));
            }
            if (!link) continue;

            const targetId = link.target_id ?? link[3];
            const targetNode = graph.getNodeById?.(targetId);
            if (!targetNode) continue;

            const targetClass = String(targetNode.comfyClass || targetNode.type || "");
            if (targetClass.includes(MOTION_DIRECTOR_CLASS)) {
                if (!directors.includes(targetNode)) directors.push(targetNode);
            }
        }
    }

    // 2. Search all nodes in graph that have an input connected from storyNode
    if (directors.length === 0 && graph._nodes) {
        for (const node of graph._nodes) {
            if (!String(node.comfyClass || node.type || "").includes(MOTION_DIRECTOR_CLASS)) continue;
            for (const input of node.inputs || []) {
                if (input.link != null) {
                    const l = graph.links?.[input.link];
                    if (l) {
                        const origId = l.origin_id ?? l[1];
                        if (origId === storyNode.id) {
                            if (!directors.includes(node)) directors.push(node);
                        }
                    }
                }
            }
        }
    }

    // 3. Auto-connect fallback: If no links exist, but there is exactly ONE Motion Director on the canvas
    if (directors.length === 0 && autoConnect && graph._nodes) {
        const allDirectors = graph._nodes.filter((n) =>
            String(n.comfyClass || n.type || "").includes(MOTION_DIRECTOR_CLASS)
        );
        if (allDirectors.length === 1) {
            const singleDirector = allDirectors[0];
            const outIdx = (storyNode.outputs || []).findIndex(
                (o) => o?.name === "storyboard" || o?.type === "MMX_STORYBOARD"
            );
            const inIdx = (singleDirector.inputs || []).findIndex(
                (i) => i?.name === "storyboard" || i?.type === "MMX_STORYBOARD"
            );
            if (outIdx !== -1 && inIdx !== -1 && typeof storyNode.connect === "function") {
                try {
                    storyNode.connect(outIdx, singleDirector, inIdx);
                    console.log("[StoryDirector] Auto-connected storyboard socket to single Motion Director on canvas.");
                } catch (e) {
                    console.warn("[StoryDirector] Auto-connect attempt warning:", e);
                }
            }
            directors.push(singleDirector);
        }
    }

    return directors;
}

function updateDirectorTaskType(directorNode, requestedMode) {
    if (!directorNode || !requestedMode) return;
    const taskWidget = (directorNode.widgets || []).find((w) => w?.name === "task_type");
    if (!taskWidget) return;

    const modeClean = String(requestedMode).toLowerCase().trim();
    const options = taskWidget.options?.values || [];
    // Match option by prefix or substring, e.g. "mixed" -> "mixed · Multi-segment Director"
    const matchedOption = options.find((opt) => {
        const o = String(opt).toLowerCase().trim();
        return (
            o === modeClean ||
            o.startsWith(modeClean + " ") ||
            o.startsWith(modeClean + "·") ||
            o.startsWith(modeClean + " -") ||
            o.includes(modeClean)
        );
    }) || requestedMode;

    const oldValue = taskWidget.value;
    taskWidget.value = matchedOption;

    if (typeof taskWidget.callback === "function") {
        try {
            taskWidget.callback(matchedOption);
        } catch (e) {
            console.warn("[StoryDirector] taskWidget callback error:", e);
        }
    }

    if (typeof directorNode.onWidgetChanged === "function") {
        try {
            directorNode.onWidgetChanged("task_type", matchedOption, oldValue, taskWidget);
        } catch (e) {
            console.warn("[StoryDirector] directorNode onWidgetChanged error:", e);
        }
    }

    // Direct update on editor if active
    const ed = directorNode._minimaxEditor;
    if (ed) {
        if (ed.populateTaskSelect && ed.globalTask) {
            ed.populateTaskSelect(ed.globalTask, matchedOption);
        }
        if (ed.globalTask) ed.globalTask.value = matchedOption;
        const targetMode = modeClean === "mixed" ? "mixed" : getDirectorMode(matchedOption);
        ed._directorMode = targetMode;
        ed.onTaskTypeChanged?.(matchedOption);
        ed.applyTaskLayout?.(targetMode);
    }
}

function applyTimelineToDirector(directorNode, timelineJson, prompt, totalDuration, directorMode) {
    if (!directorNode || !timelineJson) return false;

    let parsed = null;
    try { parsed = JSON.parse(timelineJson); } catch (e) {}
    const isMixed = (parsed && String(parsed.timelineMode || "").toLowerCase() === "mixed") || String(directorMode).toLowerCase() === "mixed";
    const modeClean = isMixed ? "mixed" : String(directorMode || "mixed").toLowerCase().trim();

    // Ensure editor is instantiated if node hasn't been mounted yet
    if (!directorNode._minimaxEditor && typeof globalThis.__MMX_INIT_DIRECTOR_EDITOR__ === "function") {
        try {
            globalThis.__MMX_INIT_DIRECTOR_EDITOR__(directorNode);
        } catch (e) {
            console.warn("[StoryDirector] initDirectorEditor warning:", e);
        }
    }

    // 1. Update task_type mode (e.g. Mixed, R2V, FL2V, T2V)
    updateDirectorTaskType(directorNode, modeClean);

    // 2. Update timeline_data widget
    const timelineWidget = (directorNode.widgets || []).find((w) => w?.name === "timeline_data");
    if (timelineWidget) {
        const oldValue = timelineWidget.value;
        timelineWidget.value = timelineJson;

        if (typeof timelineWidget.callback === "function") {
            try {
                timelineWidget.callback(timelineJson);
            } catch (e) {
                console.warn("[StoryDirector] timelineWidget callback warning:", e);
            }
        }

        if (typeof directorNode.onWidgetChanged === "function") {
            try {
                directorNode.onWidgetChanged("timeline_data", timelineJson, oldValue, timelineWidget);
            } catch (e) {
                console.warn("[StoryDirector] directorNode onWidgetChanged warning:", e);
            }
        }
    }

    // 3. Update global_prompt widget if provided
    if (prompt) {
        const promptWidget = (directorNode.widgets || []).find((w) => w?.name === "global_prompt");
        if (promptWidget) {
            const oldPrompt = promptWidget.value;
            promptWidget.value = prompt;
            if (typeof promptWidget.callback === "function") {
                try {
                    promptWidget.callback(prompt);
                } catch (e) {
                    console.warn("[StoryDirector] promptWidget callback warning:", e);
                }
            }
            if (typeof directorNode.onWidgetChanged === "function") {
                try {
                    directorNode.onWidgetChanged("global_prompt", prompt, oldPrompt, promptWidget);
                } catch (e) {
                    console.warn("[StoryDirector] promptWidget onWidgetChanged warning:", e);
                }
            }
        }
    }

    // 4. Update total_frames widget if duration is provided
    if (totalDuration != null && totalDuration > 0) {
        const totalFramesWidget = (directorNode.widgets || []).find((w) => w?.name === "total_frames");
        if (totalFramesWidget) {
            const fpsWidget = (directorNode.widgets || []).find((w) => w?.name === "frame_rate");
            const fps = Number(fpsWidget?.value || 24);
            const calculatedFrames = Math.max(5, Math.round(Number(totalDuration) * fps));
            totalFramesWidget.value = calculatedFrames;
            if (typeof totalFramesWidget.callback === "function") {
                try {
                    totalFramesWidget.callback(calculatedFrames);
                } catch (e) {
                    console.warn("[StoryDirector] totalFramesWidget callback warning:", e);
                }
            }
        }
    }

    // 5. Live refresh MiniMaxH3MotionDirectorEditor if instantiated
    const ed = directorNode._minimaxEditor;
    if (ed) {
        try {
            const initTotal = Math.max(0, parseInt(ed.totalFramesWidget?.value || 124, 10));
            const initFps = Number(ed.frameRateWidget?.value || 24);
            if (typeof ed._loadNativeTimelineState === "function") {
                ed.timeline = ed._loadNativeTimelineState(timelineJson, initTotal, initFps);
            }
            ed.ensureContextLinks?.();
            ed.syncFrameRateUI?.(ed.timeline?.frameRate || initFps);

            let parsed = null;
            try { parsed = JSON.parse(timelineJson); } catch (e) {}

            if (parsed && String(parsed.timelineMode || "").toLowerCase() === "mixed") {
                ed.mixedTimeline = normalizeMixedTimeline(parsed);
                if (directorNode.id != null) ed.mixedTimeline.nodeId = String(directorNode.id);
                ed._directorMode = "mixed";
                if (ed._mixedController) {
                    try { ed._mixedController.setState(ed.mixedTimeline); } catch (e) {}
                }
            }

            const targetMode = isMixed
                ? "mixed"
                : getDirectorMode(modeClean || ed.taskTypeWidget?.value || "r2v");
            ed._directorMode = targetMode;
            if (targetMode === "mixed") {
                // mixed mode owns editor.mixedTimeline
            } else if (targetMode === "video") {
                ed.restoreVideoFromTimeline?.();
            } else if (targetMode === "prompt_batch" || targetMode === "image_batch") {
                if (ed.timeline) {
                    delete ed.timeline.batchWorkspace;
                    delete ed.timeline.batchTaskWorkspaces;
                    delete ed.timeline.videoWorkspace;
                }
                ensureImageBatchTimeline(ed);
            } else {
                ed.ensureGenTimeline?.();
            }

            ed.populateTaskSelect?.(ed.globalTask, ed.taskTypeWidget?.value);
            ed.ensureContextLinks?.();
            ed.applyTaskLayout?.("video");
            ed.setEditMode?.(ed.timeline?.editMode || "global");
            ed.selectedIndex = 0;
            ed.updateSelectionUI?.();
            ed.commit?.(true, { syncTimeline: false });
            ed._externalGroupsSyncSig = null;
            ed.syncExternalGroupsTimeline?.();
            ed.scheduleRender?.();

            if (mode === "mixed") {
                if (ed._mixedController && ed.mixedTimeline) {
                    try { ed._mixedController.setState(ed.mixedTimeline); } catch (e) {}
                }
            } else if (ed.isImageBatch?.()) {
                ed.renderImageBatchGroups?.();
            }

            if (ed._directorModalController?.isOpen) {
                if (mode === "mixed" && ed._mixedController && ed.mixedTimeline) {
                    try { ed._mixedController.setState(ed.mixedTimeline); } catch (e) {}
                } else if (ed.isImageBatch?.()) {
                    ed.renderImageBatchGroups?.();
                }
                ed._directorModalController.setPage?.("generation");
            }
        } catch (e) {
            console.warn("[StoryDirector] Live Editor sync error:", e);
        }
    }

    // 6. Refresh timeline editor controller if open
    if (directorNode._directorTimelineController?.timelineWidget) {
        try {
            directorNode._directorTimelineController.timelineWidget.value = timelineJson;
            directorNode._directorTimelineController.syncExternalGroupsTimeline?.();
        } catch (e) {
            console.warn("[StoryDirector] Controller sync error:", e);
        }
    }

    // 7. Request LiteGraph node and canvas redraw
    directorNode.setDirtyCanvas?.(true, true);
    app.graph?.setDirtyCanvas?.(true, true);

    return true;
}

function openDirectorModal(directorNode) {
    if (!directorNode) return false;

    if (!directorNode._minimaxEditor && typeof globalThis.__MMX_INIT_DIRECTOR_EDITOR__ === "function") {
        try {
            globalThis.__MMX_INIT_DIRECTOR_EDITOR__(directorNode);
        } catch (e) {}
    }

    const ed = directorNode._minimaxEditor;
    if (ed) {
        // Re-hydrate state from timelineWidget before opening modal
        const raw = String(ed.timelineWidget?.value || "").trim();
        if (raw.startsWith("{")) {
            try {
                const initTotal = Math.max(0, parseInt(ed.totalFramesWidget?.value || 124, 10));
                const initFps = Number(ed.frameRateWidget?.value || 24);
                if (typeof ed._loadNativeTimelineState === "function") {
                    ed.timeline = ed._loadNativeTimelineState(raw, initTotal, initFps);
                }
                let parsed = null;
                try { parsed = JSON.parse(raw); } catch (e) {}
                if (parsed && String(parsed.timelineMode || "").toLowerCase() === "mixed") {
                    ed.mixedTimeline = (typeof normalizeMixedTimeline === "function")
                        ? normalizeMixedTimeline(parsed)
                        : parsed;
                    if (directorNode.id != null) ed.mixedTimeline.nodeId = String(directorNode.id);
                    if (ed._mixedController) {
                        try { ed._mixedController.setState(ed.mixedTimeline); } catch (e) {}
                    }
                } else {
                    const taskKey = resolveTaskKey(parsed?.global?.taskType || ed.taskTypeWidget?.value || "");
                    const targetMode = getDirectorMode(taskKey || parsed?.timelineMode || "r2v");
                    ed._directorMode = targetMode;
                    if (targetMode === "prompt_batch" || targetMode === "image_batch") {
                        if (ed.timeline) {
                            delete ed.timeline.batchWorkspace;
                            delete ed.timeline.batchTaskWorkspaces;
                            delete ed.timeline.videoWorkspace;
                        }
                        ensureImageBatchTimeline(ed);
                    }
                    if (ed.globalTask) ed.globalTask.value = taskKey;
                    if (ed.taskTypeWidget) ed.taskTypeWidget.value = taskKey;
                    ed.populateTaskSelect?.(ed.globalTask, taskKey);
                    ed.applyTaskLayout?.("video");
                    if (ed.isImageBatch?.()) {
                        ed.renderImageBatchGroups?.();
                    }
                }
            } catch (err) {}
        }
    }

    // Direct modal controller
    if (ed?._directorModalController) {
        if (!ed._directorModalController.isOpen) {
            ed._directorModalController.open();
        }
        if (ed._directorMode === "mixed") {
            if (!ed._mixedController) {
                ed.applyTaskLayout?.("mixed");
            }
            if (ed._mixedController && ed.mixedTimeline) {
                try { ed._mixedController.setState(ed.mixedTimeline); } catch (e) {}
            }
        } else if (ed.isImageBatch?.()) {
            ed.renderImageBatchGroups?.();
        }
        ed._directorModalController.setPage?.("generation");
        return true;
    }

    // Fallback: look for the open button in director DOM container
    const container = directorNode._minimaxDomWidget?.element;
    if (container) {
        const openBtn = container.querySelector?.('[data-a="open-director"]') || container.querySelector?.("button");
        if (openBtn) {
            openBtn.click();
            return true;
        }
    }

    return false;
}

async function executeStoryNodeSubgraph(storyNode) {
    const promptData = await app.graphToPrompt();
    if (!promptData || !promptData.output) {
        throw new Error("Impossible de compiler le graphe ComfyUI.");
    }

    const targetNodeId = String(storyNode.id);
    if (!promptData.output[targetNodeId]) {
        throw new Error(`Le nœud Story Director #${targetNodeId} n'a pas été trouvé dans le prompt.`);
    }

    // Recursively collect upstream dependencies of storyNode only
    const upstreamIds = new Set();
    function collectUpstream(nodeId) {
        if (upstreamIds.has(nodeId)) return;
        upstreamIds.add(nodeId);
        const nodeInfo = promptData.output[nodeId];
        if (!nodeInfo || !nodeInfo.inputs) return;
        for (const [key, val] of Object.entries(nodeInfo.inputs)) {
            if (Array.isArray(val) && val.length === 2) {
                const srcId = String(val[0]);
                if (promptData.output[srcId]) {
                    collectUpstream(srcId);
                }
            }
        }
    }

    collectUpstream(targetNodeId);

    // Build the isolated subgraph output: strictly exclude any downstream video / render nodes
    const partialOutput = {};
    for (const id of upstreamIds) {
        partialOutput[id] = promptData.output[id];
    }

    const payload = {
        client_id: api.clientId,
        prompt: partialOutput,
        extra_data: {
            extra_pnginfo: {
                workflow: promptData.workflow,
            },
        },
    };

    const res = await api.fetchApi("/prompt", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
    });

    if (!res.ok) {
        const txt = await res.text();
        throw new Error(`Erreur d'exécution ComfyUI (${res.status}): ${txt}`);
    }

    return await res.json();
}

async function triggerStoryboardGeneration(storyNode, btnWidget) {
    const originalLabel = btnWidget?.name || t("generateSyncButton");
    if (btnWidget) {
        btnWidget.name = t("generateSyncBusy");
        storyNode.setDirtyCanvas?.(true, true);
    }

    try {
        const getVal = (name, fallback) => {
            const w = (storyNode.widgets || []).find((w) => w?.name === name);
            return w?.value !== undefined ? w.value : fallback;
        };

        const directorMode = String(getVal("director_mode", "mixed"));
        const totalDuration = Number(getVal("total_duration_sec", 15.0));
        const shotDuration = Number(getVal("shot_duration_sec", 5.0));

        // Check if an AI model input is wired into storyNode
        const modelInput = (storyNode.inputs || []).find((i) => i.name === "model");
        const hasModelConnected = modelInput && modelInput.link != null;

        let generationHandledBySubgraph = false;

        // Mode A: If a diffusion model is connected, execute ONLY the Story Director subgraph via ComfyUI
        if (hasModelConnected) {
            try {
                await executeStoryNodeSubgraph(storyNode);
                generationHandledBySubgraph = true;
            } catch (subgraphErr) {
                console.warn("[StoryDirector] Subgraph execution queue failed, falling back to HTTP route:", subgraphErr);
            }
        }

        // Mode B: Fast HTTP route execution (standalone or fallback)
        if (!generationHandledBySubgraph) {
            const payload = {
                node_id: String(storyNode.id),
                story: String(getVal("story", "")),
                total_duration_sec: totalDuration,
                shot_duration_sec: shotDuration,
                aspect_ratio: String(getVal("aspect_ratio", "16:9 (864x480)")),
                director_mode: directorMode,
                style_prompt: String(getVal("style_prompt", "")),
                negative_prompt: String(getVal("negative_prompt", "")),
                ai_story_engine: String(getVal("ai_story_engine", "")),
                llm_url: String(getVal("llm_url", "")),
                llm_model: String(getVal("llm_model", "")),
                image_steps: Number(getVal("image_steps", 20)),
                image_cfg: Number(getVal("image_cfg", 6.5)),
                seed: Number(getVal("seed", 0)),
            };

            const response = await api.fetchApi("/minimax/storyboard/generate", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            });

            if (!response.ok) {
                const errText = await response.text();
                throw new Error(`Server returned HTTP ${response.status}: ${errText}`);
            }

            const data = await response.json();
            if (!data?.ok) {
                throw new Error(data?.error || "Unknown server error during storyboard generation.");
            }

            storyNode._lastGeneratedTimeline = data.timeline_data;
            storyNode._lastGeneratedPrompt = data.prompt;
            storyNode._lastGeneratedSubjects = data.subjects;
            storyNode._lastGeneratedScenes = data.scenes;

            const directors = findConnectedDirectors(storyNode, true);
            let syncCount = 0;
            for (const dir of directors) {
                if (applyTimelineToDirector(dir, data.timeline_data, data.prompt, totalDuration, directorMode)) {
                    syncCount++;
                }
            }

            const sceneCount = data.scenes?.length || 0;
            if (btnWidget) {
                btnWidget.name = `✅ ${sceneCount} ${t("scenesCount")} ${t("generateSyncSuccess").slice(2, 20)}...`;
                storyNode.setDirtyCanvas?.(true, true);
                setTimeout(() => {
                    btnWidget.name = originalLabel;
                    storyNode.setDirtyCanvas?.(true, true);
                }, 3500);
            }

            if (directors.length === 0) {
                alert(t("syncNoTarget"));
            }
        } else {
            // When handled by subgraph, update button label and let WebSocket listener complete sync
            if (btnWidget) {
                btnWidget.name = "⏳ Exécution IA ComfyUI...";
                storyNode.setDirtyCanvas?.(true, true);
            }
        }
    } catch (err) {
        console.error("[StoryDirector] Storyboard generation error:", err);
        alert(t("errorGenerating") + (err?.message || err));
        if (btnWidget) {
            btnWidget.name = originalLabel;
            storyNode.setDirtyCanvas?.(true, true);
        }
    }
}

function ensureStoryDirectorWidgets(node) {
    if (!node.widgets) node.widgets = [];

    const hasGenBtn = node.widgets.some((w) => w._mmxIsGenBtn || w.name === t("generateSyncButton"));
    if (!hasGenBtn) {
        const btnGen = node.addWidget(
            "button",
            t("generateSyncButton"),
            null,
            function () {
                triggerStoryboardGeneration(node, this);
            },
            { serialize: false }
        );
        btnGen._mmxIsGenBtn = true;
    }

    const hasOpenBtn = node.widgets.some((w) => w._mmxIsOpenBtn || w.name === t("openDirectorButton"));
    if (!hasOpenBtn) {
        const btnOpen = node.addWidget(
            "button",
            t("openDirectorButton"),
            null,
            function () {
                const directors = findConnectedDirectors(node, true);
                if (!directors.length) {
                    alert(t("syncNoTarget"));
                    return;
                }
                const success = openDirectorModal(directors[0]);
                if (!success) {
                    alert(t("noDirectorOpened"));
                }
            },
            { serialize: false }
        );
        btnOpen._mmxIsOpenBtn = true;
    }
}

app.registerExtension({
    name: "MiniMaxH3.MotionDirector.StoryDirector",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== STORY_DIRECTOR_CLASS) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const res = onNodeCreated?.apply(this, arguments);
            ensureStoryDirectorWidgets(this);
            return res;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const res = onConfigure?.apply(this, arguments);
            ensureStoryDirectorWidgets(this);
            return res;
        };

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            const res = onExecuted?.apply(this, arguments);

            if (message?.timeline_json) {
                const timelineStr = Array.isArray(message.timeline_json)
                    ? message.timeline_json[0]
                    : message.timeline_json;

                const promptStr = Array.isArray(message.h3_prompt)
                    ? message.h3_prompt[0]
                    : message.h3_prompt;

                const directorMode = (this.widgets || []).find((w) => w?.name === "director_mode")?.value || "mixed";
                const totalDur = (this.widgets || []).find((w) => w?.name === "total_duration_sec")?.value || 15.0;

                if (timelineStr && typeof timelineStr === "string") {
                    this._lastGeneratedTimeline = timelineStr;
                    this._lastGeneratedPrompt = promptStr;
                    const directors = findConnectedDirectors(this, true);
                    for (const dir of directors) {
                        applyTimelineToDirector(dir, timelineStr, promptStr, totalDur, directorMode);
                    }
                }
            }

            return res;
        };
    },

    setup() {
        // Listen to server broadcast events when storyboard generation completes
        api.addEventListener("minimax_storyboard_generated", (event) => {
            const detail = event?.detail;
            if (!detail?.timeline_data) return;

            const graph = app.graph;
            if (!graph) return;

            const storyNodeId = detail.node_id;
            let targetDirectors = [];

            if (storyNodeId) {
                const storyNode = findNodeInGraph(graph, storyNodeId);
                if (storyNode) {
                    storyNode._lastGeneratedTimeline = detail.timeline_data;
                    storyNode._lastGeneratedPrompt = detail.prompt;
                    targetDirectors = findConnectedDirectors(storyNode, true);

                    // Reset button label on the story node if it was busy
                    const genBtn = (storyNode.widgets || []).find((w) => w?._mmxIsGenBtn);
                    if (genBtn) {
                        genBtn.name = `✅ ${detail.num_scenes || ""} ${t("scenesCount")} ${t("generateSyncSuccess").slice(2, 20)}...`;
                        storyNode.setDirtyCanvas?.(true, true);
                        setTimeout(() => {
                            genBtn.name = t("generateSyncButton");
                            storyNode.setDirtyCanvas?.(true, true);
                        }, 3500);
                    }
                }
            }

            // Fallback: update any MotionDirector connected to any StoryDirector
            if (!targetDirectors.length) {
                const allStoryNodes = (graph._nodes || []).filter(
                    (n) => String(n.comfyClass || n.type || "") === STORY_DIRECTOR_CLASS
                );
                for (const sn of allStoryNodes) {
                    targetDirectors.push(...findConnectedDirectors(sn, true));
                }
            }

            for (const dir of targetDirectors) {
                applyTimelineToDirector(
                    dir,
                    detail.timeline_data,
                    detail.prompt,
                    detail.total_duration,
                    detail.director_mode || "mixed"
                );
            }
        });

        // Listen to director execution ingestion
        api.addEventListener("minimax_storyboard_applied", (event) => {
            const detail = event?.detail;
            if (!detail?.node_id || !detail?.timeline_data) return;

            const graph = app.graph;
            if (!graph) return;

            const directorNode = findNodeInGraph(graph, detail.node_id);
            if (directorNode) {
                applyTimelineToDirector(
                    directorNode,
                    detail.timeline_data,
                    detail.prompt,
                    detail.total_duration,
                    detail.director_mode || "mixed"
                );
            }
        });
    },
});
