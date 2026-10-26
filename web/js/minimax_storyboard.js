// MiniMax H3 AI Story Director — Frontend Extension
// Author: Jonathan97480 & Contributors
// Licensed under GNU GPL v3.0

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { getLocale } from "./minimax_i18n.js";

const STORY_DIRECTOR_CLASS = "MiniMaxH3StoryDirector";
const MOTION_DIRECTOR_CLASS = "MiniMaxH3MotionDirector";

const I18N = {
    fr: {
        generateSyncButton: "🎬 Générer & Synchroniser le Scénario",
        generateSyncBusy: "⏳ Génération en cours...",
        generateSyncSuccess: "✨ Scénario généré et synchronisé !",
        openDirectorButton: "👁️ Consulter / Modifier dans le Motion Director",
        syncNoTarget: "⚠️ Aucun Motion Director connecté sur le socket 'storyboard'. Reliez la sortie 'storyboard' à un Motion Director.",
        noDirectorOpened: "⚠️ Impossible d'ouvrir le panneau du Motion Director.",
        scenesCount: "scènes",
        duration: "durée",
        statusReady: "Prêt à générer le scénario",
        timelineUpdated: "✨ Timeline générée et injectée dans le Motion Director !",
        errorGenerating: "❌ Erreur lors de la génération du scénario : ",
    },
    en: {
        generateSyncButton: "🎬 Generate & Sync Storyboard",
        generateSyncBusy: "⏳ Generating storyboard...",
        generateSyncSuccess: "✨ Storyboard generated & synced!",
        openDirectorButton: "👁️ Inspect / Edit in Motion Director",
        syncNoTarget: "⚠️ No Motion Director connected to 'storyboard' output. Connect the 'storyboard' socket to a Motion Director.",
        noDirectorOpened: "⚠️ Unable to open Motion Director panel.",
        scenesCount: "scenes",
        duration: "duration",
        statusReady: "Ready to generate storyboard",
        timelineUpdated: "✨ Storyboard generated and applied to Motion Director!",
        errorGenerating: "❌ Error generating storyboard: ",
    },
    zh: {
        generateSyncButton: "🎬 生成并同步分镜剧本",
        generateSyncBusy: "⏳ 正在生成分镜...",
        generateSyncSuccess: "✨ 分镜与时间轴已成功同步到 Motion Director！",
        openDirectorButton: "👁️ 在 Motion Director 中查看 / 编辑",
        syncNoTarget: "⚠️ 未在 'storyboard' 输出端连接 Motion Director 节点。请将 'storyboard' 接口连接至 Motion Director。",
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

function findConnectedDirectors(storyNode) {
    const graph = storyNode?.graph || app.graph;
    if (!graph) return [];

    const directors = [];
    const storyboardOutput = (storyNode.outputs || []).find(
        (o) => o?.name === "storyboard" || o?.type === "MMX_STORYBOARD"
    );

    if (storyboardOutput?.links) {
        for (const linkId of storyboardOutput.links) {
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

    // Fallback: search all nodes in graph that have an input connected from storyNode
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

    return directors;
}

function applyTimelineToDirector(directorNode, timelineJson, prompt, totalDuration) {
    if (!directorNode || !timelineJson) return false;

    // 1. Update timeline_data widget
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

    // 2. Update global_prompt widget if provided
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

    // 3. Update total_frames widget if duration is provided
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

    // 4. Live refresh MiniMaxH3MotionDirectorEditor if instantiated
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

            const mode = ed.getDirectorMode ? ed.getDirectorMode() : (ed._directorMode || "mixed");
            ed._directorMode = mode;
            if (mode === "mixed") {
                // mixed mode owns editor.mixedTimeline
            } else if (mode === "video") {
                ed.restoreVideoFromTimeline?.();
            } else {
                ed.ensureGenTimeline?.();
            }

            ed.ensureContextLinks?.();
            ed.applyTaskLayout?.(mode);
            ed.populateTaskSelect?.(ed.globalTask, ed.taskTypeWidget?.value);
            ed.setEditMode?.(ed.timeline?.editMode || "global");
            ed.selectedIndex = 0;
            ed.updateSelectionUI?.();
            ed.commit?.(true, { syncTimeline: false });
            ed._externalGroupsSyncSig = null;
            ed.syncExternalGroupsTimeline?.();
            ed.scheduleRender?.();
        } catch (e) {
            console.warn("[StoryDirector] Live Editor sync error:", e);
        }
    }

    // 5. Refresh timeline editor controller if open
    if (directorNode._directorTimelineController?.timelineWidget) {
        try {
            directorNode._directorTimelineController.timelineWidget.value = timelineJson;
            directorNode._directorTimelineController.syncExternalGroupsTimeline?.();
        } catch (e) {
            console.warn("[StoryDirector] Controller sync error:", e);
        }
    }

    return true;
}

function openDirectorModal(directorNode) {
    if (!directorNode) return false;

    // Direct modal controller
    const ed = directorNode._minimaxEditor;
    if (ed?._directorModalController) {
        if (!ed._directorModalController.isOpen) {
            ed._directorModalController.open();
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

        const payload = {
            node_id: String(storyNode.id),
            story: String(getVal("story", "")),
            total_duration_sec: Number(getVal("total_duration_sec", 15.0)),
            shot_duration_sec: Number(getVal("shot_duration_sec", 5.0)),
            aspect_ratio: String(getVal("aspect_ratio", "16:9 (864x480)")),
            director_mode: String(getVal("director_mode", "mixed")),
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

        const directors = findConnectedDirectors(storyNode);
        let syncCount = 0;
        for (const dir of directors) {
            if (applyTimelineToDirector(dir, data.timeline_data, data.prompt, payload.total_duration_sec)) {
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
                const directors = findConnectedDirectors(node);
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

                if (timelineStr && typeof timelineStr === "string") {
                    this._lastGeneratedTimeline = timelineStr;
                    this._lastGeneratedPrompt = promptStr;
                    const directors = findConnectedDirectors(this);
                    for (const dir of directors) {
                        applyTimelineToDirector(dir, timelineStr, promptStr);
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
                const storyNode = graph.getNodeById?.(storyNodeId);
                if (storyNode) {
                    storyNode._lastGeneratedTimeline = detail.timeline_data;
                    storyNode._lastGeneratedPrompt = detail.prompt;
                    targetDirectors = findConnectedDirectors(storyNode);
                }
            }

            // Fallback: update any MotionDirector connected to any StoryDirector
            if (!targetDirectors.length) {
                const allStoryNodes = (graph._nodes || []).filter(
                    (n) => String(n.comfyClass || n.type || "") === STORY_DIRECTOR_CLASS
                );
                for (const sn of allStoryNodes) {
                    targetDirectors.push(...findConnectedDirectors(sn));
                }
            }

            for (const dir of targetDirectors) {
                applyTimelineToDirector(dir, detail.timeline_data, detail.prompt, detail.total_duration);
            }
        });

        // Listen to director execution ingestion
        api.addEventListener("minimax_storyboard_applied", (event) => {
            const detail = event?.detail;
            if (!detail?.node_id || !detail?.timeline_data) return;

            const graph = app.graph;
            if (!graph) return;

            const directorNode = graph.getNodeById?.(detail.node_id);
            if (directorNode) {
                applyTimelineToDirector(directorNode, detail.timeline_data, detail.prompt);
            }
        });
    },
});
