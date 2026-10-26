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
        syncButton: "🎬 Synchroniser avec le Director",
        syncSuccess: "✨ Storyboard synchronisé avec le Motion Director !",
        syncNoTarget: "⚠️ Aucun Motion Director connecté sur le socket 'storyboard'.",
        connectedDirector: "Connecté au Director : ",
        scenesCount: "scènes",
        duration: "durée",
        statusReady: "Prêt à générer le scénario",
        timelineUpdated: "✨ Timeline générée et injectée dans le Motion Director !",
    },
    en: {
        syncButton: "🎬 Sync with Motion Director",
        syncSuccess: "✨ Storyboard synced to Motion Director!",
        syncNoTarget: "⚠️ No Motion Director connected to 'storyboard' output.",
        connectedDirector: "Connected to Director: ",
        scenesCount: "scenes",
        duration: "duration",
        statusReady: "Ready to generate storyboard",
        timelineUpdated: "✨ Timeline generated and applied to Motion Director!",
    },
    zh: {
        syncButton: "🎬 同步到 Director 导演节点",
        syncSuccess: "✨ 分镜与时间轴已成功同步到 Motion Director！",
        syncNoTarget: "⚠️ 未在 'storyboard' 输出端连接 Motion Director 节点。",
        connectedDirector: "已连接 Director 节点：",
        scenesCount: "镜头",
        duration: "时长",
        statusReady: "分镜剧本生成器就绪",
        timelineUpdated: "✨ 分镜时间轴已成功生成并注入到 Motion Director！",
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

    if (!storyboardOutput?.links) return [];

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
        if (targetClass.includes("MiniMaxH3MotionDirector")) {
            directors.push(targetNode);
        }
    }

    return directors;
}

function applyTimelineToDirector(directorNode, timelineJson) {
    if (!directorNode || !timelineJson) return false;

    const timelineWidget = (directorNode.widgets || []).find((w) => w?.name === "timeline_data");
    if (!timelineWidget) return false;

    const oldValue = timelineWidget.value;
    timelineWidget.value = timelineJson;

    if (typeof timelineWidget.callback === "function") {
        try {
            timelineWidget.callback(timelineJson);
        } catch (e) {
            console.warn("timelineWidget callback error:", e);
        }
    }

    if (typeof directorNode.onWidgetChanged === "function") {
        try {
            directorNode.onWidgetChanged("timeline_data", timelineJson, oldValue, timelineWidget);
        } catch (e) {
            console.warn("directorNode onWidgetChanged error:", e);
        }
    }

    // Refresh timeline editor modal if already open
    if (directorNode._directorTimelineController?.timelineWidget) {
        try {
            directorNode._directorTimelineController.timelineWidget.value = timelineJson;
            directorNode._directorTimelineController.syncExternalGroupsTimeline?.();
        } catch (e) {
            console.warn("Controller sync error:", e);
        }
    }

    return true;
}

app.registerExtension({
    name: "MiniMaxH3.MotionDirector.StoryDirector",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== STORY_DIRECTOR_CLASS) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const res = onNodeCreated?.apply(this, arguments);

            // Add manual sync button
            this.addWidget(
                "button",
                t("syncButton"),
                null,
                () => {
                    const directors = findConnectedDirectors(this);
                    if (!directors.length) {
                        alert(t("syncNoTarget"));
                        return;
                    }

                    // Check if node has cached timeline data
                    if (this._lastGeneratedTimeline) {
                        let count = 0;
                        for (const dir of directors) {
                            if (applyTimelineToDirector(dir, this._lastGeneratedTimeline)) {
                                count++;
                            }
                        }
                        if (count > 0) {
                            alert(t("syncSuccess"));
                        }
                    } else {
                        alert(t("syncSuccess") + "\n(Le storyboard sera généré et transféré lors de l'exécution du workflow)");
                    }
                },
                { serialize: false }
            );

            return res;
        };

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            const res = onExecuted?.apply(this, arguments);

            if (message?.timeline_json) {
                const timelineStr = Array.isArray(message.timeline_json)
                    ? message.timeline_json[0]
                    : message.timeline_json;

                if (timelineStr && typeof timelineStr === "string") {
                    this._lastGeneratedTimeline = timelineStr;
                    const directors = findConnectedDirectors(this);
                    for (const dir of directors) {
                        applyTimelineToDirector(dir, timelineStr);
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

            // Update matching or all connected Director nodes
            const storyNodeId = detail.node_id;
            let targetDirectors = [];

            if (storyNodeId) {
                const storyNode = graph.getNodeById?.(storyNodeId);
                if (storyNode) {
                    storyNode._lastGeneratedTimeline = detail.timeline_data;
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
                applyTimelineToDirector(dir, detail.timeline_data);
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
                applyTimelineToDirector(directorNode, detail.timeline_data);
            }
        });
    },
});
