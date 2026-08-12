import { app } from "../../scripts/app.js";

const NODE_IDS = new Set(["HumanPartsUltra", "LayerMask: HumanPartsUltra"]);
const DETAIL_METHODS = new Set([
    "VITMatte",
    "VITMatte(local)",
    "vitmatte-base-composition-1k",
    "PyMatting",
    "GuidedFilter",
]);

// Original LayerStyle order, before the three anatomy controls existed.
const LEGACY_ORDER = [
    "face",
    "hair",
    "glasses",
    "top_clothes",
    "bottom_clothes",
    "torso_skin",
    "left_arm",
    "right_arm",
    "left_leg",
    "right_leg",
    "left_foot",
    "right_foot",
    "detail_method",
    "detail_erode",
    "detail_dilate",
    "black_point",
    "white_point",
    "process_detail",
    "device",
    "max_megapixels",
];

// The first implementation appended the anatomy controls for compatibility.
const APPENDED_SCHEMA_ORDER = [
    ...LEGACY_ORDER,
    "eyes",
    "breasts",
    "groin",
];

// Current schema and desired visual order.
const INLINE_SCHEMA_ORDER = [
    "face",
    "eyes",
    "hair",
    "glasses",
    "top_clothes",
    "bottom_clothes",
    "torso_skin",
    "breasts",
    "groin",
    "left_arm",
    "right_arm",
    "left_leg",
    "right_leg",
    "left_foot",
    "right_foot",
    "detail_method",
    "detail_erode",
    "detail_dilate",
    "black_point",
    "white_point",
    "process_detail",
    "device",
    "max_megapixels",
];

function valuesByName(order, values) {
    return new Map(order.map((name, index) => [name, values[index]]));
}

function normalizeSerializedValues(values) {
    if (!Array.isArray(values)) return values;
    if (DETAIL_METHODS.has(values[15])) return values;
    if (!DETAIL_METHODS.has(values[12])) return values;

    const sourceOrder = values.length >= APPENDED_SCHEMA_ORDER.length
        ? APPENDED_SCHEMA_ORDER
        : LEGACY_ORDER;
    const oldValues = valuesByName(sourceOrder, values);
    return INLINE_SCHEMA_ORDER.map((name) => oldValues.get(name) ?? false);
}

function applySerializedValues(node, values) {
    if (!Array.isArray(values)) return;
    const widgets = new Map((node.widgets ?? []).map((widget) => [widget.name, widget]));
    INLINE_SCHEMA_ORDER.forEach((name, index) => {
        const widget = widgets.get(name);
        if (widget && values[index] !== undefined) widget.value = values[index];
    });
}

app.registerExtension({
    name: "HumanPartsUrutora.workflow-value-migration",

    beforeConfigureGraph(workflow) {
        for (const node of workflow?.nodes ?? []) {
            if (!NODE_IDS.has(node.type)) continue;
            node.widgets_values = normalizeSerializedValues(node.widgets_values);
        }
    },

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!NODE_IDS.has(nodeData.name)) return;

        const originalConfigure = nodeType.prototype.onConfigure;

        nodeType.prototype.onConfigure = function (info) {
            let normalizedValues;
            if (Array.isArray(info?.widgets_values)) {
                normalizedValues = normalizeSerializedValues(info.widgets_values);
                info.widgets_values = normalizedValues;
            }
            const result = originalConfigure?.apply(this, arguments);
            // LiteGraph versions differ on whether widget values are applied
            // before or during onConfigure. Name-based assignment covers both.
            applySerializedValues(this, normalizedValues);
            return result;
        };
    },
});
