import { app } from "../../scripts/app.js";

const NODE_IDS = new Set(["HumanPartsUrutora", "LayerMask: HumanPartsUrutora"]);
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

// Previous inline schema, before breast/groin were replaced with face skin.
const PREVIOUS_INLINE_SCHEMA_ORDER = [
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

// Skin-only schema, before all preserved facial features became selectable.
const SKIN_ONLY_SCHEMA_ORDER = [
    "face",
    "face_skin",
    "eyes",
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

// Current schema and desired visual order.
const INLINE_SCHEMA_ORDER = [
    "face",
    "face_skin",
    "eyebrows",
    "eyes",
    "nose",
    "mouth",
    "lips",
    "ears",
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

function valuesByName(order, values) {
    return new Map(order.map((name, index) => [name, values[index]]));
}

function normalizeSerializedValues(values) {
    if (!Array.isArray(values)) return values;
    if (DETAIL_METHODS.has(values[19])) return values;

    let sourceOrder;
    if (DETAIL_METHODS.has(values[14])) {
        sourceOrder = SKIN_ONLY_SCHEMA_ORDER;
    } else if (DETAIL_METHODS.has(values[15])) {
        sourceOrder = PREVIOUS_INLINE_SCHEMA_ORDER;
    } else if (DETAIL_METHODS.has(values[12])) {
        sourceOrder = values.length >= APPENDED_SCHEMA_ORDER.length
            ? APPENDED_SCHEMA_ORDER
            : LEGACY_ORDER;
    } else {
        return values;
    }
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
