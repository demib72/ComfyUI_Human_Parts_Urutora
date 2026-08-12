from .comfy_compat import ComfyExtension, io

from .nodes import HumanPartsUrutoraMaskGenerator
from .nodes_urutora import HumanPartsUrutora, LayerStyleHumanPartsUrutora

__all__ = [
    "HumanPartsUrutoraMaskGenerator",
    "HumanPartsUrutora",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
    "comfy_entrypoint",
]
WEB_DIRECTORY = "./js"

# Always publish the traditional mappings. Some transitional ComfyUI builds
# expose comfy_api.latest but still discover external custom nodes through this
# contract. Current builds can use comfy_entrypoint below.
NODE_CLASS_MAPPINGS = {
    "HumanPartsUrutoraMaskGenerator": HumanPartsUrutoraMaskGenerator,
    "LayerMask: HumanPartsUrutora": LayerStyleHumanPartsUrutora,
    "HumanPartsUrutora": HumanPartsUrutora,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "HumanPartsUrutoraMaskGenerator": "🧍 Human Parts Urutora mask generator",
    "LayerMask: HumanPartsUrutora": "🧍 Human Parts Urutora",
    "HumanPartsUrutora": "🧍 Human Parts Urutora",
}


class HumanPartsUrutoraExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            HumanPartsUrutoraMaskGenerator,
            LayerStyleHumanPartsUrutora,
            HumanPartsUrutora,
        ]


async def comfy_entrypoint() -> HumanPartsUrutoraExtension:
    return HumanPartsUrutoraExtension()
