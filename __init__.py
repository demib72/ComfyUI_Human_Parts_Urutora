from comfy_api.latest import ComfyExtension, io

from .nodes import HumanParts
from .nodes_ultra import HumanPartsUltra, LayerStyleHumanPartsUltra

__all__ = ["HumanParts", "HumanPartsUltra", "comfy_entrypoint"]
WEB_DIRECTORY = "./js"


class HumanPartsExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [HumanParts, LayerStyleHumanPartsUltra, HumanPartsUltra]


async def comfy_entrypoint() -> HumanPartsExtension:
    return HumanPartsExtension()
