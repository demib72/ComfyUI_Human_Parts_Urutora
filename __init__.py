__all__ = ["HumanParts", "HumanPartsUltra"]

from .nodes import HumanParts
from .nodes_ultra import HumanPartsUltra

NODE_CLASS_MAPPINGS = {
    "HumanParts": HumanParts,
    # Preserve the exact class identifier used by existing LayerStyle workflows.
    "LayerMask: HumanPartsUltra": HumanPartsUltra,
    "HumanPartsUltra": HumanPartsUltra,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "HumanParts": "🧍 Human Parts mask generator",
    "LayerMask: HumanPartsUltra": "🧍 Human Parts Ultra",
    "HumanPartsUltra": "🧍 Human Parts Ultra",
}
