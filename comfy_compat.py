"""Compatibility helpers for both legacy and V3 ComfyUI installations."""

from __future__ import annotations


try:
    from comfy_api.latest import ComfyExtension, io

    HAS_COMFY_V3 = True
except (ImportError, ModuleNotFoundError):
    # Older ComfyUI releases do not provide comfy_api.  The node classes still
    # expose their legacy INPUT_TYPES/FUNCTION/RETURN_TYPES contract, but they
    # need small placeholders so their V3 schema methods can be defined safely.
    HAS_COMFY_V3 = False

    class ComfyExtension:
        pass

    class _NodeOutput(tuple):
        def __new__(cls, *values):
            instance = super().__new__(cls, values)
            instance.result = values
            return instance

    class _ComfyNode:
        pass

    class _Schema:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class _Input:
        @classmethod
        def Input(cls, name, **kwargs):
            return {"name": name, "type": cls.__name__, **kwargs}

    class _Output:
        @classmethod
        def Output(cls, name, **kwargs):
            return {"name": name, "type": cls.__name__, **kwargs}

    class _Image(_Input, _Output):
        pass

    class _Mask(_Input, _Output):
        pass

    class _Boolean(_Input):
        pass

    class _Combo(_Input):
        pass

    class _Int(_Input):
        pass

    class _Float(_Input):
        pass

    class _NumberDisplay:
        slider = "slider"

    class _IO:
        ComfyNode = _ComfyNode
        NodeOutput = _NodeOutput
        Schema = _Schema
        Image = _Image
        Mask = _Mask
        Boolean = _Boolean
        Combo = _Combo
        Int = _Int
        Float = _Float
        NumberDisplay = _NumberDisplay

    io = _IO()
