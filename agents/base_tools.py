from typing import Any, Callable, Dict, List, Optional, TypedDict, Union


class ToolTypeArgs(TypedDict, total=False):
    """Tool type arguments definition."""
    type: str
    description: str


class ToolType:
    """A tool type definition."""

    def __init__(
        self,
        name: str,
        description: str,
        function: Callable,
        args: Dict[str, Dict[str, Any]],
    ):
        self.name = name
        self.description = description
        self.function = function
        self.args = args


ToolTypeList = List[ToolType]


class Tool:
    """Base class for tools."""

    def __init__(self, tool_types: ToolTypeList):
        self.tool_types = tool_types 