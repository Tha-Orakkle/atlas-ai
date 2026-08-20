from atlas_ai.tools.calculator import calculate_tool, calculate
from atlas_ai.tools.current_time import current_time_tool, get_current_time

TOOLS = {
    "get_current_time": {
        "schema": current_time_tool,
        "function": get_current_time
    },
    "calculate": {
        "schema": calculate_tool,
        "function": calculate
    }
}
