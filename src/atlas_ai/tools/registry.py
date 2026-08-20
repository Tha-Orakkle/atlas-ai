from atlas_ai.tools.definitions import get_current_time
from atlas_ai.tools.schemas import current_time_tool
from atlas_ai.tools.calculator import calculate_tool, calculate

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
