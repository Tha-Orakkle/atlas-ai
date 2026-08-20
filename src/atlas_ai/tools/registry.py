from atlas_ai.tools.definitions import get_current_time
from atlas_ai.tools.schemas import current_time_tool

TOOLS = {
    "get_current_time": {
        "schema": current_time_tool,
        "function": get_current_time
    }
}
