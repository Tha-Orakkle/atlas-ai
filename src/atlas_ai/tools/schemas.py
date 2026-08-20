current_time_tool = {
    "type": "function",
    "name": "get_current_time",
    "description": (
        "Get the current time for a location using an IANA time zone "
        "identifier such as Africa/Lagos or America/New_York. "
        "Defaults to Africa/Lagos when no timezone is provided."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "timezone": {
                "type": ["string", "null"],
                "description": (
                    "An IANA time zone identifier, "
                    "e.g. Africa/Lagos or America/New_York."
                )
            }
        },
        "required": ["timezone"],
        "additionalProperties": False
    },
    "strict": True
}
