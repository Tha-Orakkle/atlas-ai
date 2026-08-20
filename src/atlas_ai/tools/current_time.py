import requests

TIME_URL = "https://timeapi.io/api/v1/time/current/zone"

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


def get_current_time(timezone: str | None = None) -> dict[str, str]:
    """
    Get the current time for a given timezone.
    Args:
        timezone (str): IANA timezone identifier e.g. Europe/Berlin.
    Returns:
        dict: A dictionary containing the current date, time, day of the week,
              timezone, and UTC offset in seconds or an error message.
    """
    timezone = timezone or "Africa/Lagos"

    try:
        response = requests.get(
            url=TIME_URL,
            params={"timezone": timezone},
            timeout=10
        )
        response.raise_for_status()
        data = response.json()

        return {
            "date": data["date"],
            "time": data["time"],
            "day_of_week": data["day_of_week"],
            "timezone": data["timezone"],
            "utc_offset_seconds": data["utc_offset_seconds"],
        }

    except requests.RequestException as exc:
        return {
            "error": f"Unable to retrieve the current time: {exc}"
        }
