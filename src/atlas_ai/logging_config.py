import logging


def configure_logging() -> None:
    """Defines logging configuration"""

    logging.basicConfig(
        level=logging.INFO,
        filename="app.log",
        format=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(name)s | "
            "%(message)s"
        )
    )
