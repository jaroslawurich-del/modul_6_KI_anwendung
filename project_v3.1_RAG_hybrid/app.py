from config.logging_config import setup_logging

from ui.layout import run_app


def main():

    logger = setup_logging()

    logger.info("Starte KI Dokumentenassistent ...")

    run_app()


if __name__ == "__main__":
    main()