"""Entry point for the voice chat assistant."""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="チチクサ - Japanese Voice Chat Assistant")
    parser.add_argument(
        "--config", "-c", default="config.yaml", help="Path to configuration file"
    )
    parser.add_argument(
        "--list-devices", action="store_true", help="List available audio devices"
    )
    parser.add_argument(
        "--list-voices", action="store_true", help="List available Japanese TTS voices"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable debug logging"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )

    if args.list_devices:
        import sounddevice as sd

        print(sd.query_devices())
        return

    if args.list_voices:
        import subprocess

        result = subprocess.run(
            ["say", "-v", "?"], capture_output=True, text=True, check=True
        )
        for line in result.stdout.splitlines():
            if "ja_JP" in line:
                print(line)
        return

    from ccxsa.app import VoiceChatApp
    from ccxsa.config import AppConfig

    config = AppConfig.load(args.config)
    app = VoiceChatApp(config)

    loop = asyncio.new_event_loop()

    def _shutdown() -> None:
        loop.create_task(app.shutdown())

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _shutdown)

    try:
        loop.run_until_complete(app.run())
    finally:
        loop.close()


if __name__ == "__main__":
    main()
