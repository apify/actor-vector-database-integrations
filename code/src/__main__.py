import asyncio
import contextlib
import logging

from apify.log import ActorLogFormatter

from .entrypoint import main

handler = logging.StreamHandler()
handler.setFormatter(ActorLogFormatter())

apify_client_logger = logging.getLogger("apify_client")
apify_client_logger.setLevel(logging.INFO)
apify_client_logger.addHandler(handler)

apify_logger = logging.getLogger("apify")
apify_logger.setLevel(logging.DEBUG)
apify_logger.addHandler(handler)

# Bug in apify 2.x: on actor exit, the platform closes the WebSocket without sending a proper
# close frame, causing a race condition in event_manager._process_platform_messages that results
# in InvalidStateError when setting result on an already-resolved Future.
# The actor exits successfully (exit_code 0) before this error occurs, so it is safe to suppress.
# Remove this workaround when upgrading apify SDK to 3.x
with contextlib.suppress(asyncio.InvalidStateError):
    asyncio.run(main())
