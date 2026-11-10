"""Saying where the app is, for a Home Assistant that is not its Supervisor.

An app installed as an app is found through the Supervisor, which knows
where it put it. Anything else -- the standalone image, a checkout being
worked on, a Home Assistant on another machine -- has to be told, and this
is how it tells.

Only the address is advertised. Home Assistant asks ``GET /info`` for
everything it actually needs, so nothing here has to be kept in step with
what the app can do.
"""

import logging
import socket

from zeroconf import IPVersion
from zeroconf.asyncio import AsyncServiceInfo, AsyncZeroconf

from . import NAME, __version__
from .protocol import API_VERSION

_LOGGER = logging.getLogger(__name__)

SERVICE_TYPE = "_voice-support._tcp.local."
"""Named after the integration that looks for it, the way _wyoming and
_esphomelib are: what hosts the code may move, what it speaks will not."""


class Advertiser:
    """The app's address on the local network, for as long as it is up."""

    def __init__(self, port: int, name: str | None = None) -> None:
        """Advertise this port, under a name unique to this machine."""
        self.port = port
        self._hostname = socket.gethostname()
        self._name = name or f"{NAME} ({self._hostname})"
        self._zeroconf: AsyncZeroconf | None = None
        self._info: AsyncServiceInfo | None = None

    async def start(self) -> None:
        """Begin answering for the service, or carry on without it."""
        try:
            self._zeroconf = AsyncZeroconf(ip_version=IPVersion.V4Only)
            self._info = AsyncServiceInfo(
                SERVICE_TYPE,
                f"{self._name}.{SERVICE_TYPE}",
                port=self.port,
                server=f"{self._hostname}.local.",
                properties={
                    "version": __version__,
                    "api_version": str(API_VERSION),
                },
            )
            await self._zeroconf.async_register_service(self._info)
        except OSError as err:
            # A container without host networking has nowhere to send this,
            # and an app nobody can find is still an app that works when it
            # is told where it is.
            _LOGGER.debug("Not advertising on the network (%s)", err)
            await self.stop()
            return
        _LOGGER.info("Advertising as %s on port %d", self._name, self.port)

    async def stop(self) -> None:
        """Stop answering, and take the name back."""
        if (self._zeroconf is not None) and (self._info is not None):
            try:
                await self._zeroconf.async_unregister_service(self._info)
            except OSError:
                pass
        if self._zeroconf is not None:
            await self._zeroconf.async_close()
        self._zeroconf = None
        self._info = None
