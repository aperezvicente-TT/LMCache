# SPDX-License-Identifier: Apache-2.0
"""Shared client heartbeat for LMCache multiprocess integrations."""

# Standard
from collections.abc import Callable
import threading

# First Party
from lmcache.logging import init_logger
from lmcache.v1.periodic_thread import PeriodicThread, ThreadLevel, ThreadRunSummary
from lmcache.v1.transport.request_client import RequestClient

logger = init_logger(__name__)


def send_ping(
    req_client: RequestClient,
    timeout: float,
    instance_id: int | None = None,
) -> bool:
    """Send one MP server health probe.

    Args:
        req_client: Request client used to send the probe.
        timeout: Maximum seconds to wait for the response.
        instance_id: Worker registration or scheduler subscription to refresh,
            or ``None`` for an untracked probe.

    Returns:
        ``True`` when the server responds healthy, otherwise ``False``.
    """
    try:
        return req_client.ping(instance_id).result(timeout=timeout)
    except TimeoutError:
        return False
    except Exception:
        logger.debug("LMCache heartbeat ping failed", exc_info=True)
        return False


class HeartbeatThread(PeriodicThread):
    """Probe an MP server and publish health transitions to an adapter.

    The recover callback runs before health is published, allowing an adapter
    to restore server-side registration before new work is admitted. Adapters
    retain ownership of when the heartbeat starts and how recovery is done.

    Recovery runs on its own thread while the heartbeat keeps pinging, so a
    slow re-registration (for example re-mapping a large shared-memory pool)
    does not silence the pings that keep the worker's server-side
    registration alive.
    """

    def __init__(
        self,
        req_client: RequestClient,
        health_event: threading.Event,
        interval: float = 10.0,
        instance_id: int | None = None,
        *,
        name: str = "lmcache-heartbeat",
    ) -> None:
        """Initialize an MP heartbeat.

        Args:
            req_client: Request client used to send PING requests.
            health_event: Adapter-owned event that represents server health.
            interval: Seconds between probes and the timeout for each probe.
            instance_id: Worker registration or scheduler subscription to
                refresh, or ``None`` for an untracked probe.
            name: Background thread name.
        """
        super().__init__(
            name=name,
            interval=interval,
            level=ThreadLevel.CRITICAL,
        )
        self._req_client = req_client
        self._health_event = health_event
        self._instance_id = instance_id
        self._recover_callback: Callable[[], bool] = lambda: True
        self._unhealthy_callback: Callable[[], None] = lambda: None
        self._healthy_callback: Callable[[], bool] = self._set_health_event
        # The running recovery thread, if any, and a count of finished
        # recoveries so that a ping that raced one is not misread.
        self._recovery_lock = threading.Lock()
        self._recovery: threading.Thread | None = None
        self._recoveries_finished = 0

    def register_recover_callback(self, callback: Callable[[], bool]) -> None:
        """Register work required before publishing a recovered server.

        The callback runs on a separate thread, while the heartbeat keeps
        pinging, after a successful ping that follows an unhealthy state
        (including one entered through :meth:`request_recovery`).

        Args:
            callback: Function returning whether recovery succeeded.
        """
        self._recover_callback = callback

    def request_recovery(self) -> None:
        """Enter degraded mode and recover on the next successful ping.

        For callers that learn from a failed request that the server lost
        this client's registration (for example after the server reaped it)
        while pings still succeed. Wakes the heartbeat so recovery starts
        without waiting for the next interval.
        """
        if self._health_event.is_set():
            logger.warning(
                "LMCache server lost this client's registration — "
                "entering degraded mode"
            )
            self._publish_unhealthy()
        self.wake()

    def register_unhealthy_callback(self, callback: Callable[[], None]) -> None:
        """Register a callback for a healthy-to-unhealthy transition.

        Args:
            callback: Function that publishes adapter-specific unhealthy state.
        """
        self._unhealthy_callback = callback

    def register_healthy_callback(self, callback: Callable[[], bool]) -> None:
        """Register a callback that publishes adapter-specific healthy state.

        Args:
            callback: Function returning whether health was published.
        """
        self._healthy_callback = callback

    def stop(self, timeout: float = 5.0) -> None:
        """Stop pinging and wait up to ``timeout`` for a running recovery.

        Args:
            timeout: Maximum seconds to wait for each of the heartbeat and
                recovery threads.
        """
        super().stop(timeout=timeout)
        self._join_recovery(timeout)

    def stop_and_wait(self) -> None:
        """Stop and wait for an in-flight callback without a join timeout."""
        self.stop()
        thread = self._thread
        if (
            thread is not None
            and thread is not threading.current_thread()
            and thread.is_alive()
        ):
            thread.join()
        self._join_recovery(None)

    def _set_health_event(self) -> bool:
        self._health_event.set()
        return True

    def _publish_unhealthy(self) -> None:
        try:
            self._unhealthy_callback()
        finally:
            self._health_event.clear()

    def _join_recovery(self, timeout: float | None) -> None:
        with self._recovery_lock:
            recovery = self._recovery
        if recovery is not None and recovery is not threading.current_thread():
            recovery.join(timeout=timeout)

    def _execute(self) -> ThreadRunSummary:
        """Run one heartbeat cycle: ping, then publish or start recovery.

        A cycle that observes a stop request returns without starting
        recovery or touching health. While a recovery runs, the cycle only
        pings; the recovery thread publishes health when it finishes.
        """
        recoveries_before = self._recoveries_finished
        healthy = send_ping(
            self._req_client,
            timeout=self.interval,
            instance_id=self._instance_id,
        )
        if self.stop_requested:
            return ThreadRunSummary(
                success=True,
                message="stop requested; skipping health update",
            )

        with self._recovery_lock:
            if self._recovery is not None:
                return ThreadRunSummary(success=True, message="recovering")
            if self._recoveries_finished != recoveries_before:
                # The ping may predate the registration that just finished.
                return ThreadRunSummary(success=True, message="recovered")
            was_healthy = self._health_event.is_set()
            if healthy and not was_healthy:
                logger.warning(
                    "LMCache server is healthy again, triggering recovery callback"
                )
                self._recovery = threading.Thread(
                    target=self._run_recovery,
                    name=f"{self.name}-recovery",
                    daemon=True,
                )
                self._recovery.start()
                return ThreadRunSummary(success=True, message="recovering")

        if not healthy and was_healthy:
            logger.warning("LMCache server is unhealthy — entering degraded mode")
            self._publish_unhealthy()

        return ThreadRunSummary(
            success=True,
            message="healthy" if healthy else "unhealthy",
        )

    def _run_recovery(self) -> None:
        """Run the recover callback, then publish health if it succeeded."""
        try:
            healthy = self._recover_callback()
        except Exception:
            logger.exception("LMCache heartbeat recovery callback failed")
            healthy = False
        # A result that lands after stop() must not publish health.
        if healthy and not self.stop_requested:
            try:
                healthy = self._healthy_callback()
            except Exception:
                logger.exception("LMCache heartbeat healthy callback failed")
                healthy = False
            if healthy:
                logger.warning(
                    "LMCache server is healthy again — resuming normal operation"
                )
        with self._recovery_lock:
            self._recovery = None
            self._recoveries_finished += 1
