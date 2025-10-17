# app/correlation_scheduler.py
import logging
import threading
import time
from datetime import datetime, timedelta
from typing import List

from .db import get_session, Interaction

logger = logging.getLogger(__name__)


class CorrelationScheduler:
    """
    Background scheduler to periodically recompute user correlations so the taste graph
    always leverages fresh similarity data.
    """

    def __init__(self, taste_graph_manager, interval_minutes: int = 60):
        self.taste_graph_manager = taste_graph_manager
        self.interval_minutes = interval_minutes
        self.is_running = False
        self.scheduler_thread = None

    def start(self):
        if self.is_running:
            logger.warning("Correlation scheduler already running")
            return
        self.is_running = True
        self.scheduler_thread = threading.Thread(target=self._run, daemon=True)
        self.scheduler_thread.start()
        logger.info(f"Correlation scheduler started (every {self.interval_minutes} minutes)")

    def stop(self):
        self.is_running = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
        logger.info("Correlation scheduler stopped")

    def _run(self):
        while self.is_running:
            try:
                self._process()
            except Exception as e:
                logger.error(f"Correlation scheduler error: {e}")
            time.sleep(self.interval_minutes * 60)

    def _process(self):
        logger.info("Correlation scheduler cycle start")
        session = get_session()
        cutoff = datetime.utcnow() - timedelta(days=7)

        # Users with recent interactions
        users_with_recent = session.session.query(Interaction.user_id).\
            filter(Interaction.created_at >= cutoff).distinct().all()
        user_ids: List[str] = [row[0] for row in users_with_recent]

        for user_id in user_ids:
            try:
                self.taste_graph_manager.compute_correlations_for_user(user_id)
            except Exception as e:
                logger.warning(f"Failed to recompute correlations for {user_id}: {e}")

        logger.info("Correlation scheduler cycle complete")

    def get_status(self):
        return {
            'is_running': self.is_running,
            'interval_minutes': self.interval_minutes,
            'thread_alive': self.scheduler_thread.is_alive() if self.scheduler_thread else False
        }


_correlation_scheduler = None


def initialize_correlation_scheduler(taste_graph_manager, interval_minutes: int = 60):
    global _correlation_scheduler
    if _correlation_scheduler is None:
        _correlation_scheduler = CorrelationScheduler(taste_graph_manager, interval_minutes)
        _correlation_scheduler.start()
    else:
        logger.warning("Correlation scheduler already initialized")
    return _correlation_scheduler


def get_correlation_scheduler():
    return _correlation_scheduler


def stop_correlation_scheduler():
    global _correlation_scheduler
    if _correlation_scheduler:
        _correlation_scheduler.stop()
        _correlation_scheduler = None
        logger.info("Correlation scheduler stopped")


