# app/auto_board_scheduler.py
import asyncio
import logging
from typing import List, Dict
from datetime import datetime, timedelta
import threading
import time

from app.db import get_session, User, Interaction, Board

logger = logging.getLogger(__name__)

class AutoBoardScheduler:
    """
    Background scheduler for automatic auto-board creation and updates.
    Runs periodically to manage auto-boards for all users.
    """
    
    def __init__(self, taste_graph_manager, check_interval_minutes: int = 30):
        self.taste_graph_manager = taste_graph_manager
        self.check_interval_minutes = check_interval_minutes
        self.is_running = False
        self.scheduler_thread = None
        
    def start(self):
        """Start the background scheduler"""
        if self.is_running:
            logger.warning("Auto-board scheduler is already running")
            return
        
        self.is_running = True
        self.scheduler_thread = threading.Thread(target=self._run_scheduler, daemon=True)
        self.scheduler_thread.start()
        logger.info(f"Auto-board scheduler started (checking every {self.check_interval_minutes} minutes)")
    
    def stop(self):
        """Stop the background scheduler"""
        self.is_running = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
        logger.info("Auto-board scheduler stopped")
    
    def _run_scheduler(self):
        """Main scheduler loop"""
        while self.is_running:
            try:
                self._process_auto_boards()
            except Exception as e:
                logger.error(f"Error in auto-board scheduler: {e}")
            
            # Sleep for the specified interval
            time.sleep(self.check_interval_minutes * 60)
    
    def _process_auto_boards(self):
        """Process auto-boards for all users"""
        logger.info("Starting auto-board processing cycle")
        
        session = get_session()
        
        # Get users who need auto-board processing
        users_to_process = self._get_users_for_auto_board_processing(session)
        
        logger.info(f"Processing auto-boards for {len(users_to_process)} users")
        
        for user_id in users_to_process:
            try:
                self._process_user_auto_boards(user_id)
            except Exception as e:
                logger.error(f"Failed to process auto-boards for user {user_id}: {e}")
        
        logger.info("Auto-board processing cycle completed")
    
    def _get_users_for_auto_board_processing(self, session) -> List[str]:
        """Get users who need auto-board processing"""
        # Get users with recent interactions (last 7 days)
        cutoff_date = datetime.utcnow() - timedelta(days=7)
        
        # Get users who have interactions but no auto-boards, or users with old auto-boards
        users_with_interactions = session.session.query(Interaction.user_id).filter(
            Interaction.created_at >= cutoff_date
        ).distinct().all()
        
        user_ids = [row[0] for row in users_with_interactions]
        
        # Filter users who need auto-board processing
        users_to_process = []
        for user_id in user_ids:
            if self._should_process_user_auto_boards(session, user_id):
                users_to_process.append(user_id)
        
        return users_to_process
    
    def _should_process_user_auto_boards(self, session, user_id: str) -> bool:
        """Check if a user needs auto-board processing"""
        # Check if user has enough interactions for auto-boards
        if not self.taste_graph_manager.auto_board_creator.is_user_ready_for_auto_boards(user_id):
            return False
        
        # Check if user has any auto-boards
        auto_boards = session.get_user_boards(user_id, board_type='auto')
        
        if not auto_boards:
            # User has no auto-boards - create them
            return True
        
        # Check if auto-boards need updating (last update was more than 3 days ago)
        for board in auto_boards:
            if board.last_activity and board.last_activity < datetime.utcnow() - timedelta(days=3):
                return True
        
        # Check if user has new interactions since last auto-board update
        latest_board_activity = max([board.last_activity for board in auto_boards if board.last_activity])
        if latest_board_activity:
            recent_interactions = session.session.query(Interaction).filter(
                Interaction.user_id == user_id,
                Interaction.created_at > latest_board_activity
            ).count()
            
            if recent_interactions >= 5:  # 5 new interactions trigger update
                return True
        
        return False
    
    def _process_user_auto_boards(self, user_id: str):
        """Process auto-boards for a specific user"""
        logger.info(f"Processing auto-boards for user {user_id}")
        
        try:
            # Check if user has existing auto-boards
            session = get_session()
            existing_auto_boards = session.get_user_boards(user_id, board_type='auto')
            
            if existing_auto_boards:
                # Update existing auto-boards
                result = self.taste_graph_manager.update_auto_boards_for_user(user_id)
                logger.info(f"Updated auto-boards for user {user_id}: {result.get('count', 0)} boards updated")
            else:
                # Create new auto-boards
                result = self.taste_graph_manager.create_auto_boards_for_user(user_id)
                logger.info(f"Created auto-boards for user {user_id}: {result.get('count', 0)} boards created")
            
            # Update correlations for the user
            try:
                self.taste_graph_manager.compute_correlations_for_user(user_id)
                logger.info(f"Updated correlations for user {user_id}")
            except Exception as e:
                logger.warning(f"Failed to update correlations for user {user_id}: {e}")
                
        except Exception as e:
            logger.error(f"Failed to process auto-boards for user {user_id}: {e}")
    
    def force_process_user(self, user_id: str):
        """Force processing of auto-boards for a specific user (for testing or manual triggers)"""
        logger.info(f"Force processing auto-boards for user {user_id}")
        self._process_user_auto_boards(user_id)
    
    def get_scheduler_status(self) -> Dict:
        """Get current scheduler status"""
        return {
            'is_running': self.is_running,
            'check_interval_minutes': self.check_interval_minutes,
            'thread_alive': self.scheduler_thread.is_alive() if self.scheduler_thread else False
        }

# Global scheduler instance
_auto_board_scheduler = None

def get_auto_board_scheduler() -> AutoBoardScheduler:
    """Get the global auto-board scheduler instance"""
    global _auto_board_scheduler
    return _auto_board_scheduler

def initialize_auto_board_scheduler(taste_graph_manager, check_interval_minutes: int = 30):
    """Initialize and start the auto-board scheduler"""
    global _auto_board_scheduler
    
    if _auto_board_scheduler is None:
        _auto_board_scheduler = AutoBoardScheduler(taste_graph_manager, check_interval_minutes)
        _auto_board_scheduler.start()
        logger.info("Auto-board scheduler initialized and started")
    else:
        logger.warning("Auto-board scheduler already initialized")
    
    return _auto_board_scheduler

def stop_auto_board_scheduler():
    """Stop the auto-board scheduler"""
    global _auto_board_scheduler
    
    if _auto_board_scheduler:
        _auto_board_scheduler.stop()
        _auto_board_scheduler = None
        logger.info("Auto-board scheduler stopped")
