"""
Google Calendar Sync using Service Account (for GitHub Actions automation)
"""

import os
import json
import tempfile
from typing import Dict
import logging
from datetime import datetime

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import pandas as pd

# Setup logging
logger = logging.getLogger(__name__)

SCOPES = ['https://www.googleapis.com/auth/calendar']


class GoogleCalendarServiceAccount:
    """Handles Google Calendar API interactions using Service Account"""
    
    def __init__(self, service_account_file: str = "service-account-key.json"):
        """
        Initialize Google Calendar sync with Service Account
        
        Args:
            service_account_file: Path to service account JSON file
        """
        self.service = None
        self.authenticate(service_account_file)
    
    def authenticate(self, service_account_file: str):
        """Authenticate with Google Calendar API using Service Account"""
        credentials = None
        
        # Try to load from file first
        if os.path.exists(service_account_file):
            logger.info(f"Loading service account from file: {service_account_file}")
            credentials = service_account.Credentials.from_service_account_file(
                service_account_file, scopes=SCOPES
            )
        else:
            # Try loading from environment variable (GitHub Actions)
            service_account_json = os.getenv('GOOGLE_SERVICE_ACCOUNT_JSON')
            if service_account_json:
                logger.info("Loading service account from environment variable")
                try:
                    # Parse JSON string
                    service_account_info = json.loads(service_account_json)
                    
                    # Create credentials from dict
                    credentials = service_account.Credentials.from_service_account_info(
                        service_account_info, scopes=SCOPES
                    )
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON in GOOGLE_SERVICE_ACCOUNT_JSON: {e}")
            else:
                raise FileNotFoundError(
                    f"Service account file '{service_account_file}' not found and "
                    "GOOGLE_SERVICE_ACCOUNT_JSON environment variable not set. "
                    "Please provide service account credentials."
                )
        
        if credentials:
            self.service = build('calendar', 'v3', credentials=credentials)
            logger.info("Successfully authenticated with Google Calendar using Service Account")
        else:
            raise ValueError("Failed to create credentials from service account")
    
    def create_event(self, event_data: Dict, calendar_id: str = 'primary') -> Dict:
        """
        Create a calendar event
        
        Args:
            event_data: Dictionary with event details (summary, description, date)
            calendar_id: Google Calendar ID
            
        Returns:
            Created event object
        """
        event = {
            'summary': event_data['summary'],
            'description': event_data.get('description', ''),
            'start': {
                'date': event_data['date'],
                'timeZone': 'America/Chicago',
            },
            'end': {
                'date': event_data['date'],
                'timeZone': 'America/Chicago',
            },
            'reminders': {
                'useDefault': False,
                'overrides': [
                    {'method': 'popup', 'minutes': 24 * 60},  # 1 day before
                    {'method': 'popup', 'minutes': 60},  # 1 hour before
                ],
            },
        }
        
        try:
            event = self.service.events().insert(
                calendarId=calendar_id, 
                body=event
            ).execute()
            return event
        except HttpError as error:
            logger.error(f"Error creating event: {error}")
            raise
    
    def event_exists(self, summary: str, date: str, calendar_id: str = 'primary') -> bool:
        """
        Check if an event already exists
        
        Args:
            summary: Event title
            date: Event date (YYYY-MM-DD)
            calendar_id: Google Calendar ID
            
        Returns:
            True if event exists, False otherwise
        """
        try:
            # Search for events on that date
            time_min = f"{date}T00:00:00Z"
            time_max = f"{date}T23:59:59Z"
            
            events_result = self.service.events().list(
                calendarId=calendar_id,
                timeMin=time_min,
                timeMax=time_max,
                q=summary,
                singleEvents=True
            ).execute()
            
            events = events_result.get('items', [])
            
            # Check if any event matches the summary
            for event in events:
                if event.get('summary') == summary:
                    return True
            
            return False
            
        except HttpError as error:
            logger.error(f"Error checking event existence: {error}")
            return False
    
    def add_events_to_calendar(self, events_df: pd.DataFrame, 
                               calendar_id: str = 'primary') -> int:
        """
        Add events from DataFrame to Google Calendar
        
        Args:
            events_df: DataFrame with events (must have: date, ticker, event_type, description)
            calendar_id: Google Calendar ID (use your calendar ID or 'primary')
            
        Returns:
            Number of events added
        """
        if events_df.empty:
            logger.warning("No events to add")
            return 0
        
        added_count = 0
        skipped_count = 0
        
        for _, row in events_df.iterrows():
            try:
                # Format event data
                date_str = pd.to_datetime(row['date']).strftime('%Y-%m-%d')
                
                # Create event summary based on type
                if row['event_type'] == 'Earnings Call':
                    summary = f"📊 {row['ticker']} Earnings"
                elif row['event_type'] == 'FOMC Meeting':
                    summary = f"🏛️ Fed FOMC Meeting"
                elif row['event_type'] == 'CPI Release':
                    summary = f"📈 CPI Report"
                elif row['event_type'] == 'Jobs Report':
                    summary = f"💼 Jobs Report (NFP)"
                else:
                    summary = f"{row['ticker']} - {row['event_type']}"
                
                # Check if event already exists
                if self.event_exists(summary, date_str, calendar_id):
                    logger.info(f"Event already exists: {summary} on {date_str}")
                    skipped_count += 1
                    continue
                
                # Create event
                event_data = {
                    'summary': summary,
                    'description': row['description'],
                    'date': date_str
                }
                
                self.create_event(event_data, calendar_id)
                logger.info(f"Added: {summary} on {date_str}")
                added_count += 1
                
            except Exception as e:
                logger.error(f"Error adding event: {e}")
                continue
        
        logger.info(f"Added {added_count} events, skipped {skipped_count} duplicates")
        return added_count


if __name__ == "__main__":
    # Test authentication
    try:
        sync = GoogleCalendarServiceAccount()
        print("✓ Service Account authentication successful!")
    except Exception as e:
        print(f"✗ Authentication failed: {e}")
