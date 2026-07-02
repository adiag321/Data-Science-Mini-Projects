# Stock Events Calendar - Project Summary

## What This Does

Automatically fetches stock earnings and economic events, then syncs them to Google Calendar.

## Files

```
Stock_event_list/
├── Stock_event_list_Main.py              # Main script (simplified)
├── google_calendar_sync.py               # OAuth (for local use)
├── google_calendar_service_account.py    # Service Account (for automation)
├── config.json                           # Your settings
├── requirements.txt                      # Dependencies
├── credentials.json                      # OAuth credentials
└── token.pickle                          # Auth token
```

## Configuration (config.json)

Edit `config.json` to customize:

- **stocks**: Your stock watchlist
- **banking_stocks**: Banking sector stocks to track
- **days_ahead**: How many days to fetch (default: 90)
- **include_economic_events**: true/false for Fed/CPI/Jobs
- **include_banking_events**: true/false for banking earnings

## How to Run

### Local (One-time setup)

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Get Google Calendar OAuth credentials:
   - Go to Google Cloud Console
   - Create project
   - Enable Google Calendar API
   - Download credentials.json

3. Run:
   ```bash
   python Stock_event_list_Main.py
   ```

4. First run: Browser opens for Google login
5. All future runs: Automatic (uses token.pickle)

### GitHub Actions (Fully Automated)

1. Create Service Account in Google Cloud
2. Download service-account-key.json
3. Share your calendar with service account email
4. Add GitHub secrets:
   - GOOGLE_SERVICE_ACCOUNT_JSON
   - GOOGLE_CALENDAR_ID
5. Create .github/workflows/sync-events.yml
6. Push to GitHub

## What It Fetches

- Stock earnings (from yfinance)
- Banking sector earnings
- Fed FOMC meetings
- CPI inflation reports
- Jobs reports (NFP)

## Output

- Console display of all events
- stock_events.csv file
- Events added to Google Calendar

## Simplified Main Script

Key improvements:
- Clear sections with headers
- Simple standalone functions (not classes)
- Easy to read and modify
- No emojis in output
- 268 lines (vs 363 before)

## Authentication

Auto-detects:
- **OAuth**: If credentials.json exists (local use)
- **Service Account**: If service-account-key.json or env variable exists (GitHub)

## Notes

- Only authenticate once locally (token persists)
- Can run manually or schedule (Task Scheduler/GitHub Actions)
- Events are deduplicated automatically
- Future dates only
