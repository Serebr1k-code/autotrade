#!/bin/bash
# Run daily trader (schedule via cron: 16:30 ET Mon-Fri)
# Example cron: 30 16 * * 1-5 cd /home/Serebr1k/stocks && ./run_daily.sh >> /tmp/trade.log 2>&1

cd "$(dirname "$0")"
echo "=== $(date) ==="
python3 trade.py
echo "Exit: $?"
