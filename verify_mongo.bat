@echo off
echo Setting up MongoDB environment variables...

set USE_MONGODB=true
set MONGODB_URI=mongodb+srv://macd_user:YOUR_PASSWORD@macd-cluster.xxxxx.mongodb.net/?retryWrites=true^&w=majority
set MONGODB_DB_NAME=macd_reversal

echo Running MongoDB verification...
python verify_mongodb.py

pause
