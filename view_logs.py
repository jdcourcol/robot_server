#!/usr/bin/env python3
"""
Log Viewer for Robot Server
View AI reasoning, robot instructions, and system events
"""

import os
import sys
from datetime import datetime

def view_logs(log_type="all", lines=50):
    """View robot server logs"""
    
    log_files = {
        "ai": "logs/ai_analysis.log",
        "robot": "logs/robot_instructions.log", 
        "system": "logs/system_events.log",
        "all": None
    }
    
    if log_type not in log_files:
        print(f"❌ Invalid log type: {log_type}")
        print(f"Available types: {', '.join(log_files.keys())}")
        return
    
    print(f"🤖 Robot Server Logs - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    if log_type == "all":
        # Show all log files
        for name, path in log_files.items():
            if name == "all":
                continue
            if os.path.exists(path):
                print(f"\n📋 {name.upper()} LOGS:")
                print("-" * 40)
                try:
                    with open(path, 'r') as f:
                        all_lines = f.readlines()
                        recent_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines
                        for line in recent_lines:
                            print(line.rstrip())
                except Exception as e:
                    print(f"❌ Error reading {path}: {e}")
            else:
                print(f"❌ Log file not found: {path}")
    else:
        # Show specific log file
        path = log_files[log_type]
        if os.path.exists(path):
            print(f"\n📋 {log_type.upper()} LOGS:")
            print("-" * 40)
            try:
                with open(path, 'r') as f:
                    all_lines = f.readlines()
                    recent_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines
                    for line in recent_lines:
                        print(line.rstrip())
            except Exception as e:
                print(f"❌ Error reading {path}: {e}")
        else:
            print(f"❌ Log file not found: {path}")

def tail_logs(log_type="all"):
    """Follow logs in real-time"""
    import time
    
    log_files = {
        "ai": "logs/ai_analysis.log",
        "robot": "logs/robot_instructions.log", 
        "system": "logs/system_events.log"
    }
    
    if log_type not in log_files:
        print(f"❌ Invalid log type: {log_type}")
        return
    
    path = log_files[log_type]
    if not os.path.exists(path):
        print(f"❌ Log file not found: {path}")
        return
    
    print(f"👀 Following {log_type} logs (Ctrl+C to stop)...")
    print("=" * 60)
    
    try:
        with open(path, 'r') as f:
            # Go to end of file
            f.seek(0, 2)
            while True:
                line = f.readline()
                if line:
                    print(line.rstrip())
                else:
                    time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n👋 Stopped following logs")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("🤖 Robot Server Log Viewer")
        print("Usage:")
        print("  python view_logs.py [ai|robot|system|all] [lines]")
        print("  python view_logs.py tail [ai|robot|system]")
        print("\nExamples:")
        print("  python view_logs.py all 100    # Show last 100 lines of all logs")
        print("  python view_logs.py ai 50      # Show last 50 lines of AI logs")
        print("  python view_logs.py tail robot # Follow robot logs in real-time")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "tail":
        if len(sys.argv) < 3:
            print("❌ Please specify log type for tail command")
            sys.exit(1)
        tail_logs(sys.argv[2])
    else:
        lines = int(sys.argv[2]) if len(sys.argv) > 2 else 50
        view_logs(command, lines)
