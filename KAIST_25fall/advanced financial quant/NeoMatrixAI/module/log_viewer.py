# python log_viewer.py --user_key {USER_KEY} --session_id {session_id} --save_log {true/false}

import json
import argparse
import websocket
import sys

log_file = None
save_log = False

def write_to_file(line):
    if save_log and log_file:
        log_file.write(line + "\n")
        log_file.flush()

def on_message(ws, message):
    try:
        parsed = json.loads(message)
        log_line = json.dumps(parsed, ensure_ascii=False)
        print(log_line)
        write_to_file(log_line)
    except Exception:
        if "\r" in message:
            line = message.split("\r")[-1]
            sys.stdout.write("\r" + line)
            sys.stdout.flush()
            write_to_file(line)
        else:
            print(message)
            write_to_file(message)

def on_close(ws, close_status_code, close_msg):
    print("🔌 Connection closed")

def on_open(ws):
    print("🚀 WebSocket connection established")

def on_error(ws, error):
    print(f"❌ Error occurred: {error}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WebSocket Real-Time Log Viewer")
    parser.add_argument("--user_key", required=True, help="User API Key")
    parser.add_argument("--session_id", required=True, help="Session ID")
    parser.add_argument("--save_log", default="false", help="Whether to save log to file (true/false)")
    args = parser.parse_args()

    save_log = args.save_log.lower() == "true"

    if save_log:
        filename = f"log_{args.session_id}.txt"
        log_file = open(filename, "a", encoding="utf-8")
        print(f"📝 Logging to {filename}")

    ws_url = f"wss://aifapbt.fin.cloud.ainode.ai/logs/ws/{args.session_id}?user_key={args.user_key}"

    ws = websocket.WebSocketApp(
        ws_url,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close
    )

    try:
        ws.run_forever()
    finally:
        if log_file:
            log_file.close()
