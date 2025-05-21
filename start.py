#!/usr/bin/env python3
"""
Start script for running both Chainlit and the Agent Portal.

Usage:
    python start.py

Environment variables:
    CHAINLIT_PORT: Port for Chainlit server (default: 8000)
    AGENT_PORTAL_PORT: Port for Agent Portal (default: 8001)
"""

import os
import sys
import subprocess
import time
import signal
import atexit

# Configure ports
CHAINLIT_PORT = int(os.getenv("CHAINLIT_PORT", 8000))
AGENT_PORTAL_PORT = int(os.getenv("AGENT_PORTAL_PORT", 8001))

# Set environment variables for child processes
os.environ["AGENT_PORTAL_ENABLED"] = "true"
os.environ["AGENT_PORTAL_URL"] = f"http://localhost:{AGENT_PORTAL_PORT}"
os.environ["DEMO_MODE"] = "true"  # For simulating agent responses

# Store processes for cleanup
processes = []

def cleanup():
    """Kill all child processes on exit"""
    print("\nShutting down servers...")
    for p in processes:
        if p.poll() is None:  # If process is still running
            try:
                p.terminate()
                time.sleep(0.5)
                if p.poll() is None:
                    p.kill()
            except Exception as e:
                print(f"Error terminating process: {e}")
    print("Shutdown complete.")


# Register cleanup handler
atexit.register(cleanup)
signal.signal(signal.SIGINT, lambda sig, frame: sys.exit(0))
signal.signal(signal.SIGTERM, lambda sig, frame: sys.exit(0))


def start_chainlit():
    """Start the Chainlit server"""
    print(f"Starting Chainlit server on port {CHAINLIT_PORT}...")
    cmd = [
        sys.executable, "-m", "chainlit", "run", "apex.py",
        "--port", str(CHAINLIT_PORT), "--host", "0.0.0.0",
        "--debug",
    ]
    env = os.environ.copy()
    process = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )
    processes.append(process)
    return process

def start_agent_portal():
    """Start the Agent Portal server"""
    print(f"Starting Agent Portal on port {AGENT_PORTAL_PORT}...")
    cmd = [
        sys.executable, "agent_portal.py"
    ]
    env = os.environ.copy()
    env["AGENT_PORTAL_PORT"] = str(AGENT_PORTAL_PORT)
    process = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )
    processes.append(process)
    return process

def monitor_output(process, prefix):
    """Monitor and print process output with prefix"""
    while True:
        if process.poll() is not None:
            break
        
        line = process.stdout.readline()
        if line:
            print(f"[{prefix}] {line.rstrip()}")
            
        line = process.stderr.readline()
        if line:
            print(f"[{prefix} ERROR] {line.rstrip()}", file=sys.stderr)

def create_required_directories():
    """Create required directories for the application"""
    dirs = ["static", "static/css", "static/js", "templates"]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"Created directory: {d}")

def main():
    """Main entry point"""
    print("=" * 50)
    print("Starting Apex AI Assistant with Human Handoff Support")
    print("=" * 50)
    
    # Create required directories
    create_required_directories()
    
    # Start servers
    portal_process = start_agent_portal()
    time.sleep(1)  # Small delay to ensure portal starts first
    chainlit_process = start_chainlit()
    
    # Print access URLs
    print("\n" + "=" * 50)
    print(f"Chainlit UI:   http://localhost:{CHAINLIT_PORT}")
    print(f"Agent Portal:  http://localhost:{AGENT_PORTAL_PORT}/agent/portal")
    print("=" * 50 + "\n")
    
    try:
        # Monitor both processes' output
        import threading
        portal_thread = threading.Thread(
            target=monitor_output, 
            args=(portal_process, "PORTAL"),
            daemon=True
        )
        chainlit_thread = threading.Thread(
            target=monitor_output, 
            args=(chainlit_process, "CHAINLIT"),
            daemon=True
        )
        
        portal_thread.start()
        chainlit_thread.start()
        
        # Wait for either process to exit
        while True:
            if portal_process.poll() is not None:
                print("Agent Portal has exited.")
                break
            if chainlit_process.poll() is not None:
                print("Chainlit has exited.")
                break
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nReceived interrupt. Shutting down...")
        sys.exit(0)


if __name__ == "__main__":
    main()
