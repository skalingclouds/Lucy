import json
import logging
import os
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any
import uuid

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, Depends, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uvicorn

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("AgentPortal")

# Initialize FastAPI app
app = FastAPI(title="Apex Agent Portal")

# In-memory storage for active handoff requests
# In production, this should be replaced with a database
active_conversations: Dict[str, Dict] = {}
active_connections: Dict[str, List[WebSocket]] = {}
conversation_messages: Dict[str, List[Dict]] = {}

# Setup static files
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/public", StaticFiles(directory="public"), name="public")

# Setup Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Placeholder for authentication (to be replaced with proper auth)
def get_current_agent(request: Request):
    # In production, implement proper authentication
    agent_id = request.headers.get("X-Agent-ID", str(uuid.uuid4()))
    agent_name = request.headers.get("X-Agent-Name", "Support Agent")
    return {"id": agent_id, "name": agent_name}

# Models
class Message(BaseModel):
    role: str
    content: str
    timestamp: Optional[str] = None

class Conversation(BaseModel):
    id: str
    user_info: Dict[str, Any]
    waiting_since: str
    status: str = "waiting"
    history: Optional[List[Message]] = []

# API Routes
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Redirect to agent portal"""
    return templates.TemplateResponse(
        "redirect.html", 
        {"request": request, "redirect_url": "/agent/portal"}
    )

@app.get("/agent/portal", response_class=HTMLResponse)
async def agent_portal(request: Request, agent: Dict = Depends(get_current_agent)):
    """Main agent portal page"""
    return templates.TemplateResponse(
        "agent_portal.html",
        {"request": request, "agent": agent}
    )

@app.get("/agent/conversation/{conversation_id}", response_class=HTMLResponse)
async def conversation_page(
    request: Request, 
    conversation_id: str, 
    agent: Dict = Depends(get_current_agent)
):
    """Page for handling a specific conversation"""
    # Check if conversation exists
    if conversation_id not in active_conversations and conversation_id not in conversation_messages:
        # For demo purposes, create a placeholder conversation
        active_conversations[conversation_id] = {
            "id": conversation_id,
            "user_info": {"name": "Demo User", "apex_id": "DEMO001"},
            "waiting_since": datetime.utcnow().isoformat(),
            "status": "waiting"
        }
        conversation_messages[conversation_id] = [
            {
                "role": "system",
                "content": "Conversation started",
                "timestamp": datetime.utcnow().isoformat()
            },
            {
                "role": "user",
                "content": "I need help with my case status",
                "timestamp": datetime.utcnow().isoformat()
            }
        ]
    
    # Get conversation data
    conversation = active_conversations.get(conversation_id, {})
    messages = conversation_messages.get(conversation_id, [])
    
    return templates.TemplateResponse(
        "conversation.html",
        {
            "request": request,
            "agent": agent,
            "conversation_id": conversation_id,
            "user_info": conversation.get("user_info", {}),
            "history": messages
        }
    )

@app.get("/api/conversations/pending")
async def get_pending_conversations(agent: Dict = Depends(get_current_agent)):
    """Get all pending conversations"""
    pending = [
        {**conv, "id": conv_id}
        for conv_id, conv in active_conversations.items()
        if conv.get("status") == "waiting"
    ]
    return pending

@app.get("/api/conversations/{conversation_id}")
async def get_conversation(conversation_id: str, agent: Dict = Depends(get_current_agent)):
    """Get conversation details"""
    if conversation_id not in active_conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    conversation = active_conversations[conversation_id]
    messages = conversation_messages.get(conversation_id, [])
    
    return {**conversation, "messages": messages}

@app.post("/api/conversations/{conversation_id}/join")
async def join_conversation(conversation_id: str, agent: Dict = Depends(get_current_agent)):
    """Join a conversation as an agent"""
    if conversation_id not in active_conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    conversation = active_conversations[conversation_id]
    conversation["status"] = "active"
    conversation["agent_id"] = agent["id"]
    conversation["agent_name"] = agent["name"]
    
    # Add system message about agent joining
    if conversation_id not in conversation_messages:
        conversation_messages[conversation_id] = []
    
    conversation_messages[conversation_id].append({
        "role": "system",
        "content": f"Agent {agent['name']} has joined the conversation",
        "timestamp": datetime.utcnow().isoformat()
    })
    
    # Broadcast message to all connections
    await broadcast_to_conversation(
        conversation_id,
        {
            "type": "agent_joined",
            "agent": agent["name"],
            "timestamp": datetime.utcnow().isoformat()
        }
    )
    
    return {"status": "joined", "conversation": conversation}

@app.post("/api/conversations/{conversation_id}/leave")
async def leave_conversation(conversation_id: str, agent: Dict = Depends(get_current_agent)):
    """Leave a conversation"""
    if conversation_id not in active_conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    conversation = active_conversations[conversation_id]
    conversation["status"] = "waiting"
    
    # Add system message about agent leaving
    if conversation_id in conversation_messages:
        conversation_messages[conversation_id].append({
            "role": "system",
            "content": f"Agent {agent['name']} has left the conversation",
            "timestamp": datetime.utcnow().isoformat()
        })
    
    # Broadcast message to all connections
    await broadcast_to_conversation(
        conversation_id,
        {
            "type": "agent_left",
            "agent": agent["name"],
            "timestamp": datetime.utcnow().isoformat()
        }
    )
    
    return {"status": "left", "conversation": conversation}

@app.get("/api/conversations/{conversation_id}/transcript")
async def get_transcript(conversation_id: str, agent: Dict = Depends(get_current_agent)):
    """Get conversation transcript as downloadable text"""
    if conversation_id not in conversation_messages:
        raise HTTPException(status_code=404, detail="Conversation transcript not found")
    
    messages = conversation_messages[conversation_id]
    
    # Format transcript
    transcript = f"Conversation ID: {conversation_id}\n"
    transcript += f"Date: {datetime.utcnow().strftime('%Y-%m-%d')}\n\n"
    
    for msg in messages:
        role = msg.get("role", "unknown").upper()
        timestamp = msg.get("timestamp", "")
        if timestamp:
            try:
                dt = datetime.fromisoformat(timestamp)
                timestamp = dt.strftime("%Y-%m-%d %H:%M:%S")
            except:
                pass
        
        transcript += f"[{timestamp}] {role}: {msg.get('content', '')}\n\n"
    
    return JSONResponse(
        content={"transcript": transcript},
        headers={"Content-Disposition": f"attachment; filename=transcript-{conversation_id}.txt"}
    )

# WebSocket endpoint for real-time communication
@app.websocket("/ws/conversation/{conversation_id}")
async def websocket_endpoint(websocket: WebSocket, conversation_id: str):
    await websocket.accept()
    
    # Add connection to active connections
    if conversation_id not in active_connections:
        active_connections[conversation_id] = []
    active_connections[conversation_id].append(websocket)
    
    try:
        # Send initial connection message
        await websocket.send_json({
            "type": "connection_established",
            "conversation_id": conversation_id,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Listen for messages
        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            # Add message to conversation history
            if conversation_id not in conversation_messages:
                conversation_messages[conversation_id] = []
            
            # Add timestamp if not provided
            if "timestamp" not in message_data:
                message_data["timestamp"] = datetime.utcnow().isoformat()
            
            conversation_messages[conversation_id].append(message_data)
            
            # Broadcast message to all connections for this conversation
            await broadcast_to_conversation(conversation_id, message_data)
            
    except WebSocketDisconnect:
        # Remove connection when disconnected
        if conversation_id in active_connections:
            active_connections[conversation_id].remove(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        if conversation_id in active_connections and websocket in active_connections[conversation_id]:
            active_connections[conversation_id].remove(websocket)

async def broadcast_to_conversation(conversation_id: str, message: dict):
    """Broadcast a message to all connections in a conversation"""
    if conversation_id in active_connections:
        dead_connections = []
        for connection in active_connections[conversation_id]:
            try:
                await connection.send_json(message)
            except Exception:
                dead_connections.append(connection)
        
        # Clean up dead connections
        for dead in dead_connections:
            if dead in active_connections[conversation_id]:
                active_connections[conversation_id].remove(dead)

# API for Lucy to create handoff requests
@app.post("/api/handoff")
async def create_handoff(handoff_data: dict):
    """Create a new handoff request from Lucy"""
    conversation_id = handoff_data.get("conversation_id")
    if not conversation_id:
        conversation_id = str(uuid.uuid4())
    
    # Store the new conversation
    active_conversations[conversation_id] = {
        "id": conversation_id,
        "user_info": handoff_data.get("user_info", {}),
        "waiting_since": datetime.utcnow().isoformat(),
        "status": "waiting"
    }
    
    # Initialize message history
    if conversation_id not in conversation_messages:
        conversation_messages[conversation_id] = []
    
    # Add any provided history
    for message in handoff_data.get("history", []):
        conversation_messages[conversation_id].append(message)
    
    # Add system message
    conversation_messages[conversation_id].append({
        "role": "system",
        "content": "Human assistance requested. Waiting for an agent to join.",
        "timestamp": datetime.utcnow().isoformat()
    })
    
    return {"status": "created", "conversation_id": conversation_id}

# Main entry point for running the app directly
if __name__ == "__main__":
    port = int(os.getenv("AGENT_PORTAL_PORT", 8000))
    log_level = os.getenv("LOG_LEVEL", "info")
    
    # Print startup banner
    print(f"\n{'='*50}")
    print(f" Apex Agent Portal starting on port {port}")
    print(f"{'='*50}\n")
    
    uvicorn.run("agent_portal:app", host="0.0.0.0", port=port, log_level=log_level, reload=True) 