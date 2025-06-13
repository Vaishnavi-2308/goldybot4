# server.py
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from typing import Dict
from goldybot import GoldyBot  # Assuming GoldyBot is defined elsewhere and imported here
import os
from dotenv import load_dotenv
#from langsmith import traceable

load_dotenv()
os.environ['LANGSMITH_PROJECT'] = os.path.basename(os.path.dirname(__file__))

app = FastAPI()
bot = GoldyBot()  # Global instance

graph_ref = bot.app

# Store active connections per user
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, user_id: str, websocket: WebSocket):
        await websocket.accept()
        self.active_connections[user_id] = websocket
        print(f"User {user_id} connected.")

    def disconnect(self, user_id: str):
        self.active_connections.pop(user_id, None)
        print(f"User {user_id} disconnected.")

    def get(self, user_id: str):
        return self.active_connections.get(user_id)

manager = ConnectionManager()


#@traceable
@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    await manager.connect(user_id, websocket)
    try:
        while True:
            data = await websocket.receive_text()
            print(f"Received from {user_id}: {data}")
            response = await bot.chat(data, user_id)
            await websocket.send_text(response)
    except WebSocketDisconnect:
        manager.disconnect(user_id)

