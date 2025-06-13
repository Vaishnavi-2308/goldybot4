# client.py
import asyncio
import websockets

async def chat():
    uri = "ws://localhost:8000/ws/test_user_123"
    async with websockets.connect(uri) as websocket:
        while True:
            msg = input("You: ")
            await websocket.send(msg)
            response = await websocket.recv()
            print(f"Bot: {response}")

asyncio.run(chat())
