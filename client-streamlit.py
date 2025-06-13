# chat_client.py
import streamlit as st
import asyncio
import websockets

# Session state for user_id and messages
if "user_id" not in st.session_state:
    st.session_state.user_id = ""
if "messages" not in st.session_state:
    st.session_state.messages = []

LOGO_URL = "https://content.sportslogos.net/logos/32/753/full/bi3fnohutcve3yvj2c7ive4hv.png"

st.image(LOGO_URL, width=400)
st.title("💬 GoldyBot")

# User ID input
if not st.session_state.user_id:
    st.session_state.user_id = st.text_input("Enter your user ID to connect", key="user_id_input")

if st.session_state.user_id:
    query = st.chat_input("Type your message")

    # Display chat history
    for role, msg in st.session_state.messages:
        with st.chat_message(role):
            st.markdown(msg)

    if query:
        st.session_state.messages.append(("user", query))
        with st.chat_message("user"):
            st.markdown(query)

        # WebSocket communication
        async def send_and_receive():
            uri = f"ws://localhost:8000/ws/{st.session_state.user_id}"
            try:
                async with websockets.connect(uri) as websocket:
                    await websocket.send(query)
                    response = await websocket.recv()
                    return response
            except Exception as e:
                return f"❌ Error: {e}"

        response = asyncio.run(send_and_receive())

        st.session_state.messages.append(("assistant", response))
        with st.chat_message("assistant"):
            st.markdown(response)
