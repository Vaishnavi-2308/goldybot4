# GoldyBot

![Goldy Gopher](https://content.sportslogos.net/logos/32/753/full/bi3fnohutcve3yvj2c7ive4hv.png)

### Installation

Clone the repository and run `uv sync`.

### Usage

Run the ollama server first - `ollama serve`

Pull the LLM - `ollama pull qwen3:1.7b` and `ollama pull nomic-embed-text`.

server - `uvicorn server:app --reload`

Streamlit client - `streamlit run client-streamlit.py`

Optionally run `langgraph dev` for the langgraph server

### Demo Video 

https://drive.google.com/file/d/1bATh-70rTLWpCQ6wsbGpnnghDXYc8YS0/view?usp=drive_link

### References

https://github.com/esurovtsev/langgraph-intro/blob/main/18_user_profiles.ipynb

https://python.langchain.com/docs/versions/migrating_memory/long_term_memory_agent/

Hotpath vs background \
https://langchain-ai.github.io/langgraph/concepts/memory