from dotenv import load_dotenv
import os

## discord
import discord
from discord.ext import commands

## reddit
from langchain_community.tools.reddit_search.tool import RedditSearchRun, RedditSearchSchema
from langchain_community.utilities.reddit_search import RedditSearchAPIWrapper

## langgraph
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor

load_dotenv()
os.environ['LANGSMITH_PROJECT'] = os.path.basename(os.path.dirname(__file__))

# Replace with your actual bot token
DISCORD_BOT_TOKEN = os.environ['DISCORD_BOT_TOKEN']

reddit_client_id = os.environ['REDDIT_CLIENT_ID']
reddit_client_secret = os.environ['REDDIT_CLIENT_SECRET']
reddit_user_agent = os.environ['REDDIT_USER_AGENT']

reddit_search_runner = RedditSearchRun(
    api_wrapper=RedditSearchAPIWrapper(
        reddit_client_id=reddit_client_id,
        reddit_client_secret=reddit_client_secret,
        reddit_user_agent=reddit_user_agent,
    )
)

def reddit_search(query: str):
    """
    Searches a set of preselected subreddits for the query.

    Args:
        query (str): a string of comma seperated keywords.

    Output:
        A string containing the results from reddit corresponding to the query.
    """
    global reddit_search_runner
    subreddits = ["uofmn", "twincities", "minneapolis"]

    result = ""
    for subreddit in subreddits:
        search_params = RedditSearchSchema(
            query=query, sort="new", time_filter="week", subreddit=subreddit, limit="5"
        )
        result += reddit_search_runner.run(tool_input=search_params.model_dump())
    
    return result


model = ChatOllama(
    model="qwen3:1.7b",
    temperature=0,
    # other params...
)

keyword_assistant = create_react_agent(
    model=model,
    tools=[],
    prompt="Your job is to extract and output comma seperated keywords and keywords only.",
    name="keyword_assistant"
)

reddit_assistant = create_react_agent(
    model=model,
    tools=[reddit_search],
    prompt=(
        "Your job is to search reddit using the given keywords and format the output"
        "so that it contains no more than 500 words."
    ),
    name="reddit_assistant"
)

supervisor = create_supervisor(
    agents=[keyword_assistant, reddit_assistant],
    model=model,
    prompt=(
        "You are a helpful assistant that helps students at the University of Minnesota, Twin Cities. "
        "You first use the keyword assistant to extract relevant keywords from any queries posed by the student users,"
        "then use the reddit search assistant to fetch the relevant results and present it to the user. "
        "Compress the final answer so that it contains no more than 2-3 lines of text."
        # "Pass only the current query's information to the agents you manage. You yourself can operate on"
        # "the memory/history of the entire conversation to finally present the output to the user."
    )
).compile()


# Enable message content intent
intents = discord.Intents.default()
intents.message_content = True

# Create bot
bot = commands.Bot(command_prefix='!', intents=intents)

@bot.event
async def on_ready():
    print(f'Logged in as {bot.user} (ID: {bot.user.id})')

@bot.event
async def on_message(message):
    # Ignore messages from the bot itself
    if message.author == bot.user:
        return

    # Log the message
    print(f'{message.author}: {message.content}')

    global supervisor
    result = supervisor.invoke(
        {"messages": [{"role": "user", "content": message.content}]}
    )
    response = result['messages'][-1].content

    while response:
        await message.channel.send(response[:2000])
        response = response[2000:]


def main():
    # print("Hello from goldybot4!")
    bot.run(DISCORD_BOT_TOKEN)


if __name__ == "__main__":
    main()
