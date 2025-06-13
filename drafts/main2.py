from dotenv import load_dotenv
import os

## discord
import discord
from discord.ext import commands

## reddit
from langchain_community.tools.reddit_search.tool import RedditSearchRun, RedditSearchSchema
from langchain_community.utilities.reddit_search import RedditSearchAPIWrapper

load_dotenv()
os.environ['LANGSMITH_PROJECT'] = os.path.basename(__file__)

# Replace with your actual bot token
DISCORD_BOT_TOKEN = os.environ['DISCORD_BOT_TOKEN']

reddit_client_id = os.environ['REDDIT_CLIENT_ID']
reddit_client_secret = os.environ['REDDIT_CLIENT_SECRET']
reddit_user_agent = os.environ['REDDIT_USER_AGENT']

reddit_search = RedditSearchRun(
    api_wrapper=RedditSearchAPIWrapper(
        reddit_client_id=reddit_client_id,
        reddit_client_secret=reddit_client_secret,
        reddit_user_agent=reddit_user_agent,
    )
)

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

    # Example: Echo message
    if message.content.startswith('!echo '):
        await message.channel.send(message.content[6:])

    # Allow commands to be processed
    await bot.process_commands(message)

## >!hello
## >Hello @username!
@bot.command()
async def hello(ctx):
    await ctx.send(f'Hello {ctx.author.display_name}!')



@bot.command()
async def search(ctx, *, message: str):
    print(message)
    search_params = RedditSearchSchema(
        query=message, sort="new", time_filter="week", subreddit="unixporn", limit="10"
    )
    global reddit_search
    result = reddit_search.run(tool_input=search_params.model_dump())
    await ctx.send(result[:2000])


def main():
    # print("Hello from goldybot4!")
    bot.run(DISCORD_BOT_TOKEN)


if __name__ == "__main__":
    main()
