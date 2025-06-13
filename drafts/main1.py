from dotenv import load_dotenv
import os

import discord
from discord.ext import commands

load_dotenv()

# Replace with your actual bot token
DISCORD_BOT_TOKEN = os.environ['DISCORD_BOT_TOKEN']

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

def main():
    # print("Hello from goldybot4!")
    bot.run(DISCORD_BOT_TOKEN)


if __name__ == "__main__":
    main()
