import os
import subprocess
from dotenv import load_dotenv

USE_OPENAI = False
USE_DEEPSEEK = True

# Load environment variables from .env file
load_dotenv()

# Initialize variables
api_key = None
api_flag = None

# Determine which API to use and retrieve the corresponding API key
if USE_OPENAI:
    api_key = os.getenv("OPENAI_API_KEY")
    api_flag = "--openai-api-key"
elif USE_DEEPSEEK:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    api_flag = "--deepseek"

# Validate that an API key was found
if not api_key:
    raise ValueError("No valid API key found in .env file.")

# Get the current working directory
current_directory = os.getcwd()

# Construct the Docker command
docker_command = [
    "docker", "run", "-it",
    "--volume", f"{current_directory}:/app",
    "paulgauthier/aider-full"
]

# Add the API flag and key to the command
docker_command.extend([api_flag, api_key])

# Execute the Docker command
try:
    subprocess.run(docker_command, check=True)
except subprocess.CalledProcessError as e:
    print(f"Error running Docker command: {e}")
except FileNotFoundError:
    print("Docker is not installed or not in your PATH.")
