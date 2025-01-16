# Fish Speech Docker

This repository provides a Docker setup for [Fish Speech](https://github.com/fishaudio/fish-speech), making it easier to run with GPU support and local model checkpoints.

## Setup

1. Clone this repository with submodules:
```bash
git clone --recursive https://github.com/YOUR_USERNAME/fish_speech_docker.git
cd fish_speech_docker
```

2. Place your model checkpoints in the `checkpoints` directory.

3. Start the container:
```bash
docker-compose up
```

4. Access the web UI at http://localhost:7860

## Directory Structure

- `fish-speech/` - The main Fish Speech repository (submodule)
- `checkpoints/` - Directory for model checkpoints
- `outputs/` - Directory for generated outputs
- `tools/` - Utility scripts
- `docker-compose.yml` - Docker configuration
