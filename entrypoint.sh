#!/bin/sh
set -e

# Start Ollama server in the background
ollama serve &

# Capture the process ID of the server
pid=$!

echo "Ollama server started with PID $pid"
echo "Waiting for server to be ready..."

# Wait for the server to be up and running
# We can check this by trying to access the API endpoint
while ! curl -s http://localhost:11434 > /dev/null; do
    echo "  - Waiting for Ollama server..."
    sleep 1
done

echo "Ollama server is ready."

# Pull the required models
echo "Pulling 'all-minilm' model..."
ollama pull all-minilm

echo "Pulling 'llava:latest' model..."
ollama pull llava:latest

echo "Models pulled successfully."

# Wait for the server process to exit
# This will keep the container running
wait $pid
