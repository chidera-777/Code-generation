#!/bin/bash
# Exit immediately if a command exits with a non-zero status
set -e

# Initialize the environment variables and load the model
python -c "from chatbot import init; init()"

# Start the Streamlit application
streamlit run chatbot.py --server.port=8501 --server.address=0.0.0.0
