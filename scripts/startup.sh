#!/bin/bash

# Activate the Python virtual environment
source /opt/venv/bin/activate

# Move to the app directory
cd /packet_buddy

# Run the Streamlit app
streamlit run packet_buddy.py
