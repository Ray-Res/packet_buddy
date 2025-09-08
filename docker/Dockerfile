FROM ubuntu:latest

ARG DEBIAN_FRONTEND=noninteractive

# Install system packages
RUN apt-get update && apt-get install -y \
    wget \
    sudo \
    dos2unix \
    python3 \
    python3-venv \
    python3-pip \
    tshark \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python3 -m venv /opt/venv

# Make sure the venv's Python and pip are used
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install all Python packages inside venv
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -U \
        streamlit \
        langchain==0.2.14 \
        langchain-community==0.2.12 \
        langchain_experimental \
        chromadb \
        tiktoken \
        pyshark \
        requests \
        jq \
        sentence-transformers==2.2.2 \
        "huggingface_hub<=0.16.4" \
        InstructorEmbedding

# Copy application code
COPY /packet_buddy /packet_buddy/
COPY /scripts /scripts/

# Convert scripts to Unix format
RUN dos2unix /scripts/startup.sh

# Set working directory
WORKDIR /packet_buddy

# Expose Streamlit default port
EXPOSE 8501

# Start the app
CMD ["/bin/bash", "/scripts/startup.sh"]
