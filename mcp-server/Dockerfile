# MCP Server Dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy MCP server requirements
COPY mcp-server/requirements.txt /app/mcp-server/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r /app/mcp-server/requirements.txt

# Copy MCP server code
COPY mcp-server/ /app/mcp-server/

# Set Python path
ENV PYTHONPATH=/app:$PYTHONPATH

# Set backend URL (can be overridden by docker-compose)
ENV BACKEND_URL=http://backend:8000

# Run the MCP server
CMD ["python", "mcp-server/server.py"]
