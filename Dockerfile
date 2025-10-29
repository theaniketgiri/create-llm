# Multi-stage Dockerfile for create-llm
# This provides both CLI scaffolding and Python training capabilities

# Stage 1: Build the TypeScript CLI
FROM node:18-alpine AS cli-builder

WORKDIR /app

# Copy package files
COPY package*.json ./
COPY tsconfig.json ./

# Install all dependencies for build
RUN npm install
RUN npm install -g typescript
COPY src/ ./src/
COPY templates/ ./templates/
RUN npm run build

# Clean up dev dependencies to reduce image size
RUN npm prune --omit=dev

# Stage 2: Runtime with both Node.js and Python
FROM python:3.11-slim

# Install Node.js
RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    && curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Copy built CLI from previous stage
COPY --from=cli-builder /app/dist /usr/local/lib/create-llm/dist
COPY --from=cli-builder /app/templates /usr/local/lib/create-llm/templates
COPY --from=cli-builder /app/package.json /usr/local/lib/create-llm/
COPY --from=cli-builder /app/node_modules /usr/local/lib/create-llm/node_modules

# Create global CLI command
RUN ln -s /usr/local/lib/create-llm/dist/index.js /usr/local/bin/create-llm && \
    chmod +x /usr/local/bin/create-llm

# Install common Python dependencies that most projects will need
RUN pip install --no-cache-dir \
    torch>=2.0.0 \
    transformers>=4.30.0 \
    tokenizers>=0.13.0 \
    tqdm>=4.65.0 \
    numpy>=1.24.0 \
    tabulate>=0.9.0 \
    datasets>=2.14.0 \
    tensorboard>=2.13.0 \
    matplotlib>=3.7.0 \
    gradio>=4.0.0 \
    pytest>=7.4.0 \
    pytest-cov>=4.1.0

# Set environment variables
ENV PYTHONPATH=/workspace
ENV NODE_PATH=/usr/local/lib/node_modules

# Create entrypoint script
COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["--help"]