# Makefile for Create-LLM Docker Management

.PHONY: help build build-all build-cli build-training test clean push

# Default target
help:
	@echo "🐳 Create-LLM Docker Management"
	@echo ""
	@echo "Available targets:"
	@echo "  build          - Build all-in-one Docker image"
	@echo "  build-all      - Build all Docker images"
	@echo "  build-cli      - Build CLI-only image"
	@echo "  build-training - Build training-only image"
	@echo "  test           - Run tests in Docker"
	@echo "  demo           - Run a complete demo"
	@echo "  clean          - Clean up Docker images and containers"
	@echo "  push           - Push images to registry"
	@echo ""
	@echo "Quick Start:"
	@echo "  make build && make demo"

# Docker image names
IMAGE_NAME = create-llm
CLI_IMAGE = $(IMAGE_NAME):cli
TRAINING_IMAGE = $(IMAGE_NAME):training
LATEST_IMAGE = $(IMAGE_NAME):latest

# Build targets
build:
	@echo "🏗️  Building all-in-one Docker image..."
	docker build -t $(LATEST_IMAGE) .
	@echo "✅ Built $(LATEST_IMAGE)"

build-cli:
	@echo "🏗️  Building CLI-only Docker image..."
	docker build -f Dockerfile.cli -t $(CLI_IMAGE) .
	@echo "✅ Built $(CLI_IMAGE)"

build-training:
	@echo "🏗️  Building training-only Docker image..."
	docker build -f Dockerfile.training -t $(TRAINING_IMAGE) .
	@echo "✅ Built $(TRAINING_IMAGE)"

build-all: build build-cli build-training
	@echo "✅ All Docker images built successfully!"

# Test targets
test: build
	@echo "🧪 Running tests..."
	@mkdir -p test-output
	@echo "Testing CLI functionality..."
	docker run --rm -v $(PWD)/test-output:/workspace $(LATEST_IMAGE) scaffold test-project --template nano --no-interactive
	@echo "Testing generated project structure..."
	@if [ -d "test-output/test-project" ]; then \
		echo "✅ Project scaffold test passed"; \
	else \
		echo "❌ Project scaffold test failed"; \
		exit 1; \
	fi
	@echo "Testing Python imports..."
	docker run --rm -v $(PWD)/test-output/test-project:/workspace $(LATEST_IMAGE) python -c "import torch; print('✅ PyTorch import successful')"
	@echo "✅ All tests passed!"

# Demo target
demo: build
	@echo "🚀 Running Create-LLM Demo"
	@echo ""
	@echo "This demo will:"
	@echo "1. Create a nano-sized LLM project"
	@echo "2. Show the generated structure"
	@echo "3. Run a quick training test"
	@echo ""
	@read -p "Press Enter to continue..." dummy
	
	@mkdir -p demo-output
	@echo "📁 Creating demo project..."
	docker run -it -v $(PWD)/demo-output:/workspace $(LATEST_IMAGE) scaffold demo-llm --template nano
	
	@echo ""
	@echo "📋 Generated project structure:"
	@docker run --rm -v $(PWD)/demo-output:/workspace $(LATEST_IMAGE) python -c "import os; [print(f'  {root[len(\"/workspace\"):].lstrip(\"/\")}/{file}') for root, dirs, files in os.walk('/workspace/demo-llm') for file in files[:3]]"
	
	@echo ""
	@echo "🧪 Testing training setup (dry run)..."
	@docker run --rm -v $(PWD)/demo-output/demo-llm:/workspace $(LATEST_IMAGE) python -c "from training.trainer import LLMTrainer; print('✅ Training modules imported successfully')"
	
	@echo ""
	@echo "✅ Demo completed! Check demo-output/demo-llm for the generated project."
	@echo "To start training: cd demo-output/demo-llm && docker run --gpus all -v \$$(pwd):/workspace $(LATEST_IMAGE) train"

# Development targets
dev: build
	@echo "🛠️  Starting development environment..."
	docker run -it --rm -v $(PWD):/workspace $(LATEST_IMAGE) shell

jupyter: build
	@echo "📓 Starting Jupyter Lab..."
	docker-compose --profile jupyter up jupyter

# Compose targets
compose-build:
	@echo "🐳 Building with Docker Compose..."
	docker-compose build

compose-cli:
	@echo "🏗️  Running CLI with Compose..."
	docker-compose --profile cli run --rm create-llm scaffold

compose-train:
	@echo "🚀 Starting training with Compose..."
	docker-compose --profile training run --rm training

compose-chat:
	@echo "💬 Starting chat interface..."
	@echo "Chat will be available at http://localhost:7860"
	docker-compose --profile chat up chat

# Cleanup targets
clean:
	@echo "🧹 Cleaning up Docker resources..."
	@docker container prune -f
	@docker image prune -f
	@docker volume prune -f
	@echo "Removing test and demo outputs..."
	@rm -rf test-output demo-output
	@echo "✅ Cleanup completed!"

clean-all: clean
	@echo "🗑️  Removing all create-llm images..."
	@docker images | grep create-llm | awk '{print $$3}' | xargs -r docker rmi -f
	@echo "✅ All images removed!"

# Registry targets (customize for your registry)
REGISTRY ?= your-registry.com
VERSION ?= latest

tag:
	@echo "🏷️  Tagging images for registry..."
	docker tag $(LATEST_IMAGE) $(REGISTRY)/$(IMAGE_NAME):$(VERSION)
	docker tag $(CLI_IMAGE) $(REGISTRY)/$(IMAGE_NAME):cli-$(VERSION)
	docker tag $(TRAINING_IMAGE) $(REGISTRY)/$(IMAGE_NAME):training-$(VERSION)

push: tag
	@echo "📤 Pushing images to registry..."
	docker push $(REGISTRY)/$(IMAGE_NAME):$(VERSION)
	docker push $(REGISTRY)/$(IMAGE_NAME):cli-$(VERSION)
	docker push $(REGISTRY)/$(IMAGE_NAME):training-$(VERSION)
	@echo "✅ Images pushed successfully!"

# Information targets
info:
	@echo "📊 Docker Environment Info"
	@echo "=========================="
	@echo "Docker version:"
	@docker --version
	@echo ""
	@echo "Docker Compose version:"
	@docker-compose --version
	@echo ""
	@echo "Available images:"
	@docker images | grep create-llm || echo "No create-llm images found"
	@echo ""
	@echo "System resources:"
	@docker system df
	@echo ""
	@if command -v nvidia-smi >/dev/null 2>&1; then \
		echo "GPU information:"; \
		nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; \
	else \
		echo "No NVIDIA GPU detected"; \
	fi

# Quick setup for new users
setup:
	@echo "🚀 Setting up Create-LLM Docker environment..."
	@echo ""
	@echo "Checking prerequisites..."
	@command -v docker >/dev/null 2>&1 || { echo "❌ Docker not found. Please install Docker first."; exit 1; }
	@command -v docker-compose >/dev/null 2>&1 || echo "⚠️  Docker Compose not found. Some features may not work."
	@echo "✅ Docker is available"
	@echo ""
	@echo "Building Docker images..."
	@$(MAKE) build
	@echo ""
	@echo "Running basic test..."
	@$(MAKE) test
	@echo ""
	@echo "🎉 Setup completed successfully!"
	@echo ""
	@echo "Next steps:"
	@echo "  1. Run 'make demo' to see a complete example"
	@echo "  2. Create your own project with 'make compose-cli'"
	@echo "  3. Check 'make help' for more options"

# Monitoring and logs
logs:
	@echo "📋 Docker Compose Logs:"
	docker-compose logs -f

status:
	@echo "📊 Service Status:"
	docker-compose ps

# Performance testing
benchmark: build
	@echo "⚡ Running performance benchmark..."
	@mkdir -p benchmark-output
	@echo "Creating small test project..."
	@docker run --rm -v $(PWD)/benchmark-output:/workspace $(LATEST_IMAGE) scaffold benchmark-test --template tiny --no-interactive
	@echo "Running training benchmark (100 steps)..."
	@time docker run --rm -v $(PWD)/benchmark-output/benchmark-test:/workspace $(LATEST_IMAGE) python training/train.py --max-steps 100 --no-tensorboard
	@echo "✅ Benchmark completed!"

# Security scan (requires Docker security tools)
security-scan:
	@echo "🔒 Running security scan..."
	@if command -v docker-scout >/dev/null 2>&1; then \
		docker scout cves $(LATEST_IMAGE); \
	elif command -v trivy >/dev/null 2>&1; then \
		trivy image $(LATEST_IMAGE); \
	else \
		echo "No security scanner found. Install docker-scout or trivy for security scanning."; \
	fi