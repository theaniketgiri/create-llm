#!/bin/bash
set -e

# create-llm Docker Quick Setup Script
# This script sets up create-llm with Docker in one command

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Create-LLM Docker Quick Setup${NC}"
echo ""

# Function to print colored output
print_step() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check prerequisites
print_info "Checking prerequisites..."

# Check Docker
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed. Please install Docker first:"
    echo "  • Windows/Mac: https://www.docker.com/products/docker-desktop"  
    echo "  • Linux: https://docs.docker.com/engine/install/"
    exit 1
fi

# Check Docker daemon
if ! docker info &> /dev/null; then
    print_error "Docker daemon is not running. Please start Docker first."
    exit 1
fi

print_step "Docker is ready"

# Check for NVIDIA Docker (optional)
if command -v nvidia-docker &> /dev/null || docker run --rm --gpus all nvidia/cuda:12.1-base nvidia-smi &> /dev/null; then
    print_step "NVIDIA Docker detected - GPU training will be available"
    GPU_SUPPORT=true
else
    print_warning "NVIDIA Docker not detected - will use CPU-only training"
    GPU_SUPPORT=false
fi

echo ""

# Get user preferences
echo -e "${BLUE}📋 Setup Options${NC}"
echo ""

# Ask if they want to build locally or use pre-built image
echo "Choose installation method:"
echo "1) Build locally (latest features, takes ~5-10 minutes)"
echo "2) Use pre-built image (faster, but may not have latest features)"
echo ""
read -p "Enter choice (1-2): " BUILD_CHOICE

case $BUILD_CHOICE in
    1)
        BUILD_LOCAL=true
        print_info "Will build Docker image locally"
        ;;
    2)
        BUILD_LOCAL=false
        print_info "Will use pre-built Docker image"
        ;;
    *)
        print_warning "Invalid choice, defaulting to local build"
        BUILD_LOCAL=true
        ;;
esac

echo ""

# Ask for project setup
read -p "Do you want to create a sample project now? (y/n): " CREATE_PROJECT
if [[ $CREATE_PROJECT =~ ^[Yy]$ ]]; then
    read -p "Enter project name (default: my-first-llm): " PROJECT_NAME
    PROJECT_NAME=${PROJECT_NAME:-my-first-llm}
    
    echo "Choose template:"
    echo "1) NANO (1M params - learn in 2 minutes)"
    echo "2) TINY (6M params - prototype in 15 minutes)" 
    echo "3) SMALL (100M params - production quality)"
    echo "4) BASE (1B params - research grade)"
    echo ""
    read -p "Enter template choice (1-4): " TEMPLATE_CHOICE
    
    case $TEMPLATE_CHOICE in
        1) TEMPLATE="nano" ;;
        2) TEMPLATE="tiny" ;;
        3) TEMPLATE="small" ;;
        4) TEMPLATE="base" ;;
        *) 
            print_warning "Invalid choice, defaulting to TINY"
            TEMPLATE="tiny"
            ;;
    esac
fi

echo ""
echo -e "${BLUE}🔧 Starting Setup...${NC}"
echo ""

# Setup working directory
WORK_DIR="$HOME/create-llm-workspace"
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

if [ "$BUILD_LOCAL" = true ]; then
    # Clone repository if building locally
    if [ ! -d "create-llm" ]; then
        print_info "Cloning create-llm repository..."
        git clone https://github.com/theaniketgiri/create-llm.git
    else
        print_info "Repository already exists, updating..."
        cd create-llm
        git pull
        cd ..
    fi
    
    cd create-llm
    
    # Build Docker image
    print_info "Building Docker image (this may take 5-10 minutes)..."
    if make build; then
        print_step "Docker image built successfully"
    else
        print_error "Failed to build Docker image"
        exit 1
    fi
    
    IMAGE_NAME="create-llm:latest"
    cd ..
else
    # Use pre-built image
    print_info "Pulling pre-built Docker image..."
    if docker pull ghcr.io/theaniketgiri/create-llm:latest; then
        docker tag ghcr.io/theaniketgiri/create-llm:latest create-llm:latest
        print_step "Pre-built image ready"
        IMAGE_NAME="create-llm:latest"
    else
        print_warning "Failed to pull pre-built image, falling back to local build..."
        BUILD_LOCAL=true
        # Repeat local build logic here...
    fi
fi

# Test the installation
print_info "Testing installation..."
if docker run --rm "$IMAGE_NAME" --help > /dev/null; then
    print_step "Installation test passed"
else
    print_error "Installation test failed"
    exit 1
fi

# Create sample project if requested
if [[ $CREATE_PROJECT =~ ^[Yy]$ ]]; then
    print_info "Creating sample project: $PROJECT_NAME ($TEMPLATE template)..."
    
    if docker run -it -v "$PWD:/workspace" "$IMAGE_NAME" scaffold "$PROJECT_NAME" --template "$TEMPLATE"; then
        print_step "Project '$PROJECT_NAME' created successfully"
        
        # Show next steps
        echo ""
        echo -e "${BLUE}🎉 Setup Complete!${NC}"
        echo ""
        echo "Your project is ready at: $WORK_DIR/$PROJECT_NAME"
        echo ""
        echo "Next steps:"
        echo ""
        echo "1. Navigate to your project:"
        echo "   cd $WORK_DIR/$PROJECT_NAME"
        echo ""
        echo "2. Add your training data:"
        echo "   # Copy text files to data/raw/"
        echo ""
        if [ "$GPU_SUPPORT" = true ]; then
            echo "3. Start training (with GPU):"
            echo "   docker run --gpus all -v \$(pwd):/workspace $IMAGE_NAME train"
        else
            echo "3. Start training (CPU only):"
            echo "   docker run -v \$(pwd):/workspace $IMAGE_NAME train"
        fi
        echo ""
        echo "4. Chat with your model:"
        echo "   docker run -p 7860:7860 -v \$(pwd):/workspace $IMAGE_NAME chat"
        echo "   # Open http://localhost:7860 in your browser"
        echo ""
        echo "5. For more options:"
        echo "   docker run -v \$(pwd):/workspace $IMAGE_NAME --help"
        
        # Create helper script
        HELPER_SCRIPT="$WORK_DIR/$PROJECT_NAME/docker-commands.sh"
        cat > "$HELPER_SCRIPT" << EOF
#!/bin/bash
# Helper script for $PROJECT_NAME

echo "🐳 Create-LLM Docker Commands for $PROJECT_NAME"
echo ""
echo "Available commands:"
echo "  ./docker-commands.sh train    - Start training"
echo "  ./docker-commands.sh chat     - Start chat interface"  
echo "  ./docker-commands.sh evaluate - Evaluate model"
echo "  ./docker-commands.sh shell    - Open interactive shell"
echo ""

case "\$1" in
    "train")
        echo "🚀 Starting training..."
        docker run --gpus all -v \$(pwd):/workspace $IMAGE_NAME train \${@:2}
        ;;
    "chat")
        echo "💬 Starting chat interface at http://localhost:7860"
        docker run -p 7860:7860 -v \$(pwd):/workspace $IMAGE_NAME chat \${@:2}
        ;;
    "evaluate")
        echo "📊 Evaluating model..."
        docker run -v \$(pwd):/workspace $IMAGE_NAME evaluate \${@:2}
        ;;
    "shell")
        echo "🐚 Opening interactive shell..."
        docker run -it -v \$(pwd):/workspace $IMAGE_NAME shell
        ;;
    *)
        docker run -v \$(pwd):/workspace $IMAGE_NAME "\$@"
        ;;
esac
EOF
        chmod +x "$HELPER_SCRIPT"
        
        print_step "Helper script created: $PROJECT_NAME/docker-commands.sh"
        
    else
        print_error "Failed to create project"
        exit 1
    fi
else
    echo ""
    echo -e "${BLUE}🎉 Setup Complete!${NC}"
    echo ""
    echo "Create-LLM is ready to use with Docker!"
    echo ""
    echo "To create your first project:"
    echo "  docker run -it -v \$(pwd):/workspace $IMAGE_NAME scaffold my-project"
    echo ""
    echo "For help:"
    echo "  docker run $IMAGE_NAME --help"
fi

echo ""
echo -e "${GREEN}📚 Documentation:${NC}"
echo "  • Quick Start: https://github.com/theaniketgiri/create-llm#quick-start"
echo "  • Docker Guide: https://github.com/theaniketgiri/create-llm/blob/main/DOCKER.md"
echo "  • Full Documentation: https://github.com/theaniketgiri/create-llm#documentation"
echo ""
echo -e "${BLUE}Happy LLM training! 🤖${NC}"