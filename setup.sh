#!/bin/bash

# =========================================
# AI Image Screener - Setup Script
# Run this after cloning the repository
# =========================================

set -e  # Exit on error

echo "================================================"
echo "AI Image Screener - Setup"
echo "================================================"
echo ""

# Check Python version
echo "📌 Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.11"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then 
    echo "❌ Error: Python 3.11+ required (found $python_version)"
    exit 1
fi
echo "✅ Python $python_version detected"
echo ""

# Create virtual environment
echo "📦 Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✅ Virtual environment created"
else
    echo "⚠️  Virtual environment already exists"
fi
echo ""

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source venv/bin/activate || {
    echo "❌ Failed to activate virtual environment"
    exit 1
}
echo "✅ Virtual environment activated"
echo ""

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip setuptools wheel > /dev/null 2>&1
echo "✅ pip upgraded"
echo ""

# Install dependencies
echo "📚 Installing dependencies..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    echo "✅ Dependencies installed"
else
    echo "❌ Error: requirements.txt not found"
    exit 1
fi
echo ""

# Create directories
echo "📁 Creating required directories..."
mkdir -p data/uploads data/reports data/cache logs
touch data/uploads/.gitkeep
touch data/reports/.gitkeep
touch data/cache/.gitkeep
touch logs/.gitkeep
echo "✅ Directories created"
echo ""

# Create .env file if not exists
echo "⚙️  Setting up environment..."
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo "✅ Created .env from .env.example"
        echo "   ⚠️  Please review and update .env with your settings"
    else
        echo "⚠️  .env.example not found, skipping .env creation"
    fi
else
    echo "⚠️  .env already exists"
fi
echo ""

# Check system dependencies
echo "🔍 Checking system dependencies..."
missing_deps=()

if ! command -v identify &> /dev/null; then
    missing_deps+=("ImageMagick")
fi

if [ ${#missing_deps[@]} -gt 0 ]; then
    echo "⚠️  Optional dependencies missing:"
    for dep in "${missing_deps[@]}"; do
        echo "   - $dep"
    done
    echo "   The app will work, but some features may be limited."
else
    echo "✅ All optional dependencies present"
fi
echo ""

# Test import
echo "🧪 Testing installation..."
if python3 -c "import fastapi, cv2, numpy, scipy, PIL, reportlab" 2>/dev/null; then
    echo "✅ All core packages import successfully"
else
    echo "❌ Some packages failed to import"
    echo "   Try: pip install -r requirements.txt"
    exit 1
fi
echo ""

echo "================================================"
echo "✨ Setup Complete!"
echo "================================================"
echo ""
echo "Next steps:"
echo ""
echo "1. Review and update .env file (optional)"
echo "2. Start the server:"
echo "   $ source venv/bin/activate"
echo "   $ python app.py"
echo ""
echo "3. Open browser:"
echo "   http://localhost:8005"
echo ""
echo "4. Or build Docker image:"
echo "   $ docker build -t ai-image-screener ."
echo "   $ docker run -p 7860:7860 ai-image-screener"
echo ""
echo "📖 Documentation: docs/"
echo "🐛 Issues: https://github.com/satyakimitra/ai-image-screener/issues"
echo ""