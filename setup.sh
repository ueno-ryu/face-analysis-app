#!/bin/bash
# Face Analysis App - Quick Setup Script
# 자동화된 환경 설정 스크립트

set -e  # 에러 발생 시 중지

echo "🚀 Face Analysis App - Quick Setup"
echo "=================================="
echo ""

# 1. 가상환경 생성
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

# 2. 가상환경 활성화
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# 3. 의존성 설치
echo "📥 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
echo "✅ Dependencies installed"

# 4. config.yaml 생성
if [ ! -f "config.yaml" ]; then
    echo "⚙️  Creating config.yaml from example..."
    if [ -f "config.yaml.example" ]; then
        cp config.yaml.example config.yaml
        echo "✅ config.yaml created"
        echo "⚠️  Please edit config.yaml with your paths:"
        echo "   - source_directory: /Volumes/01022610461/_PRJ/entire/"
        echo "   - output_directory: ./classified_output/"
    else
        echo "❌ config.yaml.example not found!"
        exit 1
    fi
else
    echo "✅ config.yaml already exists"
fi

# 5. 필수 디렉토리 생성
echo "📁 Creating required directories..."
mkdir -p data logs embeddings review_queue error_files classified_output
echo "✅ Directories created"

# 6. 샘플 이미지 확인
echo "🔍 Checking sample images..."
SAMPLE_COUNT=$(find samples -type f -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" 2>/dev/null | wc -l | tr -d ' ')
if [ "$SAMPLE_COUNT" -gt 0 ]; then
    echo "✅ Found $SAMPLE_COUNT sample images"
else
    echo "⚠️  No sample images found!"
    echo "   Please place sample images in samples/person_01/ through samples/person_35/"
fi

echo ""
echo "🎉 Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Edit config.yaml if needed"
echo "  2. Run: python main.py --mode rebuild-embeddings"
echo "  3. Run: python main.py --mode scan"
echo ""
