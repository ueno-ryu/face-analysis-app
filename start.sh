#!/bin/bash
# Face Analysis App - Quick Start Script
# 간단한 작업 시작을 위한 스크립트

# 가상환경 활성화
source venv/bin/activate

# 사용법 출력
echo "🎯 Face Analysis App - Quick Start"
echo "=================================="
echo ""
echo "Available commands:"
echo ""
echo "  1. Rebuild embeddings:"
echo "     python main.py --mode rebuild-embeddings"
echo ""
echo "  2. Scan and classify:"
echo "     python main.py --mode scan"
echo ""
echo "  3. Review low-confidence detections:"
echo "     python main.py --mode review"
echo ""
echo "  4. Full pipeline (scan + review):"
echo "     python main.py --mode full"
echo ""
echo "  5. Resume from checkpoint:"
echo "     python main.py --mode resume"
echo ""
echo "Or use this interactive mode:"
echo "  bash start.sh"
