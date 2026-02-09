# 재개 방법 (Resume Instructions)

**일시 중단 시점**: 2026-02-10 00:50
**완료된 임베딩**: 30/35 (85%)
**PID**: 41716 (종료됨)

## 완료된 파일
person_01 ~ person_30.npy (30개, 88MB)

## 남은 파일
person_31 ~ person_35 (5개)

## 재개 방법

### 1. 임베딩 생성 재개
```bash
cd /Volumes/01022610461/_PRJ/face-analysis-app
arch -arm64 python3 src/generate_embeddings_deepface.py
```

### 2. 전체 파이프라인 시작 (임베딩 완료 후)
```bash
# 전체 분류 실행
python main.py --mode full

# 또는 스캔만 실행
python main.py --mode scan
```

## 주의사항
- 시스템 Python 사용 (venv_arm64 아님)
- DeepFace VGG-Face + RetinaFace 사용
- 처리 속도: 약 1.2초/이미지
