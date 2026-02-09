# Face Analysis App

인사 사진 자동 분류 시스템 - 17,000개 파일에서 35명의 얼굴을 감지하고 분류합니다.

## 프로젝트 개요

이 프로젝트는 대용량의 이미지/비디오 파일에서 특정 인물들의 얼굴을 자동으로 감지하고 분류하는 시스템입니다.

## 현재 상태 (Current Status)

**진행률: 0% (0 / 17,302 파일 처리됨)**

**환경 설정 완료:**
- ✅ Python arm64 가상환경 설정 (venv_arm64)
- ✅ insightface 0.2.1 설치 완료
- ✅ 임베딩 생성 스크립트 실행 중 (백그라운드)

- **📋 프로젝트 문서**: [docs/face-analysis-app-handover.md](docs/face-analysis-app-handover.md)
- **🔍 의존성 검증**: [VERIFICATION_REPORT.md](VERIFICATION_REPORT.md)
- **📝 실수 내역**: [docs/MISTAKES_AND_CORRECTIONS.md](docs/MISTAKES_AND_CORRECTIONS.md)

### 실제 데이터 현황

- **📁 원본 소스**: `/Volumes/01022610461/_PRJ/entire`
  - 이미지: 12,550개
  - 동영상: 565개
  - 총계: 17,302개 (11GB)
  - 디스크 여유: 146GB (15% 남음 - ⚠️ 주의 필요)

- **📸 샘플 이미지**: 3,106개 확인 (person_01~person_07)
- **💾 데이터베이스**: 초기화됨 (metadata.db)
- **📤 출력 폴더**: 35개 person 폴더 생성됨 (모두 비어있음)
- **🔢 임베딩**: 미생성 (embeddings/ 비어있음)

### 준비 완료 항목

- ✅ config.yaml 설정 완료 (InsightFace buffalo_l 모델)
- ✅ 디렉토리 구조 준비 완료
- ✅ 데이터베이스 스키마 초기화
- ✅ 의존성 설치 (insightface 0.2.1 - 버전 차이 주의)

### 대기 중인 항목

- ⏳ 샘플 임베딩 벡터 생성 미진행
- ⏳ 얼굴 감지 및 분류 미진행
- ⏳ person_08~person_35 샘플 이미지 필요

### 다음 단계 (Next Steps)

다음 단계에 대한 자세한 내용은 [인수 문서의 섹션 10](docs/face-analysis-app-handover.md#section-10)을 참조하세요.

### 주요 기능

- **자동 얼굴 감지**: InsightFace 라이브러리를 사용한 고품질 얼굴 감지
- **임베딩 기반 분류**: 35명의 샘플 이미지에서 특징 추출 및 유사도 비교
- **GPU 가속**: Apple M1 Metal 가속 지원 (CoreMLExecutionProvider)
- **GUI 검토 인터페이스**: 낮은 신뢰도 결과에 대한 수동 검토 도구
- **병렬 처리**: 8코어 병렬 처리로 빠른 분류
- **체크포인트 시스템**: 중단된 작업 이어서 하기
- **SQLite 메타데이터**: 모든 분류 결과 데이터베이스 저장

## 시스템 요구사항

- macOS (Apple M1 칩)
- Python 3.9+
- 160GB 이상의 디스크 공간
- Xcode Command Line Tools (for Metal support)

## 설치 방법

### 🚀 빠른 시작 (권장)

```bash
# 저장소 클론
git clone https://github.com/ueno-ryu/face-analysis-app.git
cd face-analysis-app

# 자동화된 설치 실행
./setup.sh

# 완료!
```

### 📋 수동 설치

### 1. 가상환경 생성 및 활성화

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. 의존성 설치

```bash
pip install -r requirements.txt
```

### 3. 설정 파일 생성

```bash
cp config.yaml.example config.yaml
# config.yaml을 환경에 맞게 수정
```

### 4. 샘플 이미지 배치

`samples/person_01/` ~ `samples/person_35/` 디렉토리에 각각 20장의 샘플 이미지를 배치합니다.

## 사용법

### 전체 파이프라인 실행

```bash
python main.py --mode full
```

### 단계별 실행

```bash
# 샘플 임베딩 재생성
python main.py --mode rebuild-embeddings

# 스캔 및 분류만
python main.py --mode scan

# GUI 검토
python main.py --mode review

# 체크포인트에서 이어서
python main.py --mode resume
```

## 프로젝트 구조

```
face-analysis-app/
├── src/                    # 소스 코드 모듈
│   ├── detector.py         # 얼굴 감지
│   ├── recognizer.py       # 얼굴 매칭
│   ├── database.py         # SQLite 메타데이터
│   ├── checkpoint.py       # 복구 시스템
│   ├── classifier.py       # 메인 분류 파이프라인
│   └── reviewer.py         # GUI 검토 인터페이스
├── samples/                # 샘플 이미지 (35개 폴더)
├── embeddings/             # 캐시된 얼굴 임베딩
├── data/                   # metadata.db, checkpoint.json
├── logs/                   # 디버그 로그 및 에러 로그
├── review_queue/           # 수동 검토 대상 파일
├── error_files/            # 처리 실패 파일
└── classified_output/      # 최종 출력 (person_01/ ~ person_35/)
```

## 데이터베이스 스키마

### files 테이블
- `file_id`: 파일 고유 ID
- `original_path`: 원본 경로
- `file_type`: 파일 유형 (image/video)
- `status`: 처리 상태 (pending/processing/completed/error)

### detections 테이블
- `detection_id`: 감지 고유 ID
- `file_id`: 파일 ID
- `person_id`: 분류된 사람 ID
- `confidence`: 신뢰도 (0-1)
- `bbox_x1, bbox_y1, bbox_x2, bbox_y2`: 얼굴 바운딩 박스
- `needs_review`: 검토 필요 여부

### copies 테이블
- `copy_id`: 복사 고유 ID
- `file_id`: 파일 ID
- `person_id`: 대상 사람 ID
- `target_path`: 대상 경로

## 설정 옵션

`config.yaml`에서 다음 옵션을 설정할 수 있습니다:

- `confidence_threshold`: 분류 임계값 (기본값: 0.75)
- `parallel_workers`: 병렬 작업자 수 (기본값: 8)
- `video_sample_fps`: 비디오 샘플링 FPS (기본값: 2)
- `checkpoint_interval`: 체크포인트 저장 간격 (기본값: 100 파일)

## 로그 및 모니터링

- `logs/processing_YYYYMMDD.log`: 전체 처리 로그
- `logs/errors.log`: 에러만 별도 기록

## 성능 최적화

- M1 Metal 가속: `CoreMLExecutionProvider` 자동 사용
- 실패 시 CPU fallback: `CPUExecutionProvider`
- 병렬 처리: CPU 코어 수에 맞춰 자동 설정

## 문제 해결

### GPU 가속이 작동하지 않을 때
```yaml
# config.yaml
providers:
  - "CPUExecutionProvider"  # CPU로 전환
```

### 샘플 이미지에서 얼굴이 감지되지 않을 때
- 이미지가 명확한지 확인
- 얼굴이 정면을 향하고 있는지 확인
- 조명이 적절한지 확인

### 메모리 부족 오류
- `batch_size`를 줄이세요
- `parallel_workers`를 줄이세요

## 라이선스

MIT License

## 기여

이 프로젝트에 기여하고 싶으시다면 Pull Request를 제출해주세요.

## 연락처

GitHub Issues: https://github.com/ueno-ryu/face-analysis-app/issues
