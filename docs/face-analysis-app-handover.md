# Face Analysis App - 프로젝트 인계 문서

**작성일**: 2026-02-09  
**데드라인**: 2026-02-09 (당일 완료 필수)  
**Repository**: https://github.com/ueno-ryu/face-analysis-app  
**Manager**: リュスケ (관리자님)

---

## 1. 프로젝트 개요

### 1.1 핵심 목적
대량의 밴드 앨범 미디어 파일(약 17,000개)을 얼굴 인식 기술로 자동 분류하여, 1번부터 35번까지 ID가 부여된 인물별 폴더에 체계적으로 정리하는 시스템 구축.

### 1.2 프로젝트 특징
- **다중 인물 처리**: 한 사진에 여러 명이 등장하므로, N번 사진에 1~5번 인물이 확인된 경우 각 폴더(1, 2, 3, 4, 5번)에 해당 사진을 복사 배치
- **혼합 미디어**: 주로 이미지 파일이나 동영상도 포함
- **정확도 우선**: 낮은 인식도 항목은 관리자 수동 검토 후 배치
- **안전성 중심**: 원본 파일 보존 + 복사본 배치 방식

### 1.3 핵심 제약사항

| 항목 | 내용 |
|------|------|
| 데드라인 | 2026-02-09 (당일 완료) |
| 총 파일 수 | 약 17,000개 (대다수 이미지, 일부 동영상) |
| 인물 수 | 35명 (ID: 1-35번) |
| 샘플 이미지 | 인물당 약 20개 (총 700개) |
| 작업 환경 | macOS (Metal 가속 사용 시도) |

---

## 2. 기술 스택 확정

### 2.1 승인된 기술 선택

| 구성 요소 | 채택 기술 | 효율성 | 근거 |
|-----------|----------|--------|------|
| 얼굴 인식 엔진 | InsightFace (buffalo_l 모델) | 9.50/10.00 | 정확도-속도 균형 최적, 대량 파일 처리 적합 |
| GPU 가속 | ONNX Runtime + CoreML (Metal) | 8.80/10.00 | macOS Metal 지원, CPU 대비 3-5배 성능 |
| 병렬 처리 | multiprocessing.Pool | 9.40/10.00 | GIL 제약 없음, CPU 코어 수 기반 자동 조정 |
| GUI 프레임워크 | Tkinter + PIL (Pillow) | 9.50/10.00 | 표준 라이브러리, 가벼운 리뷰 인터페이스 구현 |
| 메타데이터 저장 | SQLite | 9.70/10.00 | 구조화된 쿼리, 트랜잭션 지원, 17K 규모 효율 관리 |
| 원본 파일 처리 | 원본 보존 + 복사본 배치 | 9.80/10.00 | 안전성 우선, 재처리 및 수정 가능 |
| 동영상 샘플링 | 1초당 2-3 프레임 | 9.30/10.00 | 처리 시간 최적화, 충분한 인물 감지 |
| Threshold 전략 | 동적 조정 (초기 0.75) | 9.60/10.00 | 데이터 기반 적응형 조정 |

### 2.2 의존성 패키지 목록

```bash
# 필수 패키지
insightface
onnxruntime-silicon  # macOS Metal 가속
opencv-python
pillow
numpy
sqlite3  # Python 표준 라이브러리
pyyaml
tqdm  # 프로그레스 바
colorama  # 터미널 색상
```

---

## 3. 시스템 아키텍처

### 3.1 파이프라인 단계

```
[a] 얼굴 인식 영역 생성
    ↓
[b] 인물 샘플 등록 (1-35번, 각 20개)
    ↓
[b-1] 머신러닝 학습 (샘플 수 증가 시 정확도 향상)
    ↓
[c] 전수 스캔 및 자동 분류
    ├─ confidence ≥ threshold → 자동 배치
    └─ confidence < threshold → 검토 대기열
    ↓
[d] 관리자 검토 (GUI)
    ├─ 얼굴 영역 클릭 → ID 지정
    └─ 미인식 영역 → 수동 바운더리 박스 그리기
    ↓
[d-1] 검토 완료 후 최종 배치
```

### 3.2 데이터 흐름

```
원본 파일 (17,000개)
    ↓
[InsightFace 얼굴 감지]
    ↓
각 얼굴 영역별 임베딩 벡터 추출
    ↓
샘플 벡터와 유사도 비교 (코사인 유사도)
    ↓
confidence ≥ threshold?
    ├─ YES → 복사본 생성 → 해당 인물 폴더 배치
    └─ NO → 검토 대기열 추가
    ↓
SQLite에 메타데이터 기록
    - 파일 경로, 인식된 인물 ID, confidence, 복사 위치
```

### 3.3 병렬 처리 구조

```python
# 개념적 구조
with multiprocessing.Pool(processes=CPU_CORES) as pool:
    batches = split_files_into_batches(all_files, batch_size=100)
    results = pool.map(process_batch, batches)
    
# 각 워커 프로세스:
# 1. 배치 내 파일 순차 처리
# 2. 얼굴 감지 → 임베딩 → 매칭 → 분류
# 3. 결과를 메인 프로세스로 반환
# 4. 메인 프로세스가 SQLite 업데이트 (동시성 제어)
```

---

## 4. 디렉토리 구조 설계

### 4.1 Repository 구조

```
face-analysis-app/
│
├── config.yaml                 # 설정 파일
├── main.py                     # 메인 진입점
├── requirements.txt            # 패키지 의존성
├── README.md                   # 프로젝트 문서
│
├── src/                        # 소스 코드
│   ├── __init__.py
│   ├── detector.py            # 얼굴 감지 모듈
│   ├── recognizer.py          # 얼굴 인식 및 매칭
│   ├── classifier.py          # 파일 분류 로직
│   ├── reviewer.py            # GUI 검토 인터페이스
│   ├── database.py            # SQLite 메타데이터 관리
│   ├── checkpoint.py          # 체크포인트 시스템
│   └── utils.py               # 유틸리티 함수
│
├── samples/                    # 인물별 샘플 이미지
│   ├── person_01/             # 1번 인물 샘플 (20개)
│   ├── person_02/
│   ├── ...
│   └── person_35/
│
├── embeddings/                 # 샘플 임베딩 벡터 캐시
│   ├── person_01.npy
│   ├── person_02.npy
│   └── ...
│
├── data/                       # 런타임 데이터
│   ├── metadata.db            # SQLite 데이터베이스
│   └── checkpoint.json        # 체크포인트 파일
│
├── logs/                       # 로그 파일
│   ├── processing_20260209.log
│   └── errors.log
│
├── review_queue/               # 검토 대기 파일 임시 저장
│
└── error_files/                # 처리 실패 파일 격리
```

### 4.2 출력 디렉토리 구조

```
[Manager 지정 경로]/classified_output/
│
├── person_01/                  # 1번 인물 폴더
│   ├── image_001.jpg
│   ├── image_045.jpg
│   └── video_012.mp4
│
├── person_02/
├── ...
└── person_35/
```

**중요**: 한 파일이 여러 폴더에 복사될 수 있음 (다중 인물 처리)

---

## 5. 구현 요구사항 상세

### 5.1 단계별 구현 명세

#### [a] 얼굴 인식 영역 생성

```python
# InsightFace 모델 로드
from insightface.app import FaceAnalysis
app = FaceAnalysis(providers=['CoreMLExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# 얼굴 감지
faces = app.get(image)
# 반환값: [Face 객체] (bbox, landmarks, embedding 포함)
```

#### [b] 인물 샘플 등록

```python
# 각 인물별 샘플 이미지 처리
for person_id in range(1, 36):
    sample_images = load_samples(f"samples/person_{person_id:02d}/")
    embeddings = []
    for img in sample_images:
        faces = app.get(img)
        if len(faces) == 1:  # 정확히 1개 얼굴만 있는 샘플 사용
            embeddings.append(faces[0].embedding)
    
    # 평균 임베딩 또는 전체 임베딩 리스트 저장
    np.save(f"embeddings/person_{person_id:02d}.npy", embeddings)
```

#### [c] 전수 스캔 및 분류

```python
def process_file(file_path, threshold=0.75):
    img = cv2.imread(file_path)
    faces = app.get(img)
    
    results = []
    for face in faces:
        best_match = None
        best_similarity = -1
        
        for person_id in range(1, 36):
            sample_embeddings = np.load(f"embeddings/person_{person_id:02d}.npy")
            similarities = [cosine_similarity(face.embedding, emb) for emb in sample_embeddings]
            max_sim = max(similarities)
            
            if max_sim > best_similarity:
                best_similarity = max_sim
                best_match = person_id
        
        if best_similarity >= threshold:
            results.append({'person_id': best_match, 'confidence': best_similarity, 'bbox': face.bbox})
        else:
            # 검토 대기열에 추가
            results.append({'person_id': None, 'confidence': best_similarity, 'bbox': face.bbox, 'needs_review': True})
    
    return results
```

**동영상 처리 특수 로직**:
```python
def process_video(video_path, threshold=0.75, sample_fps=2):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps / sample_fps)
    
    detected_persons = set()
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            faces = app.get(frame)
            for face in faces:
                # [c]와 동일한 매칭 로직
                person_id, confidence = match_face(face)
                if confidence >= threshold:
                    detected_persons.add(person_id)
        
        frame_count += 1
    
    cap.release()
    
    # 검출된 모든 인물 폴더에 동영상 전체 복사
    return list(detected_persons)
```

#### [d] 관리자 검토 GUI

**Tkinter + PIL 구조**:
```python
import tkinter as tk
from PIL import Image, ImageTk, ImageDraw

class ReviewGUI:
    def __init__(self, review_queue):
        self.root = tk.Tk()
        self.canvas = tk.Canvas(width=1200, height=800)
        self.current_image = None
        self.bboxes = []  # [(x1, y1, x2, y2, person_id, confidence), ...]
        
    def load_next_image(self):
        # review_queue에서 다음 이미지 로드
        # 자동 감지된 bbox + 예측 ID 오버레이
        pass
    
    def on_bbox_click(self, event):
        # 클릭한 bbox 식별
        # 1-35번 ID 선택 팝업 표시
        pass
    
    def on_drag_start(self, event):
        # 새 bbox 그리기 시작
        pass
    
    def on_drag_end(self, event):
        # bbox 완성 → ID 선택 팝업
        pass
    
    def save_and_next(self):
        # 현재 이미지의 모든 bbox → 해당 인물 폴더에 복사
        # SQLite에 기록
        # 다음 이미지 로드
        pass
```

**키보드 단축키** (미확인 - Manager 확인 필요):
- `1-9`: 빠른 ID 지정
- `Space`: 다음 이미지
- `S`: 건너뛰기 (나중에 재검토)

#### [d-1] 부분 인식 케이스 처리

```
시나리오: 사진 X에 3명 얼굴 감지
- 얼굴 A: person_05, confidence 0.92 → 자동 배치
- 얼굴 B: person_12, confidence 0.88 → 자동 배치
- 얼굴 C: person_??, confidence 0.62 → 검토 대기

처리 방식:
1. 얼굴 A, B → person_05, person_12 폴더에 사진 X 즉시 복사
2. 사진 X를 review_queue/에도 복사 (얼굴 C 검토용)
3. GUI에서 Manager가 얼굴 C를 person_18로 지정
4. person_18 폴더에 사진 X 추가 복사
5. SQLite에 (file_X, person_05), (file_X, person_12), (file_X, person_18) 3개 레코드 생성
```

---

### 5.2 SQLite 데이터베이스 스키마

```sql
-- 파일 정보 테이블
CREATE TABLE files (
    file_id INTEGER PRIMARY KEY AUTOINCREMENT,
    original_path TEXT NOT NULL UNIQUE,
    file_type TEXT NOT NULL,  -- 'image' or 'video'
    file_size INTEGER,
    processed_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    status TEXT DEFAULT 'pending'  -- 'pending', 'processed', 'error'
);

-- 얼굴 감지 결과 테이블
CREATE TABLE detections (
    detection_id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id INTEGER NOT NULL,
    person_id INTEGER,  -- NULL이면 미인식
    confidence REAL,
    bbox_x1 INTEGER,
    bbox_y1 INTEGER,
    bbox_x2 INTEGER,
    bbox_y2 INTEGER,
    needs_review BOOLEAN DEFAULT 0,
    FOREIGN KEY (file_id) REFERENCES files(file_id)
);

-- 복사 이력 테이블
CREATE TABLE copies (
    copy_id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id INTEGER NOT NULL,
    person_id INTEGER NOT NULL,
    target_path TEXT NOT NULL,
    copied_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (file_id) REFERENCES files(file_id)
);

-- 체크포인트 테이블
CREATE TABLE checkpoints (
    checkpoint_id INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_number INTEGER NOT NULL,
    processed_files_count INTEGER NOT NULL,
    checkpoint_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    status TEXT DEFAULT 'active'  -- 'active', 'completed'
);

-- 처리된 파일 추적 (병렬 처리용)
CREATE TABLE processed_files (
    file_id INTEGER PRIMARY KEY,
    worker_id INTEGER,
    status TEXT,  -- 'processing', 'completed', 'failed'
    processed_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (file_id) REFERENCES files(file_id)
);

-- 인덱스 생성 (성능 최적화)
CREATE INDEX idx_detections_file_id ON detections(file_id);
CREATE INDEX idx_detections_person_id ON detections(person_id);
CREATE INDEX idx_copies_file_id ON copies(file_id);
CREATE INDEX idx_copies_person_id ON copies(person_id);
```

---

### 5.3 설정 파일 (config.yaml)

```yaml
project:
  name: "Face Analysis App"
  version: "1.0.0"
  deadline: "2026-02-09"

paths:
  source_directory: "/path/to/17000_files"  # Manager 지정 필요
  output_directory: "/path/to/classified_output"  # Manager 지정 필요
  samples_directory: "./samples"
  embeddings_directory: "./embeddings"
  database_path: "./data/metadata.db"
  review_queue: "./review_queue"
  error_files: "./error_files"
  logs_directory: "./logs"

recognition:
  model_name: "buffalo_l"  # InsightFace 모델
  confidence_threshold: 0.75  # 초기값, 동적 조정 가능
  det_size: [640, 640]
  providers: ["CoreMLExecutionProvider", "CPUExecutionProvider"]

processing:
  batch_size: 100
  parallel_workers: 8  # CPU 코어 수 기반 조정
  video_sample_fps: 2  # 1초당 2프레임 샘플링
  checkpoint_interval: 100  # 100개 파일마다 체크포인트 저장
  max_retries: 3  # 에러 발생 시 재시도 횟수

thresholds:
  auto_adjust: true  # 동적 threshold 조정 활성화
  adjustment_batch_size: 500  # 500개 처리 후 통계 분석
  target_review_ratio_min: 0.10  # 검토 대상 10% 미만 시 threshold 상향
  target_review_ratio_max: 0.30  # 검토 대상 30% 초과 시 threshold 하향
  adjustment_step: 0.05

gui:
  window_width: 1200
  window_height: 800
  image_display_max_width: 1000
  image_display_max_height: 700

logging:
  level: "INFO"  # DEBUG, INFO, WARNING, ERROR
  max_log_size_mb: 10
  backup_count: 5
  
monitoring:
  terminal_refresh_rate: 0.5  # 0.5초마다 터미널 업데이트
  show_progress_bar: true
  use_colors: true
```

---

## 6. 성능 및 품질 기준

### 6.1 정량적 목표

| 메트릭 | 목표 값 | 측정 방법 |
|--------|---------|-----------|
| 처리 완료 시간 | 24시간 이내 (2월 9일 내) | 전체 파이프라인 종료 시각 |
| 이미지 처리 속도 | 5-10개/초 (Metal 가속 시) | 배치 처리 시간 측정 |
| 동영상 처리 속도 | 30-60초/파일 | 개별 동영상 처리 시간 |
| 자동 분류 비율 | 70-90% | (자동 배치 / 전체) × 100 |
| 검토 대상 비율 | 10-30% | (검토 대기 / 전체) × 100 |
| 에러 발생률 | < 1% | (에러 파일 / 전체) × 100 |

### 6.2 Threshold 동적 조정 전략

```python
def adjust_threshold(current_threshold, review_ratio, config):
    target_min = config['target_review_ratio_min']
    target_max = config['target_review_ratio_max']
    step = config['adjustment_step']
    
    if review_ratio < target_min:
        # 검토 대상이 너무 적음 → threshold 상향 (더 엄격하게)
        new_threshold = min(current_threshold + step, 0.95)
        print(f"⬆ Threshold 상향: {current_threshold:.2f} → {new_threshold:.2f}")
        return new_threshold
    elif review_ratio > target_max:
        # 검토 대상이 너무 많음 → threshold 하향 (더 관대하게)
        new_threshold = max(current_threshold - step, 0.60)
        print(f"⬇ Threshold 하향: {current_threshold:.2f} → {new_threshold:.2f}")
        return new_threshold
    else:
        print(f"✓ Threshold 유지: {current_threshold:.2f} (적정 범위)")
        return current_threshold
```

**조정 타이밍**: 500개 파일 처리 후 Manager에게 제안 → 엔터키로 승인/거부

---

## 7. 에러 처리 및 재시작 전략

### 7.1 에러 계층 구조

| 에러 수준 | 유형 | 처리 방식 |
|-----------|------|-----------|
| **파일 수준** | 파일 손상, 읽기 실패 | `error_files/`로 이동 → 로그 기록 → 다음 파일 진행 |
| **처리 수준** | 얼굴 인식 실패, 모델 에러 | 최대 3회 재시도 → 실패 시 검토 대기열 추가 |
| **시스템 수준** | 메모리 부족, 디스크 공간 부족 | 즉시 전체 중단 → 체크포인트 저장 → 에러 리포트 |

### 7.2 체크포인트 재시작

```python
def resume_from_checkpoint():
    conn = sqlite3.connect('data/metadata.db')
    cursor = conn.cursor()
    
    # 마지막 완료된 체크포인트 조회
    cursor.execute("""
        SELECT batch_number, processed_files_count 
        FROM checkpoints 
        WHERE status = 'completed' 
        ORDER BY checkpoint_id DESC 
        LIMIT 1
    """)
    last_checkpoint = cursor.fetchone()
    
    if last_checkpoint:
        batch_num, processed_count = last_checkpoint
        print(f"🔄 체크포인트에서 재시작: Batch {batch_num}, {processed_count}개 파일 처리 완료")
        
        # 미처리 파일 목록 추출
        cursor.execute("""
            SELECT file_id, original_path 
            FROM files 
            WHERE status = 'pending' OR status = 'error'
            ORDER BY file_id
        """)
        remaining_files = cursor.fetchall()
        return remaining_files
    else:
        print("ℹ️  체크포인트 없음 - 처음부터 시작")
        return get_all_files()
```

### 7.3 병렬 처리 재시도 로직

```python
def process_batch_with_retry(batch, max_retries=3):
    results = []
    for file_path in batch:
        retry_count = 0
        success = False
        
        while retry_count < max_retries and not success:
            try:
                result = process_file(file_path)
                results.append(result)
                success = True
            except Exception as e:
                retry_count += 1
                logger.warning(f"⚠ 재시도 {retry_count}/{max_retries}: {file_path} - {str(e)}")
                time.sleep(1)  # 재시도 전 대기
        
        if not success:
            logger.error(f"❌ 처리 실패: {file_path} (최대 재시도 초과)")
            move_to_error_folder(file_path)
    
    return results
```

---

## 8. 터미널 모니터링 UI

### 8.1 실시간 대시보드 구조

```
====================================================================
  Face Analysis App - 진행 상황
====================================================================
전체 진행률: [████████████░░░░░░░░] 60.5% (10,285 / 17,000 파일)
예상 남은 시간: 4시간 23분

현재 배치: Batch #103 (10,201 - 10,300)
현재 파일: /path/to/album_2024/IMG_5432.jpg

--------------------------------------------------------------------
실시간 통계
--------------------------------------------------------------------
✓ 처리 완료:     10,285 파일
✓ 자동 분류:      8,120 파일 (79.0%)
⚠ 검토 대기:      2,050 파일 (19.9%)
❌ 에러 발생:        115 파일 (1.1%)

인물별 분류 현황 (Top 5):
  #05: 1,234 파일 | #12: 1,089 파일| #03: 892 파일
  #18: 765 파일  | #22: 623 파일

--------------------------------------------------------------------
현재 Threshold: 0.75 (자동 조정 활성화)
병렬 워커: 8개 프로세스 가동 중
--------------------------------------------------------------------
[로그] 10:34:21 - INFO - Batch #103 처리 시작
[로그] 10:34:25 - WARNING - 낮은 confidence 검출: IMG_5432.jpg (0.68)
====================================================================
```

### 8.2 구현 라이브러리

```python
from tqdm import tqdm
from colorama import Fore, Style, init

init(autoreset=True)

# 프로그레스 바
with tqdm(total=17000, desc="전체 진행률", unit="파일") as pbar:
    for batch in batches:
        process_batch(batch)
        pbar.update(len(batch))

# 색상 코딩
print(Fore.GREEN + "✓ 처리 완료: 10,285 파일")
print(Fore.YELLOW + "⚠ 검토 대기: 2,050 파일")
print(Fore.RED + "❌ 에러 발생: 115 파일")
```

---

## 9. oh-my-claudecode 통합 인터페이스

### 9.1 CLI 진입점

```bash
# 전체 파이프라인 실행
python main.py --mode full

# 얼굴 감지 및 분류만 (검토 제외)
python main.py --mode scan

# 검토 GUI만 실행
python main.py --mode review

# 체크포인트에서 재시작
python main.py --mode resume

# 샘플 임베딩 재생성
python main.py --mode rebuild-embeddings
```

### 9.2 Orchestration 상태 모니터링

**status.json** (실시간 업데이트):
```json
{
  "timestamp": "2026-02-09T10:34:25Z",
  "status": "processing",
  "current_batch": 103,
  "total_files": 17000,
  "processed_files": 10285,
  "auto_classified": 8120,
  "needs_review": 2050,
  "errors": 115,
  "current_threshold": 0.75,
  "estimated_completion": "2026-02-09T18:45:00Z"
}
```

---

## 10. 미확인 사항 체크리스트

**다음 Claude 인스턴스가 Manager님께 확인해야 할 질문 목록입니다.**

### 10.1 필수 확인 사항 (즉시 확인 필요)

| # | 질문 | 중요도 | 영향 범위 |
|---|------|--------|-----------|
| 1 | **원본 파일 디렉토리 구조**: 17,000개 파일이 단일 평면 폴더에 있습니까, 아니면 서브폴더 계층 구조입니까? | 🔴 높음 | 파일 스캔 로직 |
| 2 | **원본 파일 절대 경로**: `/path/to/17000_files` 실제 경로를 지정해주십시오 | 🔴 높음 | config.yaml |
| 3 | **출력 디렉토리 절대 경로**: `/path/to/classified_output` 실제 경로를 지정해주십시오 | 🔴 높음 | config.yaml |
| 4 | **샘플 이미지 준비 상태**: 35명 × 20개 = 700개 샘플이 이미 준비되어 있습니까? | 🔴 높음 | 파이프라인 시작 가능 여부 |
| 5 | **디스크 여유 공간**: 원본 파일 총 용량과 출력 디렉토리의 여유 공간을 확인해주십시오 | 🟡 중간 | 복사 작업 가능 여부 |

### 10.2 선택적 확인 사항 (구현 중 결정 가능)

| # | 질문 | 기본값 | 비고 |
|---|------|--------|------|
| 6 | **GUI 키보드 단축키**: 1-9 숫자키로 빠른 ID 지정 기능이 필요합니까? | 미구현 | 검토 효율성 향상 |
| 7 | **파일 명명 규칙**: 원본 파일명에 패턴이 있습니까? (예: `YYYYMMDD_album_001.jpg`) | 없음 | 메타데이터 추출 가능성 |
| 8 | **병렬 워커 수**: CPU 코어 수 기반 자동 설정(8개) vs 수동 지정? | 자동(8) | 성능 최적화 |
| 9 | **로그 레벨**: INFO(표준) vs DEBUG(상세) vs WARNING(경고만)? | INFO | 로깅 세부 정도 |
| 10 | **하드링크 사용**: 디스크 공간 절약을 위해 하드링크 사용 시도? | 미사용 | 복사본 방식 대신 |

### 10.3 기술적 검증 필요 사항

| # | 항목 | 검증 방법 |
|---|------|-----------|
| 11 | **macOS Metal 가속**: M1/M2/M3 칩 여부 및 `onnxruntime-silicon` 설치 가능 여부 | `python -c "import onnxruntime; print(onnxruntime.get_available_providers())"` 실행 |
| 12 | **InsightFace 모델 다운로드**: buffalo_l 모델 자동 다운로드 또는 수동 설치 필요 | 첫 실행 시 자동 다운로드 시도, 실패 시 수동 설치 안내 |
| 13 | **SQLite 쓰기 권한**: `./data/` 디렉토리 생성 및 쓰기 권한 확인 | 초기화 스크립트에서 자동 생성 시도 |

---

## 11. Manager 선호 프로토콜

### 11.1 커뮤니케이션 규칙

**호칭**: "Manager (관리자)님" 사용  
**어조**: 겸손하고 전문적, 진지하고 절제된 톤, 유머 최소화  
**언어**: 한국어 기본, 기술 용어는 영어 유지 후 한국어 설명 병기  

### 11.2 핵심 프로토콜

**Rule 0 (자율 실행)**:
- DESIGN 단계에서는 권한 요청 금지
- 최적 설계/일정/전략을 구조적 근거와 함께 즉시 생성
- Manager 확인 후 EXECUTION 단계 진입

**Rule 1 (Canvas 제한)**:
- Canvas(Artifact) 사용은 명시적 허가 없이 절대 금지
- 파일 생성은 `/mnt/user-data/outputs/`에 직접 저장

**Rule 2 (실행 승인)**:
- 모든 실행은 Manager 승인 필요
- 단, 세션 내 권한 위임 시 자율 실행 가능 (삭제/수정 제외)

### 11.3 응답 형식

**마무리**: 모든 응답은 정확히 3개의 "명확화 질문"으로 종료  
- 단, 모호함이 없고 작업이 완전히 자기완결적이면 "질문 없음." 명시

**형식**:
- 테이블 형식 선호
- 이모지 최소화, HTML 태그 최소화
- 다이어그램은 효율성 정당화 시만 사용 (3개 이상 관계 or 복잡한 흐름)

**Triple-Option Rule**:
- 모든 제안은 3개 옵션을 테이블로 제시
- 각 옵션을 Efficiency & Feasibility 두 축으로 `/10.00` 평가 (예: 9.50/10.00)

**Quest System**:
- 작업을 Quest로 제시
- Progress Bar: `[##########----------]` (진행률 = 완료/전체 × 100%)
- Todo List 포함

---

## 12. 다음 단계 행동 지침

### 12.1 즉시 수행 작업 (Manager 확인 후)

1. **미확인 사항 질의** (섹션 10 체크리스트 기반)
   - 원본 경로, 출력 경로, 샘플 준비 상태 확인
   
2. **Repository 초기화**
   ```bash
   cd face-analysis-app
   mkdir -p src data logs embeddings review_queue error_files
   mkdir -p samples/person_{01..35}
   ```

3. **의존성 설치**
   ```bash
   pip install insightface onnxruntime-silicon opencv-python pillow numpy pyyaml tqdm colorama
   ```

4. **config.yaml 생성**
   - Manager 제공 경로로 수정

5. **샘플 임베딩 생성**
   ```bash
   python main.py --mode rebuild-embeddings
   ```

### 12.2 구현 우선순위

| 우선순위 | 모듈 | 이유 |
|---------|------|------|
| P0 (최우선) | `detector.py`, `recognizer.py` | 핵심 얼굴 인식 기능 |
| P0 | `database.py` | 메타데이터 관리 필수 |
| P0 | `checkpoint.py` | 재시작 전략 필수 (deadline 압박) |
| P1 (높음) | `classifier.py` | 파일 분류 및 복사 로직 |
| P1 | `main.py` | CLI 진입점 및 orchestration |
| P2 (중간) | `reviewer.py` | GUI 검토 인터페이스 |
| P3 (낮음) | `utils.py` | 유틸리티 함수 |

### 12.3 테스트 전략

**단계적 검증**:
1. 샘플 10개 파일로 전체 파이프라인 테스트
2. 배치 100개로 성능 및 병렬 처리 검증
3. Threshold 동적 조정 로직 검증
4. 체크포인트 재시작 시나리오 테스트
5. 전체 17,000개 파일 처리

---

## 13. 위험 요소 및 완화 전략

| 위험 요소 | 발생 가능성 | 영향도 | 완화 전략 |
|-----------|-------------|--------|-----------|
| 샘플 이미지 부족 (< 20개/인물) | 🟡 중간 | 🔴 높음 | 가용 샘플로 먼저 진행, 추가 샘플 점진적 추가 |
| Metal 가속 미작동 | 🟡 중간 | 🟡 중간 | CPU fallback 자동 전환, 처리 시간 증가 감수 |
| 디스크 공간 부족 | 🟢 낮음 | 🔴 높음 | 사전 공간 확인, 필요 시 하드링크 전환 |
| 동영상 처리 병목 | 🟡 중간 | 🟡 중간 | 샘플링 FPS 조정 (2 → 1), 병렬 처리 강화 |
| 검토 대상 과다 (> 30%) | 🟡 중간 | 🔴 높음 | Threshold 하향 조정, 샘플 품질 개선 |
| 처리 중단 (시스템 재시작) | 🟢 낮음 | 🟡 중간 | 체크포인트 시스템으로 복구 |

---

## 14. 성공 기준

**프로젝트 완료 조건**:
✅ 17,000개 파일 중 최소 99% 처리 완료 (에러 < 1%)  
✅ 각 인물 폴더에 해당 인물이 포함된 모든 파일 복사 배치  
✅ SQLite 메타데이터에 전체 분류 이력 기록  
✅ 2026-02-09 23:59:59 이전 완료  

**품질 기준**:
✅ 자동 분류 비율 70-90%  
✅ 검토 대상 비율 10-30%  
✅ Manager 검토 작업 시간 < 전체 처리 시간의 20%  

---

## 15. 참고 자료

**InsightFace 공식 문서**: https://github.com/deepinsight/insightface  
**ONNX Runtime CoreML Provider**: https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html  
**Tkinter 공식 문서**: https://docs.python.org/3/library/tkinter.html  
**SQLite Python API**: https://docs.python.org/3/library/sqlite3.html  

---

## 문서 버전

**Version**: 1.0.0  
**작성일**: 2026-02-09  
**작성자**: Claude (Sonnet 4.5)  
**검토자**: リュスケ (Manager)  
**다음 업데이트**: Manager 피드백 반영 후

---

**이 문서를 다음 Claude 인스턴스에게 전달하면, 섹션 10의 체크리스트를 기반으로 Manager님께 즉시 질문하고 프로젝트를 이어받을 수 있습니다.**
