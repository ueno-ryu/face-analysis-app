# Face Analysis App - Manager Handoff Document

## ✅ Implementation Status: COMPLETE

All core modules have been implemented and committed to local Git.

---

## 🚀 Immediate Action Required for Manager

### Step 1: Create GitHub Repository

1. Go to https://github.com/new
2. Create a new repository named `face-analysis-app`
3. **DO NOT** initialize with README, .gitignore, or license (we already have them)
4. Click "Create repository"
5. Run these commands in the project directory:

```bash
cd /Volumes/01022610461/_PRJ/face-analysis-app
git remote remove origin
git remote add origin https://github.com/ueno-ryu/face-analysis-app.git
git push -u origin main
```

### Step 2: Place Sample Images (700 images total)

**CRITICAL**: The system needs sample images to generate embeddings before it can classify files.

For each person ID from 1 to 35:
1. Navigate to `/Volumes/01022610461/_PRJ/face-analysis-app/samples/`
2. Place **20 high-quality sample images** into `person_XX/` folder
3. Each sample image must contain **exactly one clear face** of that person

Example structure:
```
samples/
├── person_01/
│   ├── sample_01.jpg
│   ├── sample_02.jpg
│   └── ... (20 images)
├── person_02/
│   └── ... (20 images)
...
└── person_35/
    └── ... (20 images)
```

**Sample Image Requirements:**
- ✅ Clear, well-lit frontal face photos
- ✅ High resolution (preferably >500x500)
- ✅ Neutral expression or variety of expressions
- ❌ No photos with multiple people
- ❌ No blurry or low-quality images
- ❌ No extreme angles or side profiles

### Step 3: Install Dependencies

```bash
cd /Volumes/01022610461/_PRJ/face-analysis-app

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Note**: The first run will download the InsightFace buffalo_l model (~300MB). This is automatic.

### Step 4: Generate Sample Embeddings

Once samples are placed:

```bash
python main.py --mode rebuild-embeddings
```

This will:
1. Load all 700 sample images (20 per person × 35 persons)
2. Detect faces and extract embeddings
3. Save embeddings to `embeddings/person_XX.npy` files

**Expected output:**
- 35 `.npy` files in `embeddings/` directory
- Each file contains embeddings for one person

### Step 5: Run Full Classification

```bash
python main.py --mode full
```

This will:
1. Scan all 17,000 files from `/Volumes/01022610461/_PRJ/entire/`
2. Detect and classify faces
3. Copy files to `classified_output/person_XX/` folders
4. Launch GUI for low-confidence detections

**Estimated time**: 2-4 hours (depending on file sizes)

---

## 📋 Execution Modes Reference

| Mode | Command | Purpose |
|------|---------|---------|
| **Rebuild Embeddings** | `python main.py --mode rebuild-embeddings` | Regenerate embeddings from sample images |
| **Scan Only** | `python main.py --mode scan` | Classify files without GUI review |
| **Review Only** | `python main.py --mode review` | Launch GUI for manual review |
| **Full Pipeline** | `python main.py --mode full` | Scan + Review (recommended) |
| **Resume** | `python main.py --mode resume` | Resume from checkpoint |

---

## 📁 Project Structure

```
face-analysis-app/
├── main.py                 # Entry point - run this!
├── config.yaml             # Configuration (M1 Metal enabled)
├── requirements.txt        # Python dependencies
├── README.md              # Full documentation (Korean)
├── .gitignore             # Git ignore rules
│
├── src/                   # Core modules
│   ├── detector.py        # Face detection (InsightFace)
│   ├── recognizer.py      # Face matching (cosine similarity)
│   ├── database.py        # SQLite metadata
│   ├── checkpoint.py      # Resumable processing
│   ├── classifier.py      # Main pipeline
│   └── reviewer.py        # GUI review interface
│
├── samples/               # ← MANAGER: Place 700 images here
│   ├── person_01/        # 20 images
│   ├── person_02/        # 20 images
│   └── ...               # (35 folders total)
│
├── embeddings/            # Auto-generated embeddings
│   ├── person_01.npy
│   ├── person_02.npy
│   └── ...               # (35 files)
│
├── data/                  # Database & checkpoints
│   ├── metadata.db       # SQLite database
│   └── checkpoint.json   # Processing state
│
├── logs/                  # Execution logs
│   ├── processing_*.log  # DEBUG level logs
│   └── errors.log        # Error-only log
│
├── classified_output/     # Final classified files
│   ├── person_01/        # All files with person 01
│   ├── person_02/        # All files with person 02
│   └── ...               # (35 folders)
│
├── review_queue/          # Low-confidence files (auto-created)
└── error_files/           # Failed files (auto-created)
```

---

## ⚙️ Key Configuration Settings

**File**: `config.yaml`

| Setting | Value | Description |
|---------|-------|-------------|
| `confidence_threshold` | 0.75 | Initial classification threshold |
| `parallel_workers` | 8 | CPU cores to use |
| `video_sample_fps` | 2 | Video frame sampling rate |
| `checkpoint_interval` | 100 | Save checkpoint every 100 files |
| `providers` | CoreML, CPU | GPU acceleration with CPU fallback |

**Auto-adjustment**: Threshold automatically adjusts every 500 files based on review ratio:
- If review ratio < 10%: Increase threshold (more strict)
- If review ratio > 30%: Decrease threshold (more lenient)

---

## 🎯 Success Criteria

The system will be successful when:

✅ **Processing**: 99% of 17,000 files processed (error rate < 1%)
✅ **Accuracy**: Each person folder contains all files with that person
✅ **Metadata**: SQLite database records all classifications
✅ **Timeline**: Completed by 2026-02-09 23:59:59

---

## 🐛 Troubleshooting

### Issue: "No sample embeddings found"

**Solution**: Run `python main.py --mode rebuild-embeddings` after placing sample images.

### Issue: "CoreMLExecutionProvider not available"

**Solution**: The system automatically falls back to CPU. No action needed.

### Issue: "Sample directory not found"

**Solution**: Ensure samples/person_01/ through samples/person_35/ exist and contain images.

### Issue: "Database locked"

**Solution**: Close any database viewers (DB Browser for SQLite) and retry.

### Issue: Out of memory

**Solution**: Reduce `parallel_workers` in config.yaml from 8 to 4.

---

## 📊 Monitoring Progress

### Real-time Logs

```bash
# Follow main log
tail -f logs/processing_*.log

# Follow errors only
tail -f logs/errors.log
```

### Database Queries

```bash
# Connect to database
sqlite3 data/metadata.db

# Check progress
SELECT status, COUNT(*) FROM files GROUP BY status;

# Check detections per person
SELECT person_id, COUNT(*) FROM detections
WHERE person_id IS NOT NULL
GROUP BY person_id
ORDER BY person_id;

# Check review queue
SELECT COUNT(*) FROM detections WHERE needs_review = 1;
```

---

## 🔄 Checkpoint & Resume

The system automatically saves checkpoints every 100 files.

**To resume interrupted processing:**

```bash
python main.py --mode resume
```

The checkpoint stores:
- Number of files processed
- Last processed file
- Current confidence threshold
- Batch number

---

## 📈 Performance Estimates

**Based on M1 Metal acceleration:**

- **Images**: ~50-100 images/second
- **Videos**: ~2-5 videos/second (depends on length)
- **Total time**: 2-4 hours for 17,000 files

**Bottlenecks:**
- Video processing (frame sampling)
- High-resolution images
- Multiple faces in one image

---

## 🤝 Support

For issues or questions:

1. Check `logs/errors.log` for error details
2. Review this handoff document
3. Check README.md for full documentation

---

## ✅ Pre-Flight Checklist

Before running `--mode full`:

- [ ] GitHub repository created and pushed
- [ ] All 35 sample directories exist
- [ ] 20 images placed in each person_XX folder (700 total)
- [ ] Dependencies installed via `pip install -r requirements.txt`
- [ ] Sample embeddings generated via `--mode rebuild-embeddings`
- [ ] Source directory path correct in config.yaml
- [ ] At least 160GB disk space available

---

## 🎉 Ready to Launch!

Once all checklist items are complete, run:

```bash
python main.py --mode full
```

The system will:
1. ✅ Detect faces in all 17,000 files
2. ✅ Classify them into 35 person folders
3. ✅ Launch GUI for low-confidence detections
4. ✅ Save progress every 100 files
5. ✅ Generate completion report

**Estimated completion**: 2-4 hours

---

*Generated: 2026-02-09*
*Implementation: Claude Sonnet 4.5*
