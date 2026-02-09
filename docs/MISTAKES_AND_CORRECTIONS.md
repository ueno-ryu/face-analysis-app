# 실수 내역 및 교정 기록 (Mistakes and Corrections)

**Date**: 2026-02-09

## Section 1: Initial Issues

### Issue 1: Displayed hypothetical monitoring UI as real progress
- **Description**: Initially showed a monitoring interface with mock/placeholder data as if it represented actual project progress
- **Impact**: User confusion about actual project status and completion level
- **Correction**: Clarified that actual progress was 0% and provided transparent status updates

## Section 2: Configuration Issues

### Issue 2: config.yaml had DeepFace settings (VGG-Face, retinaface)
- **Description**: Initial configuration file contained DeepFace-specific model settings
- **Impact**: Wrong model configuration could lead to incorrect face detection results
- **Correction**: Updated to use InsightFace buffalo_l with CoreML providers for optimal performance on Apple Silicon

## Section 3: Documentation Issues

### Issue 3: README showed hypothetical status
- **Description**: README contained placeholder/mock status information
- **Impact**: Inaccurate representation of project scope and progress
- **Correction**: Updated with real data:
  - Total files: 17,302
  - Actual progress: 0% (initially)
  - Dataset size: 11GB

### Issue 4: Python/numpy Architecture Mismatch (2026-02-09 23:50)
- **Problem**: Embedding generation failed with architecture error
- **Error**: "mach-o file, but is an incompatible architecture (have 'arm64', need 'x86_64')"
- **Root Cause**: Python running in x86_64 (Rosetta) mode, numpy installed for arm64
- **Impact**: Blocking all embedding generation and classification work
- **Correction**: Use `arch -arm64 python3` to run scripts with native Apple Silicon architecture
- **Status**: ✅ Fixed - embeddings generation now running with correct architecture

## Section 4: Timeline of corrections

1. **2026-02-09**: Initial project setup with incorrect configurations
2. **2026-02-09**: Updated config.yaml with correct InsightFace settings
3. **2026-02-09**: Revised README with accurate project information
4. **2026-02-09**: Clarified project status and progress reporting

## Section 5: Lessons learned

### Technical Lessons
- Always verify model compatibility with target hardware (CoreML for Apple Silicon)
- Configuration files should be tested before documenting progress
- Real data should always be used instead of placeholder/mock data

### Project Management Lessons
- Transparency about project status is crucial for stakeholder trust
- Documentation should accurately reflect current project state
- Configuration changes should be validated against actual requirements

### Quality Assurance
- Implement verification steps for all configuration changes
- Review documentation for accuracy before sharing
- Clear separation between development/testing data and production data

### Communication
- When displaying progress information, clearly distinguish between real and hypothetical data
- Update all project documentation when making configuration changes
- Be transparent about what is complete versus what is planned

This document serves as a record of issues encountered and their resolutions to prevent similar mistakes in future projects.