# Quy trình Phát triển Phần mềm — Nhóm 6 (8-10 người)

> **Dự án:** Hệ thống Tự động Căn chỉnh và Làm rõ nét Ảnh chụp Tài liệu
> **Tham khảo:** Git Flow (Atlassian), Trunk-based Development (Google), GitLab Flow

---

## 1. Đánh giá Hiện trạng Codebase

### Vấn đề hiện tại

| # | Vấn đề | Mức độ nghiêm trọng | Ảnh hưởng |
|---|--------|---------------------|-----------|
| 1 | Chưa khởi tạo Git repository | **Nghiêm trọng** | Không thể phối hợp, không có lịch sử thay đổi |
| 2 | Không có cấu trúc thư mục project | **Nghiêm trọng** | 8-10 người sẽ tạo file lung tung, xung đột liên tục |
| 3 | Không có `.gitignore` | **Cao** | Commit nhầm file rác, notebook checkpoint, `__pycache__` |
| 4 | Không có `requirements.txt` | **Cao** | Mỗi người cài phiên bản thư viện khác nhau → bug "chạy máy tôi được" |
| 5 | Không có quy ước đặt tên, code style | **Cao** | Code review khó, merge conflict nhiều |
| 6 | Không có CI/CD | **Trung bình** | Không tự động kiểm tra code trước khi merge |
| 7 | Chỉ có 1 file tài liệu `.md` | **Thông tin** | Cần bổ sung code implementation |

---

## 2. Cấu trúc Project đề xuất

```
document-scanner/
│
├── README.md                      # Giới thiệu project, hướng dẫn cài đặt & chạy
├── BaiToan.md                     # Mô tả bài toán (đã có)
├── TEAM_WORKFLOW.md               # File này — quy trình phối hợp team
├── requirements.txt               # Các thư viện Python cần cài
├── .gitignore                     # File/thư mục không track bởi Git
├── .flake8                        # Cấu hình code style (tùy chọn)
│
├── src/                           # === SOURCE CODE CHÍNH ===
│   ├── __init__.py
│   ├── main.py                    # Entry point — pipeline chính
│   │
│   ├── detection/                 # Module 1: Phát hiện vùng tài liệu
│   │   ├── __init__.py
│   │   ├── edge_detector.py       # Canny edge detection
│   │   ├── contour_finder.py      # Tìm contour, xấp xỉ đa giác
│   │   └── document_locator.py    # Kết hợp edge + contour → 4 đỉnh
│   │
│   ├── transform/                 # Module 2: Biến đổi phối cảnh
│   │   ├── __init__.py
│   │   ├── perspective.py         # Perspective warp
│   │   └── utils.py               # Sắp xếp đỉnh, tính kích thước
│   │
│   ├── enhancement/               # Module 3: Xử lý ánh sáng & tương phản
│   │   ├── __init__.py
│   │   ├── adaptive_threshold.py  # Nhị phân hóa thích nghi
│   │   ├── morphology.py          # Phép toán hình thái học
│   │   └── shadow_removal.py      # Loại bỏ bóng đổ
│   │
│   ├── sharpening/                # Module 4: Tăng cường chi tiết & khử nhiễu
│   │   ├── __init__.py
│   │   ├── unsharp_mask.py        # Unsharp masking
│   │   ├── highpass_filter.py     # Bộ lọc thông cao
│   │   └── denoising.py           # Bilateral filter, NLM
│   │
│   └── utils/                     # Tiện ích dùng chung
│       ├── __init__.py
│       ├── image_io.py            # Đọc/ghi ảnh
│       └── visualization.py       # Hiển thị kết quả
│
├── tests/                         # === UNIT TESTS ===
│   ├── __init__.py
│   ├── test_detection.py
│   ├── test_transform.py
│   ├── test_enhancement.py
│   └── test_sharpening.py
│
├── notebooks/                     # === JUPYTER NOTEBOOKS (thử nghiệm) ===
│   ├── 01_exploration.ipynb
│   ├── 02_detection_demo.ipynb
│   ├── 03_transform_demo.ipynb
│   ├── 04_enhancement_demo.ipynb
│   └── 05_full_pipeline.ipynb
│
├── data/                          # === DỮ LIỆU ===
│   ├── input/                     # Ảnh đầu vào mẫu
│   ├── output/                    # Kết quả xử lý
│   └── ground_truth/              # Ảnh chuẩn để so sánh (nếu có)
│
└── docs/                          # === TÀI LIỆU BỔ SUNG ===
    ├── architecture.md            # Mô tả kiến trúc
    └── api_reference.md           # Tham chiếu API các module
```

### Nguyên tắc phân chia

> **Mỗi module = 1 thư mục riêng → Mỗi nhóm nhỏ (2-3 người) phụ trách 1 module → Giảm xung đột file tối đa.**

| Module | Thư mục | Số người | Nhiệm vụ |
|--------|---------|----------|-----------|
| Detection | `src/detection/` | 2-3 | Phát hiện vùng tài liệu |
| Transform | `src/transform/` | 1-2 | Biến đổi phối cảnh |
| Enhancement | `src/enhancement/` | 2-3 | Xử lý ánh sáng, bóng đổ |
| Sharpening | `src/sharpening/` | 2-3 | Làm sắc nét, khử nhiễu |
| Integration | `src/main.py` + `tests/` | 1 (Lead) | Tích hợp pipeline, test |

---

## 3. Git Branching Strategy — Modified Git Flow

### 3.1. Sơ đồ nhánh

```
main (production-ready)
 │
 ├── develop (integration branch)
 │    │
 │    ├── feature/detection-canny-edge        ← Nhóm Detection
 │    ├── feature/detection-contour-finder     ← Nhóm Detection
 │    ├── feature/transform-perspective-warp   ← Nhóm Transform
 │    ├── feature/enhancement-adaptive-thresh  ← Nhóm Enhancement
 │    ├── feature/enhancement-shadow-removal   ← Nhóm Enhancement
 │    ├── feature/sharpening-unsharp-mask      ← Nhóm Sharpening
 │    ├── feature/sharpening-denoising         ← Nhóm Sharpening
 │    │
 │    ├── fix/detection-corner-case            ← Sửa bug
 │    └── refactor/utils-cleanup               ← Refactor
 │
 ├── release/v1.0                              ← Chuẩn bị release
 └── hotfix/critical-crash                     ← Sửa lỗi khẩn cấp trên main
```

### 3.2. Mô tả từng loại nhánh

| Nhánh | Tạo từ | Merge vào | Mục đích | Ai được merge? |
|-------|--------|-----------|----------|----------------|
| `main` | — | — | Code ổn định, đã test, sẵn sàng nộp bài | Chỉ Lead |
| `develop` | `main` | `main` (qua release) | Nhánh tích hợp, nơi các feature gộp lại | Lead review & merge |
| `feature/*` | `develop` | `develop` | Phát triển tính năng mới | Tác giả tạo MR, reviewer approve |
| `fix/*` | `develop` | `develop` | Sửa bug | Tương tự feature |
| `release/*` | `develop` | `main` + `develop` | Chuẩn bị phiên bản release | Lead |
| `hotfix/*` | `main` | `main` + `develop` | Sửa lỗi nghiêm trọng trên main | Lead |

### 3.3. Quy ước đặt tên nhánh

```
<type>/<module>-<mô-tả-ngắn>
```

**Ví dụ:**
- `feature/detection-canny-edge`
- `feature/enhancement-adaptive-threshold`
- `fix/transform-corner-sorting-bug`
- `refactor/utils-image-io-cleanup`

**Type hợp lệ:** `feature`, `fix`, `refactor`, `docs`, `test`, `experiment`

---

## 4. Quy trình Phát triển một Tính năng Mới (Step-by-step)

### Bước 1: Nhận task & Tạo nhánh

```bash
# 1. Đảm bảo develop mới nhất
git checkout develop
git pull origin develop

# 2. Tạo nhánh feature mới
git checkout -b feature/detection-canny-edge

# 3. Kiểm tra đang ở đúng nhánh
git branch
```

### Bước 2: Phát triển (Code)

```bash
# Code trong module của mình
# Ví dụ: chỉ sửa file trong src/detection/

# Commit thường xuyên với message rõ ràng
git add src/detection/edge_detector.py
git commit -m "feat(detection): implement Canny edge detection with auto threshold"

# Tiếp tục code...
git add src/detection/edge_detector.py tests/test_detection.py
git commit -m "test(detection): add unit tests for edge detection"
```

### Bước 3: Sync với develop (QUAN TRỌNG — tránh xung đột)

```bash
# Trước khi tạo MR, luôn rebase lên develop mới nhất
git checkout develop
git pull origin develop
git checkout feature/detection-canny-edge
git rebase develop

# Nếu có conflict → giải quyết từng file → git add → git rebase --continue
# Nếu quá phức tạp → git rebase --abort và dùng merge thay thế:
# git merge develop
```

### Bước 4: Push & Tạo Merge Request (MR)

```bash
# Push nhánh lên remote
git push origin feature/detection-canny-edge
```

Sau đó vào GitLab/GitHub tạo MR với format:

```
Title: feat(detection): Implement Canny edge detection module

## Mô tả
- Implement hàm detect_edges() sử dụng Canny Edge Detector
- Tự động tính ngưỡng dựa trên median của ảnh
- Hỗ trợ tùy chỉnh sigma cho Gaussian blur

## Thay đổi
- Thêm mới: src/detection/edge_detector.py
- Thêm mới: tests/test_detection.py
- Sửa: src/detection/__init__.py (export mới)

## Cách test
1. Chạy: python -m pytest tests/test_detection.py -v
2. Hoặc mở notebook: notebooks/02_detection_demo.ipynb

## Screenshots (nếu có)
[Ảnh before/after]

## Checklist
- [ ] Code chạy được, không lỗi
- [ ] Có docstring cho hàm public
- [ ] Đã viết test
- [ ] Đã rebase lên develop mới nhất
- [ ] Không commit file rác (.pyc, checkpoint, data lớn)
```

### Bước 5: Code Review & Approve

**Quy tắc review:**
- Mỗi MR cần **ít nhất 1 reviewer** (người khác module) approve
- Reviewer kiểm tra:
  - Code có chạy được không
  - Logic có đúng không
  - Có test không
  - Code style có nhất quán không
  - Có ảnh hưởng module khác không
- Nếu cần sửa → Comment → Tác giả sửa & push thêm commit → Review lại

### Bước 6: Merge & Cleanup

```bash
# Sau khi MR được approve, Lead merge vào develop (trên GitLab/GitHub)
# Sau đó, tác giả xóa nhánh cũ:
git checkout develop
git pull origin develop
git branch -d feature/detection-canny-edge
git push origin --delete feature/detection-canny-edge
```

---

## 5. Quy ước Commit Message

Sử dụng **Conventional Commits** (chuẩn của Angular, Vue, nhiều dự án lớn):

```
<type>(<scope>): <mô tả ngắn gọn>

[body - tùy chọn: giải thích chi tiết]
```

### Type

| Type | Khi nào dùng | Ví dụ |
|------|-------------|-------|
| `feat` | Thêm tính năng mới | `feat(detection): add contour finder with Douglas-Peucker` |
| `fix` | Sửa bug | `fix(transform): correct corner point sorting order` |
| `refactor` | Tái cấu trúc (không thay đổi behavior) | `refactor(enhancement): extract shadow removal to separate function` |
| `test` | Thêm/sửa test | `test(sharpening): add tests for bilateral filter parameters` |
| `docs` | Cập nhật tài liệu | `docs: update README with installation steps` |
| `style` | Format code (không thay đổi logic) | `style: fix PEP8 warnings in detection module` |
| `chore` | Cập nhật build, config, dependencies | `chore: add numpy to requirements.txt` |

### Scope (phạm vi)

| Scope | Module |
|-------|--------|
| `detection` | `src/detection/` |
| `transform` | `src/transform/` |
| `enhancement` | `src/enhancement/` |
| `sharpening` | `src/sharpening/` |
| `utils` | `src/utils/` |
| `pipeline` | `src/main.py` |
| *(bỏ trống)* | Toàn project |

---

## 6. Quy trình Release

### 6.1. Khi nào release?

```
Sprint/Milestone kết thúc
       │
       ▼
Tất cả feature branches đã merge vào develop?
       │
       ├── CHƯA → Chờ hoặc hoãn feature sang sprint sau
       │
       └── RỒI → Tạo release branch
                     │
                     ▼
              ┌──────────────────┐
              │ release/v1.0     │
              │ • Fix bug cuối   │
              │ • Cập nhật docs  │
              │ • Test toàn bộ   │
              └────────┬─────────┘
                       │
                       ▼
              Merge vào main + tag version
              Merge ngược vào develop
```

### 6.2. Các bước release chi tiết

```bash
# 1. Tạo nhánh release từ develop
git checkout develop
git pull origin develop
git checkout -b release/v1.0

# 2. Trên release branch — CHỈ sửa bug, cập nhật version, docs
#    KHÔNG thêm feature mới

# 3. Test toàn bộ pipeline
python -m pytest tests/ -v
python src/main.py --input data/input/ --output data/output/

# 4. Khi ổn định → Merge vào main
git checkout main
git merge release/v1.0
git tag -a v1.0 -m "Release v1.0: Full document scanning pipeline"
git push origin main --tags

# 5. Merge ngược vào develop (đảm bảo develop có các bugfix từ release)
git checkout develop
git merge release/v1.0
git push origin develop

# 6. Xóa nhánh release
git branch -d release/v1.0
git push origin --delete release/v1.0
```

### 6.3. Hotfix (sửa lỗi khẩn cấp trên main)

```bash
# 1. Tạo hotfix từ main
git checkout main
git checkout -b hotfix/fix-crash-on-grayscale

# 2. Sửa lỗi, test

# 3. Merge vào BOTH main và develop
git checkout main
git merge hotfix/fix-crash-on-grayscale
git tag -a v1.0.1 -m "Hotfix: fix crash on grayscale input"

git checkout develop
git merge hotfix/fix-crash-on-grayscale
```

---

## 7. Chiến lược Giảm Xung đột cho Team 8-10 người

### 7.1. Nguyên tắc vàng

| # | Nguyên tắc | Giải thích |
|---|-----------|------------|
| 1 | **Mỗi người chỉ sửa file trong module mình** | Detection team không sửa file trong `src/enhancement/` |
| 2 | **Pull develop hàng ngày** | `git pull origin develop` mỗi sáng trước khi code |
| 3 | **Feature branch ngắn** | Mỗi nhánh tồn tại tối đa 2-3 ngày, tránh diverge quá xa |
| 4 | **Commit nhỏ, thường xuyên** | Dễ resolve conflict hơn commit lớn |
| 5 | **Không sửa file chung khi không cần** | `main.py`, `__init__.py` chỉ Lead sửa |
| 6 | **Communicate trước khi sửa file chung** | Nhắn nhóm: "Tôi cần sửa main.py, ai đang sửa không?" |

### 7.2. File có nguy cơ xung đột cao

| File | Nguy cơ | Giải pháp |
|------|---------|-----------|
| `src/main.py` | **Rất cao** | Chỉ Lead sửa, các module expose API qua `__init__.py` |
| `requirements.txt` | **Cao** | Thêm dependency qua MR riêng, không gộp với feature MR |
| `src/utils/*.py` | **Cao** | Mỗi người thêm function mới ở cuối file, không sửa function có sẵn |
| `notebooks/*.ipynb` | **Rất cao** | Mỗi người có notebook riêng, không chia sẻ notebook |
| `__init__.py` | **Trung bình** | Chỉ thêm import mới, không sửa import cũ |

### 7.3. Phân công cụ thể cho 8-10 người

```
┌─────────────────────────────────────────────────────────────┐
│                        TEAM LEAD (1 người)                  │
│  • Quản lý nhánh main, develop                              │
│  • Review & merge MR                                        │
│  • Viết src/main.py (pipeline integration)                  │
│  • Giải quyết conflict phức tạp                             │
└──────────────────────────┬──────────────────────────────────┘
                           │
         ┌─────────────────┼──────────────────┐
         │                 │                  │
         ▼                 ▼                  ▼
┌─────────────┐  ┌──────────────┐  ┌──────────────┐
│ Sub-team A  │  │ Sub-team B   │  │ Sub-team C   │
│ (2-3 người) │  │ (2-3 người)  │  │ (2-3 người)  │
│             │  │              │  │              │
│ Detection   │  │ Enhancement  │  │ Sharpening   │
│ + Transform │  │ + Shadow     │  │ + Denoising  │
│             │  │   Removal    │  │              │
│ Thư mục:    │  │ Thư mục:     │  │ Thư mục:     │
│ detection/  │  │ enhancement/ │  │ sharpening/  │
│ transform/  │  │              │  │              │
└─────────────┘  └──────────────┘  └──────────────┘
```

---

## 8. Quy trình Daily — Team hoạt động hàng ngày

### 8.1. Checklist mỗi sáng (5 phút)

```bash
# 1. Pull develop mới nhất
git checkout develop
git pull origin develop

# 2. Rebase nhánh feature của mình
git checkout feature/my-feature
git rebase develop

# 3. Kiểm tra có conflict không
#    - Nếu CÓ → giải quyết ngay
#    - Nếu KHÔNG → tiếp tục code
```

### 8.2. Standup meeting (15 phút, mỗi ngày hoặc cách ngày)

Mỗi người trả lời 3 câu:
1. **Hôm qua làm gì?** — "Hoàn thành hàm detect_edges(), đã push commit"
2. **Hôm nay làm gì?** — "Viết test cho detect_edges(), bắt đầu contour_finder"
3. **Có blocker không?** — "Cần utils/image_io.py hỗ trợ đọc ảnh từ URL"

### 8.3. Checklist trước khi tạo MR

- [ ] Code chạy được trên máy mình
- [ ] Đã chạy `python -m pytest tests/` — PASS
- [ ] Đã rebase lên develop mới nhất — không conflict
- [ ] Commit message theo quy ước
- [ ] Không commit file rác (`.pyc`, `__pycache__`, `.ipynb_checkpoints`, data lớn)
- [ ] MR description đầy đủ (mô tả, cách test, screenshots)

---

## 9. Ví dụ Thực tế: Phát triển module Shadow Removal

### Scenario: Thành viên Minh (Sub-team B) nhận task "Implement shadow removal"

**Ngày 1:**
```bash
# Minh pull develop và tạo nhánh
git checkout develop && git pull origin develop
git checkout -b feature/enhancement-shadow-removal

# Code shadow_removal.py
# Commit
git add src/enhancement/shadow_removal.py
git commit -m "feat(enhancement): implement background estimation using dilation"
```

**Ngày 2:**
```bash
# Sáng — sync develop
git checkout develop && git pull origin develop
git checkout feature/enhancement-shadow-removal
git rebase develop   # Không conflict vì chỉ sửa file riêng

# Tiếp tục code + viết test
git add src/enhancement/shadow_removal.py tests/test_enhancement.py
git commit -m "feat(enhancement): add shadow subtraction and normalization"
git commit -m "test(enhancement): add unit tests for shadow removal"
```

**Ngày 3:**
```bash
# Final sync + push
git rebase develop
git push origin feature/enhancement-shadow-removal

# Tạo MR trên GitLab/GitHub
# Assign reviewer: Lan (Sub-team B) + Lead
# Chờ review...
```

**Ngày 4:**
```bash
# Reviewer comment: "Nên thêm parameter cho kernel_size"
# Minh sửa theo comment
git add src/enhancement/shadow_removal.py
git commit -m "fix(enhancement): add configurable kernel_size parameter"
git push origin feature/enhancement-shadow-removal

# Reviewer approve → Lead merge vào develop
# Minh cleanup
git checkout develop && git pull origin develop
git branch -d feature/enhancement-shadow-removal
```

---

## 10. So sánh với Quy trình của các Đội nhóm lớn

| Tiêu chí | Google (Trunk-based) | Atlassian (Git Flow) | **Nhóm 6 (Đề xuất)** |
|----------|---------------------|---------------------|----------------------|
| Nhánh chính | 1 (main) | 2 (main + develop) | **2 (main + develop)** |
| Feature branch | Rất ngắn (<1 ngày) | Dài (tuần) | **2-3 ngày** |
| Code review | Bắt buộc (1+ reviewer) | Bắt buộc | **1 reviewer + Lead** |
| CI/CD | Tự động test mỗi commit | Tự động | **Thủ công (pytest)** |
| Release | Continuous (nhiều lần/ngày) | Scheduled | **Theo milestone** |
| Phù hợp khi | Team rất giỏi, code test tốt | Team lớn, release có lịch | **Team học thuật, 8-10 người** |

### Tại sao chọn Modified Git Flow?

1. **Git Flow gốc** quá phức tạp cho team học thuật → Đơn giản hóa, bỏ bớt ceremony
2. **Trunk-based** yêu cầu CI/CD mạnh + team có kỷ luật cao → Chưa phù hợp
3. **Modified Git Flow** giữ lại cấu trúc rõ ràng (main/develop/feature) nhưng giảm quy trình → Phù hợp nhất

---

## 11. Công cụ hỗ trợ

| Mục đích | Công cụ đề xuất | Ghi chú |
|----------|----------------|---------|
| Source control | **GitHub** hoặc **GitLab** | Tạo private repo, invite team |
| MR / Code Review | GitHub Pull Requests / GitLab MR | Enforce 1 reviewer approve |
| Communication | Zalo / Slack / Discord | Channel riêng cho từng sub-team |
| Task tracking | GitHub Issues / Trello / Notion | Gán task cho từng người |
| CI (tùy chọn) | GitHub Actions | Tự động chạy pytest khi tạo MR |

---

## 12. Quick Reference — Lệnh Git thường dùng

```bash
# === HÀNG NGÀY ===
git checkout develop && git pull origin develop     # Sync develop
git checkout -b feature/xxx                         # Tạo nhánh mới
git add <files> && git commit -m "type(scope): msg" # Commit
git rebase develop                                  # Sync trước khi push
git push origin feature/xxx                         # Push để tạo MR

# === KHI CÓ CONFLICT ===
git rebase develop                     # Nếu conflict xuất hiện:
# 1. Mở file conflict, sửa thủ công (tìm <<<<<<< ======= >>>>>>>)
# 2. git add <file-đã-sửa>
# 3. git rebase --continue
# Nếu quá rối: git rebase --abort

# === SAU KHI MR MERGED ===
git checkout develop && git pull origin develop
git branch -d feature/xxx                           # Xóa nhánh local
git push origin --delete feature/xxx                # Xóa nhánh remote

# === XEM TRẠNG THÁI ===
git status                    # Xem file thay đổi
git log --oneline -10         # Xem 10 commit gần nhất
git branch -a                 # Xem tất cả nhánh
git diff                      # Xem thay đổi chưa commit
```

---

## 13. Checklist Khởi tạo Project (Làm 1 lần bởi Lead)

- [ ] Tạo repository trên GitHub/GitLab
- [ ] `git init` + push cấu trúc thư mục ban đầu
- [ ] Tạo nhánh `develop` từ `main`
- [ ] Set `develop` là default branch
- [ ] Bật branch protection cho `main` (cấm push trực tiếp)
- [ ] Bật branch protection cho `develop` (yêu cầu MR + 1 approve)
- [ ] Thêm `.gitignore` cho Python
- [ ] Thêm `requirements.txt`
- [ ] Invite tất cả thành viên vào repo
- [ ] Tạo labels cho Issues: `detection`, `transform`, `enhancement`, `sharpening`, `bug`, `docs`
- [ ] Tạo milestone cho từng giai đoạn phát triển
- [ ] Gửi link tài liệu này cho cả team đọc
