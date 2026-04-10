# /gpu-exec — Autonomous GPU Pipeline Skill

Claude Code 세션과 무관하게 장기 GPU 실험(RunPod) 또는 로컬 장기 작업을 끝까지 자동 실행하는 스킬.

스킬은 "실행자"가 아니라 **스크립트 생성기 + crontab 등록기**입니다.
생성된 `monitor.sh` + `gate_eval.py`가 세션 없이 OS crontab에서 독립 실행됩니다.

## Recognition Pattern

- `/gpu-exec`, `gpu-exec`, `GPU 실험 실행`, `파이프라인 등록`, `RunPod 파이프라인`
- `GPU 실험 자동화`, `session-independent`, `crontab pipeline`

## Quick Start

### 1. pipeline_spec.json 작성

```bash
cp ~/.claude/plugins/marketplaces/runpod-mcp-marketplace/skills/gpu-exec/templates/pipeline_spec.example.json .omc/gpu-exec/pipeline_spec.json
# 편집: pipeline_id, phases, gate conditions, email 설정
```

### 2. 파이프라인 생성 + crontab 등록

```bash
# RunPod 모드
export RUNPOD_API_KEY="your-key"
python3 ~/.claude/plugins/marketplaces/runpod-mcp-marketplace/skills/gpu-exec/scripts/generate_pipeline.py \
  --spec .omc/gpu-exec/pipeline_spec.json

# 로컬 모드 (pipeline_spec에 "mode": "local" 설정 후)
python3 ~/.claude/plugins/marketplaces/runpod-mcp-marketplace/skills/gpu-exec/scripts/generate_pipeline.py \
  --spec .omc/gpu-exec/pipeline_spec.json
```

생성 결과:
```
.omc/gpu-exec/
  monitor.sh      # crontab에 자동 등록됨 (*/10 * * * *)
  gate_eval.py    # gate 조건 평가기
  state.json      # 런타임 상태 (IDLE)
  monitor.log     # cron 실행 로그
  .env            # RUNPOD_API_KEY (chmod 600)
```

### 3. 스크립트에 sentinel 추가 (필수)

훈련 스크립트 **마지막 줄**에 반드시:
```bash
mv "${DONE_DIR}/.done.tmp" "${DONE_DIR}/.done"
```

### 4. 상태 확인

```bash
python3 ~/.claude/plugins/marketplaces/runpod-mcp-marketplace/skills/gpu-exec/scripts/generate_pipeline.py \
  --spec .omc/gpu-exec/pipeline_spec.json --status
```

출력 예시:
```
=== gpu-exec status: exp-036-phaseABC ===
  State:   POLLING
  Mode:    runpod
  Phase:   0
  Pod ID:  abc123xyz
  Retries: 2
  Updated: 2026-04-09T14:30:00Z
  Cron:    registered
```

### 5. GATE_FAIL 또는 ERROR 후 재시작

```bash
python3 ~/.claude/plugins/marketplaces/runpod-mcp-marketplace/skills/gpu-exec/scripts/generate_pipeline.py \
  --spec .omc/gpu-exec/pipeline_spec.json --resume
```

- `GATE_FAIL` → 현재 phase 유지, retry_count=0, IDLE로 리셋 + cron 재등록
- `ERROR` → phase 0부터 전체 재시작 (잔여 pod는 수동 삭제 필요)

## State Machine

```
RunPod 모드:
IDLE → CREATING_POD → WAITING_FOR_SSH → POD_READY → UPLOADING →
RUNNING → POLLING → GATE_CHECK → DOWNLOADING → PHASE_DONE
  ↓ (마지막 phase)       ↓ (다음 phase 있음)
PIPELINE_DONE         CREATING_POD (다음 phase)

종료 상태:
GATE_FAIL — pod 삭제 + 이메일 + cron 제거 (--resume으로 재시작)
ERROR     — pod 삭제(있으면) + 이메일 + cron 제거 (--resume으로 재시작)

Local 모드 (mode: "local"):
IDLE → RUNNING → POLLING → GATE_CHECK → PHASE_DONE → PIPELINE_DONE
```

## Gate Condition 문법

```
"condition": "rho>=0.30 AND auc_prc>=0.40 AND precision_at_recall>=0.30"
```

- 지원 연산자: `>=`, `<=`, `==`, `!=`, `>`, `<`
- 지원 논리: `AND` (OR 미지원)
- 메트릭 키 형식: `\w+` (영숫자+밑줄, 공백 불가 — `val_loss` O, `val loss` X)

gate_eval.py exit 코드:
- `0` = 모든 조건 통과
- `1` = 하나 이상 실패 → GATE_FAIL 상태 전이
- `2` = 시스템 오류 (키 누락/NaN/JSON 파싱 실패) → ERROR 상태 전이

로컬 테스트:
```bash
python3 .omc/gpu-exec/gate_eval.py \
  --metrics '{"rho": 0.35, "auc_prc": 0.42}' \
  --condition "rho>=0.30 AND auc_prc>=0.40" --phase test
# {"pass": true, "metrics": {...}, ...}
```

## pipeline_spec.json 필드

| 필드 | 필수 | 기본값 | 설명 |
|------|------|--------|------|
| `pipeline_id` | ✓ | — | 파이프라인 고유 ID (cron marker로 사용) |
| `mode` | | `"runpod"` | `"runpod"` 또는 `"local"` |
| `max_runtime_hours` | | — | 전체 실행 시간 상한 (초과 시 ERROR) |
| `phases[].id` | ✓ | — | phase 식별자 |
| `phases[].script` | ✓ | — | 실행할 bash 스크립트 경로 (PROJECT_DIR 기준) |
| `phases[].sentinel_dir` | | PROJECT_DIR | `.done` 파일이 생성될 디렉토리 |
| `phases[].upload` | | — | RunPod 전용 업로드 설정 |
| `phases[].outputs` | | — | 다운로드할 결과 파일 (RunPod 모드) |
| `phases[].gate` | | — | gate 미설정 시 자동 통과 |
| `phases[].gate.metrics_file` | | — | PROJECT_DIR 기준 metrics JSON 경로 |
| `phases[].gate.condition` | | — | gate 조건 문자열 |
| `notifications.email` | | — | 알림 이메일 (빈 문자열 시 알림 스킵) |
| `runpod.gpu_type_id` | RunPod | — | GPU 타입 |
| `runpod.image_name` | RunPod | — | Docker 이미지 |
| `runpod.network_volume_id` | RunPod | — | 네트워크 볼륨 ID |

## 환경 변수

| 변수 | 필수 | 설명 |
|------|------|------|
| `RUNPOD_API_KEY` | RunPod 모드 | RunPod GraphQL API 키 |
| `SSH_KEY_PATH` | RunPod 모드 | SSH private key 경로 (기본: `~/.ssh/id_ed25519`) |

`RUNPOD_API_KEY`는 generate_pipeline.py 실행 시 `.omc/gpu-exec/.env`에 저장됩니다 (chmod 600).
cron은 이 `.env`를 자동으로 source합니다.

## 주의사항

- **sentinel 필수**: 훈련 스크립트 마지막 줄에 `mv "${DONE_DIR}/.done.tmp" "${DONE_DIR}/.done"` 없으면 POLLING 단계에서 영구 대기
- **SIGKILL 후 cron 잔존 가능**: 서버 재부팅 등 비정상 종료 시 crontab 항목이 남을 수 있음 → `crontab -l | grep gpu-exec` 확인 후 수동 제거
- **동시 실행 보호**: flock으로 monitor.sh 중복 실행 방지 (이전 실행이 끝나기 전까지 다음 cron 인스턴스는 즉시 종료)
- **RunPod API 재시도**: 일시적 오류 시 3회 재시도 (10s, 20s, 40s). 3회 실패 → ERROR

## Files

```
~/.claude/plugins/marketplaces/runpod-mcp-marketplace/skills/gpu-exec/
  SKILL.md                          # 이 파일
  templates/
    gate_eval.py.tmpl               # gate 조건 평가기 (복사 사용, 치환 없음)
    monitor.sh.tmpl                 # state machine 오케스트레이터 (__KEY__ 치환)
    pipeline_spec.example.json      # 주석 포함 예시 spec
  scripts/
    generate_pipeline.py            # 생성기 + crontab 등록기
```
