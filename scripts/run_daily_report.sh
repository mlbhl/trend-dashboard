#!/usr/bin/env bash
# Trend Dashboard 일일 리포트 실행 래퍼 (cron용)
#
# 화~토 하루 3번(8:10/12:10/16:10 KST) 시도하되, 멱등 가드로 실제 실행은 1회.
# 미국장 종가가 KST 다음날 새벽에 확정되므로 화~토가 각 거래일 종가를 하루 안에 반영.
# 8:10에 PC가 꺼져 있었으면 다음 시각에 따라잡는다.
set -uo pipefail

# cron은 최소 환경으로 실행되므로 HOME/PATH를 명시한다 (git push에 gh 크리덴셜 필요).
export HOME="/home/byoun"
export PATH="/home/byoun/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH"

PROJECT_DIR="/home/byoun/projects/trend-dashboard"
PYTHON="/home/byoun/miniconda3/bin/python3"
cd "$PROJECT_DIR" || exit 1
mkdir -p reports

{
  echo "===== $(date '+%Y-%m-%d %H:%M:%S %Z') 실행 시작 ====="
  today=$(date +%F)
  report="reports/report_${today}.html"

  # 멱등 가드: 오늘치 리포트가 이미 있으면 스킵
  if [ -f "$report" ]; then
    echo "오늘(${today}) 이미 생성됨 — 스킵"
  else
    "$PYTHON" scripts/generate_report.py
    code=$?
    echo "python 종료 코드: ${code}"

    if [ "$code" -eq 0 ] && [ -f "$report" ]; then
      cp "$report" reports/latest.html
      git add reports
      if git diff --staged --quiet -- reports; then
        echo "변경 없음 — 커밋 생략"
      else
        git commit -m "Update daily report" -- reports
        git push || echo "git push 실패 — 리포트 파일은 보존됨, 다음 커밋 때 함께 푸시"
      fi
    else
      echo "리포트 생성 실패 — 다음 시각에 재시도"
    fi
  fi
  echo "===== $(date '+%Y-%m-%d %H:%M:%S %Z') 실행 종료 ====="
} >> reports/cron.log 2>&1
