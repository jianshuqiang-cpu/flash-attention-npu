#!/usr/bin/env bash
# Copyright (c) 2026, flash-attention-npu CI maintainers.
#
# 扫描宿主机 Ascend NPU，选一张可用于 CI 的卡。
#
# 用法:
#   bash ci/detect_npu.sh --summary      # 打印所有 NPU 与选中卡
#   bash ci/detect_npu.sh --candidates   # 打印 CI 加锁尝试顺序
#   bash ci/detect_npu.sh --env          # 打印 "export NPU_SELECTED_DEVICE=<id>"
#   eval "$(bash ci/detect_npu.sh --env)"
#
# 选卡策略:
#   1. 优先 health=OK 且 free>0 的卡，按 id 升序
#   2. 其次 free>0 但 health!=OK 的卡，按 id 升序
#   3. 没有候选时返回非 0
#
# 环境变量:
#   CI_REQUIRE_HEALTHY_NPU  (默认 false) 为 true 时只接受 health=OK 的卡
#   NPU_SELECTED_DEVICE     可手动指定，跳过自动选卡

set -euo pipefail

NPU_SMI_BIN="${NPU_SMI_BIN:-npu-smi}"
REQUIRE_HEALTHY="${CI_REQUIRE_HEALTHY_NPU:-false}"

log() { printf '[detect_npu] %s\n' "$*"; }
die() { printf '[detect_npu][ERROR] %s\n' "$*" >&2; exit 1; }

command -v "$NPU_SMI_BIN" >/dev/null 2>&1 || die "npu-smi not found in PATH; is Ascend driver installed?"

# 解析 npu-smi info 输出，得到 "id|name|health|free|soc" 行
# 输出示例片段:
#   +---------------------------+-----------------+-------------------------------------------------+
#   | NPU   Name                | Health          | Power(W)    Temp(C)           Hugepages-Usage(page) |
#   | chip                      | Bus-Id          | AICore(%)   Memory-Usage(MB)                  |
#   +===========================+=================+=================================================+
#   | 0     910B3               | OK              | 67.5        42                0    / 0           |
#   | 0                         | 0000:C1:00.0    | 0           0    / 0           |
#   +===========================+=================+=================================================+
parse_npu_smi() {
  "$NPU_SMI_BIN" info 2>/dev/null | awk '
    /^\| [0-9]+ +/ {
      id=$2; name=$3; health=$4;
      # 该行不含 free 信息，free 信息在下一行 Memory-Usage
      next_id=id; next_name=name; next_health=health;
    }
    /Memory-Usage/ {
      # 形如 | 0  | 0000:C1:00.0 | 0  | 0    / 0  |
      # 解析 used / total
      line=$0;
      match(line, /[0-9]+ *\/ *[0-9]+/);
      if (RSTART > 0) {
        sub=/.*\//; 
        split(substr(line, RSTART, RLENGTH), parts, "/");
        used=parts[1]+0; total=parts[2]+0;
        free=total-used;
        if (next_id != "") {
          printf "%s|%s|%s|%d|%s\n", next_id, next_name, next_health, free, "ascend910b";
          next_id="";
        }
      }
    }
  '
}

# 生成候选列表: "id|name|health|free|soc"
build_candidates() {
  local rows healthy_free any_free
  rows="$(parse_npu_smi)"
  if [ -z "$rows" ]; then
    return 0
  fi

  healthy_free=""
  any_free=""
  while IFS='|' read -r id name health free soc; do
    [ -z "$id" ] && continue
    if [ "${free:-0}" -gt 0 ]; then
      if [ "$health" = "OK" ]; then
        healthy_free+="${id}|${name}|${health}|${free}|${soc}"$'\n'
      else
        any_free+="${id}|${name}|${health}|${free}|${soc}"$'\n'
      fi
    fi
  done <<< "$rows"

  # 优先健康且空闲，再空闲但告警
  printf '%s' "$healthy_free"
  if [ "$REQUIRE_HEALTHY" != "true" ]; then
    printf '%s' "$any_free"
  fi
}

select_device() {
  # 手动指定优先
  if [ -n "${NPU_SELECTED_DEVICE:-}" ]; then
    echo "$NPU_SELECTED_DEVICE"
    return 0
  fi
  local first
  first="$(build_candidates | head -n1 | cut -d'|' -f1)"
  if [ -z "$first" ]; then
    return 1
  fi
  echo "$first"
}

case "${1:---summary}" in
  --summary)
    echo "Detected NPU devices:"
    rows="$(parse_npu_smi)"
    if [ -z "$rows" ]; then
      die "no NPU detected by npu-smi"
    fi
    while IFS='|' read -r id name health free soc; do
      [ -z "$id" ] && continue
      printf '  - id=%s name=%s health=%s free=%s soc=%s\n' \
        "$id" "$name" "$health" "$free" "$soc"
    done <<< "$rows"
    sel="$(select_device || true)"
    if [ -n "$sel" ]; then
      meta="$(build_candidates | grep "^${sel}|" || true)"
      if [ -n "$meta" ]; then
        IFS='|' read -r id name health free soc <<< "$meta"
        printf 'Selected NPU: id=%s name=%s health=%s free=%s soc=%s\n' \
          "$id" "$name" "$health" "$free" "$soc"
      else
        printf 'Selected NPU: id=%s\n' "$sel"
      fi
    else
      die "no selectable NPU (free memory > 0) found"
    fi
    ;;
  --candidates)
    cands="$(build_candidates)"
    if [ -z "$cands" ]; then
      die "no candidate NPU"
    fi
    echo "CI will try NPU in this order:"
    while IFS='|' read -r id name health free soc; do
      [ -z "$id" ] && continue
      printf '  - id=%s name=%s health=%s free=%s\n' "$id" "$name" "$health" "$free"
    done <<< "$cands"
    ;;
  --env)
    sel="$(select_device || true)"
    if [ -z "$sel" ]; then
      die "no selectable NPU"
    fi
    printf 'export NPU_SELECTED_DEVICE=%s\n' "$sel"
    ;;
  *)
    die "unknown argument: $1 (use --summary|--candidates|--env)"
    ;;
esac
