#!/usr/bin/env bash
#
# bench.sh - LLM Inference Benchmark Suite
#
# Benchmarks ziglm against llama.cpp (llama-cli) and Ollama on identical models
# and prompts, comparing prefill speed, generation speed, and output parity.
#
# Dependencies: bash, awk, sed, grep, curl (standard POSIX / Unix tools)
#

set -euo pipefail

# -----------------------------------------------------------------------------
# Configuration Defaults
# -----------------------------------------------------------------------------
MODEL_PATH="/home/simonuwu/models/gemma4-q4/gemma-4-E2B_q4_0-it.gguf"
OLLAMA_MODEL=""
PROMPT="Explain how quicksort works in three sentences."
MAX_TOKENS=64
NUM_RUNS=3
DO_WARMUP=1
CHART_FILE="benchmark.svg"
REPORT_FILE="benchmark.md"
DO_CHART=1
DO_REPORT=1
TARGETS_ARG=""

ZIGLM_BIN="./zig-out/bin/ziglm"
LLAMA_BIN="$(command -v llama-cli || true)"
OLLAMA_HOST="${OLLAMA_HOST:-http://localhost:11434}"

# ANSI Colors
CLR_RESET="\033[0m"
CLR_BOLD="\033[1m"
CLR_DIM="\033[2m"
CLR_CYAN="\033[36m"
CLR_GREEN="\033[32m"
CLR_YELLOW="\033[33m"
CLR_RED="\033[31m"

# -----------------------------------------------------------------------------
# Usage & Help
# -----------------------------------------------------------------------------
usage() {
    cat << HELP_EOF
Usage: $(basename "$0") [OPTIONS]

Benchmarks ziglm against llama.cpp and Ollama using the same model and prompt.
Measures prefill tokens/s, generation tokens/s, and compares generated outputs.

Options:
  -m, --model PATH        Path to GGUF model file
                          (default: ${MODEL_PATH})
  -p, --prompt TEXT       Prompt string for evaluation
                          (default: "${PROMPT}")
  -n, --tokens NUM        Max new tokens to generate (default: ${MAX_TOKENS})
  -r, --runs NUM          Number of benchmark runs to average (default: ${NUM_RUNS})
  -t, --targets LIST      Comma-separated list of targets to benchmark:
                          ziglm-gpu, ziglm-cpu, llama-gpu, llama-cpu, ollama
                          (default: all detected/available targets)
      --ollama-model NAME Ollama model tag to test against (default: auto-detected)
  -c, --chart FILE        SVG chart output file (default: ${CHART_FILE})
  -o, --report FILE       Markdown report output file (default: ${REPORT_FILE})
      --no-chart          Disable SVG chart generation
      --no-report         Disable Markdown report generation
      --no-warmup         Skip unmeasured warmup run (default: runs 1 warmup)
  -h, --help              Show this help message and exit

Examples:
  ./bench.sh
  ./bench.sh -m /path/to/model.gguf -n 128 -r 5
  ./bench.sh -t ziglm-gpu,llama-gpu -p "Write a hello world program in Zig."
HELP_EOF
}

# -----------------------------------------------------------------------------
# CLI Argument Parsing
# -----------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        -m|--model)
            MODEL_PATH="$2"; shift 2 ;;
        -p|--prompt)
            PROMPT="$2"; shift 2 ;;
        -n|--tokens)
            MAX_TOKENS="$2"; shift 2 ;;
        -r|--runs)
            NUM_RUNS="$2"; shift 2 ;;
        -t|--targets)
            TARGETS_ARG="$2"; shift 2 ;;
        --ollama-model)
            OLLAMA_MODEL="$2"; shift 2 ;;
        -c|--chart)
            CHART_FILE="$2"; shift 2 ;;
        -o|--report)
            REPORT_FILE="$2"; shift 2 ;;
        --no-chart)
            DO_CHART=0; shift ;;
        --no-report)
            DO_REPORT=0; shift ;;
        --no-warmup)
            DO_WARMUP=0; shift ;;
        -h|--help)
            usage; exit 0 ;;
        *)
            echo -e "${CLR_RED}Error:${CLR_RESET} Unknown option: $1" >&2
            usage
            exit 1
            ;;
    esac
done

# -----------------------------------------------------------------------------
# Prerequisite & Environment Check
# -----------------------------------------------------------------------------
echo -e "${CLR_BOLD}LLM Inference Benchmark${CLR_RESET}"
echo -e "${CLR_DIM}Initializing environment and verifying tools...${CLR_RESET}"

if [[ ! -f "$MODEL_PATH" ]]; then
    # Search common fallback locations
    FALLBACK="$(find "${HOME}/models" -name "*.gguf" 2>/dev/null | head -n 1 || true)"
    if [[ -n "$FALLBACK" && -f "$FALLBACK" ]]; then
        echo -e "${CLR_YELLOW}Notice:${CLR_RESET} Specified model not found. Using fallback: $FALLBACK"
        MODEL_PATH="$FALLBACK"
    else
        echo -e "${CLR_RED}Error:${CLR_RESET} Model file not found: $MODEL_PATH" >&2
        exit 1
    fi
fi

# Ensure ziglm binary exists; build ReleaseFast if missing
if [[ ! -x "$ZIGLM_BIN" ]]; then
    echo -e "${CLR_YELLOW}Notice:${CLR_RESET} $ZIGLM_BIN not found. Building ReleaseFast..."
    zig build -Doptimize=ReleaseFast
fi

# Check CUDA capability
HAS_CUDA=0
GPU_NAME="None"
if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
    HAS_CUDA=1
    GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
fi

# Check Ollama status
HAS_OLLAMA=0
if curl -s -f "${OLLAMA_HOST}/api/tags" &>/dev/null; then
    HAS_OLLAMA=1
    if [[ -z "$OLLAMA_MODEL" ]]; then
        # Try to match model basename or pick existing model
        MODEL_BASE="$(basename "$MODEL_PATH" .gguf | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9._-]/-/g')"
        TAGS_JSON="$(curl -s "${OLLAMA_HOST}/api/tags")"
        if echo "$TAGS_JSON" | grep -q "\"gemma4-2b\""; then
            OLLAMA_MODEL="gemma4-2b"
        elif echo "$TAGS_JSON" | grep -q "\"${MODEL_BASE}\""; then
            OLLAMA_MODEL="${MODEL_BASE}"
        else
            FIRST_MODEL="$(echo "$TAGS_JSON" | grep -oE '"name":"[^"]+"' | head -n 1 | cut -d'"' -f4 || true)"
            OLLAMA_MODEL="$FIRST_MODEL"
        fi
    fi
fi

# Build list of active targets
ACTIVE_TARGETS=()
if [[ -n "$TARGETS_ARG" ]]; then
    IFS=',' read -ra TARGETS_LIST <<< "$TARGETS_ARG"
    for t in "${TARGETS_LIST[@]}"; do
        t="$(echo "$t" | tr -d ' ')"
        ACTIVE_TARGETS+=("$t")
    done
else
    # Auto-detection
    if [[ $HAS_CUDA -eq 1 ]]; then
        ACTIVE_TARGETS+=("ziglm-gpu")
        [[ -n "$LLAMA_BIN" ]] && ACTIVE_TARGETS+=("llama-gpu")
        [[ $HAS_OLLAMA -eq 1 && -n "$OLLAMA_MODEL" ]] && ACTIVE_TARGETS+=("ollama")
    fi
    ACTIVE_TARGETS+=("ziglm-cpu")
    [[ -n "$LLAMA_BIN" ]] && ACTIVE_TARGETS+=("llama-cpu")
fi

echo -e "Model:      ${CLR_CYAN}${MODEL_PATH}${CLR_RESET} ($(du -h "$MODEL_PATH" | cut -f1))"
echo -e "Prompt:     \"${PROMPT}\""
echo -e "Tokens:     ${MAX_TOKENS} | Runs: ${NUM_RUNS} | Warmup: $([[ $DO_WARMUP -eq 1 ]] && echo "yes" || echo "no")"
[[ $HAS_CUDA -eq 1 ]] && echo -e "GPU:        ${CLR_GREEN}${GPU_NAME}${CLR_RESET}"
echo -e "Targets:    ${ACTIVE_TARGETS[*]}"
echo ""

# -----------------------------------------------------------------------------
# Execution Functions
# -----------------------------------------------------------------------------

# Run ziglm (GPU or CPU)
exec_ziglm() {
    local use_gpu="$1"
    local gpu_flag=""
    [[ "$use_gpu" -eq 1 ]] && gpu_flag="--gpu"

    local raw_out
    raw_out="$(LC_ALL=C "$ZIGLM_BIN" run -m "$MODEL_PATH" -p "$PROMPT" -n "$MAX_TOKENS" --greedy $gpu_flag 2>&1)"

    local prefill gen total_ms
    prefill="$(echo "$raw_out" | grep -oE "Prefill:[[:space:]]+[0-9.]+[[:space:]]+tok/s" | awk '{print $2}' | tail -n 1)"
    gen="$(echo "$raw_out" | grep -oE "Generation:[[:space:]]+[0-9.]+[[:space:]]+tok/s" | awk '{print $2}' | tail -n 1)"
    total_ms="$(echo "$raw_out" | grep -oE "Total time:[[:space:]]+[0-9.]+[[:space:]]+ms" | awk '{print $3}' | tail -n 1)"

    local response
    response="$(echo "$raw_out" | sed -n '/Prompt:/,/─\{10,\}/p' | grep -v 'Prompt:' | grep -v '─\{10,\}' | sed '/^[[:space:]]*$/d')"

    echo "${prefill:-0.0}|${gen:-0.0}|${total_ms:-0.0}|${response}"
}

# Run llama-cli (GPU or CPU)
exec_llama() {
    local use_gpu="$1"
    local ngl="0"
    [[ "$use_gpu" -eq 1 ]] && ngl="99"

    local raw_out
    raw_out="$(LC_ALL=C "$LLAMA_BIN" -m "$MODEL_PATH" -p "$PROMPT" -n "$MAX_TOKENS" --temp 0 -ngl "$ngl" -st --no-warmup --simple-io --reasoning off 2>&1)"

    local prefill gen
    prefill="$(echo "$raw_out" | grep -oE "Prompt:[[:space:]]+[0-9.]+[[:space:]]+t/s" | awk '{print $2}' | tail -n 1)"
    gen="$(echo "$raw_out" | grep -oE "Generation:[[:space:]]+[0-9.]+[[:space:]]+t/s" | awk '{print $2}' | tail -n 1)"

    local response
    response="$(echo "$raw_out" | sed -n '/> '"$PROMPT"'/!b;n;:a;/\[ Prompt:/!{p;n;ba}' | sed '/^[[:space:]]*$/d')"
    if [[ -z "$response" ]]; then
        response="$(echo "$raw_out" | grep -v "^Loading model" | grep -v "▄▄" | grep -v "build " | grep -v "available commands" | grep -v "Exiting\.\.\." | grep -v "\[ Prompt:" | sed '/^[[:space:]]*$/d')"
    fi

    local total_ms
    total_ms="$(awk -v p="${prefill:-0}" -v g="${gen:-0}" -v n="$MAX_TOKENS" 'BEGIN {
        if (p > 0 && g > 0) printf "%.1f", ((16.0 / p) + (n / g)) * 1000.0; else print "0.0"
    }')"

    echo "${prefill:-0.0}|${gen:-0.0}|${total_ms}|${response}"
}

# Run Ollama via REST API
exec_ollama() {
    local req_json
    if command -v jq &>/dev/null; then
        req_json="$(jq -n \
            --arg model "$OLLAMA_MODEL" \
            --arg prompt "$PROMPT" \
            --argjson tokens "$MAX_TOKENS" \
            '{model: $model, prompt: $prompt, stream: false, think: false, options: {num_predict: $tokens, temperature: 0.0}}')"
    else
        req_json="{\"model\":\"$OLLAMA_MODEL\",\"prompt\":\"$PROMPT\",\"stream\":false,\"think\":false,\"options\":{\"num_predict\":$MAX_TOKENS,\"temperature\":0.0}}"
    fi

    local res
    res="$(curl -s -X POST "${OLLAMA_HOST}/api/generate" -H "Content-Type: application/json" -d "$req_json")"

    local p_count p_ns e_count e_ns total_ns response
    if command -v jq &>/dev/null; then
        p_count="$(echo "$res" | jq -r '.prompt_eval_count // 0')"
        p_ns="$(echo "$res" | jq -r '.prompt_eval_duration // 0')"
        e_count="$(echo "$res" | jq -r '.eval_count // 0')"
        e_ns="$(echo "$res" | jq -r '.eval_duration // 0')"
        total_ns="$(echo "$res" | jq -r '.total_duration // 0')"
        response="$(echo "$res" | jq -r '.response // ""')"
    else
        p_count="$(echo "$res" | grep -oE '"prompt_eval_count":[0-9]+' | cut -d: -f2 || echo 0)"
        p_ns="$(echo "$res" | grep -oE '"prompt_eval_duration":[0-9]+' | cut -d: -f2 || echo 0)"
        e_count="$(echo "$res" | grep -oE '"eval_count":[0-9]+' | cut -d: -f2 || echo 0)"
        e_ns="$(echo "$res" | grep -oE '"eval_duration":[0-9]+' | cut -d: -f2 || echo 0)"
        total_ns="$(echo "$res" | grep -oE '"total_duration":[0-9]+' | cut -d: -f2 || echo 0)"
        response="$(echo "$res" | sed -n 's/.*"response":"\([^"]*\)".*/\1/p')"
    fi

    local prefill gen total_ms
    prefill="$(awk -v c="$p_count" -v ns="$p_ns" 'BEGIN { if (ns > 0) printf "%.1f", c / (ns / 1000000000.0); else print "0.0" }')"
    gen="$(awk -v c="$e_count" -v ns="$e_ns" 'BEGIN { if (ns > 0) printf "%.1f", c / (ns / 1000000000.0); else print "0.0" }')"
    total_ms="$(awk -v ns="$total_ns" 'BEGIN { if (ns > 0) printf "%.1f", ns / 1000000.0; else print "0.0" }')"

    echo "${prefill:-0.0}|${gen:-0.0}|${total_ms}|${response}"
}

# -----------------------------------------------------------------------------
# Benchmark Runner Loop
# -----------------------------------------------------------------------------
TMP_DATA="$(mktemp /tmp/bench_data_XXXXXX.txt)"
TMP_RESP="$(mktemp /tmp/bench_resp_XXXXXX.txt)"
trap 'rm -f "$TMP_DATA" "$TMP_RESP"' EXIT

echo -e "Starting test matrix (${#ACTIVE_TARGETS[@]} targets, ${NUM_RUNS} runs each)..."
echo "-------------------------------------------------------------------------------"

for target in "${ACTIVE_TARGETS[@]}"; do
    target_name=""
    case "$target" in
        ziglm-gpu)  target_name="ziglm (GPU)" ;;
        ziglm-cpu)  target_name="ziglm (CPU)" ;;
        llama-gpu)  target_name="llama.cpp (GPU)" ;;
        llama-cpu)  target_name="llama.cpp (CPU)" ;;
        ollama)     target_name="Ollama (GPU)" ;;
        *)          target_name="$target" ;;
    esac

    echo -ne "Testing ${CLR_BOLD}${target_name}${CLR_RESET} ... "

    # Warmup
    if [[ $DO_WARMUP -eq 1 ]]; then
        case "$target" in
            ziglm-gpu)  exec_ziglm 1 >/dev/null 2>&1 || true ;;
            ziglm-cpu)  exec_ziglm 0 >/dev/null 2>&1 || true ;;
            llama-gpu)  exec_llama 1 >/dev/null 2>&1 || true ;;
            llama-cpu)  exec_llama 0 >/dev/null 2>&1 || true ;;
            ollama)     exec_ollama >/dev/null 2>&1 || true ;;
        esac
    fi

    # Benchmark Iterations
    p_sum="0.0"
    g_sum="0.0"
    t_sum="0.0"
    captured_response=""

    for ((r=1; r<=NUM_RUNS; r++)); do
        run_res=""
        case "$target" in
            ziglm-gpu)  run_res="$(exec_ziglm 1)" ;;
            ziglm-cpu)  run_res="$(exec_ziglm 0)" ;;
            llama-gpu)  run_res="$(exec_llama 1)" ;;
            llama-cpu)  run_res="$(exec_llama 0)" ;;
            ollama)     run_res="$(exec_ollama)" ;;
        esac

        cur_p="$(echo "$run_res" | cut -d'|' -f1)"
        cur_g="$(echo "$run_res" | cut -d'|' -f2)"
        cur_t="$(echo "$run_res" | cut -d'|' -f3)"
        captured_response="$(echo "$run_res" | cut -d'|' -f4-)"

        p_sum="$(awk -v a="$p_sum" -v b="$cur_p" 'BEGIN { printf "%.2f", a + b }')"
        g_sum="$(awk -v a="$g_sum" -v b="$cur_g" 'BEGIN { printf "%.2f", a + b }')"
        t_sum="$(awk -v a="$t_sum" -v b="$cur_t" 'BEGIN { printf "%.2f", a + b }')"
    done

    avg_p="$(awk -v sum="$p_sum" -v n="$NUM_RUNS" 'BEGIN { printf "%.1f", sum / n }')"
    avg_g="$(awk -v sum="$g_sum" -v n="$NUM_RUNS" 'BEGIN { printf "%.1f", sum / n }')"
    avg_t="$(awk -v sum="$t_sum" -v n="$NUM_RUNS" 'BEGIN { printf "%.1f", sum / n }')"

    echo -e "Prefill: ${CLR_CYAN}${avg_p} t/s${CLR_RESET} | Gen: ${CLR_GREEN}${avg_g} t/s${CLR_RESET} | Latency: ${avg_t} ms"

    echo "${target_name}|${avg_p}|${avg_g}|${avg_t}" >> "$TMP_DATA"
    echo -e "### ${target_name}\n${captured_response}\n" >> "$TMP_RESP"
done

echo "-------------------------------------------------------------------------------"

# -----------------------------------------------------------------------------
# Visual Charts & Tables
# -----------------------------------------------------------------------------

# 1. Terminal Bar Chart
echo -e "\n${CLR_BOLD}Performance Chart (Tokens / Second)${CLR_RESET}"
awk -F'|' '
BEGIN {
  max_val = 1.0
  count = 0
}
NF >= 3 {
  names[count] = $1
  prefill[count] = $2 + 0.0
  gen[count] = $3 + 0.0
  if (prefill[count] > max_val) max_val = prefill[count]
  if (gen[count] > max_val) max_val = gen[count]
  count++
}
END {
  bar_max_width = 38
  printf "\n%-18s %-12s %10s   %s\n", "Engine", "Metric", "Rate", "Visual Comparison"
  print "─────────────────────────────────────────────────────────────────────────────"
  for (i = 0; i < count; i++) {
    pw = int((prefill[i] / max_val) * bar_max_width)
    gw = int((gen[i] / max_val) * bar_max_width)
    if (pw < 1 && prefill[i] > 0) pw = 1
    if (gw < 1 && gen[i] > 0) gw = 1

    pbar = ""
    for (b = 0; b < pw; b++) pbar = pbar "█"
    gbar = ""
    for (b = 0; b < gw; b++) gbar = gbar "█"

    printf "%-18s %-12s %7.1f t/s   \033[36m%s\033[0m\n", names[i], "Prefill", prefill[i], pbar
    printf "%-18s %-12s %7.1f t/s   \033[32m%s\033[0m\n", "", "Generation", gen[i], gbar
    if (i < count - 1) print "  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -"
  }
  print "─────────────────────────────────────────────────────────────────────────────"
  printf "Legend: \033[36m██ Prefill\033[0m  |  \033[32m██ Generation\033[0m\n\n"
}
' "$TMP_DATA"

# 2. Terminal Table
echo -e "${CLR_BOLD}Summary Table${CLR_RESET}"
printf "%-18s | %14s | %14s | %12s\n" "Engine" "Prefill Rate" "Generation Rate" "Latency"
echo "-------------------+----------------+----------------+-------------"
awk -F'|' '{
  printf "%-18s | %10.1f t/s | %10.1f t/s | %9.1f ms\n", $1, $2, $3, $4
}' "$TMP_DATA"
echo "-------------------+----------------+----------------+-------------"

# 3. Text Parity / Result Comparison
echo -e "\n${CLR_BOLD}Generated Text Parity Comparison${CLR_RESET}"
echo "============================================================================="
while IFS='|' read -r target_name p g t; do
    echo -e "${CLR_CYAN}[${target_name}]${CLR_RESET}"
    sed -n "/### ${target_name}/,/^### /{//!p}" "$TMP_RESP" | sed '/^[[:space:]]*$/d'
    echo ""
done < "$TMP_DATA"
echo "============================================================================="

# -----------------------------------------------------------------------------
# SVG Chart Generation
# -----------------------------------------------------------------------------
if [[ $DO_CHART -eq 1 ]]; then
    SUBTITLE="Model: $(basename "$MODEL_PATH") | Tokens: ${MAX_TOKENS} | Avg of ${NUM_RUNS} runs"
    awk -F'|' -v title="LLM Inference Benchmark: ziglm vs llama.cpp vs Ollama" -v subtitle="$SUBTITLE" '
    BEGIN {
      count = 0
      max_val = 1.0
    }
    NF >= 3 {
      names[count] = $1
      prefill[count] = $2 + 0.0
      gen[count] = $3 + 0.0
      if (prefill[count] > max_val) max_val = prefill[count]
      if (gen[count] > max_val) max_val = gen[count]
      count++
    }
    END {
      tick_step = 50
      if (max_val > 500) tick_step = 100
      if (max_val < 100) tick_step = 20
      if (max_val < 30) tick_step = 5
      nice_max = (int(max_val / tick_step) + 1) * tick_step

      width = 850
      top_margin = 110
      row_height = 56
      bottom_margin = 55
      plot_x = 160
      plot_width = 600
      height = top_margin + count * row_height + bottom_margin
      plot_bottom = height - bottom_margin

      printf "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 %d %d\" width=\"%d\" height=\"%d\" style=\"background:#0f172a;font-family:-apple-system,BlinkMacSystemFont,\\x27Segoe UI\\x27,Roboto,sans-serif;\">\n", width, height, width, height
      printf "  <rect width=\"100%%\" height=\"100%%\" fill=\"#0f172a\" rx=\"12\"/>\n"
      printf "  <rect x=\"16\" y=\"16\" width=\"%d\" height=\"%d\" fill=\"#1e293b\" rx=\"8\" stroke=\"#334155\" stroke-width=\"1\"/>\n", width - 32, height - 32

      printf "  <text x=\"40\" y=\"50\" fill=\"#f8fafc\" font-size=\"18\" font-weight=\"700\">%s</text>\n", title
      printf "  <text x=\"40\" y=\"72\" fill=\"#94a3b8\" font-size=\"12\">%s</text>\n", subtitle

      printf "  <rect x=\"%d\" y=\"40\" width=\"12\" height=\"12\" fill=\"#38bdf8\" rx=\"3\"/>\n", width - 260
      printf "  <text x=\"%d\" y=\"51\" fill=\"#e2e8f0\" font-size=\"12\" font-weight=\"500\">Prefill (tok/s)</text>\n", width - 242
      printf "  <rect x=\"%d\" y=\"40\" width=\"12\" height=\"12\" fill=\"#34d399\" rx=\"3\"/>\n", width - 140
      printf "  <text x=\"%d\" y=\"51\" fill=\"#e2e8f0\" font-size=\"12\" font-weight=\"500\">Generation (tok/s)</text>\n", width - 122

      num_ticks = int(nice_max / tick_step)
      for (t = 0; t <= num_ticks; t++) {
        val = t * tick_step
        tx = plot_x + (val / nice_max) * plot_width
        printf "  <line x1=\"%.1f\" y1=\"%d\" x2=\"%.1f\" y2=\"%d\" stroke=\"#334155\" stroke-width=\"1\" stroke-dasharray=\"3,3\"/>\n", tx, top_margin - 15, tx, plot_bottom
        printf "  <text x=\"%.1f\" y=\"%d\" fill=\"#94a3b8\" font-size=\"11\" text-anchor=\"middle\">%d</text>\n", tx, plot_bottom + 16, val
      }
      printf "  <text x=\"%.1f\" y=\"%d\" fill=\"#64748b\" font-size=\"11\" text-anchor=\"middle\">Tokens / Second (higher is better)</text>\n", plot_x + plot_width / 2.0, plot_bottom + 34

      for (i = 0; i < count; i++) {
        y0 = top_margin + i * row_height
        printf "  <text x=\"%d\" y=\"%d\" fill=\"#f1f5f9\" font-size=\"13\" font-weight=\"600\" text-anchor=\"end\">%s</text>\n", plot_x - 14, y0 + 24, names[i]

        pw = (prefill[i] / nice_max) * plot_width
        if (pw < 2 && prefill[i] > 0) pw = 2
        printf "  <rect x=\"%d\" y=\"%d\" width=\"%.1f\" height=\"14\" fill=\"#38bdf8\" rx=\"3\"/>\n", plot_x, y0 + 4, pw
        printf "  <text x=\"%.1f\" y=\"%d\" fill=\"#38bdf8\" font-size=\"11\" font-weight=\"600\">%.1f t/s</text>\n", plot_x + pw + 6, y0 + 15, prefill[i]

        gw = (gen[i] / nice_max) * plot_width
        if (gw < 2 && gen[i] > 0) gw = 2
        printf "  <rect x=\"%d\" y=\"%d\" width=\"%.1f\" height=\"14\" fill=\"#34d399\" rx=\"3\"/>\n", plot_x, y0 + 22, gw
        printf "  <text x=\"%.1f\" y=\"%d\" fill=\"#34d399\" font-size=\"11\" font-weight=\"600\">%.1f t/s</text>\n", plot_x + gw + 6, y0 + 33, gen[i]
      }

      printf "</svg>\n"
    }
    ' "$TMP_DATA" > "$CHART_FILE"

    echo -e "Saved SVG chart to ${CLR_GREEN}${CHART_FILE}${CLR_RESET}"
fi

# -----------------------------------------------------------------------------
# Markdown Report Generation
# -----------------------------------------------------------------------------
if [[ $DO_REPORT -eq 1 ]]; then
    {
        echo "# Inference Benchmark: ziglm vs llama.cpp vs Ollama"
        echo ""
        echo "- **Date**: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "- **Model**: \`$(basename "$MODEL_PATH")\`"
        echo "- **Prompt**: \"${PROMPT}\""
        echo "- **Max Tokens**: ${MAX_TOKENS}"
        echo "- **Runs**: ${NUM_RUNS} iterations"
        if [[ $HAS_CUDA -eq 1 ]]; then
            echo "- **GPU**: \`${GPU_NAME}\`"
        fi
        echo ""
        echo "## Performance Results"
        echo ""
        echo "| Engine | Prefill Rate | Generation Rate | Latency |"
        echo "| :--- | :---: | :---: | :---: |"
        awk -F'|' '{
          printf "| **%s** | %.1f tok/s | %.1f tok/s | %.1f ms |\n", $1, $2, $3, $4
        }' "$TMP_DATA"
        echo ""
        if [[ $DO_CHART -eq 1 && -f "$CHART_FILE" ]]; then
            echo "## Performance Chart"
            echo ""
            echo "![Benchmark Chart](${CHART_FILE})"
            echo ""
        fi
        echo "## Output Parity & Quality"
        echo ""
        while IFS='|' read -r target_name p g t; do
            echo "### ${target_name}"
            echo ""
            echo "> $(sed -n "/### ${target_name}/,/^### /{//!p}" "$TMP_RESP" | sed '/^[[:space:]]*$/d' | tr '\n' ' ')"
            echo ""
        done < "$TMP_DATA"
    } > "$REPORT_FILE"

    echo -e "Saved Markdown report to ${CLR_GREEN}${REPORT_FILE}${CLR_RESET}"
fi

echo -e "\n${CLR_GREEN}Benchmark completed successfully.${CLR_RESET}"
