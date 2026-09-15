#!/usr/bin/env bash
# run_tests.sh -- run every test layer and aggregate results.
#
# Layers:
#   1. test_eval.py    -> env logic (graders, prereqs, full happy paths easy/medium/hard)
#   2. test_crisis.py  -> crisis flow + regression on easy/hard
#   3. test_api.py     -> FastAPI HTTP contract tests
#
# Usage:
#   chmod +x run_tests.sh
#   ./run_tests.sh
#
# Run from project root.

set -uo pipefail

if [ -t 1 ]; then
  RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BOLD='\033[1m'; NC='\033[0m'
else
  RED='' GREEN='' YELLOW='' BOLD='' NC=''
fi

SCRIPTS=("tests/test_eval.py" "tests/test_crisis.py" "tests/test_api.py")
RESULTS=()
TOTAL_OK=0
TOTAL_FAIL=0

printf "\n${BOLD}==== Test suite ====${NC}\n"
printf "Running %d test files: %s\n\n" "${#SCRIPTS[@]}" "${SCRIPTS[*]}"

for script in "${SCRIPTS[@]}"; do
  if [ ! -f "$script" ]; then
    printf "${YELLOW}SKIP${NC}   %s (not found in $(pwd))\n" "$script"
    RESULTS+=("SKIP   $script")
    continue
  fi

  printf "${BOLD}--- %s ---${NC}\n" "$script"
  if python3 "$script"; then
    RESULTS+=("${GREEN}PASS${NC}   $script")
    TOTAL_OK=$((TOTAL_OK + 1))
  else
    RC=$?
    RESULTS+=("${RED}FAIL${NC}   $script (exit $RC)")
    TOTAL_FAIL=$((TOTAL_FAIL + 1))
  fi
  printf "\n"
done

printf "${BOLD}==== Summary ====${NC}\n"
for line in "${RESULTS[@]}"; do
  printf "  %b\n" "$line"
done
printf "\nFiles passed: %d   Files failed: %d\n\n" "$TOTAL_OK" "$TOTAL_FAIL"

if [ "$TOTAL_FAIL" -gt 0 ]; then
  printf "${RED}${BOLD}Some tests failed.${NC}\n\n"
  exit 1
fi

printf "${GREEN}${BOLD}All test files passed.${NC}\n\n"
exit 0
