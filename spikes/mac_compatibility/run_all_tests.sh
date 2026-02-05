#!/bin/bash
#
# Master test runner for macOS compatibility
#
# Runs all test scripts in sequence and reports overall results.
# Tests can be run individually or all at once.

set -e  # Exit on first error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test results
declare -a TEST_RESULTS
declare -a TEST_NAMES

run_test() {
    local test_name="$1"
    local test_script="$2"

    echo ""
    echo "========================================"
    echo "Running: $test_name"
    echo "========================================"

    if python3 "$test_script"; then
        TEST_RESULTS+=("PASS")
        echo -e "${GREEN}✓ $test_name PASSED${NC}"
    else
        TEST_RESULTS+=("FAIL")
        echo -e "${RED}✗ $test_name FAILED${NC}"
    fi

    TEST_NAMES+=("$test_name")
}

echo "========================================"
echo "macOS Compatibility Test Suite"
echo "========================================"
echo ""
echo "This will run all compatibility tests."
echo "Tests will be run in sequence."
echo ""

# Check if we're in the right directory
if [ ! -f "README.md" ]; then
    echo -e "${RED}Error: Must be run from spikes/mac_compatibility/${NC}"
    exit 1
fi

# Run all tests in order
run_test "1. Environment Setup" "test_environment.py"
run_test "2. JPype + JVM" "test_jpype_mac.py"
run_test "3. Autoplot Bridge" "test_autoplot_mac.py"
run_test "4. Knowledge Modules" "test_knowledge_mac.py"
run_test "5. Agent Integration" "test_agent_mac.py"

# Final summary
echo ""
echo "========================================"
echo "FINAL SUMMARY"
echo "========================================"

PASS_COUNT=0
FAIL_COUNT=0

for i in "${!TEST_NAMES[@]}"; do
    test_name="${TEST_NAMES[$i]}"
    result="${TEST_RESULTS[$i]}"

    if [ "$result" = "PASS" ]; then
        echo -e "${GREEN}✓${NC} $test_name"
        ((PASS_COUNT++))
    else
        echo -e "${RED}✗${NC} $test_name"
        ((FAIL_COUNT++))
    fi
done

echo ""
echo "Passed: $PASS_COUNT / ${#TEST_NAMES[@]}"
echo "Failed: $FAIL_COUNT / ${#TEST_NAMES[@]}"

if [ $FAIL_COUNT -eq 0 ]; then
    echo ""
    echo -e "${GREEN}========================================"
    echo "✓ ALL TESTS PASSED"
    echo "macOS compatibility confirmed!"
    echo -e "========================================${NC}"
    exit 0
else
    echo ""
    echo -e "${RED}========================================"
    echo "✗ SOME TESTS FAILED"
    echo "Check output above for details"
    echo -e "========================================${NC}"
    exit 1
fi
