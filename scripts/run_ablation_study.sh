#!/bin/bash
# Master script for running complete ablation study
# Usage: bash scripts/run_ablation_study.sh

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SEED=42
PYTHON_CMD="python"
SCRIPT_DIR="scripts"
TRAINING_SCRIPT="${SCRIPT_DIR}/train_augmented.py"

# Log file
LOG_DIR="outputs/ablation_study/logs"
mkdir -p "${LOG_DIR}"
MASTER_LOG="${LOG_DIR}/ablation_study_$(date +%Y%m%d_%H%M%S).log"

# Function to print colored messages
print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

# Function to log messages
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "${MASTER_LOG}"
}

# Function to run experiment with error handling
run_experiment() {
    local exp_name=$1
    local exp_flags=$2
    local exp_log="${LOG_DIR}/${exp_name}_$(date +%Y%m%d_%H%M%S).log"
    
    print_header "Experiment: ${exp_name}"
    log "Starting experiment: ${exp_name}"
    log "Command: ${PYTHON_CMD} ${TRAINING_SCRIPT} ${exp_flags}"
    
    # Run experiment
    if ${PYTHON_CMD} ${TRAINING_SCRIPT} ${exp_flags} 2>&1 | tee "${exp_log}"; then
        print_success "${exp_name} completed successfully"
        log "Experiment ${exp_name} completed successfully"
        return 0
    else
        print_error "${exp_name} failed"
        log "Experiment ${exp_name} failed (see ${exp_log})"
        return 1
    fi
}

# Function to check if baseline exists
check_baseline() {
    if [ ! -d "outputs/runs" ]; then
        return 1
    fi
    
    # Look for baseline or no_aug experiment
    if ls outputs/runs/*baseline_no_aug* 1> /dev/null 2>&1; then
        return 0
    fi
    
    return 1
}

# Function to estimate total time
estimate_time() {
    local num_experiments=$1
    local hours_per_exp=2.5
    local total_hours=$(echo "$num_experiments * $hours_per_exp" | bc)
    
    echo "${total_hours}"
}

# Main execution
main() {
    print_header "ABLATION STUDY - MASTER RUNNER"
    echo ""
    
    log "Starting ablation study"
    log "Random seed: ${SEED}"
    log "Master log: ${MASTER_LOG}"
    
    # Check Python environment
    print_info "Checking Python environment..."
    if ! command -v ${PYTHON_CMD} &> /dev/null; then
        print_error "Python not found. Make sure conda environment is activated."
        exit 1
    fi
    
    python_version=$(${PYTHON_CMD} --version)
    print_success "Python: ${python_version}"
    
    # Check training script exists
    if [ ! -f "${TRAINING_SCRIPT}" ]; then
        print_error "Training script not found: ${TRAINING_SCRIPT}"
        exit 1
    fi
    print_success "Training script found"
    
    echo ""
    
    # Check if baseline exists
    if check_baseline; then
        print_info "Baseline experiment already exists, skipping..."
        EXPERIMENTS=(
            "pitch_only:--pitch-only --two-stage --seed ${SEED}"
            "stretch_only:--stretch-only --two-stage --seed ${SEED}"
            "specaug_only:--specaug-only --two-stage --seed ${SEED}"
            "full_augment:--full-augment --two-stage --seed ${SEED}"
        )
    else
        print_warning "No baseline found, including baseline experiment"
        EXPERIMENTS=(
            "baseline:--no-augment --two-stage --seed ${SEED}"
            "pitch_only:--pitch-only --two-stage --seed ${SEED}"
            "stretch_only:--stretch-only --two-stage --seed ${SEED}"
            "specaug_only:--specaug-only --two-stage --seed ${SEED}"
            "full_augment:--full-augment --two-stage --seed ${SEED}"
        )
    fi
    
    # Estimate time
    num_exp=${#EXPERIMENTS[@]}
    estimated_hours=$(estimate_time ${num_exp})
    
    print_header "EXPERIMENT PLAN"
    echo "Total experiments: ${num_exp}"
    echo "Estimated time: ~${estimated_hours} hours"
    echo ""
    
    for i in "${!EXPERIMENTS[@]}"; do
        IFS=':' read -r exp_name exp_flags <<< "${EXPERIMENTS[$i]}"
        echo "$((i+1)). ${exp_name}"
    done
    
    echo ""
    print_warning "This will take approximately ${estimated_hours} hours"
    read -p "Do you want to continue? (y/n) " -n 1 -r
    echo
    
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_info "Ablation study cancelled"
        exit 0
    fi
    
    echo ""
    log "User confirmed, starting experiments"
    
    # Run all experiments
    success_count=0
    fail_count=0
    start_time=$(date +%s)
    
    for i in "${!EXPERIMENTS[@]}"; do
        IFS=':' read -r exp_name exp_flags <<< "${EXPERIMENTS[$i]}"
        
        echo ""
        print_info "Experiment $((i+1))/${num_exp}"
        
        if run_experiment "${exp_name}" "${exp_flags}"; then
            ((success_count++))
        else
            ((fail_count++))
            print_warning "Continuing with remaining experiments..."
        fi
        
        # Calculate and display progress
        elapsed=$(($(date +%s) - start_time))
        remaining=$((num_exp - i - 1))
        if [ ${i} -gt 0 ]; then
            avg_time=$((elapsed / (i + 1)))
            est_remaining=$((avg_time * remaining))
            est_hours=$((est_remaining / 3600))
            est_mins=$(((est_remaining % 3600) / 60))
            print_info "Estimated time remaining: ${est_hours}h ${est_mins}m"
        fi
        
        echo ""
    done
    
    # Final summary
    total_time=$(($(date +%s) - start_time))
    hours=$((total_time / 3600))
    minutes=$(((total_time % 3600) / 60))
    
    print_header "ABLATION STUDY COMPLETE"
    log "Ablation study complete"
    
    echo ""
    print_info "Total time: ${hours}h ${minutes}m"
    print_success "Successful: ${success_count}/${num_exp}"
    
    if [ ${fail_count} -gt 0 ]; then
        print_error "Failed: ${fail_count}/${num_exp}"
        log "Some experiments failed (${fail_count}/${num_exp})"
    fi
    
    echo ""
    print_header "NEXT STEPS"
    echo ""
    echo "1. Collect and analyze results:"
    echo "   ${PYTHON_CMD} ${SCRIPT_DIR}/collect_ablation_results.py"
    echo ""
    echo "2. View experiment dashboard:"
    echo "   ${PYTHON_CMD} ${SCRIPT_DIR}/track_experiments.py --no-watch"
    echo ""
    echo "3. Review output:"
    echo "   - Individual logs: ${LOG_DIR}/"
    echo "   - Master log: ${MASTER_LOG}"
    echo "   - Experiment outputs: outputs/runs/"
    echo ""
    
    log "Master log saved to ${MASTER_LOG}"
}

# Run main function
main