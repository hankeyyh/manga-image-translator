#!/bin/bash
# Quick deployment script for Modal
#
# Usage:
#   ./deploy/deploy.sh [command]
#
# Commands:
#   setup     - Initial setup (create secrets)
#   deploy    - Deploy the application
#   models    - Download models to volume
#   test      - Run smoke tests
#   logs      - View logs
#   cleanup   - Clean up old results
#   help      - Show this help

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_header() {
    echo -e "\n${BLUE}═══════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════${NC}\n"
}

# Check if Modal is installed
check_modal() {
    if ! command -v modal &> /dev/null; then
        print_error "Modal CLI not found. Installing..."
        pip install modal
    fi
}

PROD_ENV_FILE=".env.prod"

# Setup: Create secrets from production env
setup() {
    print_header "Modal Setup"

    check_modal

    print_info "Checking for ${PROD_ENV_FILE} (production env for Modal secrets)..."
    if [ ! -f "$PROD_ENV_FILE" ]; then
        if [ -f .env ]; then
            cp .env "$PROD_ENV_FILE"
            print_warning "Created ${PROD_ENV_FILE} from .env"
            print_info "Edit ${PROD_ENV_FILE}: set production SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY"
            print_info "Then run: ./deploy/deploy.sh setup"
            exit 0
        fi
        print_error "${PROD_ENV_FILE} not found"
        print_info "Create it from local .env, then set production Supabase values:"
        print_info "  cp .env ${PROD_ENV_FILE}"
        exit 1
    fi

    print_info "Checking if MT_WEB_NONCE exists in ${PROD_ENV_FILE}..."
    if ! grep -q "^MT_WEB_NONCE=" "$PROD_ENV_FILE"; then
        print_warning "MT_WEB_NONCE not found in ${PROD_ENV_FILE}"
        print_info "Generating and adding MT_WEB_NONCE..."
        NONCE=$(openssl rand -hex 32)
        echo "MT_WEB_NONCE=$NONCE" >> "$PROD_ENV_FILE"
        print_success "MT_WEB_NONCE added to ${PROD_ENV_FILE}"
    fi

    if grep -Eq "^[[:space:]]*SUPABASE_URL[[:space:]]*=[[:space:]]*['\"]?https?://(127\.0\.0\.1|localhost)" "$PROD_ENV_FILE"; then
        print_error "SUPABASE_URL in ${PROD_ENV_FILE} points to localhost; Modal cannot reach it."
        print_info "Set SUPABASE_URL to your cloud Supabase project URL, then re-run setup."
        exit 1
    fi

    if grep -Eq "^[[:space:]]*SUPABASE_SERVICE_ROLE_KEY[[:space:]]*=[[:space:]]*['\"]?[[:space:]]*$" "$PROD_ENV_FILE"; then
        print_error "SUPABASE_SERVICE_ROLE_KEY in ${PROD_ENV_FILE} is empty."
        print_info "Set the cloud project's service_role key, then re-run setup."
        exit 1
    fi

    print_info "Creating Modal secret from ${PROD_ENV_FILE}..."
    modal secret create manga-translator-env --from-dotenv "$PROD_ENV_FILE" --force

    print_success "Setup completed!"
    print_info "Next steps:"
    print_info "  1. Run: ./deploy/deploy.sh deploy"
    print_info "  2. Run: ./deploy/deploy.sh models"
    print_info "  3. Run: ./deploy/deploy.sh test"
}

# Deploy the application
deploy() {
    print_header "Deploying to Modal"

    check_modal

    print_info "Deploying application..."
    modal deploy deploy/modal_app.py

    print_success "Deployment completed!"
    print_info "Your app is now live!"
    print_info "Next step: ./deploy/deploy.sh models (to download models)"
}

# Download models
download_models() {
    print_header "Downloading Models"

    check_modal

    print_info "Starting model download (this may take 30-60 minutes)..."
    print_warning "This will download ~5.2GB of models"

    modal run deploy/modal_app.py::download_models

    print_success "Models downloaded!"
}

# Run smoke tests
# Args:
#   $1 - test case name (optional): health, queue, translate_image, translate_form_image, translate_batch_json, results, or all
#   $2... - image args (optional): repeated -i/--image <path> or legacy single image path
run_tests() {
    local test_case="all"
    local image_args=()

    if [ $# -gt 0 ] && [[ "$1" != -* ]]; then
        test_case="$1"
        shift
    fi

    while [ $# -gt 0 ]; do
        case "$1" in
            -i|--image)
                if [ -n "${2:-}" ]; then
                    image_args+=("$1" "$2")
                    shift 2
                else
                    print_error "$1 requires an image path"
                    exit 1
                fi
                ;;
            *)
                if [ -f "$1" ]; then
                    image_args+=("--image" "$1")
                else
                    print_warning "Ignoring unknown test argument: $1"
                fi
                shift
                ;;
        esac
    done

    print_header "Running Smoke Tests"

    # Get Modal URL
    print_info "Fetching your Modal app URL..."
    MODAL_USERNAME=$(modal profile current)

    if [ -z "$MODAL_USERNAME" ]; then
        print_error "Could not determine Modal username"
        print_info "Please run tests manually:"
        print_info "  python deploy/smoke_test.py --url https://your-username--manga-translator-web.modal.run"
        exit 1
    fi

    MODAL_URL="https://${MODAL_USERNAME}--manga-translator-web.modal.run"
    print_info "Testing URL: $MODAL_URL"

    # Check if smoke_test.py dependencies are installed
    if ! python -c "import requests" 2>/dev/null; then
        print_info "Installing test dependencies..."
        pip install requests Pillow
    fi

    # Build test command with optional image parameters
    local test_cmd=(python deploy/smoke_test.py --url "$MODAL_URL" --verbose)

    if [ "$test_case" != "all" ]; then
        test_cmd+=(--test "$test_case")
    fi

    if [ ${#image_args[@]} -gt 0 ]; then
        print_info "Using image args: ${image_args[*]}"
        test_cmd+=("${image_args[@]}")
    fi
    
    # Run the test
    if [ "$test_case" = "all" ]; then
        print_info "Running all tests..."
    else
        print_info "Running $test_case test..."
    fi

    "${test_cmd[@]}"

    print_success "Tests completed!"
}

# View logs
view_logs() {
    print_header "Viewing Logs"

    check_modal

    print_info "Following logs (Ctrl+C to exit)..."
    modal logs manga-translator --follow
}

# Cleanup old results
cleanup_results() {
    print_header "Cleaning Up Results"

    check_modal

    print_info "Cleaning up results older than 7 days..."
    modal run deploy/modal_app.py::cleanup_old_results --max-age-days 7

    print_success "Cleanup completed!"
}

# Show help
show_help() {
    cat << EOF
Manga Image Translator - Modal Deployment Script

Usage: ./deploy/deploy.sh [command]

Commands:
  setup     Initial setup (create Modal secrets from .env.prod)
  deploy    Deploy the application to Modal
  models    Download models to persistent volume
  test [case] [-i image ...]  Run smoke tests against deployed app
            Available test cases:
              all       - Run all tests (default)
              health    - Test /health endpoint
              queue     - Test /queue-size endpoint
              translate_image - Test /translate/image endpoint
              translate_form_image - Test /translate/with-form/image endpoint
              translate_batch_json - Test /translate/batch/json endpoint
              results   - Test /results/list endpoint
            Optional image parameters:
              -i/--image <path> can be repeated for multiple images
  logs      View application logs
  cleanup   Clean up old result files
  help      Show this help message

Quick Start:
  1. Keep .env for local development
  2. vim .env.prod  # Production keys; set cloud SUPABASE_URL
  3. ./deploy/deploy.sh setup
  4. ./deploy/deploy.sh deploy
  5. ./deploy/deploy.sh models
  6. ./deploy/deploy.sh test

Examples:
  ./deploy/deploy.sh test                          # Run all tests with default image
  ./deploy/deploy.sh test health                   # Run only health check
  ./deploy/deploy.sh test translate_image -i my_manga.jpg
  ./deploy/deploy.sh test translate_batch_json -i img1.jpg -i img2.jpg

For more information, see deploy/README_modal.md

EOF
}

# Main script
main() {
    case "${1:-help}" in
        setup)
            setup
            ;;
        deploy)
            deploy
            ;;
        models)
            download_models
            ;;
        test)
            shift
            run_tests "$@"
            ;;
        logs)
            view_logs
            ;;
        cleanup)
            cleanup_results
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            print_error "Unknown command: $1"
            show_help
            exit 1
            ;;
    esac
}

main "$@"
