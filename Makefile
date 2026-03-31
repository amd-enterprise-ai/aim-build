# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

# ==============================================================================
# Makefile for building AIM containers
# ==============================================================================
# This Makefile builds AIM container images using the standard AIM conventions:
#
#   - Image naming: per-model repos  aim-{org}-{model}:{version}
#   - Name sanitization: lowercase, non-alphanumeric → "-", collapse separators
#   - Base image source: reads assets/{ACCELERATOR_TYPE}/{ORG}/{MODEL}/config.yaml
#   - OCI labels: generated via aim_utils.generate_labels
#
# Local dev builds use a ".0-dev" version suffix (e.g. 0.10.0-dev) to
# distinguish them from published releases which use full semver tags.
#
# Common targets:
#   make build           - Build base + default model image
#   make build-model ORG=<org> MODEL=<model>  - Build a specific model image
#   make all-models      - Build, tag, and push all model images
#   make test            - Run unit tests
#   make dev-setup       - Set up development environment
#
# Examples:
#   make build-model ORG=meta-llama MODEL=Llama-3.1-70B-Instruct
#   make build-model ORG=Qwen MODEL=Qwen3-32B
# ==============================================================================

# ==============================================================================
# Configuration
# ==============================================================================

# Import default configuration values
-include makefile-defaults.mk

# If zen5 flag is set, include zen5 makefile (must be before ACCELERATOR_TYPE
# default and computed variables so overrides take effect)
ifeq ($(BUILD_ZEN5), true)
include makefile-zen5.mk
endif

# Accelerator type: instinct (GPU) or epyc
ACCELERATOR_TYPE ?= instinct

# ==============================================================================
# Computed Variables
# ==============================================================================

BASE_REGISTRY_NAMESPACE   := $(shell python -m aim_utils.config_utils get base_image.registry_host --canonical_name base --assets_path assets/$(ACCELERATOR_TYPE) 2>/dev/null || echo $(BASE_REGISTRY_NAMESPACE))
BASE_REPOSITORY := $(shell python -m aim_utils.config_utils get base_image.base_repository --canonical_name base --assets_path assets/$(ACCELERATOR_TYPE) 2>/dev/null || echo $(BASE_REPOSITORY))
BASE_TAG        := $(shell python -m aim_utils.config_utils get base_image.base_tag --canonical_name base --assets_path assets/$(ACCELERATOR_TYPE) 2>/dev/null || echo $(BASE_TAG))

# Dockerfile paths
ifeq ($(ACCELERATOR_TYPE),epyc)
  DOCKERFILE_BASE = docker/Dockerfile.aim-epyc-base
  AIM_BASE_REPOSITORY = aim-epyc-base
  AIM_REPOSITORY = aim-epyc
else
  DOCKERFILE_BASE = docker/Dockerfile.aim-base
endif
DOCKERFILE_MODEL = docker/Dockerfile.aim

# ==============================================================================
# Name Sanitization
# ==============================================================================
# Docker-safe names: lowercase → replace non-[a-z0-9._-] with "-" → collapse → strip
ORG_SANITIZED   = $(shell echo '$(ORG)' | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9._-]/-/g' | sed 's/[-_.][-._ ]*/-/g' | sed 's/^[-_.]//;s/[-_.]$$//')
MODEL_SANITIZED = $(shell echo '$(MODEL)' | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9._-]/-/g' | sed 's/[-_.][-._ ]*/-/g' | sed 's/^[-_.]//;s/[-_.]$$//')

# ==============================================================================
# Base Image Version
# ==============================================================================
# Base container version from pyproject.toml (semantic versioning)
AIM_BASE_IMAGE_TAG = $(shell python -c "import tomllib; f=open('pyproject.toml', 'rb'); print(tomllib.load(f)['project']['version'])")

# Base image references
# Example: aim-base:0.10
LOCAL_BASE_IMAGE_REF  = $(AIM_BASE_REPOSITORY):$(AIM_BASE_IMAGE_TAG)
REMOTE_BASE_IMAGE_REF = $(AIM_REGISTRY_HOSTNAME)/$(AIM_REGISTRY_NAMESPACE)/$(AIM_BASE_REPOSITORY):$(AIM_BASE_IMAGE_TAG)

# ==============================================================================
# Model Image References
# ==============================================================================
# Per-model repositories: aim-{org_sanitized}-{name_sanitized}:{version}
# Local dev builds use {base_version}.0-dev (no registry lookup needed)
# Example: aim-meta-llama-llama-3-1-8b-instruct:0.10.0-dev
MODEL_IMAGE_REPOSITORY = $(AIM_REPOSITORY)-$(ORG_SANITIZED)-$(MODEL_SANITIZED)
MODEL_TAG              = $(AIM_BASE_IMAGE_TAG).0-dev
LOCAL_MODEL_IMAGE_REF  = $(MODEL_IMAGE_REPOSITORY):$(MODEL_TAG)
REMOTE_MODEL_IMAGE_REF = $(AIM_REGISTRY_HOSTNAME)/$(AIM_REGISTRY_NAMESPACE)/$(MODEL_IMAGE_REPOSITORY):$(MODEL_TAG)

# ==============================================================================
# Model Base Image (from assets config)
# ==============================================================================
# Base image info is read from assets/{ACCELERATOR_TYPE}/{ORG}/{MODEL}/config.yaml
MODEL_BASE_REGISTRY   = $(shell python -m aim_utils.config_utils get base_image.registry_host --canonical_name $(ORG)/$(MODEL) --assets_path assets/$(ACCELERATOR_TYPE) 2>/dev/null || echo $(AIM_REGISTRY_HOSTNAME))
MODEL_BASE_NAMESPACE  = $(shell python -m aim_utils.config_utils get base_image.base_registry_namespace --canonical_name $(ORG)/$(MODEL) --assets_path assets/$(ACCELERATOR_TYPE) 2>/dev/null || echo $(AIM_REGISTRY_NAMESPACE))
MODEL_BASE_REPOSITORY = $(shell python -m aim_utils.config_utils get base_image.base_repository --canonical_name $(ORG)/$(MODEL) --assets_path assets/$(ACCELERATOR_TYPE) 2>/dev/null || echo $(AIM_BASE_REPOSITORY))
MODEL_BASE_TAG        = $(shell python -m aim_utils.config_utils get base_image.base_tag --canonical_name $(ORG)/$(MODEL) --assets_path assets/$(ACCELERATOR_TYPE) 2>/dev/null || echo $(AIM_BASE_IMAGE_TAG))

# ==============================================================================
# OCI Metadata Labels (via aim_utils.generate_labels)
# ==============================================================================
# Generates OCI-standard --label flags for docker buildx build.
# Uses LABELS_IS_DOCKER_FORMAT to output "--label key=value" format.
LABELS_COMMON_ENV = \
	LABELS_SERVER_URL=https://github.com \
	LABELS_REPOSITORY=silogen/aim-build \
	LABELS_SHA=$$(git rev-parse HEAD) \
	LABELS_TIMESTAMP=$$(date -u +"%Y-%m-%dT%H:%M:%SZ") \
	LABELS_UPDATED_AT=$$(date -u +"%Y-%m-%dT%H:%M:%SZ") \
	LABELS_IS_DOCKER_FORMAT=true

LABELS_BASE = $(shell $(LABELS_COMMON_ENV) \
	LABELS_IS_BASE=true \
	LABELS_VERSION_NUMBER=$(AIM_BASE_IMAGE_TAG) \
	python -m aim_utils.generate_labels assets/$(ACCELERATOR_TYPE)/base/metadata.yaml)

LABELS_MODEL = $(shell $(LABELS_COMMON_ENV) \
	LABELS_IS_BASE=false \
	LABELS_ORG=$(ORG) \
	LABELS_MODEL_NAME=$(MODEL) \
	LABELS_VERSION_NUMBER=$(MODEL_TAG) \
	python -m aim_utils.generate_labels assets/$(ACCELERATOR_TYPE)/$(ORG)/$(MODEL)/metadata.yaml)

# ==============================================================================
# Targets
# ==============================================================================
.PHONY: all build build-base build-model build-all-models \
        tag tag-base tag-model tag-all-models \
        push push-base push-model push-all-models \
        all-models clean clean-all-models

# ==============================================================================
# Main Targets
# ==============================================================================

# Default target: build base and default model image
all: build

# Complete workflow: build, tag, and push all model-specific containers
all-models: build-all-models tag-all-models push-all-models

# ==============================================================================
# Build Targets
# ==============================================================================

# Build base and default model images
build: build-base build-model

# Build the base AIM container (contains AIM runtime, no model-specific profiles)
build-base:
	@echo ">>> Building base image: $(LOCAL_BASE_IMAGE_REF)"
	docker buildx build \
		--build-arg BASE_REGISTRY_NAMESPACE=$(BASE_REGISTRY_NAMESPACE) \
		--build-arg BASE_REPOSITORY=$(BASE_REPOSITORY) \
		--build-arg BASE_TAG=$(BASE_TAG) \
		$(LABELS_BASE) \
		-t $(LOCAL_BASE_IMAGE_REF) -f $(DOCKERFILE_BASE) .

# Build the model-specific AIM container
# Uses base image reference from assets/{ACCELERATOR_TYPE}/{ORG}/{MODEL}/config.yaml
build-model: build-base tag-base
	@echo ">>> Building model-specific image: $(LOCAL_MODEL_IMAGE_REF) for $(ORG)/$(MODEL)"
	@echo "    Base image: $(MODEL_BASE_REGISTRY)/$(MODEL_BASE_NAMESPACE)/$(MODEL_BASE_REPOSITORY):$(MODEL_BASE_TAG)"
	docker buildx build \
		--build-arg BASE_REGISTRY_NAMESPACE=$(MODEL_BASE_REGISTRY)/$(MODEL_BASE_NAMESPACE) \
		--build-arg BASE_REPOSITORY=$(MODEL_BASE_REPOSITORY) \
		--build-arg BASE_TAG=$(MODEL_BASE_TAG) \
		--build-arg ORG=$(ORG) \
		--build-arg MODEL=$(MODEL) \
		--build-arg ACCELERATOR_TYPE=$(ACCELERATOR_TYPE) \
		$(LABELS_MODEL) \
		-t $(LOCAL_MODEL_IMAGE_REF) \
		-f $(DOCKERFILE_MODEL) .

# Build all model-specific containers for all organization/model profiles
build-all-models: build-base tag-base
	@echo ">>> Building model-specific containers for all profiles"
	@for org_dir in profiles/*/; do \
		if [ "$$(basename $$org_dir)" != "general" ]; then \
			org=$$(basename $$org_dir); \
			for model_dir in $$org_dir*/; do \
				if [ -d "$$model_dir" ]; then \
					model=$$(basename $$model_dir); \
					echo ">>> Building container for $$org/$$model"; \
					$(MAKE) build-model ORG=$$org MODEL=$$model || exit 1; \
				fi; \
			done; \
		fi; \
	done

# ==============================================================================
# Tag Targets
# ==============================================================================

# Tag all images for remote registry
tag: tag-base tag-model

# Tag the base image
tag-base: build-base
	@echo ">>> Tagging base image for registry: $(REMOTE_BASE_IMAGE_REF)"
	docker tag $(LOCAL_BASE_IMAGE_REF) $(REMOTE_BASE_IMAGE_REF)

# Tag the model-specific image
tag-model:
	@echo ">>> Tagging model-specific image for registry: $(REMOTE_MODEL_IMAGE_REF)"
	docker tag $(LOCAL_MODEL_IMAGE_REF) $(REMOTE_MODEL_IMAGE_REF)

# Tag all model-specific containers for all profiles
tag-all-models:
	@echo ">>> Tagging all model-specific containers for remote registry"
	@for org_dir in profiles/*/; do \
		if [ "$$(basename $$org_dir)" != "general" ]; then \
			org=$$(basename $$org_dir); \
			for model_dir in $$org_dir*/; do \
				if [ -d "$$model_dir" ]; then \
					model=$$(basename $$model_dir); \
					echo ">>> Tagging container for $$org/$$model"; \
					$(MAKE) tag-model ORG=$$org MODEL=$$model || exit 1; \
				fi; \
			done; \
		fi; \
	done

# ==============================================================================
# Push Targets
# ==============================================================================

# Push all images to the remote registry
push: push-base push-model

# Push the base image
push-base: tag-base
	@echo ">>> Pushing base image: $(REMOTE_BASE_IMAGE_REF)"
	docker push $(REMOTE_BASE_IMAGE_REF)

# Push the model-specific image
push-model: tag-model
	@echo ">>> Pushing model-specific image: $(REMOTE_MODEL_IMAGE_REF)"
	docker push $(REMOTE_MODEL_IMAGE_REF)

# Push all model-specific containers for all profiles
push-all-models:
	@echo ">>> Pushing all model-specific containers to remote registry"
	@for org_dir in profiles/*/; do \
		if [ "$$(basename $$org_dir)" != "general" ]; then \
			org=$$(basename $$org_dir); \
			for model_dir in $$org_dir*/; do \
				if [ -d "$$model_dir" ]; then \
					model=$$(basename $$model_dir); \
					echo ">>> Pushing container for $$org/$$model"; \
					$(MAKE) push-model ORG=$$org MODEL=$$model || exit 1; \
				fi; \
			done; \
		fi; \
	done

# ==============================================================================
# Dependency Management
# ==============================================================================

# Update dependencies in requirements.txt based on pyproject.toml
.PHONY: update-dependencies
update-dependencies:
	pip install pip-tools
	python -m piptools compile --upgrade --resolver backtracking -o requirements/requirements.txt pyproject.toml

.PHONY: update-test-dependencies
update-test-dependencies: update-dependencies
	python -m piptools compile --extra test --upgrade --resolver backtracking -o requirements/test-requirements.txt pyproject.toml

.PHONY: update-dev-dependencies
update-dev-dependencies: update-dependencies
	python -m piptools compile --extra dev --upgrade --resolver backtracking -o requirements/dev-requirements.txt pyproject.toml

.PHONY: update-evaluation-dependencies
update-evaluation-dependencies: update-dependencies
	python -m piptools compile --extra evaluation --upgrade --resolver backtracking -o requirements/evaluation-requirements.txt pyproject.toml

# ==============================================================================
# Testing Targets
# ==============================================================================

# Run tests
.PHONY: test
test: update-test-dependencies
	@echo ">>> Running unit tests (excluding integration tests)"
	pip install -r requirements/requirements.txt
	pip install -r requirements/test-requirements.txt
	pip install --no-deps .
	python3 -m pytest tests/ -v -m "not integration"

# Run tests with coverage report
.PHONY: test-cov
test-cov:
	@echo ">>> Running unit tests with coverage (excluding integration tests)"
	python3 -m pytest tests/ -v -m "not integration" --cov=src/aim_runtime --cov=src/aim_common --cov=src/aim_utils --cov-report=term --cov-report=html

# Run CI tests with dedicated coverage report
.PHONY: test-ci-cov
test-ci-cov:
	@echo ">>> Running CI tests with coverage (report in htmlcov-ci/)"
	python3 -m pytest tests/ci/ -v --cov=ci --cov-config=.coveragerc-ci

# Run integration tests (requires GPU environment)
.PHONY: test-integration
test-integration:
	@echo ">>> Running integration tests (requires GPU/ROCm environment)"
	python3 -m pytest tests/ -v -m integration

# Run all tests including integration tests
.PHONY: test-all
test-all:
	@echo ">>> Running all tests (unit + integration)"
	python3 -m pytest tests/ -v

# Run tests and open coverage report
.PHONY: test-cov-open
test-cov-open: test-cov
	@echo ">>> Opening coverage report"
	@command -v xdg-open >/dev/null 2>&1 && xdg-open htmlcov/index.html || echo "Coverage report saved to htmlcov/index.html"

# ==============================================================================
# Development Targets
# ==============================================================================

# Run pre-commit hooks
.PHONY: lint
lint:
	@echo ">>> Running pre-commit hooks"
	pre-commit run --all-files

# Install development dependencies
.PHONY: dev-setup
dev-setup: update-dev-dependencies
	@echo ">>> Setting up development environment"
	pip install -r requirements/requirements.txt
	pip install -r requirements/dev-requirements.txt
	pip install --no-deps -e .
	pre-commit install

# ==============================================================================
# Cleanup Targets
# ==============================================================================

# Clean up local images
clean:
	@echo ">>> Cleaning up local Docker images"
	docker image rm $(LOCAL_BASE_IMAGE_REF) || true
	docker image rm $(LOCAL_MODEL_IMAGE_REF) || true
	docker image rm $(REMOTE_BASE_IMAGE_REF) || true
	docker image rm $(REMOTE_MODEL_IMAGE_REF) || true

# Clean up all model-specific images
clean-all-models:
	@echo ">>> Cleaning up all model-specific Docker images"
	@for org_dir in profiles/*/; do \
		if [ "$$(basename $$org_dir)" != "general" ]; then \
			org=$$(basename $$org_dir); \
			for model_dir in $$org_dir*/; do \
				if [ -d "$$model_dir" ]; then \
					model=$$(basename $$model_dir); \
					echo ">>> Removing images for $$org/$$model"; \
					$(MAKE) clean ORG=$$org MODEL=$$model || true; \
				fi; \
			done; \
		fi; \
	done
