# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

# ==============================================================================
# Makefile for building AIM containers
# ==============================================================================
# This Makefile provides targets for building, tagging, and pushing AIM
# container images. It supports both base images (with AIM runtime) and
# model-specific images (with model profiles).
#
# Common targets:
#   make build           - Build base and default model images
#   make all-models      - Build, tag, and push all model images
#   make test            - Run unit tests
#   make dev-setup       - Set up development environment
#
# Override variables:
#   make build-model ORG=meta-llama MODEL=Llama-3.1-70B-Instruct
# ==============================================================================

# ==============================================================================
# Configuration
# ==============================================================================

# Import default configuration values
-include makefile-defaults.mk

# ==============================================================================
# Computed Variables
# ==============================================================================

# Dockerfile paths
DOCKERFILE_BASE  = docker/Dockerfile.aim-base
DOCKERFILE_MODEL = docker/Dockerfile.aim

# Convert organization and model names to lowercase for Docker tag safety
ORG_LOWER   = $(shell echo $(ORG) | tr '[:upper:]' '[:lower:]')
MODEL_LOWER = $(shell echo $(MODEL) | tr '[:upper:]' '[:lower:]')

# Base container version (semantic versioning)
# Used for: aim-base:0.3
AIM_BASE_IMAGE_TAG = $(shell python -c "import tomllib; f=open('pyproject.toml', 'rb'); print(tomllib.load(f)['project']['version'])")

# Base image references
# Example: aim-base:0.3
LOCAL_BASE_IMAGE_REF  = $(AIM_BASE_REPOSITORY):$(AIM_BASE_IMAGE_TAG)
REMOTE_BASE_IMAGE_REF = $(AIM_REGISTRY_HOSTNAME)/$(AIM_REGISTRY_NAMESPACE)/$(AIM_BASE_REPOSITORY):$(AIM_BASE_IMAGE_TAG)

# Model-specific image references
# Example: aim:0.3.0-meta-llama-llama-3.1-8b-instruct-v20251001
MODEL_TAG              = $(AIM_BASE_IMAGE_TAG)-$(ORG_LOWER)-$(MODEL_LOWER)-$(DATE_VERSION)
LOCAL_MODEL_IMAGE_REF  = $(AIM_REPOSITORY):$(MODEL_TAG)
REMOTE_MODEL_IMAGE_REF = $(AIM_REGISTRY_HOSTNAME)/$(AIM_REGISTRY_NAMESPACE)/$(AIM_REPOSITORY):$(MODEL_TAG)



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
		-t $(LOCAL_BASE_IMAGE_REF) -f $(DOCKERFILE_BASE) .

# Build the model-specific AIM container
build-model: build-base tag-base
	@echo ">>> Building model-specific image: $(LOCAL_MODEL_IMAGE_REF) for $(ORG)/$(MODEL)"
	docker buildx build \
		--build-arg BASE_REGISTRY_NAMESPACE=$(AIM_REGISTRY_HOSTNAME)/$(AIM_REGISTRY_NAMESPACE) \
		--build-arg BASE_REPOSITORY=$(AIM_BASE_REPOSITORY) \
		--build-arg BASE_TAG=$(AIM_BASE_IMAGE_TAG) \
		--build-arg ORG=$(ORG) \
		--build-arg MODEL=$(MODEL) \
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
	python3 -m pytest tests/ -v -m "not integration" --cov=src/aim_runtime --cov-report=term --cov-report=html

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
					org_lower=$$(echo $$org | tr '[:upper:]' '[:lower:]'); \
					model_lower=$$(echo $$model | tr '[:upper:]' '[:lower:]'); \
					local_image="aim:$(AIM_BASE_IMAGE_TAG)-$$org_lower-$$model_lower-$(DATE_VERSION)"; \
					remote_image="$(AIM_REGISTRY_HOSTNAME)/$(AIM_REGISTRY_NAMESPACE)/aim:$(AIM_BASE_IMAGE_TAG)-$$org_lower-$$model_lower-$(DATE_VERSION)"; \
					echo ">>> Removing $$local_image and $$remote_image"; \
					docker image rm $$local_image $$remote_image || true; \
				fi; \
			done; \
		fi; \
	done
