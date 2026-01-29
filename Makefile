# ConsciousnessX Production Makefile

.PHONY: help setup build up down logs clean test lint deploy

help: ## Show this help
	@echo "ConsciousnessX Production Commands"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

setup: ## Initial setup
	@echo "Running setup..."
	@python scripts/setup.py

build: ## Build Docker images
	@echo "Building Docker images..."
	@docker-compose build

up: ## Start all services
	@echo "Starting services..."
	@docker-compose up -d

down: ## Stop all services
	@echo "Stopping services..."
	@docker-compose down

logs: ## Show service logs
	@docker-compose logs -f

logs-api: ## Show API logs
	@docker-compose logs -f consciousnessx-api

logs-webui: ## Show WebUI logs
	@docker-compose logs -f consciousnessx-webui

clean: ## Clean up containers and volumes
	@echo "Cleaning up..."
	@docker-compose down -v
	@rm -rf uploads/* logs/* data/*

test: ## Run tests
	@echo "Running tests..."
	@pytest tests/ -v

test-unit: ## Run unit tests
	@pytest tests/unit -v

test-integration: ## Run integration tests
	@pytest tests/integration -v

lint: ## Run linters
	@echo "Running linters..."
	@black --check src/
	@mypy src/

format: ## Format code
	@black src/

migrate: ## Run database migrations
	@echo "Running migrations..."
	@docker-compose exec consciousnessx-api alembic upgrade head

seed: ## Seed database with test data
	@echo "Seeding database..."
	@python scripts/seed_data.py

backup: ## Backup database
	@echo "Creating backup..."
	@./scripts/backup.sh

monitor: ## Open monitoring dashboard
	@echo "Opening monitoring..."
	@open http://localhost:3000 || xdg-open http://localhost:3000 || echo "Open http://localhost:3000 in your browser"

api-docs: ## Open API documentation
	@echo "Opening API docs..."
	@open http://localhost:8000/docs || xdg-open http://localhost:8000/docs || echo "Open http://localhost:8000/docs in your browser"

webui: ## Open WebUI
	@echo "Opening WebUI..."
	@open http://localhost:8501 || xdg-open http://localhost:8501 || echo "Open http://localhost:8501 in your browser"

deploy-staging: ## Deploy to staging
	@echo "Deploying to staging..."
	@./scripts/deploy.sh staging

deploy-production: ## Deploy to production
	@echo "Deploying to production..."
	@./scripts/deploy.sh production

health: ## Check service health
	@curl -f http://localhost:8000/health && echo "API: ✅" || echo "API: ❌"
	@curl -f http://localhost:8501/_stcore/health && echo "WebUI: ✅" || echo "WebUI: ❌"
	@curl -f http://localhost:9090/-/healthy && echo "Prometheus: ✅" || echo "Prometheus: ❌"

# Shortcuts
start: up ## Start services (alias for up)
stop: down ## Stop services (alias for down)
restart: down up ## Restart services
