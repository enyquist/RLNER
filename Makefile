init: ## Initialize Project
	@python3.8 -m venv venv
	@./venv/bin/python3 -m pip install -U pip setuptools wheel
	@./venv/bin/python3 -m pip install -r requirements/requirements.dev.txt
	@./venv/bin/python3 -m pip install -e . --no-deps
	@./venv/bin/python3 -m pre_commit install --install-hooks --overwrite

clean:  ## remove build artifacts
	rm -rf build dist venv pip-wheel-metadata htmlcov
	find . -name .tox | xargs rm -rf
	find . -name __pycache__ | xargs rm -rf
	find . -name .pytest_cache | xargs rm -rf
	find . -name *.egg-info | xargs rm -rf
	find . -name setup-py-dev-links | xargs rm -rf

update: clean init

lint: ## Run linters
	@./venv/bin/pre-pre_commit run -a

test: lint ## Run tests
	@./venv/bin/pytest -vv --durations=10 --cov-fail-under=1 --cov=rlner --cov-report html tests/

docs: init ##
	@rm -rf docs.log build/html/
	@./venv/bin/sphinx-build -c docs.log docs build/html/
