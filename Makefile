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

rebuild-notebooks: ## Re-run notebooks for latest outputs
	./venv/bin/python3 src/docs/run-notebooks.py; \
	username=`whoami`; \
	date; \
	for f in $(shell find notebooks/ -type f -name "*.ipynb" | sort); do \
		echo "Scrub username from '$${f}'"; \
		sed -i"" "s/$${username}/jdoe/g" $${f}; \
		date; \
	done

docs-build:  # Build docs
	# run sphinx to build docs
	./venv/bin/sphinx-build -c docs/ -w docs.log docs/ docs/_build/html/
	mkdir -p docs/_build/html/_static/notebooks
	cp notebooks/*.ipynb docs/_build/html/_static/notebooks

docs: rebuild-notebooks docs-build ## Build documentation and API docs

serve-docs: docs  # Serve docs in web-browser
	firefox docs/_build/html/index.html
