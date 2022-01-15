default:
	@echo "\"just publish\"?"

publish:
	@if [ "$(git rev-parse --abbrev-ref HEAD)" != "main" ]; then exit 1; fi
	gh release create
	flit publish

clean:
	@find . | grep -E "(__pycache__|\.pyc|\.pyo$)" | xargs rm -rf
	@rm -rf src/*.egg-info/ build/ dist/ .tox/

format:
	isort .
	black .
	blacken-docs README.md

lint:
	black --check .
	flake8 .
