VERSION=$(shell python3 -c "import perfplot; print(perfplot.__version__)")

default:
	@echo "\"make publish\"?"

tag:
	@if [ "$(shell git rev-parse --abbrev-ref HEAD)" != "master" ]; then exit 1; fi
	@echo "Tagging v$(VERSION)..."
	git tag v$(VERSION)
	git push --tags

upload: setup.py
	@if [ "$(shell git rev-parse --abbrev-ref HEAD)" != "master" ]; then exit 1; fi
	rm -f dist/*
	python3 setup.py sdist
	python3 setup.py bdist_wheel --universal
	twine upload dist/*

publish: tag upload

clean:
	@find . | grep -E "(__pycache__|\.pyc|\.pyo$\)" | xargs rm -rf
	@rm -rf *.egg-info/ build/ dist/

black:
	black setup.py perfplot/ test/*.py

lint:
	black --check setup.py perfplot/ test/*.py
	flake8 setup.py perfplot/ test/*.py
