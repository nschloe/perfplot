VERSION=$(shell python -c "import perfplot; print(perfplot.__version__)")

default:
	@echo "\"make publish\"?"

README.rst: README.md
	pandoc README.md -o README.rst
	python setup.py check -r -s || exit 1

upload: setup.py README.rst
	@if [ "$(shell git rev-parse --abbrev-ref HEAD)" != "master" ]; then exit 1; fi
	rm -f dist/*
	python setup.py bdist_wheel --universal
	gpg --detach-sign -a dist/*
	twine upload dist/*

tag:
	@if [ "$(shell git rev-parse --abbrev-ref HEAD)" != "master" ]; then exit 1; fi
	@echo "Tagging v$(VERSION)..."
	git tag v$(VERSION)
	git push --tags

publish: tag upload

clean:
	rm -f README.rst
