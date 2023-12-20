.SILENT: ;

setup:
	rm -rf venv build dist *.egg-info
	python -m venv venv

install: 
	pip install --upgrade pip
	python setup.py install

clean:
	rm -rf build *.egg-info *.log report.tar.gz

uninstall:
	rm -rf venv build dist *.egg-info
	find . -type d -name '__pycache__' -exec rm -r {} +

build:
	python setup.py sdist bdist_wheel

publish:
	python -m twine upload dist/*

help:
	echo 'Usage:'   
	echo '    make <command> [options]'
	echo 'Commands:'
	echo '    setup        Sets up a fresh virtual environment.'
	echo '    install      Installs necessary packages.'
	echo '    clean        Removes dist and build directories.'
	echo '    uninstall    Removes the virtual environment, dist, build and all __pycache__ directories.'