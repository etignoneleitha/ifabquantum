# xg-lesioni

A library for QAOA with bayesian optimisation.

## Usage

### Poetry

Open the terminal and:
- Enter the repo: ```cd <path-to-this-repo-folder>```
- Install a poetry virtual environment: ```poetry install```
- Look at the available arguments (```poetry run python src/main_qutip.py --help```) and run the program, e.g.: ```poetry run python src/main_qutip.py --num_nodes 6 --p 2```

NOTE: In order to install the poetry environment you **need Python >= 3.7.1 and < 3.8** on your computer, in order to make [Pulser](https://pypi.org/project/pulser/) work properly. If you don't have a Python version satisfying these requirements on your computer, use Docker or Anaconda.

### Docker

Open the terminal and:
- Enter the repo: ```cd <path-to-this-repo-folder>```
- Build the image: ```docker build -t qaoa-pipeline .```
- Run the container and enter: ```docker run --rm -v <absolute-path-to-folder-src-in-this-repo>:/qaoa-pipeline/src -v <absolute-path-to-folder-output-in-this-repo>:/qaoa-pipeline/output -it qaoa-pipeline```
- Once inside the container, you can look at the available arguments (```poetry run python src/main_qutip.py --help```) and run the program, e.g.: ```poetry run python src/main_qutip.py --num_nodes 6 --p 2```

### Anaconda
TODO
