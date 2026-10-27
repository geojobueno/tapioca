# How to install tapIOca

```bash
git clone https://github.com/ggciag/tapioca.git #or https://github.com/geojobueno/tapioca.git
cd tapioca
conda env create -f environment.yml -y
pip install -e .
```

To check that everything works, run the tests (with pytest, included in `environment.yml`) from the root of the repository:

```bash
pytest
```