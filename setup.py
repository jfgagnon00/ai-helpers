from setuptools import find_namespace_packages, setup

def _dependencies():
    return [
        "contractions",
        "fanalysis",
        "inflect",
        "ipykernel",
        "ipywidgets",
        "ipywidgets",
        "jupyterlab",
        "keras",
        "matplotlib",
        "mlxtend",
        "nltk",
        "numpy",
        "pandas",
        "scikit-learn",
        "scipy",
        "seaborn",
        "sympy",
        "tensorflow",
        "torch",
        "tqdm",
    ]

setup(name="ai-helpers",
      packages=find_namespace_packages(where="src"),
      package_dir={"": "src"},
      version="0.0.1",
      description="Helpers AI.",
      author="Jean-Francois Gagnon",
      install_requires=_dependencies())
