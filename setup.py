from setuptools import setup, find_packages

setup(
    name="alphai_watson",
    version="0.3.0",
    packages=find_packages(exclude=["doc", "tests*"]),
    url="https://github.com/alpha-i/tool-watson-python",
    license="",
    author="Gabriele Alese, Daniele Murroni",
    author_email="gabriele.alese@alphai.co, daniele.murroni@alpha-i.co",
    install_requires=[
        "numpy",
        "pandas==0.22",
        "h5py==2.7.1",
        "scipy",
        "sklearn",
        "numpy",
        "contexttimer",
        "tables==3.4.2"
    ],
    description="The detectives' helper"
)
