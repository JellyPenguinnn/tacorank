from setuptools import find_packages, setup


setup(
    packages=(
        find_packages("src")
        + find_packages(".", include=("benchmarks", "benchmarks.*"))
    ),
    package_dir={"": "src", "benchmarks": "benchmarks"},
)
