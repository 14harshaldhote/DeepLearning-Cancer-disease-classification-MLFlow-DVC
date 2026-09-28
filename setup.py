import setuptools

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()


__version__ = "2.0.0"

REPO_NAME = "DeepLearning-Cancer-disease-classification-MLFlow-DVC"
AUTHOR_USER_NAME = "14harshaldhote"
SRC_REPO = "cnnClassifier"
AUTHOR_EMAIL = "dhoteharshal16@gmail.com"


setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="Chest CT cancer classifier: DVC pipeline, MLflow tracking, FastAPI + Grad-CAM",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    },
    python_requires=">=3.10,<3.12",
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
)
