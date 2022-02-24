import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='toolbox',
    version='0.0.1',
    author='Thomas Champion',
    author_email="thomas@thomaschampion.net",
    description='Collection of useful scripts for use in ML analysis/modeling',
    long_description_content_type="text/markdown",
    url='https://github.com/tlchampion/toolbox',
    project_urls={
        "Bug Tracker": "https://github.com/tlchampion/toolbox/issues"
    },
    license='MIT',
    packages=['toolbox'],
    install_requires=['os', 'scikit-learn', 'pandas',
                      'numpy', 'path', 'itertools', 'matplotlib', 'zipfile']
)
