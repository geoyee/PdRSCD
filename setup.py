import setuptools

with open("README.md", "r", encoding='utf-8') as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as fr:
    requires = fr.readlines()
for i in range(len(requires)):
    requires[i] = requires[i].strip().replace('>', '=')

setuptools.setup(
    name="ppcd",
    version="0.1",
    author="geoyee",
    author_email="geoyee@yeah.net",
    description="SDK about pdrscd",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/geoyee/PdRSCD",
    packages=setuptools.find_packages(),
    install_requires=requires,
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)