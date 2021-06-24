import setuptools

# with open("README.md", "r", encoding='utf-8') as fh:
#     long_description = fh.read()

long_description = "PdRSCD（PaddlePaddle Remote Sensing Change Detection）\
                    是一个基于飞桨PaddlePaddle的一个用于遥感变化检测的工具。\
                    详细说明和使用方法请参照github。"

with open("requirements.txt", "r") as fr:
    requires = fr.readlines()
for i in range(len(requires)):
    requires[i] = requires[i].strip()
    if requires[i] == 'GDAL':  # GDAL无法通过pip安转
        del requires[i]

setuptools.setup(
    name="ppcd",
    version="0.1.2",
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