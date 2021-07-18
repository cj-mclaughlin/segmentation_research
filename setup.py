import setuptools
import os

CWD = os.path.abspath(os.path.dirname(__file__))
NAME = 'segmentation_research'
REQUIRES_PYTHON = '>=3.0.0'

try:
    with open(os.path.join(CWD, 'requirements.txt'), encoding='utf-8') as f:
        REQUIRED = f.read().split('\n')
except:
    REQUIRED = []

EXTRAS = {
    'tests': ['pytest']
}

setuptools.setup(
    name=NAME,
    python_requires=REQUIRES_PYTHON,
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    include_package_data=True,
)