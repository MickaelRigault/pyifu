# pyifu
Python library to manipulate Integral Field Unit (IFU) cubes

[![PyPI](https://img.shields.io/pypi/v/pyifu.svg?style=flat-square)](https://pypi.python.org/pypi/pyifu)

# Installation
using pip:
```
pip install pyifu
```

using git:
```
git clone https://github.com/MickaelRigault/pyifu.git
cd pyifu
python setup.py install
```

### Dependencies
- propobject (install automatically if you used pip otherwise `pip install propobject`)
- astropy >= 1.3

# Usage
To load a Spectrum file:
```
from pyifu import load_spectrum
spec = load_spectrum(SPECTRUMFILE)
```

To load a IFU Cube file:
```
from pyifu import load_cube
cube = load_cube(CUBEFILE)
```
