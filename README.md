![Python 3](https://img.shields.io/badge/python-3-blue.svg)
# geoviz
 **Geoviz is a library with learning purposes for geometric visualization of the transformations inside a feedforward neural network.**

<img src="./docs/img/Figure 299_v2.png" alt="Example img" width="450px">

## Installation
To install geoviz, first download or clone the code from the repository: 

```bash
git clone https://github.com/fpier21/geoviz.git 
```
Then, inside the main folder just install it via pip.  
The library requires torch to be installed. In case you haven't it already, you can install geoviz with pytorch in CPU platform via: 

```bash
pip install . --extra-index-url https://download.pytorch.org/whl/cpu
```

If instead you want the full CUDA platform pytorch just execute: 

```bash
pip install .
```