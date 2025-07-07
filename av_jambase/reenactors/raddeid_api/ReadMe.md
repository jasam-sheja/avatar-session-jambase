# Environment Setup
## Python Environment
Python 3.11+
PyTorch 2.0+
<!-- please add the necessary steps to setup the environment -->

# API Documentation
minimalist API for the Animator model.

```python
class AnimatorAPI:
    def __init__(self):
        # add initialization steps here
    def prep_frame(self, frame: Dict[str, Any])
        # add preprocessing steps here
    def animate(self, source: Dict[str, Any], reference: Dict[str, Any], driving: Dict[str, Any]) -> Tensor:
        # add animation steps here
```

NOTE: All computations are done on the GPU and inorder to keep the application real-time, the model should be optimized and avoid unnecessary computations.

## Install Other Dependencies
```bash
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
pip install kornia basicsr facexlib
```

### Fixes:
#### ```ModuleNotFoundError: No module named 'torchvision.transforms.functional_tensor'```
Solution from https://github.com/Daethyra/FreeStream/issues/64

Open the file `.../envs/pytorch3d/lib/python3.11/site-packages/basicsr/data/degradations.py`
and change the import statement from:
```python
from torchvision.transforms.functional_tensor import ...
```
to:
```python
from torchvision.transforms.functional import ...
```

#### `#error -- unsupported GNU version! gcc versions later than 11 are not supported! The nvcc flag '-allow-unsupported-compiler' can be used to override this version check; however, using an unsupported host compiler may cause compilation failure or incorrect run time execution. Use at your own risk.`
check the version of gcc and g++:
```bash
gcc --version
g++ --version
```
if the version is greater than 11, you can use the following command to set the version to 11:
```bash
sudo apt install gcc-11 g++-11
export CC=gcc-11
export CXX=g++-11
```
then rerun the installation command