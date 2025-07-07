# Environment Setup
## Python Environment
Python 3.11+
PyTorch 2.0+
no other dependencies are required for this project.

# API Documentation
minimalist API for the Animator model.

```python
class AnimatorAPI:
    def __init__(self):
        # add initialization steps here
    def prep_frame(self, frame: Dict[str, Any]):
        # add preprocessing steps here
    def animate(self, source: Dict[str, Any], reference: Dict[str, Any], driving: Dict[str, Any]) -> Tensor:
        # add animation steps here
```

NOTE: All computations are done on the GPU and inorder to keep the application real-time, the model should be optimized and avoid unnecessary computations.