import transformers
from transformers.models.qwen2_vl import Qwen2VLImageProcessor
import inspect

# 查看类的定义位置
print(inspect.getfile(Qwen2VLImageProcessor))