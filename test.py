import torch
print(torch.__version__)         # 应显示正确版本
print(torch.cuda.is_available()) # 应该返回True
print(torch.cuda.get_device_name(0))  # 显示GPU型号