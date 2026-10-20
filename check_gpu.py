"""Проверка, что PyTorch видит видеокарту: python check_gpu.py"""
import torch

print("PyTorch:", torch.__version__)
print("Сборка с CUDA:", torch.version.cuda or "нет (CPU-версия)")
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name}, {props.total_memory / 1024 ** 3:.1f} ГБ")
    x = torch.randn(1024, 1024, device="cuda")
    print("Тест вычислений на GPU:", "OK" if torch.isfinite(x @ x).all() else "ошибка")
else:
    print("GPU недоступна — обучение пойдёт на CPU")
