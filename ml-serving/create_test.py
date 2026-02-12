import json
import numpy as np

# Создаем тестовый тензор правильной формы: [1, 1, 28, 28]
test_data = np.ones((1, 1, 28, 28)) * 0.1

data = {
    "input": test_data.tolist()
}

with open("correct_input.json", "w") as f:
    json.dump(data, f)

print("Файл correct_input.json создан с правильной формой [1, 1, 28, 28]")
