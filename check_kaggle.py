import kagglehub
import os

path = kagglehub.dataset_download(
    "iamsouravbanerjee/animal-image-dataset-90-different-animals"
)

print(path)

for root, dirs, files in os.walk(path):
    print(root, len(files))