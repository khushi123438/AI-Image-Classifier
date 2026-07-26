import kagglehub
import os
import shutil

# Download dataset
path = kagglehub.dataset_download(
    "iamsouravbanerjee/animal-image-dataset-90-different-animals"
)

print("Dataset location:", path)


# Tumhare project ka dataset folder
target = "dataset"

os.makedirs(target, exist_ok=True)


# Required classes
animals = [
    "cat",
    "dog",
    "horse",
    "lion"
]


for animal in animals:

    source_folder = None

    # dataset ke andar folder search karega
    for root, dirs, files in os.walk(path):

        for d in dirs:
            if d.lower() == animal:
                source_folder = os.path.join(root,d)
                break

        if source_folder:
            break


    if source_folder:

        destination = os.path.join(target, animal)

        shutil.copytree(
            source_folder,
            destination,
            dirs_exist_ok=True
        )

        print(animal,"copied")

    else:
        print(animal,"not found")