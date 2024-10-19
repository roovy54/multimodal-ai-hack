import tarfile

file_path = "public_dataset_updated.tar.gz"

with tarfile.open(file_path, "r:gz") as tar:
    tar.extractall("/")
    tar.close()
