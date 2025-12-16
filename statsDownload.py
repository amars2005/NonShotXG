import kagglehub

# Download latest version
path = kagglehub.dataset_download("saurabhshahane/statsbomb-football-data")

print("Path to dataset files:", path)