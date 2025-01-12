import os

# Create necessary directories
base_dir = os.path.dirname(os.path.abspath(__file__))
instance_dir = os.path.join(base_dir, 'instance')
uploads_dir = os.path.join(instance_dir, 'uploads')
resumes_dir = os.path.join(uploads_dir, 'resumes')

# Create directories if they don't exist
for directory in [instance_dir, uploads_dir, resumes_dir]:
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")
