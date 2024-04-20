import os
import json
import subprocess

# Get the list of imported libraries
imported_libraries = [
    "streamlit",
    "streamlit_chat",
    "streamlit_lottie",
    "st_pages",

    "vertexai",
    "langchain_google_vertexai",
    "fpdf",
    
    "langchain_core",
    "langgraph",
    "langchain_community",
    "langchain_chroma",
    "langchain",
    "torch",
    "tensorflow",
    "transformers",
    "sentence-transformers",
    "langchain_experimental",
    
]

# Dictionary to store library versions
library_versions = {}

# Iterate through each library
for library in imported_libraries:
    try:
        # Use subprocess to execute pip show command to get the version
        result = subprocess.run(['pip', 'show', library], capture_output=True, text=True)
        output = result.stdout.strip()
        lines = output.split('\n')
        version_line = [line for line in lines if line.startswith("Version:")][0]
        version = version_line.split(":")[1].strip()
        library_versions[library] = version
    except Exception as e:
        print(f"Error getting version for {library}: {e}")

# Write library versions to a text file
with open("library_versions.txt", "w") as file:
    for library, version in library_versions.items():
        file.write(f"{library}=={version}\n")

print("Library versions saved to library_versions.txt")
