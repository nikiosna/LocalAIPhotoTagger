# Local AI Photo Tagger

This project is a suite of Python scripts designed to automate the process of organizing and tagging a local photo library using a local AI model. It can classify images with descriptive names, generate and embed tags directly into image metadata, and clean up tag inconsistencies across your entire collection.

The scripts are designed to work with any local AI model that is compatible with the OpenAI API standard, such as LM Studio, Ollama, etc.

## Features

-   **Image Classification**: Automatically renames image files based on their content and generates a JSON file with a title, description, and keywords.
-   **AI-Powered Tagging**: Generates relevant tags for images and writes them to XMP and EXIF metadata.
-   **Tag Deduplication**: Scans the entire library for similar or duplicate tags (e.g., "cat" vs "cats") and intelligently merges them.
-   **Recursive Processing**: Scripts can process images in nested subdirectories.
-   **Metadata Focused**: Writes tags directly to image files (XMP and EXIF), ensuring compatibility with photo management software like Adobe Lightroom or digiKam.

## Prerequisites

-   Python 3.7+
-   A running local AI model server (e.g., LM Studio) that exposes an OpenAI-compatible API endpoint.
-   `exiftool` must be installed and available in your system's PATH. You can install it from the [official website](https://exiftool.org/install.html).

## Scripts

### 1. `image_classifier.py`

This script processes a directory of images, using an AI model to understand the content of each image. For each image, it:

1.  Generates a descriptive title, a short description, a list of keywords, and a new filename.
2.  Copies the original image to a `processed` subdirectory with the new, descriptive filename.
3.  Saves the generated metadata (title, description, keywords) into a JSON file with the same name as the new image.

**Usage:**

```bash
python image_classifier.py <path_to_your_images>
```

### 2. `image_tagger.py`

This script recursively scans a directory for images and uses an AI model to generate a list of relevant tags. It then writes these tags directly into the image's metadata.

-   **XMP (`Subject` tag)**: This is the primary method used and is widely supported by photo management software.
-   **EXIF (`UserComment` tag)**: This is used as a fallback for compatible image formats (JPEG, TIFF).

**Usage:**

```bash
# Replace existing tags (default)
python image_tagger.py <path_to_your_images>

# Append new tags to existing ones
python image_tagger.py <path_to_your_images> --append-tags
```

### 3. `tag_deduplicator.py`

Over time, AI-generated tags can become inconsistent (e.g., "Dslr", "DSLR", "Digital SLR"). This script solves that problem by:

1.  Scanning all images in a directory and collecting every unique tag.
2.  Sending the list of unique tags to an AI model to get a "deduplication map" (e.g., `{"Dslr": "DSLR", "Digital SLR": "DSLR"}`).
3.  Applying these corrections to the XMP metadata of all affected images.
4.  Generating a `tag_deduplication_report.json` file with a summary of the changes.

**Usage:**

```bash
# Analyze and apply changes
python tag_deduplicator.py <path_to_your_images>

# See proposed changes without applying them
python tag_deduplicator.py <path_to_your_images> --dry-run
```

## General Options

All scripts support the following command-line arguments:

-   `--api-url`: The URL of your local AI model's API endpoint (default: `http://1227.0.0.1:1234`).
-   `--model`: The name of the model to use.
-   `-v` or `--verbose`: Enable more detailed logging.
