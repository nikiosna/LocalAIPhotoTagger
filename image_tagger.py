#!/usr/bin/env python3
"""
Image Tagger Script

This script processes images in a directory, calls a local AI model to generate
tags, and stores these tags in both EXIF metadata and JSON backup files.
"""

import json
import base64
import requests  # type: ignore
import argparse
import subprocess
from pathlib import Path
from typing import List, Optional, Dict, Tuple
import logging
import piexif

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImageTagger:
    """Main class for tagging images with AI-generated keywords."""

    def __init__(self, api_url: str = "http://127.0.0.1:1234", model: str = "google/gemma-3n-e4b", append_tags: bool = False):
        self.api_url = api_url
        self.model = model
        self.append_tags = append_tags
        # EXIF-supported formats
        self.exif_formats = {'.jpg', '.jpeg', '.tiff', '.tif'}
        # All supported image formats
        self.supported_formats = {'.jpg', '.jpeg', '.tiff', '.tif', '.png', '.bmp', '.webp'}
        
        # EXIF UserComment character code for UTF-8
        self.utf8_code = b'UNICODE\x00'

        # AI prompt template for generating tags
        self.prompt = """Generate a concise JSON object containing a list of relevant tags (keywords) for the following image. The JSON object should have a single key "tags" with a list of strings.

Example:
{
  "tags": ["tag1", "tag2", "tag3"]
}

The result must be a valid JSON object. Focus on descriptive, specific keywords that would be useful for searching and categorizing the image."""

    def normalize_tags(self, tags: List[str]) -> List[str]:
        """Normalize tag capitalization - capitalize first character of each word."""
        normalized_tags = []
        for tag in tags:
            tag = tag.strip()
            if tag:
                # Split on whitespace and capitalize each word
                words = tag.split()
                capitalized_words = [word.capitalize() for word in words]
                normalized_tag = ' '.join(capitalized_words)
                normalized_tags.append(normalized_tag)
        
        return normalized_tags

    def get_existing_xmp_tags(self, image_path: Path) -> List[str]:
        """Read existing XMP Subject tags from an image using exiftool."""
        try:
            result = subprocess.run(['exiftool', '-j', '-xmp:subject', str(image_path)],
                                  capture_output=True, text=True, check=True)
            data = json.loads(result.stdout)
            if data and len(data) > 0:
                subject = data[0].get('Subject', [])
                if isinstance(subject, list):
                    return subject
                elif isinstance(subject, str):
                    # Fallback for comma-separated strings
                    return [tag.strip() for tag in subject.split(',') if tag.strip()]
            return []
        except Exception as e:
            logger.debug(f"Could not read XMP tags from {image_path}: {e}")
            return []

    def write_tags_to_xmp(self, image_path: Path, tags: List[str]) -> bool:
        """Write tags to the image's XMP Subject field using exiftool."""
        if not tags:
            return False

        try:
            if self.append_tags:
                # Get existing tags and merge with new ones
                existing_tags = self.get_existing_xmp_tags(image_path)
                # Combine and deduplicate tags
                all_tags = list(dict.fromkeys(existing_tags + tags))  # Preserves order and removes duplicates
                logger.info(f"Appending {len(tags)} new tags to {len(existing_tags)} existing tags")
            else:
                # Replace existing tags
                all_tags = tags
                logger.info(f"Replacing existing tags with {len(tags)} new tags")

            # Prepare the tag arguments for exiftool with proper quoting
            tag_args = []
            for tag in all_tags:
                tag_args.append(f'-xmp:subject+={tag}')
            
            # First, clear existing XMP subject tags
            clear_cmd = ['exiftool', '-overwrite_original', '-xmp:subject=', str(image_path)]
            subprocess.run(clear_cmd, capture_output=True, text=True, check=True)
            
            # Then add new tags
            if tag_args:
                write_cmd = ['exiftool', '-overwrite_original'] + tag_args + [str(image_path)]
                subprocess.run(write_cmd, capture_output=True, text=True, check=True)
            
            logger.info(f"Successfully wrote {len(all_tags)} XMP tags to {image_path}")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to write XMP tags to {image_path}: {e}")
            return False
        except Exception as e:
            logger.error(f"Error writing XMP tags to {image_path}: {e}")
            return False

    def is_supported_image(self, file_path: Path) -> bool:
        """Check if the file is a supported image format."""
        return file_path.suffix.lower() in self.supported_formats

    def supports_exif(self, file_path: Path) -> bool:
        """Check if the file format supports EXIF metadata."""
        return file_path.suffix.lower() in self.exif_formats

    def encode_image_to_base64(self, image_path: Path) -> str:
        """Encode image file to a base64 string."""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            logger.error(f"Error encoding image {image_path}: {e}")
            raise

    def get_ai_tags(self, image_path: Path) -> Optional[List[str]]:
        """Call the local AI model to get tags for an image."""
        try:
            image_base64 = self.encode_image_to_base64(image_path)
            
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": self.prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
                            }
                        ]
                    }
                ],
                "max_tokens": 300,
                "temperature": 0.3  # Lower temperature for more consistent results
            }

            response = requests.post(
                f"{self.api_url}/v1/chat/completions",
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=60  # Increased timeout
            )

            if response.status_code == 200:
                result = response.json()
                content = result.get('choices', [{}])[0].get('message', {}).get('content', '')
                
                try:
                    # Clean the response more thoroughly
                    content = content.strip()
                    if content.startswith('```json'):
                        content = content[7:]
                    elif content.startswith('```'):
                        content = content[3:]
                    if content.endswith('```'):
                        content = content[:-3]
                    content = content.strip()
                    
                    tags_data = json.loads(content)
                    if isinstance(tags_data, dict) and 'tags' in tags_data and isinstance(tags_data['tags'], list):
                        # Filter and clean tags
                        clean_tags = []
                        for tag in tags_data['tags']:
                            if isinstance(tag, str) and tag.strip():
                                clean_tags.append(tag.strip())
                        return clean_tags if clean_tags else None
                    else:
                        logger.error(f"Invalid JSON structure for tags in response for {image_path}")
                        return None
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON response for {image_path}: {e}")
                    logger.debug(f"Raw content: {content}")
                    return None
            else:
                logger.error(f"API request failed for {image_path}: {response.status_code} - {response.text}")
                return None

        except requests.exceptions.Timeout:
            logger.error(f"API request timed out for {image_path}")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"API request error for {image_path}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error calling AI model for {image_path}: {e}")
            return None

    def write_tags_to_exif(self, image_path: Path, tags: List[str]) -> bool:
        """Write tags to the image's EXIF UserComment field with proper encoding."""
        if not tags or not self.supports_exif(image_path):
            return False

        try:
            # Join tags into a comma-separated string
            tag_string = ", ".join(tags)
            logger.debug(f"Writing tags to EXIF: {tag_string}")

            # Load existing EXIF data
            try:
                exif_dict = piexif.load(str(image_path))
            except Exception as e:
                logger.warning(f"Could not load existing EXIF data for {image_path}. Creating new EXIF data. Error: {e}")
                exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}

            # Encode the tag string properly for EXIF UserComment
            # EXIF UserComment should start with 8-byte character code
            user_comment = self.utf8_code + tag_string.encode('utf-8')
            exif_dict['Exif'][piexif.ExifIFD.UserComment] = user_comment

            # Get the EXIF bytes
            exif_bytes = piexif.dump(exif_dict)

            # Write the new EXIF data to the image
            piexif.insert(exif_bytes, str(image_path))
            logger.info(f"Successfully wrote {len(tags)} tags to EXIF for {image_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to write EXIF data to {image_path}: {e}")
            return False

    def verify_tags(self, image_path: Path, expected_tags: List[str]) -> bool:
        """Verify that tags were written correctly to both EXIF and XMP."""
        success = True
        
        # Check EXIF tags if supported
        if self.supports_exif(image_path):
            try:
                exif_dict = piexif.load(str(image_path))
                user_comment = exif_dict.get('Exif', {}).get(piexif.ExifIFD.UserComment, b'')
                
                if user_comment.startswith(self.utf8_code):
                    stored_tags_str = user_comment[len(self.utf8_code):].decode('utf-8')
                    stored_tags = [tag.strip() for tag in stored_tags_str.split(',')]
                    
                    # Check if all expected tags are present
                    expected_set = set(expected_tags)
                    stored_set = set(stored_tags)
                    
                    if expected_set != stored_set:
                        logger.warning(f"EXIF tag verification failed for {image_path}. Expected: {expected_tags}, Found: {stored_tags}")
                        success = False
                    else:
                        logger.debug(f"EXIF tag verification successful for {image_path}")
                else:
                    logger.warning(f"No valid EXIF tags found for {image_path}")
                    success = False
                    
            except Exception as e:
                logger.error(f"Error verifying EXIF tags for {image_path}: {e}")
                success = False
        
        # Check XMP tags
        try:
            xmp_tags = self.get_existing_xmp_tags(image_path)
            expected_set = set(expected_tags)
            xmp_set = set(xmp_tags)
            
            if expected_set != xmp_set:
                logger.warning(f"XMP tag verification failed for {image_path}. Expected: {expected_tags}, Found: {xmp_tags}")
                success = False
            else:
                logger.debug(f"XMP tag verification successful for {image_path}")
                
        except Exception as e:
            logger.error(f"Error verifying XMP tags for {image_path}: {e}")
            success = False
            
        return success

    def process_image(self, image_path: Path) -> Tuple[bool, Optional[List[str]]]:
        """Process a single image: get tags, write to EXIF and XMP."""
        logger.info(f"Processing image: {image_path}")
        
        # Get tags from AI
        raw_tags = self.get_ai_tags(image_path)
        if not raw_tags:
            logger.warning(f"Could not generate tags for {image_path}")
            return False, None
        
        # Normalize tag capitalization
        tags = self.normalize_tags(raw_tags)
        
        logger.info(f"Generated {len(tags)} tags for {image_path}: {', '.join(tags)}")
        
        success = True
        
        # Write to XMP (primary method for tags)
        xmp_success = self.write_tags_to_xmp(image_path, tags)
        if not xmp_success:
            logger.warning(f"Failed to write XMP tags for {image_path}")
            success = False
        
        # Write to EXIF if supported (backup method)
        if self.supports_exif(image_path):
            exif_success = self.write_tags_to_exif(image_path, tags)
            if not exif_success:
                logger.warning(f"Failed to write EXIF tags for {image_path}")
                # Don't fail completely if XMP succeeded
                if not xmp_success:
                    success = False
        else:
            logger.info(f"EXIF not supported for {image_path}, skipping EXIF write")
        
        # Verify tags were written correctly
        if success and not self.verify_tags(image_path, tags):
            logger.warning(f"Tag verification failed for {image_path}")
            success = False
        
        return success, tags

    def process_directory(self, input_dir: Path) -> Dict[str, int]:
        """Process all supported images in a given directory recursively."""
        logger.info(f"Scanning directory recursively: {input_dir}")
        
        if not input_dir.exists() or not input_dir.is_dir():
            logger.error(f"Invalid input directory: {input_dir}")
            return {"total_files": 0, "processed_successfully": 0, "failed": 0}
        
        # Use rglob to recursively find all image files
        image_files: List[Path] = []
        for ext in self.supported_formats:
            # Search for files with each supported extension recursively
            pattern = f"**/*{ext}"
            image_files.extend(input_dir.rglob(pattern))
            # Also search for uppercase extensions
            pattern_upper = f"**/*{ext.upper()}"
            image_files.extend(input_dir.rglob(pattern_upper))
        
        # Remove duplicates and ensure they are files
        image_files = list(set([p for p in image_files if p.is_file()]))
        
        if not image_files:
            logger.warning("No supported image files found to process recursively.")
            return {"total_files": 0, "processed_successfully": 0, "failed": 0}

        logger.info(f"Found {len(image_files)} supported images across all subdirectories.")
        
        # Log directory structure being processed
        directories_found = set()
        for img_file in image_files:
            directories_found.add(img_file.parent)
        
        logger.info(f"Processing images in {len(directories_found)} directories:")
        for directory in sorted(directories_found):
            rel_path = directory.relative_to(input_dir) if directory != input_dir else Path(".")
            img_count = len([f for f in image_files if f.parent == directory])
            logger.info(f"  {rel_path}: {img_count} images")
        
        processed_successfully = 0
        failed = 0
        
        for image_path in image_files:
            try:
                success, tags = self.process_image(image_path)
                if success:
                    processed_successfully += 1
                else:
                    failed += 1
            except Exception as e:
                logger.error(f"Unexpected error processing {image_path}: {e}")
                failed += 1
        
        return {
            "total_files": len(image_files),
            "processed_successfully": processed_successfully,
            "failed": failed
        }

def main():
    """Main function to run the image tagger."""
    parser = argparse.ArgumentParser(description="Tag images with AI-generated keywords using XMP and EXIF metadata.")
    parser.add_argument("input_dir", help="Input directory containing images to tag.")
    parser.add_argument("--api-url", default="http://127.0.0.1:1234", help="AI model API URL.")
    parser.add_argument("--model", default="google/gemma-3n-e4b", help="AI model name.")
    parser.add_argument("--append-tags", action="store_true", help="Append new tags to existing ones instead of replacing them.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging.")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        
    tagger = ImageTagger(api_url=args.api_url, model=args.model, append_tags=args.append_tags)
    input_path = Path(args.input_dir)
    
    if not input_path.is_dir():
        logger.error(f"Input path is not a valid directory: {input_path}")
        return 1
        
    try:
        results = tagger.process_directory(input_path)
        
        print("\n=== Tagging Complete ===")
        print(f"Total files: {results['total_files']}")
        print(f"Successfully processed: {results['processed_successfully']}")
        print(f"Failed: {results['failed']}")
        
        if results['failed'] > 0:
            print(f"\nSome files failed to process. Check the logs for details.")
            
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user.")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        return 1
        
    return 0

if __name__ == "__main__":
    exit(main())