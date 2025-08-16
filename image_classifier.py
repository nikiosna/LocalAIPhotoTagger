#!/usr/bin/env python3
"""
Image Classifier Script

This script processes images in a directory, calls a local AI model to generate
descriptions and new filenames, and outputs structured JSON data.
"""

import os
import json
import base64
import requests  # type: ignore
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImageClassifier:
    """Main class for processing images with AI model."""
    
    def __init__(self, api_url: str = "http://127.0.0.1:1234", model: str = "google/gemma-3n-e4b"):
        self.api_url = api_url
        self.model = model
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
        
        # AI prompt template
        self.prompt = """Generate a JSON object describing an image using the following fields:

title: A short title of the picture
filename: The image filename (e.g., "sunset.jpg")
description: A brief description of what's in the image
keywords: A list of relevant tags

The result must be a valid JSON object matching this structure."""

    def is_image_file(self, file_path: Path) -> bool:
        """Check if file is a supported image format."""
        return file_path.suffix.lower() in self.supported_formats

    def encode_image_to_base64(self, image_path: Path) -> str:
        """Encode image file to base64 string."""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            logger.error(f"Error encoding image {image_path}: {e}")
            raise

    def call_ai_model(self, image_path: Path) -> Optional[Dict]:
        """Call the local AI model with the image and prompt."""
        try:
            # Encode image to base64
            image_base64 = self.encode_image_to_base64(image_path)
            
            # Prepare the request payload
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": self.prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 500,
                "temperature": 0.7
            }
            
            # Make API request
            response = requests.post(
                f"{self.api_url}/v1/chat/completions",
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result.get('choices', [{}])[0].get('message', {}).get('content', '')
                
                # Try to parse JSON from the response
                try:
                    # Clean the response - sometimes AI models wrap JSON in markdown
                    content = content.strip()
                    if content.startswith('```json'):
                        content = content[7:]
                    if content.endswith('```'):
                        content = content[:-3]
                    content = content.strip()
                    
                    return json.loads(content)
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON response for {image_path}: {e}")
                    logger.error(f"Raw content: {content}")
                    return None
            else:
                logger.error(f"API request failed for {image_path}: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Error calling AI model for {image_path}: {e}")
            return None

    def process_image(self, image_path: Path, output_dir: Path) -> bool:
        """Process a single image and create renamed file + JSON."""
        logger.info(f"Processing image: {image_path}")
        
        # Call AI model
        ai_result = self.call_ai_model(image_path)
        if not ai_result:
            return False
        
        # Validate required fields
        required_fields = ['title', 'filename', 'description', 'keywords']
        if not all(field in ai_result for field in required_fields):
            logger.error(f"AI response missing required fields for {image_path}")
            return False
        
        try:
            # Get the AI-suggested filename
            suggested_filename = ai_result.get('filename', '')
            if not suggested_filename:
                logger.error(f"No filename suggested for {image_path}")
                return False
            
            # Ensure the filename has the same extension as the original
            original_ext = image_path.suffix
            suggested_path = Path(suggested_filename)
            
            # If suggested filename doesn't have extension, add original extension
            if not suggested_path.suffix:
                suggested_filename = f"{suggested_filename}{original_ext}"
            elif suggested_path.suffix.lower() != original_ext.lower():
                # Replace extension with original extension
                suggested_filename = f"{suggested_path.stem}{original_ext}"
            
            # Create new path in output directory
            new_image_path = output_dir / suggested_filename
            
            # Ensure filename is unique
            counter = 1
            base_stem = Path(suggested_filename).stem
            while new_image_path.exists():
                new_image_path = output_dir / f"{base_stem}_{counter}{original_ext}"
                counter += 1
            
            # Copy image file to new location with new name
            import shutil
            shutil.copy2(image_path, new_image_path)
            logger.info(f"Created: {new_image_path.name}")
            
            # Create JSON file with same base name
            json_filename = new_image_path.stem + ".json"
            json_path = output_dir / json_filename
            
            # Save AI result as JSON
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(ai_result, f, indent=2, ensure_ascii=False)
            logger.info(f"Created: {json_filename}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing file {image_path}: {e}")
            return False

    def scan_directory(self, input_dir: Path) -> List[Path]:
        """Scan directory for image files."""
        image_files: List[Path] = []
        
        if not input_dir.exists():
            logger.error(f"Input directory does not exist: {input_dir}")
            return image_files
        
        if not input_dir.is_dir():
            logger.error(f"Input path is not a directory: {input_dir}")
            return image_files
        
        for file_path in input_dir.iterdir():
            if file_path.is_file() and self.is_image_file(file_path):
                image_files.append(file_path)
        
        logger.info(f"Found {len(image_files)} image files in {input_dir}")
        return image_files

    def process_directory(self, input_dir: Path, output_dir: Optional[Path] = None) -> Dict[str, int]:
        """Process all images in a directory."""
        if output_dir is None:
            output_dir = input_dir / "processed"
        
        # Create output directory if it doesn't exist
        output_dir.mkdir(exist_ok=True)
        
        # Scan for image files
        image_files = self.scan_directory(input_dir)
        
        if not image_files:
            logger.warning("No image files found to process")
            return {"total_files": 0, "processed_successfully": 0, "failed": 0}
        
        processed_successfully = 0
        failed = 0
        
        # Process each image
        for image_path in image_files:
            try:
                success = self.process_image(image_path, output_dir)
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
    """Main function to run the image classifier."""
    parser = argparse.ArgumentParser(description="Classify images using local AI model")
    parser.add_argument("input_dir", help="Input directory containing images")
    parser.add_argument("-o", "--output", help="Output directory (default: input_dir/processed)")
    parser.add_argument("--api-url", default="http://127.0.0.1:1234", help="AI model API URL")
    parser.add_argument("--model", default="google/gemma-3n-e4b", help="AI model name")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize classifier
    classifier = ImageClassifier(api_url=args.api_url, model=args.model)
    
    # Process directory
    input_path = Path(args.input_dir)
    output_path = Path(args.output) if args.output else None
    
    try:
        results = classifier.process_directory(input_path, output_path)
        
        # Print summary
        print(f"\n=== Processing Complete ===")
        print(f"Total files: {results['total_files']}")
        print(f"Successfully processed: {results['processed_successfully']}")
        print(f"Failed: {results['failed']}")
        
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())