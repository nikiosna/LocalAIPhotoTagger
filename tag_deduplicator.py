#!/usr/bin/env python3
"""
Tag Deduplicator Script

This script collects all tags from images in a directory, uses AI to deduplicate
similar tags, and applies the changes back to all images.
"""

import json
import requests
import argparse
import subprocess
from pathlib import Path
from typing import List, Dict, Set, Optional, Tuple
import logging
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TagDeduplicator:
    """Main class for deduplicating similar tags across images."""

    def __init__(self, api_url: str = "http://127.0.0.1:1234", model: str = "qwen/qwen3-8b", use_chunking: bool = False):
        self.api_url = api_url
        self.model = model
        self.use_chunking = use_chunking  # Split tags alphabetically by starting character
        # Supported image formats (same as image_tagger.py)
        self.supported_formats = {'.jpg', '.jpeg', '.tiff', '.tif', '.png', '.bmp', '.webp'}
        
        # Configuration for AI model behavior
        self.ADD_NOTHINK_TO_PROMPT = True
        
        # AI prompt for tag deduplication
        self.deduplication_prompt = """You are a tag deduplication expert. I will provide you with a list of tags used across multiple images. Your task is to identify similar or duplicate tags and create a mapping to deduplicate them.

Rules:
1. Group similar tags that refer to the same concept (e.g., "Portrait" and "Portraits", "Landscape" and "Landscapes")
2. Prefer more specific and descriptive tags over generic ones
3. Maintain consistent capitalization (Title Case)
4. Keep technical and domain-specific terms precise
5. Only merge tags that are truly similar in meaning
6. Consider variations in spelling, hyphenation, and pluralization
7. Merge synonymous terms (e.g., "Automobile" and "Car", "Building" and "Architecture")

Please respond with a JSON object where each key is an "old tag" and each value is the "new tag" it should be renamed to. Only include tags that need to be changed - don't include tags that should remain unchanged.

Example format:
{
  "old tag 1": "new tag 1",
  "old tag 2": "new tag 1",
  "old tag 3": "new tag 3"
}

Here are the tags to deduplicate:"""

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
            # Prepare the tag arguments for exiftool with proper quoting
            tag_args = []
            for tag in tags:
                tag_args.append(f'-xmp:subject+={tag}')
            
            # First, clear existing XMP subject tags
            clear_cmd = ['exiftool', '-overwrite_original', '-xmp:subject=', str(image_path)]
            subprocess.run(clear_cmd, capture_output=True, text=True, check=True)
            
            # Then add new tags
            if tag_args:
                write_cmd = ['exiftool', '-overwrite_original'] + tag_args + [str(image_path)]
                subprocess.run(write_cmd, capture_output=True, text=True, check=True)
            
            logger.info(f"Successfully updated XMP tags for {image_path}")
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

    def collect_all_tags(self, input_dir: Path) -> Tuple[Dict[str, Set[str]], Set[str]]:
        """
        Collect all tags from all images in the directory.
        Returns: (tag_to_files_mapping, all_unique_tags)
        """
        logger.info(f"Collecting tags from all images in: {input_dir}")
        
        if not input_dir.exists() or not input_dir.is_dir():
            logger.error(f"Invalid input directory: {input_dir}")
            return {}, set()
        
        # Find all image files recursively
        image_files: List[Path] = []
        for ext in self.supported_formats:
            pattern = f"**/*{ext}"
            image_files.extend(input_dir.rglob(pattern))
            pattern_upper = f"**/*{ext.upper()}"
            image_files.extend(input_dir.rglob(pattern_upper))
        
        # Remove duplicates and ensure they are files
        image_files = list(set([p for p in image_files if p.is_file()]))
        
        if not image_files:
            logger.warning("No supported image files found.")
            return {}, set()

        logger.info(f"Found {len(image_files)} images to analyze")
        
        # Collect tags from all images
        tag_to_files = defaultdict(set)
        all_tags = set()
        
        for image_path in image_files:
            try:
                tags = self.get_existing_xmp_tags(image_path)
                if tags:
                    logger.debug(f"Found {len(tags)} tags in {image_path}: {tags}")
                    for tag in tags:
                        tag_to_files[tag].add(str(image_path))
                        all_tags.add(tag)
                else:
                    logger.debug(f"No tags found in {image_path}")
            except Exception as e:
                logger.error(f"Error reading tags from {image_path}: {e}")
        
        logger.info(f"Collected {len(all_tags)} unique tags from {len(image_files)} images")
        return dict(tag_to_files), all_tags

    def chunk_tags_alphabetically(self, tags: Set[str]) -> Dict[str, List[str]]:
        """Split tags into chunks based on their starting character (A, B, C, etc.)."""
        chunks: Dict[str, List[str]] = {}
        
        for tag in tags:
            # Get the first character and convert to uppercase
            first_char = tag[0].upper() if tag else 'OTHER'
            
            # Group by alphabetic character, put non-alphabetic in 'OTHER'
            if first_char.isalpha():
                chunk_key = first_char
            else:
                chunk_key = 'OTHER'
            
            if chunk_key not in chunks:
                chunks[chunk_key] = []
            chunks[chunk_key].append(tag)
        
        # Sort tags within each chunk
        for chunk_key in chunks:
            chunks[chunk_key].sort()
        
        return chunks

    def get_ai_deduplication_mapping(self, tags: Set[str], dry_run: bool = False) -> Optional[Dict[str, str]]:
        """Call the AI model to get tag deduplication mapping."""
        if not tags:
            return {}
        
        # If chunking is enabled, process tags in alphabetical chunks
        if self.use_chunking:
            return self.get_ai_deduplication_mapping_chunked(tags, dry_run)
        else:
            return self.get_ai_deduplication_mapping_single(tags, dry_run)

    def get_ai_deduplication_mapping_single(self, tags: Set[str], dry_run: bool = False) -> Optional[Dict[str, str]]:
        """Process all tags in a single AI request (original behavior)."""
        try:
            tags_list = sorted(list(tags))
            tags_text = "\n".join([f"- {tag}" for tag in tags_list])
            
            full_prompt = f"{self.deduplication_prompt}\n\n{tags_text}"
            
            # Add /no_think directive if enabled
            if self.ADD_NOTHINK_TO_PROMPT:
                full_prompt += "\n\n/no_think"
            
            if dry_run:
                logger.info("AI Prompt being sent:")
                logger.info('"""')
                logger.info(full_prompt)
                logger.info('"""')
            
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": full_prompt
                    }
                ],
                "max_tokens": 2000,
                "temperature": 0.1  # Low temperature for consistent results
            }

            logger.info(f"Sending {len(tags)} tags to AI model for deduplication...")
            response = requests.post(
                f"{self.api_url}/v1/chat/completions",
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=120
            )

            return self.parse_ai_response(response, tags)

        except requests.exceptions.Timeout:
            logger.error("AI API request timed out")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"AI API request error: {e}")
            return None
        except Exception as e:
            logger.error(f"Error calling AI model: {e}")
            return None

    def get_ai_deduplication_mapping_chunked(self, tags: Set[str], dry_run: bool = False) -> Optional[Dict[str, str]]:
        """Process tags in alphabetical chunks for large datasets."""
        logger.info(f"Processing {len(tags)} tags in alphabetical chunks...")
        
        # Split tags into alphabetical chunks
        chunks = self.chunk_tags_alphabetically(tags)
        logger.info(f"Split tags into {len(chunks)} alphabetical chunks: {sorted(chunks.keys())}")
        
        combined_mapping = {}
        
        for chunk_key in sorted(chunks.keys()):
            chunk_tags = chunks[chunk_key]
            logger.info(f"Processing chunk '{chunk_key}' with {len(chunk_tags)} tags...")
            
            try:
                chunk_mapping = self.get_ai_deduplication_mapping_single(set(chunk_tags), dry_run)
                
                if chunk_mapping is None:
                    logger.error(f"Failed to process chunk '{chunk_key}', skipping...")
                    continue
                
                # Merge the chunk mapping into the combined mapping
                combined_mapping.update(chunk_mapping)
                logger.info(f"Chunk '{chunk_key}' suggested {len(chunk_mapping)} tag changes")
                
            except Exception as e:
                logger.error(f"Error processing chunk '{chunk_key}': {e}")
                continue
        
        logger.info(f"Combined chunked processing: {len(combined_mapping)} total tag changes suggested")
        return combined_mapping

    def parse_ai_response(self, response, tags: Set[str]) -> Optional[Dict[str, str]]:
        """Parse the AI response and extract the tag mapping."""
        if response.status_code == 200:
            result = response.json()
            content = result.get('choices', [{}])[0].get('message', {}).get('content', '')
            
            try:
                # Clean the response more thoroughly
                content = content.strip()
                
                # Remove thinking tags if present
                if '<think>' in content:
                    # Extract content between </think> and end, or find JSON block
                    think_end = content.find('</think>')
                    if think_end != -1:
                        content = content[think_end + 8:].strip()
                
                # Remove code blocks
                if content.startswith('```json'):
                    content = content[7:]
                elif content.startswith('```'):
                    content = content[3:]
                if content.endswith('```'):
                    content = content[:-3]
                content = content.strip()
                
                # Try to find JSON object in the content
                json_start = content.find('{')
                json_end = content.rfind('}')
                if json_start != -1 and json_end != -1 and json_end > json_start:
                    content = content[json_start:json_end + 1]
                
                mapping = json.loads(content)
                if isinstance(mapping, dict):
                    return mapping
                else:
                    logger.error("AI response is not a valid dictionary")
                    return None
                    
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse AI response: {e}")
                logger.debug(f"Raw content: {content}")
                return None
        else:
            logger.error(f"AI API request failed: {response.status_code} - {response.text}")
            return None

    def apply_tag_changes(self, input_dir: Path, tag_mapping: Dict[str, str], tag_to_files: Dict[str, Set[str]]) -> Dict[str, int]:
        """Apply tag changes to all affected images."""
        logger.info(f"Applying tag changes to images...")
        
        # Group files by the changes they need
        files_to_update = set()
        for old_tag in tag_mapping.keys():
            if old_tag in tag_to_files:
                files_to_update.update(tag_to_files[old_tag])
        
        logger.info(f"Need to update {len(files_to_update)} files")
        
        updated_successfully = 0
        failed = 0
        
        for file_path_str in files_to_update:
            try:
                file_path = Path(file_path_str)
                if not file_path.exists():
                    logger.warning(f"File no longer exists: {file_path}")
                    failed += 1
                    continue
                
                # Get current tags
                current_tags = self.get_existing_xmp_tags(file_path)
                if not current_tags:
                    logger.debug(f"No tags found in {file_path}, skipping")
                    continue
                
                # Apply mapping to tags
                updated_tags = []
                changes_made = False
                
                for tag in current_tags:
                    if tag in tag_mapping:
                        new_tag = tag_mapping[tag]
                        updated_tags.append(new_tag)
                        changes_made = True
                        logger.debug(f"Changed '{tag}' to '{new_tag}' in {file_path}")
                    else:
                        updated_tags.append(tag)
                
                # Remove duplicates while preserving order
                final_tags = []
                seen = set()
                for tag in updated_tags:
                    if tag not in seen:
                        final_tags.append(tag)
                        seen.add(tag)
                
                if changes_made:
                    # Write updated tags back to the image
                    if self.write_tags_to_xmp(file_path, final_tags):
                        updated_successfully += 1
                        logger.info(f"Updated tags in {file_path}")
                    else:
                        failed += 1
                        logger.error(f"Failed to update tags in {file_path}")
                else:
                    logger.debug(f"No changes needed for {file_path}")
                    
            except Exception as e:
                logger.error(f"Error updating {file_path_str}: {e}")
                failed += 1
        
        return {
            "files_updated": updated_successfully,
            "files_failed": failed,
            "total_files_processed": len(files_to_update)
        }

    def save_deduplication_report(self, output_path: Path, tag_mapping: Dict[str, str], 
                                tag_to_files: Dict[str, Set[str]], results: Dict[str, int]):
        """Save a detailed report of the deduplication process."""
        report = {
            "timestamp": str(Path().cwd()),  # Using current time would be better but this is simpler
            "tag_mapping": tag_mapping,
            "affected_files_count": {tag: len(files) for tag, files in tag_to_files.items() if tag in tag_mapping},
            "results": results,
            "summary": {
                "total_tags_changed": len(tag_mapping),
                "total_files_affected": results.get("total_files_processed", 0),
                "files_updated_successfully": results.get("files_updated", 0),
                "files_failed": results.get("files_failed", 0)
            }
        }
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            logger.info(f"Deduplication report saved to: {output_path}")
        except Exception as e:
            logger.error(f"Failed to save report: {e}")

    def run_deduplication(self, input_dir: Path, dry_run: bool = False) -> bool:
        """Run the complete deduplication process."""
        logger.info("Starting tag deduplication process...")
        
        # Step 1: Collect all tags
        tag_to_files, all_tags = self.collect_all_tags(input_dir)
        
        if not all_tags:
            logger.warning("No tags found in any images. Nothing to deduplicate.")
            return False
        
        logger.info(f"Found {len(all_tags)} unique tags across all images")
        
        # Step 2: Get AI deduplication mapping
        tag_mapping = self.get_ai_deduplication_mapping(all_tags, dry_run)
        
        if not tag_mapping:
            logger.warning("No tag changes suggested by AI or AI call failed.")
            return False
        
        # Filter out no-op changes where old_tag == new_tag
        filtered_mapping = {old_tag: new_tag for old_tag, new_tag in tag_mapping.items() if old_tag != new_tag}
        
        if not filtered_mapping:
            logger.info("No duplicate tags found. All tags are already unique.")
            return True
        
        # Update tag_mapping to use the filtered version
        tag_mapping = filtered_mapping
        
        # Display proposed changes
        logger.info("Proposed tag changes:")
        for old_tag, new_tag in tag_mapping.items():
            affected_files = len(tag_to_files.get(old_tag, set()))
            logger.info(f"  '{old_tag}' -> '{new_tag}' (affects {affected_files} files)")
        
        if dry_run:
            logger.info("Dry run mode - no changes will be applied")
            return True
        
        # Step 3: Apply changes
        results = self.apply_tag_changes(input_dir, tag_mapping, tag_to_files)
        
        # Step 4: Save report
        report_path = input_dir / "tag_deduplication_report.json"
        self.save_deduplication_report(report_path, tag_mapping, tag_to_files, results)
        
        # Summary
        logger.info("=== Deduplication Complete ===")
        logger.info(f"Tags changed: {len(tag_mapping)}")
        logger.info(f"Files updated successfully: {results['files_updated']}")
        logger.info(f"Files failed: {results['files_failed']}")
        logger.info(f"Total files processed: {results['total_files_processed']}")
        
        return results['files_failed'] == 0

def main():
    """Main function to run the tag deduplicator."""
    parser = argparse.ArgumentParser(description="Deduplicate similar tags across images using AI.")
    parser.add_argument("input_dir", help="Input directory containing images with tags.")
    parser.add_argument("--api-url", default="http://127.0.0.1:1234", help="AI model API URL.")
    parser.add_argument("--model", default="qwen/qwen3-8b", help="AI model name for deduplication.")
    parser.add_argument("--dry-run", action="store_true", help="Show proposed changes without applying them.")
    parser.add_argument("--chunk", action="store_true", help="Split tags alphabetically into chunks for large datasets (A, B, C, etc.).")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging.")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        
    deduplicator = TagDeduplicator(api_url=args.api_url, model=args.model, use_chunking=args.chunk)
    input_path = Path(args.input_dir)
    
    if not input_path.is_dir():
        logger.error(f"Input path is not a valid directory: {input_path}")
        return 1
        
    try:
        success = deduplicator.run_deduplication(input_path, dry_run=args.dry_run)
        return 0 if success else 1
            
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user.")
        return 1
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        return 1

if __name__ == "__main__":
    exit(main())