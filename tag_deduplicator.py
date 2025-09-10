#!/usr/bin/env python3
"""
Embedding-Based Tag Deduplicator Script

This script uses text embeddings to identify semantically similar tags across images
and automatically deduplicate them based on configurable similarity thresholds.
"""

import json
import requests
import argparse
import subprocess
import numpy as np
from pathlib import Path
from typing import List, Dict, Set, Optional, Tuple, Any
import logging
from collections import defaultdict, Counter
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import itertools

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EmbeddingTagDeduplicator:
    """Advanced tag deduplicator using text embeddings for semantic similarity."""

    def __init__(self, 
                 embedding_api_url: str = "http://localhost:1234",
                 embedding_model: str = "text-embedding-nomic-embed-text-v1.5",
                 similarity_threshold: float = 0.2,
                 min_cluster_size: int = 2,
                 batch_size: int = 100):
        """
        Initialize the embedding-based tag deduplicator.
        
        Args:
            embedding_api_url: URL for the embedding API
            embedding_model: Name of the embedding model to use
            similarity_threshold: Cosine similarity threshold for grouping tags (0.0-1.0)
            min_cluster_size: Minimum number of tags required to form a cluster
            batch_size: Number of tags to process in each embedding batch
        """
        self.embedding_api_url = embedding_api_url
        self.embedding_model = embedding_model
        self.similarity_threshold = similarity_threshold
        self.min_cluster_size = min_cluster_size
        self.batch_size = batch_size
        
        # Supported image formats
        self.supported_formats = {'.jpg', '.jpeg', '.tiff', '.tif', '.png', '.bmp', '.webp'}
        
        # Cache for embeddings to avoid recomputation
        self.embedding_cache: Dict[str, np.ndarray] = {}
        
        logger.info(f"Initialized EmbeddingTagDeduplicator with similarity threshold: {similarity_threshold}")

    def get_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """
        Get embeddings for a list of texts, using cache when possible.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors as numpy arrays
        """
        # Check cache first
        cached_embeddings = []
        uncached_texts = []
        uncached_indices = []
        
        for i, text in enumerate(texts):
            if text in self.embedding_cache:
                cached_embeddings.append((i, self.embedding_cache[text]))
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # Get embeddings for uncached texts
        new_embeddings = []
        if uncached_texts:
            try:
                logger.debug(f"Getting embeddings for {len(uncached_texts)} new texts")
                url = f"{self.embedding_api_url}/v1/embeddings"
                response = requests.post(
                    url,
                    json={
                        "model": self.embedding_model,
                        "input": uncached_texts
                    },
                    timeout=60
                )
                response.raise_for_status()
                
                embedding_data = response.json()["data"]
                for i, data in enumerate(embedding_data):
                    embedding = np.array(data["embedding"])
                    
                    # Validate embedding
                    if len(embedding) == 0:
                        logger.error(f"Received empty embedding for text: '{uncached_texts[i]}'")
                        raise ValueError(f"Empty embedding received for text: '{uncached_texts[i]}'")
                    
                    if np.any(np.isnan(embedding)) or np.any(np.isinf(embedding)):
                        logger.error(f"Received invalid embedding (NaN/inf) for text: '{uncached_texts[i]}'")
                        raise ValueError(f"Invalid embedding (NaN/inf) received for text: '{uncached_texts[i]}'")
                    
                    text = uncached_texts[i]
                    # Cache the embedding
                    self.embedding_cache[text] = embedding
                    new_embeddings.append((uncached_indices[i], embedding))
                    
            except Exception as e:
                logger.error(f"Failed to get embeddings: {e}")
                raise
        
        # Combine cached and new embeddings in correct order
        all_embeddings = cached_embeddings + new_embeddings
        all_embeddings.sort(key=lambda x: x[0])  # Sort by original index
        
        return [embedding for _, embedding in all_embeddings]

    def calculate_similarity_matrix(self, embeddings: List[np.ndarray]) -> np.ndarray:
        """
        Calculate cosine similarity matrix for embeddings.
        
        Args:
            embeddings: List of embedding vectors
            
        Returns:
            Similarity matrix as numpy array
        """
        if not embeddings:
            return np.array([])
        
        # Stack embeddings into matrix
        embedding_matrix = np.vstack(embeddings)
        
        # Check for invalid values in embeddings
        if np.any(np.isnan(embedding_matrix)) or np.any(np.isinf(embedding_matrix)):
            logger.error("Found NaN or infinite values in embeddings")
            raise ValueError("Invalid values (NaN or inf) found in embeddings")
        
        # Normalize embeddings to ensure valid cosine similarity calculation
        # This prevents numerical issues with very small or zero-norm vectors
        norms = np.linalg.norm(embedding_matrix, axis=1, keepdims=True)
        zero_norm_mask = norms.flatten() == 0
        
        if np.any(zero_norm_mask):
            logger.warning(f"Found {np.sum(zero_norm_mask)} zero-norm embeddings, replacing with small random values")
            # Replace zero-norm vectors with small random vectors
            for i in np.where(zero_norm_mask)[0]:
                embedding_matrix[i] = np.random.normal(0, 1e-8, embedding_matrix.shape[1])
            # Recalculate norms
            norms = np.linalg.norm(embedding_matrix, axis=1, keepdims=True)
        
        # Normalize embeddings
        normalized_embeddings = embedding_matrix / norms
        
        # Calculate cosine similarity
        similarity_matrix = cosine_similarity(normalized_embeddings)
        
        # Ensure similarity values are in valid range [-1, 1]
        similarity_matrix = np.clip(similarity_matrix, -1.0, 1.0)
        
        # Check for invalid values in similarity matrix
        if np.any(np.isnan(similarity_matrix)) or np.any(np.isinf(similarity_matrix)):
            logger.error("Found NaN or infinite values in similarity matrix")
            raise ValueError("Invalid values (NaN or inf) found in similarity matrix")
        
        return similarity_matrix

    def find_similar_tag_clusters(self, tags: List[str], embeddings: List[np.ndarray]) -> List[List[int]]:
        """
        Find clusters of similar tags using DBSCAN clustering.
        
        Args:
            tags: List of tag strings
            embeddings: List of corresponding embedding vectors
            
        Returns:
            List of clusters, where each cluster is a list of tag indices
        """
        if len(tags) < self.min_cluster_size:
            return []
        
        # Calculate similarity matrix
        similarity_matrix = self.calculate_similarity_matrix(embeddings)
        
        # Convert similarity to distance (1 - similarity)
        distance_matrix = 1 - similarity_matrix
        
        # Validate distance matrix
        if np.any(distance_matrix < 0):
            logger.error("Distance matrix contains negative values")
            logger.debug(f"Min distance: {np.min(distance_matrix)}, Max distance: {np.max(distance_matrix)}")
            logger.debug(f"Similarity matrix range: [{np.min(similarity_matrix)}, {np.max(similarity_matrix)}]")
            raise ValueError("Distance matrix contains negative values")
        
        if np.any(np.isnan(distance_matrix)) or np.any(np.isinf(distance_matrix)):
            logger.error("Distance matrix contains NaN or infinite values")
            raise ValueError("Distance matrix contains invalid values (NaN or inf)")
        
        # Use DBSCAN for clustering
        # eps is the maximum distance between samples in the same cluster
        eps = 1 - self.similarity_threshold
        clustering = DBSCAN(eps=eps, min_samples=self.min_cluster_size, metric='precomputed')
        cluster_labels = clustering.fit_predict(distance_matrix)
        
        # Group indices by cluster label
        clusters = defaultdict(list)
        for idx, label in enumerate(cluster_labels):
            if label != -1:  # -1 indicates noise/outlier
                clusters[label].append(idx)
        
        # Convert to list of clusters
        cluster_list = [cluster for cluster in clusters.values() if len(cluster) >= self.min_cluster_size]
        
        logger.info(f"Found {len(cluster_list)} clusters from {len(tags)} tags")
        for i, cluster in enumerate(cluster_list):
            cluster_tags = [tags[idx] for idx in cluster]
            logger.debug(f"Cluster {i}: {cluster_tags}")
        
        return cluster_list

    def select_best_representative_tag(self, tag_indices: List[int], tags: List[str], 
                                     tag_frequencies: Dict[str, int]) -> str:
        """
        Select the best representative tag from a cluster.
        
        Selection criteria (in order of priority):
        1. Most frequent tag (appears in most images)
        2. Shortest tag (more concise)
        3. Alphabetically first (consistent ordering)
        
        Args:
            tag_indices: Indices of tags in the cluster
            tags: List of all tags
            tag_frequencies: Dictionary mapping tags to their frequency counts
            
        Returns:
            The selected representative tag
        """
        cluster_tags = [tags[idx] for idx in tag_indices]
        
        # Sort by frequency (descending), then by length (ascending), then alphabetically
        def sort_key(tag):
            frequency = tag_frequencies.get(tag, 0)
            return (-frequency, len(tag), tag.lower())
        
        sorted_tags = sorted(cluster_tags, key=sort_key)
        selected_tag = sorted_tags[0]
        
        logger.debug(f"Selected '{selected_tag}' as representative from: {cluster_tags}")
        return selected_tag

    def create_tag_mapping(self, tags: List[str], tag_frequencies: Dict[str, int]) -> Dict[str, str]:
        """
        Create a mapping from old tags to new tags based on embedding similarity.
        
        Args:
            tags: List of unique tags
            tag_frequencies: Dictionary mapping tags to their frequency counts
            
        Returns:
            Dictionary mapping old tags to new representative tags
        """
        if not tags:
            return {}
        
        logger.info(f"Creating tag mapping for {len(tags)} unique tags")
        
        # Process tags in batches to avoid memory issues
        all_embeddings = []
        for i in range(0, len(tags), self.batch_size):
            batch_tags = tags[i:i + self.batch_size]
            batch_embeddings = self.get_embeddings(batch_tags)
            all_embeddings.extend(batch_embeddings)
        
        # Find clusters of similar tags
        clusters = self.find_similar_tag_clusters(tags, all_embeddings)
        
        # Create mapping from clusters
        tag_mapping = {}
        for cluster in clusters:
            representative_tag = self.select_best_representative_tag(cluster, tags, tag_frequencies)
            
            for tag_idx in cluster:
                old_tag = tags[tag_idx]
                if old_tag != representative_tag:
                    tag_mapping[old_tag] = representative_tag
        
        logger.info(f"Created mapping for {len(tag_mapping)} tags into {len(clusters)} clusters")
        return tag_mapping

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
            
            logger.debug(f"Successfully updated XMP tags for {image_path}")
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

    def collect_all_tags(self, input_dir: Path) -> Tuple[Dict[str, Set[str]], Dict[str, int]]:
        """
        Collect all tags from all images in the directory.
        
        Returns:
            Tuple of (tag_to_files_mapping, tag_frequencies)
        """
        logger.info(f"Collecting tags from all images in: {input_dir}")
        
        if not input_dir.exists() or not input_dir.is_dir():
            logger.error(f"Invalid input directory: {input_dir}")
            return {}, {}
        
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
            return {}, {}

        logger.info(f"Found {len(image_files)} images to analyze")
        
        # Collect tags from all images
        tag_to_files = defaultdict(set)
        tag_counter: Counter[str] = Counter()
        
        for image_path in image_files:
            try:
                tags = self.get_existing_xmp_tags(image_path)
                if tags:
                    logger.debug(f"Found {len(tags)} tags in {image_path}: {tags}")
                    for tag in tags:
                        tag_to_files[tag].add(str(image_path))
                        tag_counter[tag] += 1
                else:
                    logger.debug(f"No tags found in {image_path}")
            except Exception as e:
                logger.error(f"Error reading tags from {image_path}: {e}")
        
        tag_frequencies = dict(tag_counter)
        logger.info(f"Collected {len(tag_frequencies)} unique tags from {len(image_files)} images")
        
        return dict(tag_to_files), tag_frequencies

    def apply_tag_changes(self, input_dir: Path, tag_mapping: Dict[str, str], 
                         tag_to_files: Dict[str, Set[str]]) -> Dict[str, int]:
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
                                tag_to_files: Dict[str, Set[str]], tag_frequencies: Dict[str, int],
                                results: Dict[str, int]):
        """Save a detailed report of the deduplication process."""
        # Create detailed cluster information
        clusters_info: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for old_tag, new_tag in tag_mapping.items():
            clusters_info[new_tag].append({
                "old_tag": old_tag,
                "frequency": tag_frequencies.get(old_tag, 0),
                "files_affected": len(tag_to_files.get(old_tag, set()))
            })
        
        report = {
            "configuration": {
                "embedding_model": self.embedding_model,
                "similarity_threshold": self.similarity_threshold,
                "min_cluster_size": self.min_cluster_size,
                "batch_size": self.batch_size
            },
            "tag_mapping": tag_mapping,
            "clusters": {
                representative_tag: {
                    "representative_tag": representative_tag,
                    "merged_tags": cluster_tags,
                    "total_frequency": sum(tag_info["frequency"] for tag_info in cluster_tags),
                    "total_files_affected": sum(tag_info["files_affected"] for tag_info in cluster_tags)
                }
                for representative_tag, cluster_tags in clusters_info.items()
            },
            "results": results,
            "summary": {
                "total_unique_tags_before": len(tag_frequencies),
                "total_tags_merged": len(tag_mapping),
                "total_clusters_created": len(clusters_info),
                "total_files_affected": results.get("total_files_processed", 0),
                "files_updated_successfully": results.get("files_updated", 0),
                "files_failed": results.get("files_failed", 0),
                "embedding_cache_size": len(self.embedding_cache)
            }
        }
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False, default=str)
            logger.info(f"Deduplication report saved to: {output_path}")
        except Exception as e:
            logger.error(f"Failed to save report: {e}")

    def run_deduplication(self, input_dir: Path, dry_run: bool = False) -> bool:
        """Run the complete embedding-based deduplication process."""
        logger.info("Starting embedding-based tag deduplication process...")
        
        # Step 1: Collect all tags
        tag_to_files, tag_frequencies = self.collect_all_tags(input_dir)
        
        if not tag_frequencies:
            logger.warning("No tags found in any images. Nothing to deduplicate.")
            return False
        
        logger.info(f"Found {len(tag_frequencies)} unique tags across all images")
        
        # Step 2: Create embedding-based tag mapping
        unique_tags = list(tag_frequencies.keys())
        tag_mapping = self.create_tag_mapping(unique_tags, tag_frequencies)
        
        if not tag_mapping:
            logger.info("No similar tags found. All tags are already unique.")
            return True
        
        # Display proposed changes
        logger.info("Proposed tag changes based on embedding similarity:")
        for old_tag, new_tag in sorted(tag_mapping.items()):
            affected_files = len(tag_to_files.get(old_tag, set()))
            old_freq = tag_frequencies.get(old_tag, 0)
            new_freq = tag_frequencies.get(new_tag, 0)
            logger.info(f"  '{old_tag}' (freq: {old_freq}) -> '{new_tag}' (freq: {new_freq}) "
                       f"(affects {affected_files} files)")
        
        if dry_run:
            logger.info("Dry run mode - no changes will be applied")
            # Still save a report for dry run
            report_path = input_dir / "tag_deduplication_report_dryrun.json"
            dummy_results = {"files_updated": 0, "files_failed": 0, "total_files_processed": 0}
            self.save_deduplication_report(report_path, tag_mapping, tag_to_files, 
                                         tag_frequencies, dummy_results)
            return True
        
        # Step 3: Apply changes
        results = self.apply_tag_changes(input_dir, tag_mapping, tag_to_files)
        
        # Step 4: Save report
        report_path = input_dir / "tag_deduplication_report_embedding.json"
        self.save_deduplication_report(report_path, tag_mapping, tag_to_files, 
                                     tag_frequencies, results)
        
        # Summary
        logger.info("=== Embedding-Based Deduplication Complete ===")
        logger.info(f"Tags merged: {len(tag_mapping)}")
        logger.info(f"Clusters created: {len(set(tag_mapping.values()))}")
        logger.info(f"Files updated successfully: {results['files_updated']}")
        logger.info(f"Files failed: {results['files_failed']}")
        logger.info(f"Total files processed: {results['total_files_processed']}")
        logger.info(f"Embeddings cached: {len(self.embedding_cache)}")
        
        return results['files_failed'] == 0

def main():
    """Main function to run the embedding-based tag deduplicator."""
    parser = argparse.ArgumentParser(
        description="Deduplicate similar tags across images using text embeddings."
    )
    parser.add_argument("input_dir", help="Input directory containing images with tags.")
    parser.add_argument("--embedding-url", default="http://localhost:1234", 
                       help="Embedding API URL.")
    parser.add_argument("--embedding-model", default="text-embedding-nomic-embed-text-v1.5",
                       help="Embedding model name.")
    parser.add_argument("--similarity-threshold", type=float, default=0.85,
                       help="Cosine similarity threshold for grouping tags (0.0-1.0).")
    parser.add_argument("--min-cluster-size", type=int, default=2,
                       help="Minimum number of tags required to form a cluster.")
    parser.add_argument("--batch-size", type=int, default=100,
                       help="Number of tags to process in each embedding batch.")
    parser.add_argument("--dry-run", action="store_true", 
                       help="Show proposed changes without applying them.")
    parser.add_argument("-v", "--verbose", action="store_true", 
                       help="Enable verbose logging.")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        
    deduplicator = EmbeddingTagDeduplicator(
        embedding_api_url=args.embedding_url,
        embedding_model=args.embedding_model,
        similarity_threshold=args.similarity_threshold,
        min_cluster_size=args.min_cluster_size,
        batch_size=args.batch_size
    )
    
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