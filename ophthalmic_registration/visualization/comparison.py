"""
Visualization and comparison tools for ophthalmic image registration.

This module provides interactive and static visualization tools for
comparing baseline, follow-up, and registered images, including
overlay modes, difference maps, and keypoint visualization.
"""

import logging
from enum import Enum
from typing import Optional, Tuple, List, Union
import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from ophthalmic_registration.core.image_data import (
    ImageData,
    TransformResult,
    FeatureMatchResult,
)

logger = logging.getLogger(__name__)


class ComparisonMode(Enum):
    """Available comparison visualization modes."""
    SIDE_BY_SIDE = "side_by_side"
    OVERLAY = "overlay"
    DIFFERENCE = "difference"
    CHECKERBOARD = "checkerboard"
    SPLIT = "split"
    FLICKER = "flicker"


class Visualizer:
    """
    Visualization tools for ophthalmic image registration results.
    
    Provides multiple comparison modes for evaluating registration
    quality and creating publication-ready figures.
    
    Attributes:
        colormap: Default colormap for difference visualization
        figure_dpi: DPI for saved figures
    
    Example:
        >>> viz = Visualizer()
        >>> 
        >>> # Side-by-side comparison
        >>> fig = viz.compare_side_by_side(baseline, followup, registered)
        >>> 
        >>> # Interactive overlay
        >>> viz.show_interactive_overlay(baseline, registered)
        >>> 
        >>> # Save comparison figure
        >>> viz.save_comparison(baseline, registered, "comparison.png")
    """
    
    def __init__(
        self,
        colormap: str = "viridis",
        figure_dpi: int = 150
    ):
        """
        Initialize visualizer.
        
        Args:
            colormap: Matplotlib colormap for difference maps
            figure_dpi: DPI for saved figures
        """
        self.colormap = colormap
        self.figure_dpi = figure_dpi
        
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib not available - some features disabled")
    
    def compare_side_by_side(
        self,
        baseline: ImageData,
        followup: ImageData,
        registered: Optional[ImageData] = None,
        titles: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (15, 5)
    ) -> 'plt.Figure':
        """
        Create side-by-side comparison figure.
        
        Args:
            baseline: Baseline/reference image
            followup: Original follow-up image
            registered: Registered follow-up image (optional)
            titles: Custom titles for each panel
            figsize: Figure size in inches
        
        Returns:
            Matplotlib figure
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("Matplotlib required for visualization")
        
        n_images = 3 if registered is not None else 2
        fig, axes = plt.subplots(1, n_images, figsize=figsize)
        
        if n_images == 2:
            axes = [axes[0], axes[1]]
        
        # Default titles
        if titles is None:
            if registered is not None:
                titles = ["Baseline", "Follow-up (Original)", "Follow-up (Registered)"]
            else:
                titles = ["Baseline", "Follow-up"]
        
        images = [baseline, followup]
        if registered is not None:
            images.append(registered)
        
        for ax, img, title in zip(axes, images, titles):
            display_img = self._prepare_for_display(img)
            ax.imshow(display_img, cmap='gray' if display_img.ndim == 2 else None)
            ax.set_title(title)
            ax.axis('off')
        
        plt.tight_layout()
        return fig
    
    def create_overlay(
        self,
        image1: ImageData,
        image2: ImageData,
        alpha: float = 0.5,
        colorize: bool = True
    ) -> np.ndarray:
        """
        Create blended overlay of two images.
        
        Args:
            image1: First image (typically baseline)
            image2: Second image (typically registered follow-up)
            alpha: Blending factor (0 = image1 only, 1 = image2 only)
            colorize: Apply different colors to each image
        
        Returns:
            Blended overlay image
        """
        img1 = self._prepare_for_display(image1)
        img2 = self._prepare_for_display(image2)
        
        # Ensure same size
        if img1.shape[:2] != img2.shape[:2]:
            h = min(img1.shape[0], img2.shape[0])
            w = min(img1.shape[1], img2.shape[1])
            img1 = img1[:h, :w]
            img2 = img2[:h, :w]
        
        if colorize:
            # Colorize: image1 in magenta, image2 in green
            if img1.ndim == 2:
                img1_color = np.stack([img1, np.zeros_like(img1), img1], axis=-1)
            else:
                img1_color = img1.copy()
                img1_color[:, :, 1] = 0  # Remove green
            
            if img2.ndim == 2:
                img2_color = np.stack([np.zeros_like(img2), img2, np.zeros_like(img2)], axis=-1)
            else:
                img2_color = img2.copy()
                img2_color[:, :, 0] = 0  # Remove red
                img2_color[:, :, 2] = 0  # Remove blue
            
            overlay = cv2.addWeighted(
                img1_color.astype(np.float32), 1 - alpha,
                img2_color.astype(np.float32), alpha,
                0
            )
        else:
            # Simple alpha blending
            if img1.ndim == 2:
                img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
            if img2.ndim == 2:
                img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
            
            overlay = cv2.addWeighted(
                img1.astype(np.float32), 1 - alpha,
                img2.astype(np.float32), alpha,
                0
            )
        
        return np.clip(overlay, 0, 255).astype(np.uint8)
    
    def create_difference_map(
        self,
        image1: ImageData,
        image2: ImageData,
        absolute: bool = True,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Create difference map between two images.
        
        Args:
            image1: First image
            image2: Second image
            absolute: Use absolute difference
            normalize: Normalize to full range
        
        Returns:
            Difference map as numpy array
        """
        img1 = self._prepare_for_display(image1).astype(np.float32)
        img2 = self._prepare_for_display(image2).astype(np.float32)
        
        # Ensure same size
        if img1.shape[:2] != img2.shape[:2]:
            h = min(img1.shape[0], img2.shape[0])
            w = min(img1.shape[1], img2.shape[1])
            img1 = img1[:h, :w]
            img2 = img2[:h, :w]
        
        # Convert to grayscale if color
        if img1.ndim == 3:
            img1 = cv2.cvtColor(img1.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
        if img2.ndim == 3:
            img2 = cv2.cvtColor(img2.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
        
        if absolute:
            diff = np.abs(img1 - img2)
        else:
            diff = img1 - img2
        
        if normalize:
            diff_min, diff_max = diff.min(), diff.max()
            if diff_max > diff_min:
                diff = (diff - diff_min) / (diff_max - diff_min) * 255
            else:
                diff = np.zeros_like(diff)
        
        return diff.astype(np.uint8)
    
    def create_checkerboard(
        self,
        image1: ImageData,
        image2: ImageData,
        grid_size: int = 50
    ) -> np.ndarray:
        """
        Create checkerboard comparison of two images.
        
        Args:
            image1: First image
            image2: Second image
            grid_size: Size of checkerboard squares in pixels
        
        Returns:
            Checkerboard comparison image
        """
        img1 = self._prepare_for_display(image1)
        img2 = self._prepare_for_display(image2)
        
        # Ensure same size
        if img1.shape[:2] != img2.shape[:2]:
            h = min(img1.shape[0], img2.shape[0])
            w = min(img1.shape[1], img2.shape[1])
            img1 = img1[:h, :w]
            img2 = img2[:h, :w]
        
        h, w = img1.shape[:2]
        
        # Create checkerboard mask
        mask = np.zeros((h, w), dtype=bool)
        for i in range(0, h, grid_size):
            for j in range(0, w, grid_size):
                if ((i // grid_size) + (j // grid_size)) % 2 == 0:
                    mask[i:i+grid_size, j:j+grid_size] = True
        
        # Apply mask
        if img1.ndim == 2:
            result = np.where(mask, img1, img2)
        else:
            result = np.where(mask[:, :, np.newaxis], img1, img2)
        
        return result.astype(np.uint8)
    
    def show_comparison(
        self,
        baseline: ImageData,
        followup: ImageData,
        registered: Optional[ImageData] = None,
        transform_result: Optional[TransformResult] = None,
        mode: ComparisonMode = ComparisonMode.SIDE_BY_SIDE
    ):
        """
        Display comparison visualization.
        
        Args:
            baseline: Baseline image
            followup: Follow-up image
            registered: Registered follow-up (optional)
            transform_result: Registration result for metrics display
            mode: Comparison mode
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("Matplotlib required for visualization")
        
        if mode == ComparisonMode.SIDE_BY_SIDE:
            fig = self.compare_side_by_side(baseline, followup, registered)
        
        elif mode == ComparisonMode.OVERLAY:
            fig, ax = plt.subplots(figsize=(10, 10))
            target = registered if registered is not None else followup
            overlay = self.create_overlay(baseline, target, alpha=0.5)
            ax.imshow(overlay)
            ax.set_title("Overlay (Baseline: Magenta, Follow-up: Green)")
            ax.axis('off')
        
        elif mode == ComparisonMode.DIFFERENCE:
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            
            # Before registration
            diff_before = self.create_difference_map(baseline, followup)
            axes[0].imshow(diff_before, cmap='hot')
            axes[0].set_title("Difference: Before Registration")
            axes[0].axis('off')
            
            # After registration
            if registered is not None:
                diff_after = self.create_difference_map(baseline, registered)
                axes[1].imshow(diff_after, cmap='hot')
                axes[1].set_title("Difference: After Registration")
            else:
                axes[1].text(0.5, 0.5, "No registered image", ha='center', va='center')
            axes[1].axis('off')
        
        elif mode == ComparisonMode.CHECKERBOARD:
            fig, ax = plt.subplots(figsize=(10, 10))
            target = registered if registered is not None else followup
            checker = self.create_checkerboard(baseline, target, grid_size=50)
            ax.imshow(checker, cmap='gray' if checker.ndim == 2 else None)
            ax.set_title("Checkerboard Comparison")
            ax.axis('off')
        
        else:
            raise ValueError(f"Unsupported comparison mode: {mode}")
        
        # Add metrics if available
        if transform_result is not None:
            self._add_metrics_annotation(fig, transform_result)
        
        plt.tight_layout()
        plt.show()
    
    def show_interactive_overlay(
        self,
        image1: ImageData,
        image2: ImageData,
        title: str = "Interactive Overlay"
    ):
        """
        Show interactive overlay with alpha slider.
        
        Args:
            image1: First image
            image2: Second image
            title: Window title
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("Matplotlib required for interactive visualization")
        
        fig, ax = plt.subplots(figsize=(10, 10))
        plt.subplots_adjust(bottom=0.15)
        
        # Initial overlay
        overlay = self.create_overlay(image1, image2, alpha=0.5)
        im = ax.imshow(overlay)
        ax.set_title(title)
        ax.axis('off')
        
        # Add slider
        ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
        slider = Slider(ax_slider, 'Alpha', 0, 1, valinit=0.5)
        
        def update(val):
            overlay = self.create_overlay(image1, image2, alpha=val)
            im.set_data(overlay)
            fig.canvas.draw_idle()
        
        slider.on_changed(update)
        plt.show()
    
    def visualize_keypoints(
        self,
        image: ImageData,
        feature_result: FeatureMatchResult,
        show_all: bool = False
    ) -> np.ndarray:
        """
        Visualize detected keypoints on an image.
        
        Args:
            image: Source image
            feature_result: Feature detection result
            show_all: Show all keypoints (vs only matched)
        
        Returns:
            Image with keypoints drawn
        """
        if not CV2_AVAILABLE:
            raise ImportError("OpenCV required for keypoint visualization")
        
        img = self._prepare_for_display(image)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        if show_all:
            keypoints = feature_result.keypoints_baseline
        else:
            # Only matched keypoints
            matched_indices = set(m.queryIdx for m in feature_result.matches)
            keypoints = [
                kp for i, kp in enumerate(feature_result.keypoints_baseline)
                if i in matched_indices
            ]
        
        result = cv2.drawKeypoints(
            img, keypoints, None,
            color=(0, 255, 0),
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )
        
        return result
    
    def visualize_matches(
        self,
        baseline: ImageData,
        followup: ImageData,
        feature_result: FeatureMatchResult,
        show_inliers_only: bool = True,
        max_matches: int = 100
    ) -> np.ndarray:
        """
        Visualize feature matches between two images.
        
        Args:
            baseline: Baseline image
            followup: Follow-up image
            feature_result: Feature matching result
            show_inliers_only: Only show RANSAC inliers
            max_matches: Maximum matches to display
        
        Returns:
            Visualization image with matches drawn
        """
        if not CV2_AVAILABLE:
            raise ImportError("OpenCV required for match visualization")
        
        img1 = self._prepare_for_display(baseline)
        img2 = self._prepare_for_display(followup)
        
        if img1.ndim == 2:
            img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        if img2.ndim == 2:
            img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
        
        # Filter matches
        if show_inliers_only and feature_result.inlier_mask is not None:
            mask = feature_result.inlier_mask.ravel()
            matches = [m for m, is_inlier in zip(feature_result.matches, mask) if is_inlier]
        else:
            matches = feature_result.matches
        
        # Limit number of matches
        if len(matches) > max_matches:
            # Sort by distance and take best
            matches = sorted(matches, key=lambda m: m.distance)[:max_matches]
        
        # Draw matches
        result = cv2.drawMatches(
            img1, feature_result.keypoints_baseline,
            img2, feature_result.keypoints_followup,
            matches, None,
            matchColor=(0, 255, 0),
            singlePointColor=(255, 0, 0),
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        
        return result
    
    def create_registration_summary(
        self,
        baseline: ImageData,
        followup: ImageData,
        registered: ImageData,
        result: TransformResult,
        figsize: Tuple[int, int] = (16, 12)
    ) -> 'plt.Figure':
        """
        Create comprehensive registration summary figure.
        
        Args:
            baseline: Baseline image
            followup: Original follow-up
            registered: Registered follow-up
            result: Registration result
            figsize: Figure size
        
        Returns:
            Matplotlib figure with summary
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("Matplotlib required for summary visualization")
        
        fig = plt.figure(figsize=figsize)
        
        # Create grid
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.2)
        
        # Row 1: Original images
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(self._prepare_for_display(baseline), cmap='gray')
        ax1.set_title("Baseline")
        ax1.axis('off')
        
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(self._prepare_for_display(followup), cmap='gray')
        ax2.set_title("Follow-up (Original)")
        ax2.axis('off')
        
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.imshow(self._prepare_for_display(registered), cmap='gray')
        ax3.set_title("Follow-up (Registered)")
        ax3.axis('off')
        
        # Overlay
        ax4 = fig.add_subplot(gs[0, 3])
        overlay = self.create_overlay(baseline, registered, alpha=0.5)
        ax4.imshow(overlay)
        ax4.set_title("Overlay")
        ax4.axis('off')
        
        # Row 2: Difference maps
        ax5 = fig.add_subplot(gs[1, 0:2])
        diff_before = self.create_difference_map(baseline, followup)
        im5 = ax5.imshow(diff_before, cmap='hot')
        ax5.set_title("Difference: Before Registration")
        ax5.axis('off')
        plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)
        
        ax6 = fig.add_subplot(gs[1, 2:4])
        diff_after = self.create_difference_map(baseline, registered)
        im6 = ax6.imshow(diff_after, cmap='hot')
        ax6.set_title("Difference: After Registration")
        ax6.axis('off')
        plt.colorbar(im6, ax=ax6, fraction=0.046, pad=0.04)
        
        # Row 3: Metrics and checkerboard
        ax7 = fig.add_subplot(gs[2, 0:2])
        checker = self.create_checkerboard(baseline, registered, grid_size=40)
        ax7.imshow(checker, cmap='gray' if checker.ndim == 2 else None)
        ax7.set_title("Checkerboard")
        ax7.axis('off')
        
        # Metrics panel
        ax8 = fig.add_subplot(gs[2, 2:4])
        ax8.axis('off')
        
        metrics_text = self._format_metrics(result)
        ax8.text(
            0.1, 0.9, metrics_text,
            transform=ax8.transAxes,
            fontsize=10,
            verticalalignment='top',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )
        ax8.set_title("Registration Metrics")
        
        return fig
    
    def save_comparison(
        self,
        baseline: ImageData,
        followup: ImageData,
        output_path: str,
        registered: Optional[ImageData] = None,
        mode: ComparisonMode = ComparisonMode.SIDE_BY_SIDE,
        dpi: Optional[int] = None
    ):
        """
        Save comparison figure to file.
        
        Args:
            baseline: Baseline image
            followup: Follow-up image
            output_path: Output file path
            registered: Registered image (optional)
            mode: Comparison mode
            dpi: Output DPI (uses default if None)
        """
        dpi = dpi or self.figure_dpi
        
        self.show_comparison(baseline, followup, registered, mode=mode)
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved comparison figure to {output_path}")
    
    def _prepare_for_display(
        self,
        image: Union[ImageData, np.ndarray]
    ) -> np.ndarray:
        """Prepare image for display (normalize to uint8)."""
        if isinstance(image, ImageData):
            img = image.pixel_array
        else:
            img = image
        
        if img.dtype == np.uint8:
            return img
        
        # Normalize to uint8
        img_float = img.astype(np.float64)
        img_min, img_max = img_float.min(), img_float.max()
        
        if img_max > img_min:
            img_normalized = (img_float - img_min) / (img_max - img_min) * 255
        else:
            img_normalized = np.zeros_like(img_float)
        
        return img_normalized.astype(np.uint8)
    
    def _add_metrics_annotation(
        self,
        fig: 'plt.Figure',
        result: TransformResult
    ):
        """Add metrics annotation to figure."""
        metrics_text = []
        
        if result.ecc_correlation is not None:
            metrics_text.append(f"ECC: {result.ecc_correlation:.4f}")
        
        if "overlap_ratio" in result.quality_metrics:
            metrics_text.append(f"Overlap: {result.quality_metrics['overlap_ratio']:.1%}")
        
        if result.feature_match_result is not None:
            metrics_text.append(f"Inliers: {result.feature_match_result.num_inliers}")
        
        if metrics_text:
            fig.text(
                0.02, 0.02,
                " | ".join(metrics_text),
                fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
            )
    
    def _format_metrics(self, result: TransformResult) -> str:
        """Format metrics for display."""
        lines = ["REGISTRATION METRICS", "=" * 30]
        
        lines.append(f"Motion Model: {result.motion_model.value}")
        lines.append(f"Time: {result.registration_time_ms:.1f} ms")
        lines.append("")
        
        if result.ecc_correlation is not None:
            lines.append(f"ECC Correlation: {result.ecc_correlation:.4f}")
            lines.append(f"ECC Converged: {result.ecc_converged}")
        
        lines.append("")
        tx, ty = result.translation
        lines.append(f"Translation: ({tx:.2f}, {ty:.2f}) px")
        
        if result.rotation_degrees is not None:
            lines.append(f"Rotation: {result.rotation_degrees:.2f}°")
        
        sx, sy = result.scale_factors
        lines.append(f"Scale: ({sx:.4f}, {sy:.4f})")
        
        if result.quality_metrics:
            lines.append("")
            lines.append("Quality Metrics:")
            for key, value in result.quality_metrics.items():
                if isinstance(value, float):
                    lines.append(f"  {key}: {value:.4f}")
                else:
                    lines.append(f"  {key}: {value}")
        
        if result.warnings:
            lines.append("")
            lines.append("Warnings:")
            for warning in result.warnings:
                lines.append(f"  - {warning[:50]}...")
        
        return "\n".join(lines)
