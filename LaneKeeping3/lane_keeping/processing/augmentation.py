"""
Data Augmentation Pipeline for Lane Detection Training.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import cv2


class LaneAugmentation:
    """
    Augmentation pipeline specifically designed for lane detection.
    
    Includes:
    - Photometric augmentations (exposure, contrast, color)
    - Geometric augmentations (perspective, crop, scale)
    - Weather/condition augmentations (rain, glare, shadows)
    - Temporal augmentations (motion blur, frame jitter)
    """
    
    def __init__(
        self,
        config: Optional[Dict] = None,
        mode: str = "train",
    ):
        """
        Initialize augmentation pipeline.
        
        Args:
            config: Augmentation configuration
            mode: 'train', 'val', or 'test'
        """
        self.config = config or self._default_config()
        self.mode = mode
        
        # Random state
        self.rng = np.random.default_rng()
    
    def _default_config(self) -> Dict:
        """Default augmentation configuration."""
        return {
            # Photometric
            'brightness_range': (-0.2, 0.2),
            'contrast_range': (0.8, 1.2),
            'saturation_range': (0.8, 1.2),
            'hue_shift_range': (-10, 10),
            
            # Geometric
            'perspective_prob': 0.5,
            'perspective_range': 0.1,
            'horizontal_flip_prob': 0.5,
            'scale_range': (0.9, 1.1),
            'crop_prob': 0.3,
            
            # Weather
            'rain_prob': 0.1,
            'shadow_prob': 0.2,
            'glare_prob': 0.1,
            'fog_prob': 0.1,
            
            # Motion
            'motion_blur_prob': 0.1,
            'motion_blur_kernel': (3, 7),
            
            # Noise
            'gaussian_noise_prob': 0.1,
            'noise_std': 10,
        }
    
    def __call__(
        self,
        image: np.ndarray,
        keypoints: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Apply augmentations.
        
        Args:
            image: Input BGR image
            keypoints: Lane keypoints [N, K, 2]
            mask: Segmentation mask
            
        Returns:
            Dictionary with augmented image, keypoints, and mask
        """
        if self.mode == "test":
            return {"image": image, "keypoints": keypoints, "mask": mask}
        
        result = {"image": image.copy()}
        if keypoints is not None:
            result["keypoints"] = keypoints.copy()
        if mask is not None:
            result["mask"] = mask.copy()
        
        # Apply augmentations
        if self.mode == "train":
            result = self._apply_photometric(result)
            result = self._apply_geometric(result)
            result = self._apply_weather(result)
            result = self._apply_motion(result)
            result = self._apply_noise(result)
        elif self.mode == "val":
            # Light augmentations for validation
            result = self._apply_photometric(result, light=True)
        
        return result
    
    def _apply_photometric(
        self,
        data: Dict,
        light: bool = False,
    ) -> Dict:
        """Apply photometric augmentations."""
        image = data["image"]
        
        # Convert to float
        image = image.astype(np.float32)
        
        # Brightness
        if not light or self.rng.random() < 0.5:
            brightness = self.rng.uniform(*self.config['brightness_range'])
            if light:
                brightness *= 0.5
            image = image + brightness * 255
        
        # Contrast
        if not light or self.rng.random() < 0.5:
            contrast = self.rng.uniform(*self.config['contrast_range'])
            if light:
                contrast = 1 + (contrast - 1) * 0.5
            image = (image - 128) * contrast + 128
        
        # Saturation and Hue (in HSV)
        if self.rng.random() < 0.5:
            image_hsv = cv2.cvtColor(
                np.clip(image, 0, 255).astype(np.uint8),
                cv2.COLOR_BGR2HSV
            ).astype(np.float32)
            
            # Saturation
            saturation = self.rng.uniform(*self.config['saturation_range'])
            image_hsv[:, :, 1] = image_hsv[:, :, 1] * saturation
            
            # Hue shift
            hue_shift = self.rng.uniform(*self.config['hue_shift_range'])
            image_hsv[:, :, 0] = (image_hsv[:, :, 0] + hue_shift) % 180
            
            image_hsv = np.clip(image_hsv, 0, 255).astype(np.uint8)
            image = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR).astype(np.float32)
        
        data["image"] = np.clip(image, 0, 255).astype(np.uint8)
        return data
    
    def _apply_geometric(self, data: Dict) -> Dict:
        """Apply geometric augmentations."""
        image = data["image"]
        h, w = image.shape[:2]
        keypoints = data.get("keypoints")
        mask = data.get("mask")
        
        # Horizontal flip
        if self.rng.random() < self.config['horizontal_flip_prob']:
            image = cv2.flip(image, 1)
            if mask is not None:
                mask = cv2.flip(mask, 1)
            if keypoints is not None:
                keypoints = keypoints.copy()
                keypoints[:, :, 0] = w - keypoints[:, :, 0]
                # Swap left and right lanes
                if keypoints.shape[0] >= 2:
                    keypoints = keypoints[::-1]
        
        # Perspective transform (simulate pitch/roll changes)
        if self.rng.random() < self.config['perspective_prob']:
            range_val = self.config['perspective_range']
            
            # Random perspective distortion
            pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
            pts2 = pts1 + self.rng.uniform(-range_val, range_val, pts1.shape) * np.array([w, h])
            pts2 = pts2.astype(np.float32)
            
            M = cv2.getPerspectiveTransform(pts1, pts2)
            image = cv2.warpPerspective(image, M, (w, h))
            
            if mask is not None:
                mask = cv2.warpPerspective(mask, M, (w, h))
            
            if keypoints is not None:
                # Transform keypoints
                kp_reshaped = keypoints.reshape(-1, 2)
                kp_h = np.hstack([kp_reshaped, np.ones((kp_reshaped.shape[0], 1))])
                kp_transformed = (M @ kp_h.T).T
                kp_transformed = kp_transformed[:, :2] / (kp_transformed[:, 2:] + 1e-10)
                keypoints = kp_transformed.reshape(keypoints.shape)
        
        # Scale
        scale = self.rng.uniform(*self.config['scale_range'])
        if scale != 1.0:
            new_h, new_w = int(h * scale), int(w * scale)
            image = cv2.resize(image, (new_w, new_h))
            
            if mask is not None:
                mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            
            if keypoints is not None:
                keypoints = keypoints * scale
            
            # Crop or pad to original size
            if scale > 1.0:
                # Crop
                start_h = (new_h - h) // 2
                start_w = (new_w - w) // 2
                image = image[start_h:start_h + h, start_w:start_w + w]
                if mask is not None:
                    mask = mask[start_h:start_h + h, start_w:start_w + w]
                if keypoints is not None:
                    keypoints[:, :, 0] -= start_w
                    keypoints[:, :, 1] -= start_h
            else:
                # Pad
                pad_h = (h - new_h) // 2
                pad_w = (w - new_w) // 2
                image = cv2.copyMakeBorder(
                    image, pad_h, h - new_h - pad_h,
                    pad_w, w - new_w - pad_w,
                    cv2.BORDER_CONSTANT, value=0
                )
                if mask is not None:
                    mask = cv2.copyMakeBorder(
                        mask, pad_h, h - new_h - pad_h,
                        pad_w, w - new_w - pad_w,
                        cv2.BORDER_CONSTANT, value=0
                    )
                if keypoints is not None:
                    keypoints[:, :, 0] += pad_w
                    keypoints[:, :, 1] += pad_h
        
        data["image"] = image
        if keypoints is not None:
            data["keypoints"] = keypoints
        if mask is not None:
            data["mask"] = mask
        
        return data
    
    def _apply_weather(self, data: Dict) -> Dict:
        """Apply weather/condition augmentations."""
        image = data["image"]
        
        # Rain effect
        if self.rng.random() < self.config['rain_prob']:
            image = self._add_rain(image)
        
        # Shadow
        if self.rng.random() < self.config['shadow_prob']:
            image = self._add_shadow(image)
        
        # Glare/lens flare
        if self.rng.random() < self.config['glare_prob']:
            image = self._add_glare(image)
        
        # Fog
        if self.rng.random() < self.config['fog_prob']:
            image = self._add_fog(image)
        
        data["image"] = image
        return data
    
    def _add_rain(self, image: np.ndarray) -> np.ndarray:
        """Add rain effect."""
        h, w = image.shape[:2]
        
        # Create rain streaks
        rain = np.zeros((h, w), dtype=np.uint8)
        num_drops = self.rng.integers(100, 500)
        
        for _ in range(num_drops):
            x = self.rng.integers(0, w)
            y = self.rng.integers(0, h)
            length = self.rng.integers(10, 30)
            thickness = self.rng.integers(1, 3)
            
            x2 = x + self.rng.integers(-5, 5)
            y2 = min(y + length, h - 1)
            
            cv2.line(rain, (x, y), (x2, y2), 200, thickness)
        
        # Apply motion blur to rain
        kernel_size = 5
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[:, kernel_size // 2] = 1 / kernel_size
        rain = cv2.filter2D(rain, -1, kernel)
        
        # Blend with image
        rain_overlay = cv2.merge([rain, rain, rain])
        alpha = 0.3
        image = cv2.addWeighted(image, 1, rain_overlay, alpha, 0)
        
        return image
    
    def _add_shadow(self, image: np.ndarray) -> np.ndarray:
        """Add shadow effect."""
        h, w = image.shape[:2]
        
        # Create random polygon shadow
        num_points = self.rng.integers(3, 6)
        points = []
        
        for _ in range(num_points):
            x = self.rng.integers(0, w)
            y = self.rng.integers(0, h)
            points.append([x, y])
        
        points = np.array(points, dtype=np.int32)
        
        # Create shadow mask
        shadow_mask = np.zeros((h, w), dtype=np.float32)
        cv2.fillPoly(shadow_mask, [points], 1.0)
        
        # Blur the mask
        shadow_mask = cv2.GaussianBlur(shadow_mask, (51, 51), 0)
        
        # Apply shadow
        shadow_intensity = self.rng.uniform(0.3, 0.7)
        shadow_mask = 1 - shadow_mask * (1 - shadow_intensity)
        
        image = (image * shadow_mask[:, :, np.newaxis]).astype(np.uint8)
        
        return image
    
    def _add_glare(self, image: np.ndarray) -> np.ndarray:
        """Add lens flare/glare effect."""
        h, w = image.shape[:2]
        
        # Random glare position (usually top portion)
        cx = self.rng.integers(w // 4, 3 * w // 4)
        cy = self.rng.integers(0, h // 3)
        
        # Create radial gradient
        y, x = np.ogrid[:h, :w]
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        
        radius = self.rng.integers(50, 200)
        glare = np.clip(1 - dist / radius, 0, 1) ** 2
        
        # Add glare
        intensity = self.rng.uniform(50, 150)
        glare_color = np.array([intensity, intensity * 0.9, intensity * 0.8])
        
        image = image.astype(np.float32)
        for c in range(3):
            image[:, :, c] += glare * glare_color[c]
        
        image = np.clip(image, 0, 255).astype(np.uint8)
        
        return image
    
    def _add_fog(self, image: np.ndarray) -> np.ndarray:
        """Add fog effect."""
        h, w = image.shape[:2]
        
        # Fog intensity increases with distance (lower in image = closer)
        fog_map = np.linspace(0, 1, h).reshape(-1, 1)
        fog_map = np.tile(fog_map, (1, w))
        
        # Add some noise
        noise = self.rng.normal(0, 0.1, (h, w))
        fog_map = np.clip(fog_map + noise, 0, 1)
        
        # Blur
        fog_map = cv2.GaussianBlur(fog_map.astype(np.float32), (21, 21), 0)
        
        # Fog color (light gray)
        fog_color = np.array([200, 200, 200], dtype=np.float32)
        
        # Blend
        intensity = self.rng.uniform(0.3, 0.6)
        fog_map = fog_map * intensity
        
        image = image.astype(np.float32)
        for c in range(3):
            image[:, :, c] = image[:, :, c] * (1 - fog_map) + fog_color[c] * fog_map
        
        image = np.clip(image, 0, 255).astype(np.uint8)
        
        return image
    
    def _apply_motion(self, data: Dict) -> Dict:
        """Apply motion blur."""
        if self.rng.random() < self.config['motion_blur_prob']:
            image = data["image"]
            
            kernel_size = self.rng.integers(*self.config['motion_blur_kernel'])
            if kernel_size % 2 == 0:
                kernel_size += 1
            
            # Create motion blur kernel (vertical for forward motion)
            kernel = np.zeros((kernel_size, kernel_size))
            kernel[:, kernel_size // 2] = 1 / kernel_size
            
            # Slight rotation for realism
            angle = self.rng.uniform(-5, 5)
            M = cv2.getRotationMatrix2D(
                (kernel_size // 2, kernel_size // 2),
                angle,
                1.0
            )
            kernel = cv2.warpAffine(kernel, M, (kernel_size, kernel_size))
            kernel = kernel / kernel.sum()
            
            image = cv2.filter2D(image, -1, kernel)
            data["image"] = image
        
        return data
    
    def _apply_noise(self, data: Dict) -> Dict:
        """Apply noise."""
        if self.rng.random() < self.config['gaussian_noise_prob']:
            image = data["image"].astype(np.float32)
            
            noise = self.rng.normal(0, self.config['noise_std'], image.shape)
            image = image + noise
            
            data["image"] = np.clip(image, 0, 255).astype(np.uint8)
        
        return data
