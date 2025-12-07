"""
Inverse Perspective Mapping (IPM) - Image to ground coordinate transformation.
"""

from typing import Optional, Tuple
import numpy as np


class InversePerspectiveMapper:
    """
    Inverse Perspective Mapping for converting image coordinates to ground/vehicle coordinates.
    
    Assumes:
    - Flat ground plane
    - Known camera intrinsics and extrinsics
    - Vehicle coordinate system: X-right, Y-forward, Z-up
    """
    
    def __init__(
        self,
        intrinsics: np.ndarray,
        extrinsics: np.ndarray,
        image_size: Tuple[int, int],
        ground_height: float = 0.0,
    ):
        """
        Initialize IPM.
        
        Args:
            intrinsics: 3x3 camera intrinsic matrix
                [[fx, 0, cx],
                 [0, fy, cy],
                 [0, 0, 1]]
            extrinsics: 4x4 camera to vehicle transformation matrix
            image_size: Image dimensions (width, height)
            ground_height: Ground plane height in vehicle frame (usually 0)
        """
        self.K = intrinsics
        self.T_cam_to_veh = extrinsics
        self.image_width, self.image_height = image_size
        self.ground_height = ground_height
        
        # Compute inverse intrinsics
        self.K_inv = np.linalg.inv(self.K)
        
        # Extract rotation and translation from extrinsics
        self.R = self.T_cam_to_veh[:3, :3]
        self.t = self.T_cam_to_veh[:3, 3]
        
        # Compute homography for flat ground
        self._compute_homography()
    
    def _compute_homography(self) -> None:
        """
        Compute homography matrix for ground plane projection.
        
        For a flat ground plane at z = ground_height, we can derive
        a homography between image and ground coordinates.
        """
        # Camera height above ground (negative z in camera frame)
        self.camera_height = self.t[2] - self.ground_height
        
        # Ground plane normal in camera frame (pointing up)
        n = self.R.T @ np.array([0, 0, 1])
        
        # Distance to ground plane
        d = self.camera_height
        
        # Homography: H = K * (R - t*n'/d) * K^-1
        # Simplified for flat ground
        self.H_img_to_ground = self._compute_ground_homography()
        
        # Inverse homography
        try:
            self.H_ground_to_img = np.linalg.inv(self.H_img_to_ground)
        except np.linalg.LinAlgError:
            self.H_ground_to_img = None
    
    def _compute_ground_homography(self) -> np.ndarray:
        """Compute image to ground homography matrix."""
        # For a pinhole camera looking at flat ground:
        # Ground point: [X, Y, 0]^T in vehicle frame
        # Camera point: R @ [X, Y, -h]^T + t
        # Image point: K @ camera_point
        
        # We invert this to get ground from image
        # This is a simplification assuming flat ground
        
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]
        
        # Camera pose
        pitch = np.arcsin(-self.R[2, 0])  # Approximate pitch angle
        h = self.camera_height
        
        # Build homography
        # This is a simplified version; full version would use extrinsics directly
        H = np.array([
            [1/fx, 0, -cx/fx],
            [0, np.cos(pitch)/fy, -cy*np.cos(pitch)/fy - np.sin(pitch)],
            [0, np.sin(pitch)/(fy*h), (1 - cy*np.sin(pitch)/fy)/h + np.cos(pitch)/h],
        ])
        
        return H
    
    def image_to_ground(
        self,
        image_points: np.ndarray,
    ) -> np.ndarray:
        """
        Convert image coordinates to ground/vehicle coordinates.
        
        Args:
            image_points: Nx2 array of (u, v) pixel coordinates
            
        Returns:
            Nx2 array of (x, y) ground coordinates in meters
            x: lateral (positive right)
            y: longitudinal (positive forward)
        """
        if len(image_points) == 0:
            return np.array([])
        
        points = np.atleast_2d(image_points)
        n_points = points.shape[0]
        
        # Convert to homogeneous
        ones = np.ones((n_points, 1))
        points_h = np.hstack([points, ones])
        
        # Apply homography
        ground_h = (self.H_img_to_ground @ points_h.T).T
        
        # Normalize
        ground_h = ground_h / (ground_h[:, 2:3] + 1e-10)
        
        # Extract x, y (ignore z which should be ~0 for ground plane)
        ground_points = ground_h[:, :2]
        
        # Filter invalid points (behind camera or too far)
        valid_mask = (ground_points[:, 1] > 0) & (ground_points[:, 1] < 100)
        ground_points[~valid_mask] = np.nan
        
        return ground_points
    
    def ground_to_image(
        self,
        ground_points: np.ndarray,
    ) -> np.ndarray:
        """
        Convert ground/vehicle coordinates to image coordinates.
        
        Args:
            ground_points: Nx2 array of (x, y) ground coordinates
            
        Returns:
            Nx2 array of (u, v) pixel coordinates
        """
        if self.H_ground_to_img is None:
            raise ValueError("Ground to image homography not available")
        
        if len(ground_points) == 0:
            return np.array([])
        
        points = np.atleast_2d(ground_points)
        n_points = points.shape[0]
        
        # Convert to homogeneous (add z=0)
        zeros = np.zeros((n_points, 1))
        ones = np.ones((n_points, 1))
        points_h = np.hstack([points, ones])
        
        # Apply inverse homography
        image_h = (self.H_ground_to_img @ points_h.T).T
        
        # Normalize
        image_h = image_h / (image_h[:, 2:3] + 1e-10)
        
        # Extract u, v
        image_points = image_h[:, :2]
        
        # Clip to image bounds
        image_points[:, 0] = np.clip(image_points[:, 0], 0, self.image_width - 1)
        image_points[:, 1] = np.clip(image_points[:, 1], 0, self.image_height - 1)
        
        return image_points
    
    def ray_to_ground(
        self,
        u: float,
        v: float,
    ) -> Tuple[float, float]:
        """
        Cast ray from image pixel to ground plane.
        
        Args:
            u: Pixel x coordinate
            v: Pixel y coordinate
            
        Returns:
            Tuple of (x, y) ground coordinates
        """
        # Normalized image coordinates
        point_norm = self.K_inv @ np.array([u, v, 1])
        
        # Ray direction in camera frame
        ray_cam = point_norm / np.linalg.norm(point_norm)
        
        # Transform to vehicle frame
        ray_veh = self.R @ ray_cam
        origin_veh = self.t
        
        # Intersect with ground plane (z = ground_height)
        # origin + t * ray = ground
        # origin_z + t * ray_z = ground_height
        if abs(ray_veh[2]) < 1e-6:
            return np.nan, np.nan
        
        t_intersect = (self.ground_height - origin_veh[2]) / ray_veh[2]
        
        if t_intersect < 0:
            return np.nan, np.nan
        
        ground_point = origin_veh + t_intersect * ray_veh
        
        return ground_point[0], ground_point[1]
    
    def create_bev_image(
        self,
        image: np.ndarray,
        output_size: Tuple[int, int] = (400, 400),
        meters_per_pixel: float = 0.1,
        y_offset: float = 5.0,
    ) -> np.ndarray:
        """
        Create bird's eye view (BEV) image.
        
        Args:
            image: Input camera image
            output_size: BEV image size (width, height)
            meters_per_pixel: Ground resolution
            y_offset: Forward offset in meters
            
        Returns:
            Bird's eye view image
        """
        import cv2
        
        bev_width, bev_height = output_size
        
        # Create mapping from BEV to image
        map_x = np.zeros((bev_height, bev_width), dtype=np.float32)
        map_y = np.zeros((bev_height, bev_width), dtype=np.float32)
        
        for bev_v in range(bev_height):
            for bev_u in range(bev_width):
                # BEV pixel to ground coordinates
                x_ground = (bev_u - bev_width // 2) * meters_per_pixel
                y_ground = (bev_height - bev_v) * meters_per_pixel + y_offset
                
                # Ground to image
                img_pts = self.ground_to_image(np.array([[x_ground, y_ground]]))
                
                if not np.isnan(img_pts[0, 0]):
                    map_x[bev_v, bev_u] = img_pts[0, 0]
                    map_y[bev_v, bev_u] = img_pts[0, 1]
                else:
                    map_x[bev_v, bev_u] = -1
                    map_y[bev_v, bev_u] = -1
        
        # Remap image
        bev_image = cv2.remap(
            image,
            map_x,
            map_y,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        
        return bev_image
    
    @classmethod
    def from_camera_params(
        cls,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        camera_height: float,
        camera_pitch: float,
        image_size: Tuple[int, int],
    ) -> "InversePerspectiveMapper":
        """
        Create IPM from simple camera parameters.
        
        Args:
            fx, fy: Focal lengths in pixels
            cx, cy: Principal point
            camera_height: Height above ground in meters
            camera_pitch: Pitch angle in radians (positive = looking down)
            image_size: Image dimensions (width, height)
            
        Returns:
            InversePerspectiveMapper instance
        """
        # Build intrinsic matrix
        K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1],
        ])
        
        # Build extrinsic matrix (camera to vehicle)
        # Assuming camera is at (0, 0, camera_height) looking forward with pitch
        R_pitch = np.array([
            [1, 0, 0],
            [0, np.cos(camera_pitch), -np.sin(camera_pitch)],
            [0, np.sin(camera_pitch), np.cos(camera_pitch)],
        ])
        
        # Camera frame: X-right, Y-down, Z-forward
        # Vehicle frame: X-right, Y-forward, Z-up
        R_cam_to_veh = np.array([
            [1, 0, 0],
            [0, 0, 1],
            [0, -1, 0],
        ])
        
        R = R_cam_to_veh @ R_pitch
        t = np.array([0, 0, camera_height])
        
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        
        return cls(K, T, image_size)
