"""
Lane Overlay Visualization.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import cv2

from lane_keeping.core.lane import Lane, LaneBoundary, LaneDetectionResult
from lane_keeping.core.steering import SteeringOutput


class LaneOverlay:
    """
    Visualize lane detection results on images.
    """
    
    # Color palette
    COLORS = {
        'left_lane': (0, 255, 0),      # Green
        'right_lane': (255, 0, 0),      # Blue
        'centerline': (0, 255, 255),    # Yellow
        'ego_lane': (0, 200, 255),      # Orange
        'lookahead': (255, 0, 255),     # Magenta
        'warning': (0, 0, 255),         # Red
        'text': (255, 255, 255),        # White
        'background': (0, 0, 0),        # Black
    }
    
    def __init__(
        self,
        show_boundaries: bool = True,
        show_centerline: bool = True,
        show_lookahead: bool = True,
        show_info: bool = True,
        show_confidence: bool = True,
        line_thickness: int = 3,
        point_radius: int = 5,
        alpha: float = 0.4,
    ):
        """
        Initialize overlay visualizer.
        
        Args:
            show_boundaries: Show lane boundaries
            show_centerline: Show lane centerline
            show_lookahead: Show lookahead point
            show_info: Show info text
            show_confidence: Show confidence values
            line_thickness: Line thickness in pixels
            point_radius: Keypoint radius
            alpha: Overlay transparency
        """
        self.show_boundaries = show_boundaries
        self.show_centerline = show_centerline
        self.show_lookahead = show_lookahead
        self.show_info = show_info
        self.show_confidence = show_confidence
        self.line_thickness = line_thickness
        self.point_radius = point_radius
        self.alpha = alpha
    
    def draw(
        self,
        image: np.ndarray,
        detection: LaneDetectionResult,
        steering: Optional[SteeringOutput] = None,
    ) -> np.ndarray:
        """
        Draw lane detection overlay on image.
        
        Args:
            image: Input BGR image
            detection: Lane detection result
            steering: Steering guidance output (optional)
            
        Returns:
            Image with overlay
        """
        output = image.copy()
        overlay = image.copy()
        
        # Draw ego lane fill
        if detection.ego_lane is not None:
            self._draw_lane_fill(overlay, detection.ego_lane)
        
        # Blend overlay
        cv2.addWeighted(overlay, self.alpha, output, 1 - self.alpha, 0, output)
        
        # Draw lane boundaries
        if self.show_boundaries:
            for boundary in detection.boundaries:
                self._draw_boundary(output, boundary)
        
        # Draw centerline
        if self.show_centerline and detection.ego_lane is not None:
            self._draw_centerline(output, detection.ego_lane)
        
        # Draw lookahead point
        if self.show_lookahead and steering is not None:
            self._draw_lookahead(output, steering, image.shape)
        
        # Draw info panel
        if self.show_info:
            self._draw_info(output, detection, steering)
        
        return output
    
    def _draw_boundary(
        self,
        image: np.ndarray,
        boundary: LaneBoundary,
    ) -> None:
        """Draw a lane boundary."""
        if boundary.image_points is None or len(boundary.image_points) < 2:
            return
        
        points = boundary.image_points.astype(np.int32)
        
        # Determine color based on position
        if boundary.id == 0:
            color = self.COLORS['left_lane']
        elif boundary.id == 1:
            color = self.COLORS['right_lane']
        else:
            color = (128, 128, 128)
        
        # Draw polyline
        cv2.polylines(
            image,
            [points],
            isClosed=False,
            color=color,
            thickness=self.line_thickness,
            lineType=cv2.LINE_AA,
        )
        
        # Draw keypoints
        for pt in points:
            cv2.circle(
                image,
                tuple(pt),
                self.point_radius,
                color,
                -1,
                lineType=cv2.LINE_AA,
            )
        
        # Draw confidence
        if self.show_confidence and len(points) > 0:
            text = f"{boundary.detection_confidence:.2f}"
            pos = (int(points[0, 0]), int(points[0, 1]) - 10)
            cv2.putText(
                image,
                text,
                pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )
    
    def _draw_lane_fill(
        self,
        image: np.ndarray,
        lane: Lane,
    ) -> None:
        """Draw filled lane area."""
        if lane.left_boundary is None or lane.right_boundary is None:
            return
        
        left_pts = lane.left_boundary.image_points
        right_pts = lane.right_boundary.image_points
        
        if left_pts is None or right_pts is None:
            return
        
        # Create polygon points
        poly_points = np.vstack([
            left_pts,
            right_pts[::-1]
        ]).astype(np.int32)
        
        # Fill polygon
        cv2.fillPoly(
            image,
            [poly_points],
            self.COLORS['ego_lane'],
        )
    
    def _draw_centerline(
        self,
        image: np.ndarray,
        lane: Lane,
    ) -> None:
        """Draw lane centerline."""
        if lane.centerline is None or lane.centerline.image_points is None:
            return
        
        points = lane.centerline.image_points.astype(np.int32)
        
        # Draw dashed line
        for i in range(0, len(points) - 1, 2):
            pt1 = tuple(points[i])
            pt2 = tuple(points[min(i + 1, len(points) - 1)])
            cv2.line(
                image,
                pt1,
                pt2,
                self.COLORS['centerline'],
                self.line_thickness,
                lineType=cv2.LINE_AA,
            )
    
    def _draw_lookahead(
        self,
        image: np.ndarray,
        steering: SteeringOutput,
        image_shape: Tuple[int, ...],
    ) -> None:
        """Draw lookahead point and guidance info."""
        h, w = image_shape[:2]
        
        # Draw lookahead marker (crosshair)
        # Convert from ground coordinates to image coordinates
        # This is a simplified visualization
        cx = w // 2 + int(steering.lookahead_x * 50)  # Scale factor
        cy = h // 3  # Approximate lookahead position in image
        
        color = self.COLORS['lookahead']
        if steering.fallback_active:
            color = self.COLORS['warning']
        
        # Draw crosshair
        size = 15
        cv2.line(image, (cx - size, cy), (cx + size, cy), color, 2)
        cv2.line(image, (cx, cy - size), (cx, cy + size), color, 2)
        
        # Draw heading indicator
        heading_length = 50
        heading_x = int(cx + heading_length * np.sin(steering.heading_error))
        heading_y = int(cy - heading_length * np.cos(steering.heading_error))
        cv2.arrowedLine(
            image,
            (cx, cy),
            (heading_x, heading_y),
            color,
            2,
            tipLength=0.3,
        )
    
    def _draw_info(
        self,
        image: np.ndarray,
        detection: LaneDetectionResult,
        steering: Optional[SteeringOutput],
    ) -> None:
        """Draw info panel."""
        h, w = image.shape[:2]
        
        # Background panel
        panel_height = 120
        cv2.rectangle(
            image,
            (10, h - panel_height - 10),
            (300, h - 10),
            self.COLORS['background'],
            -1,
        )
        cv2.rectangle(
            image,
            (10, h - panel_height - 10),
            (300, h - 10),
            self.COLORS['text'],
            1,
        )
        
        # Text info
        y_offset = h - panel_height
        line_height = 20
        
        # Detection info
        lines = [
            f"Lanes: {detection.num_boundaries}",
            f"Inference: {detection.total_time_ms:.1f}ms",
        ]
        
        if detection.ego_lane is not None:
            conf = detection.ego_lane.confidence
            lines.append(f"Confidence: {conf:.2f}")
        
        if steering is not None:
            lines.extend([
                f"Lat. Offset: {steering.lateral_offset:.2f}m",
                f"Heading Err: {np.degrees(steering.heading_error):.1f}°",
            ])
            
            if steering.fallback_active:
                lines.append(f"FALLBACK: {steering.fallback_reason[:20]}")
        
        for i, line in enumerate(lines):
            cv2.putText(
                image,
                line,
                (20, y_offset + (i + 1) * line_height),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                self.COLORS['text'],
                1,
                cv2.LINE_AA,
            )
    
    def draw_bev(
        self,
        detection: LaneDetectionResult,
        steering: Optional[SteeringOutput] = None,
        size: Tuple[int, int] = (400, 400),
        scale: float = 5.0,
    ) -> np.ndarray:
        """
        Draw bird's eye view of lane detection.
        
        Args:
            detection: Lane detection result
            steering: Steering output
            size: BEV image size
            scale: Pixels per meter
            
        Returns:
            BEV visualization image
        """
        w, h = size
        bev = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Draw grid
        grid_spacing = int(10 * scale)  # 10m grid
        for i in range(0, w, grid_spacing):
            cv2.line(bev, (i, 0), (i, h), (40, 40, 40), 1)
        for i in range(0, h, grid_spacing):
            cv2.line(bev, (0, i), (w, i), (40, 40, 40), 1)
        
        # Draw vehicle position (bottom center)
        vehicle_pos = (w // 2, h - 20)
        cv2.circle(bev, vehicle_pos, 10, (255, 255, 255), -1)
        
        # Draw lane boundaries in ground coordinates
        for boundary in detection.boundaries:
            if boundary.ground_points is None:
                continue
            
            # Convert to BEV coordinates
            points = boundary.ground_points.copy()
            points[:, 0] = w // 2 + points[:, 0] * scale
            points[:, 1] = h - points[:, 1] * scale
            
            points = points[points[:, 1] > 0]  # Filter points above image
            points = points[points[:, 1] < h]
            
            if len(points) < 2:
                continue
            
            points = points.astype(np.int32)
            
            color = self.COLORS['left_lane'] if boundary.id == 0 else self.COLORS['right_lane']
            cv2.polylines(bev, [points], False, color, 2)
        
        # Draw centerline
        if detection.ego_lane and detection.ego_lane.centerline:
            cl = detection.ego_lane.centerline
            if cl.ground_points is not None:
                points = cl.ground_points.copy()
                points[:, 0] = w // 2 + points[:, 0] * scale
                points[:, 1] = h - points[:, 1] * scale
                
                points = points[points[:, 1] > 0]
                points = points[points[:, 1] < h]
                
                if len(points) >= 2:
                    points = points.astype(np.int32)
                    cv2.polylines(bev, [points], False, self.COLORS['centerline'], 2)
        
        # Draw lookahead point
        if steering:
            lh_x = int(w // 2 + steering.lookahead_x * scale)
            lh_y = int(h - steering.lookahead_y * scale)
            
            if 0 < lh_x < w and 0 < lh_y < h:
                cv2.circle(bev, (lh_x, lh_y), 8, self.COLORS['lookahead'], -1)
                cv2.line(bev, vehicle_pos, (lh_x, lh_y), self.COLORS['lookahead'], 1)
        
        # Add scale indicator
        cv2.putText(
            bev,
            "10m",
            (grid_spacing + 5, h - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (100, 100, 100),
            1,
        )
        
        return bev
