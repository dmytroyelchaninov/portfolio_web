import os
import numpy as np
import cv2 as cv
from PIL import Image
from ultralytics import YOLO
from sklearn.cluster import DBSCAN

model_dir = os.path.join(os.path.dirname(__file__), 'yolos')
model1 = os.path.join(model_dir, 'part1.pt')
model2 = os.path.join(model_dir, 'part2.pt')


class Game():
    def __init__(self):
        """Initializes the Main class with default error state."""
        self._error = False

class Board(Game):
    def __init__(self, img_path):
        super().__init__()
        self._img_path = img_path
        self._img = self._load_image()
        self._img_size = self._img_size()

    def _load_image(self):
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
        try:
            if not self._img_path.lower().endswith(valid_extensions):
                raise ValueError(f"File at path {self._img_path} is not a valid image file.")
        except ValueError:
            print("Wrong file extension")
            return None
        img = cv.imread(self._img_path, cv.IMREAD_COLOR)
        if img is None:
            try:
                os.remove(self._img_path)
                print(f"Image at path {self._img_path} could not be loaded and has been deleted.")
            except OSError as e:
                print(f"Error deleting file: {e}")
            raise ValueError(f"Image at path {self._img_path} could not be loaded.")
        
        return img

    def _img_size(self):
        return (self._img.shape[0], self._img.shape[1])
    
    def get_image(self):
        return self._img    

    def save_image(self, filename):
        cv.imwrite(filename, self._img)

class BoardTransform(Board):
    def __init__(self, board):
        self._img = board._img
        self._center = self._default_center()
        self._outer_ellipse = None
        self._inner_ellipse = None
        self._img_size = board._img_size
        

    def _default_center(self):
        return (self._img.shape[1] // 2, self._img.shape[0] // 2)

    def calc_circle_center_least_squares(self, points):
        x = points[:, 1]
        y = points[:, 0]
        
        x = points[:, 1]
        y = points[:, 0]

        center_x = np.mean(x)
        center_y = np.mean(y)

        return int(center_x), int(center_y)

    def crop_radial(self, center=None, crop_factor=0.95):
        if center is None:
            center = self._center       
        min_dimension = min(self._img.shape[0], self._img.shape[1])
        radius = int(min_dimension * crop_factor/2)
        mask = np.zeros_like(self._img, dtype=np.uint8)
        cv.circle(mask, center, radius, (255, 255, 255), thickness=-1)

        cropped_img = cv.bitwise_and(self._img, mask)
        self._img = cropped_img

        return cropped_img

    def crop_ellipse(self, ellipse, outer_padding_factor=0.03, inner_padding_factor=0.0):
        (center_x, center_y), (major_axis_length, minor_axis_length), angle = ellipse
        
        padded_major_axis = int(major_axis_length * (1 + outer_padding_factor))
        padded_minor_axis = int(minor_axis_length * (1 + outer_padding_factor))
        
        outer_padded_ellipse = ((center_x, center_y), (padded_major_axis, padded_minor_axis), angle)
        
        mask = np.zeros_like(self._img, dtype=np.uint8)
        cv.ellipse(mask, outer_padded_ellipse, (255, 255, 255), thickness=-1)
        
        inner_major_axis = int(major_axis_length * inner_padding_factor)
        inner_minor_axis = int(minor_axis_length * inner_padding_factor)
        
        inner_padded_ellipse = ((center_x, center_y), (inner_major_axis, inner_minor_axis), angle)

        if inner_padding_factor > 0:
            cv.ellipse(mask, inner_padded_ellipse, (0, 0, 0), thickness=-1)
        masked_img = cv.bitwise_and(self._img, mask)
        self._img = masked_img
        
        return self._img

    def crop_around(self, predictions, square_size=50):
        img = self._img
        mask = np.zeros_like(img)

        padding = square_size // 2
        
        for prediction in predictions:
            x, y = prediction
            x1 = max(0, int(x - padding))
            y1 = max(0, int(y - padding))
            x2 = min(img.shape[1], int(x + padding))
            y2 = min(img.shape[0], int(y + padding))
            
            mask[y1:y2, x1:x2] = img[y1:y2, x1:x2]
            
        return mask

    def resize_image(self, target_size=None, half_size=False, predictions=None):
        
        if target_size is None:
            target_size = self._img_size
        if half_size:
            target_size = (target_size[0] // 2, target_size[1] // 2)

        resize_factor_x = target_size[0] / self._img.shape[1]
        resize_factor_y = target_size[1] / self._img.shape[0]

        resized_image = cv.resize(self._img, target_size, interpolation=cv.INTER_AREA if resize_factor_x < 1 or resize_factor_y < 1 else cv.INTER_CUBIC)

        if len(resized_image.shape) == 2:
            resized_image = cv.cvtColor(resized_image, cv.COLOR_GRAY2BGR)

        new_center_x = int(self._center[0] * resize_factor_x)
        new_center_y = int(self._center[1] * resize_factor_y)
        new_center = (new_center_x, new_center_y)

        transformed_predictions = None
        if predictions is not None:
            transformed_predictions = []
            for x, y in predictions:
                new_x = int(x * resize_factor_x)
                new_y = int(y * resize_factor_y)
                transformed_predictions.append((new_x, new_y))

        self._img = resized_image
        self._center = new_center
        
        return transformed_predictions

    def final_crop(self, ellipse, predictions=None, padding=0.03):

        (center_x, center_y), (major_axis_length, minor_axis_length), angle = ellipse
        radius = int(max(major_axis_length, minor_axis_length) / 2 * (1 + padding))
        
        mask = np.zeros((self._img.shape[0], self._img.shape[1]), dtype=np.uint8)
        cv.circle(mask, (int(center_x), int(center_y)), radius, 255, thickness=-1)
        
        if len(self._img.shape) == 3:
            mask = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
        masked_img = cv.bitwise_and(self._img, mask)
        
        x, y, w, h = cv.boundingRect(cv.cvtColor(mask, cv.COLOR_BGR2GRAY))
        
        cropped_img = masked_img[y:y+h, x:x+w]
        self._img = cropped_img

        new_center_x = int(center_x - x)
        new_center_y = int(center_y - y)
        self._center = (new_center_x, new_center_y)

        self._outer_ellipse = (
            (new_center_x, new_center_y), 
            self._outer_ellipse[1], 
            self._outer_ellipse[2]
        )
        
        self._inner_ellipse = (
            (new_center_x, new_center_y), 
            self._inner_ellipse[1], 
            self._inner_ellipse[2]
        )

        transformed_predictions = None
        if predictions is not None:
            transformed_predictions = []
            for pred_x, pred_y in predictions:
                new_x = pred_x - x
                new_y = pred_y - y
                if 0 <= new_x < w and 0 <= new_y < h:
                    transformed_predictions.append((new_x, new_y))

        return transformed_predictions

    def _generate_gray_ranges(self, lower_scale=1.0, upper_scale=1.0):

        base_ranges = [
        (np.array([0, 0, 60]), np.array([180, 50, 200])),  # Dark gray to light gray
        (np.array([0, 0, 200]), np.array([180, 50, 240]))  # Light gray, stopping before white
        ]
        adjusted_ranges = []
        for lower, upper in base_ranges:
            adjusted_lower = np.clip(lower * lower_scale, 0, 255).astype(int)
            adjusted_upper = np.clip(upper * upper_scale, 0, 254).astype(int)
            adjusted_ranges.append((adjusted_lower, adjusted_upper))
        return adjusted_ranges

    def _generate_green_ranges(self, lower_scale=1.0, upper_scale=1.0):
        base_ranges = [
            (np.array([35, 50, 50]), np.array([85, 255, 255])),   # Main green range
            (np.array([30, 40, 40]), np.array([90, 255, 255])),   # Slightly wider range
        ]
        adjusted_ranges = []
        for lower, upper in base_ranges:
            adjusted_lower = np.clip(lower * lower_scale, 0, 255).astype(int)
            adjusted_upper = np.clip(upper * upper_scale, 0, 255).astype(int)
            adjusted_ranges.append((adjusted_lower, adjusted_upper))
        return adjusted_ranges

    def _generate_red_ranges(self, lower_scale=1.0, upper_scale=1.0):
        lower_red1 = np.clip(np.array([0, 70, 50]) * lower_scale, 0, 255).astype(int)
        upper_red1 = np.clip(np.array([10, 255, 255]) * upper_scale, 0, 255).astype(int)
        lower_red2 = np.clip(np.array([170, 70, 50]) * lower_scale, 0, 255).astype(int)
        upper_red2 = np.clip(np.array([180, 255, 255]) * upper_scale, 0, 255).astype(int)
        adjusted_ranges = [(lower_red1, upper_red1), (lower_red2, upper_red2)]
        return adjusted_ranges

    def _generate_red_ranges_new(self, lower_scale=0.5, upper_scale=3.0, avoid_lower_scale=1.0, avoid_upper_scale=1.0):
        target_colors = [
            [216, 72, 72],  # d84848
            [216, 62, 62],  # d83e3e
            [214, 68, 69],  # d64445
            [151, 40, 47],  # 97282f
            [197, 81, 81],  # c55151
            [212, 74, 72],  # d44a48
            [239, 71, 70],  # ef4746
            [203, 64, 67],  # cb4043
            [221, 72, 76],  # dd484c
            [212, 63, 65],  # d43f41
        ]
        avoid_colors = [
            [254, 115, 112],  # fe7370
            [130, 87, 78],    # 82574e
            [99, 59, 51],     # 633b33
            [231, 70, 85],    # e74655
            [217, 72, 75],    # d9484b
        ]

        adjusted_ranges = []
        for color in target_colors:
            hsv_color = cv.cvtColor(np.uint8([[color]]), cv.COLOR_RGB2HSV)[0][0]
            lower_bound = np.clip(hsv_color * lower_scale, 0, 255).astype(int)
            upper_bound = np.clip(hsv_color * upper_scale, 0, 255).astype(int)
            adjusted_ranges.append((lower_bound, upper_bound))

        filtered_ranges = []
        for lower, upper in adjusted_ranges:
            intersects = False
            for avoid_color in avoid_colors:
                avoid_hsv = cv.cvtColor(np.uint8([[avoid_color]]), cv.COLOR_RGB2HSV)[0][0]
                avoid_lower_bound = np.clip(avoid_hsv * avoid_lower_scale, 0, 255).astype(int)
                avoid_upper_bound = np.clip(avoid_hsv * avoid_upper_scale, 0, 255).astype(int)
                if (np.all(lower <= avoid_upper_bound) and np.all(upper >= avoid_lower_bound)):
                    intersects = True
                    break
            if not intersects:
                filtered_ranges.append((lower, upper))

        return filtered_ranges


    def _generate_blue_ranges(self, lower_scale=1.0, upper_scale=1.0):
        base_ranges = [
            (np.array([90, 50, 70]), np.array([130, 255, 255])),  # Main blue range
            (np.array([85, 50, 50]), np.array([135, 255, 255])),  # Slightly wider range
        ]
        adjusted_ranges = []
        for lower, upper in base_ranges:
            adjusted_lower = np.clip(lower * lower_scale, 0, 255).astype(int)
            adjusted_upper = np.clip(upper * upper_scale, 0, 255).astype(int)
            adjusted_ranges.append((adjusted_lower, adjusted_upper))
        return adjusted_ranges
    
    def overlay_mask_on_image(self, image, mask, alpha=0.5, color=(0, 255, 0)):
        mask_colored = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)

        mask_colored[:, :, 0] = np.where(mask > 0, color[0], 0)
        mask_colored[:, :, 1] = np.where(mask > 0, color[1], 0)
        mask_colored[:, :, 2] = np.where(mask > 0, color[2], 0)

        overlay = cv.addWeighted(image, 1, mask_colored, alpha, 0)
        overlay_img = cv.cvtColor(overlay, cv.COLOR_BGR2RGB)

        return overlay_img

    def apply_masks(self, colors, lower_scale=1.0, upper_scale=1.0):
        
        if len(self._img.shape) == 2:
            self._img = cv.cvtColor(self._img, cv.COLOR_GRAY2BGR)

        hsv = cv.cvtColor(self._img, cv.COLOR_BGR2HSV)
        combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

        for color in colors:
            if color == 'gray':
                ranges = self._generate_gray_ranges(lower_scale, upper_scale)
            elif color == 'green':
                ranges = self._generate_green_ranges(lower_scale, upper_scale)

            elif color == 'red':

                # ranges = self._generate_red_ranges(lower_scale, upper_scale)
                # NO RANGES, DONT TOUCH !!!
                ranges = self._generate_red_ranges()

            elif color == 'blue':
                ranges = self._generate_blue_ranges(lower_scale, upper_scale)
            else:
                raise ValueError("Unsupported color. Use 'gray', 'green', 'red', or 'blue'.")

            for lower, upper in ranges:
                mask = cv.inRange(hsv, lower, upper)
                combined_mask = cv.bitwise_or(combined_mask, mask)

        self._img = cv.bitwise_and(self._img, self._img, mask=combined_mask)
        
        return self._img, combined_mask
    
    def gray_thresh(self, threshold=30):
        gray_image = cv.cvtColor(self._img, cv.COLOR_BGR2GRAY)
        _, thresh = cv.threshold(gray_image, threshold, 255, cv.THRESH_BINARY)

        return thresh

    def find_center(self, min_radius=13, max_radius=30, param1=70, param2=20):
        self._img = cv.cvtColor(self._img, cv.COLOR_BGR2GRAY)
        circles = cv.HoughCircles(
            self._img,
            cv.HOUGH_GRADIENT,
            dp=1,
            minDist=1,
            param1=param1,
            param2=param2,
            minRadius=min_radius,
            maxRadius=max_radius)
        if circles is None:
            return None, None

        circles = np.uint16(np.around(circles))
        centers = [(circle[0], circle[1]) for circle in circles[0, :]]
        sum_x = sum([c[0] for c in centers])
        sum_y = sum([c[1] for c in centers])
        num_circles = len(centers)
        center_mean = (int(sum_x / num_circles), int(sum_y / num_circles))
        radii = [circle[2] for circle in circles[0, :]]
        radius_mean = int(np.mean(radii))
        self._center = center_mean   

        return center_mean, radius_mean, circles
    
    def dbscan_clustering(self, eps=7, min_samples=10, plot=False, threshold=30):
        thresh = self.gray_thresh(threshold=threshold)
        coords = cv.findNonZero(thresh)

        if coords is not None:
            coords = coords.reshape(-1, 2)
        else:
            coords = np.empty((0, 2))
        
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
        labels = db.labels_

        unique_labels, counts = np.unique(labels, return_counts=True)
        sorted_indices = np.argsort(-counts)
        
        if len(unique_labels) == 0:
            raise ValueError("No clusters found by DBSCAN.")
        
        largest_cluster_label = unique_labels[sorted_indices[0]]
        largest_cluster_coords = coords[labels == largest_cluster_label]

        if len(unique_labels) > 1:
            smaller_cluster_label = unique_labels[sorted_indices[1]]
            smaller_cluster_coords = coords[labels == smaller_cluster_label]
        else:
            smaller_cluster_coords = np.empty((0, 2))

        return largest_cluster_coords, smaller_cluster_coords

    def gather_extreme_points(self, coords=None, rotations=14):
        if coords is None:
            thresh = self.gray_thresh(threshold=30)
            coords = cv.findNonZero(thresh)
            if coords is not None:
                coords = coords.reshape(-1, 2)
            else:
                coords = np.empty((0, 2))
        all_extreme_points = []
        original_coords = coords.copy()
        for i in range(0,rotations):
            angle = i * 120 / rotations
            rotated_coords = self.rotate_coordinates_around_center(original_coords, angle, self._center)
            extreme_points = self.extract_extreme_points(rotated_coords)
            extreme_points_rotated_back = self.rotate_coordinates_around_center(extreme_points, -angle, self._center)
            all_extreme_points.append(extreme_points_rotated_back)
        all_extreme_points = np.vstack(all_extreme_points)

        return all_extreme_points

    def extract_extreme_points(self, coords):
        if len(coords) < 5:
            raise ValueError("Not enough points to extract extremities.")

        max_x_idx = np.argmax(coords[:, 1])
        min_x_idx = np.argmin(coords[:, 1])
        max_y_idx = np.argmax(coords[:, 0])
        min_y_idx = np.argmin(coords[:, 0])

        extreme_points = np.array([
            coords[max_x_idx],
            coords[min_x_idx],
            coords[max_y_idx],
            coords[min_y_idx]
        ])

        return extreme_points

    def rotate_coordinates_around_center(self, coords, angle, center):
        translated_coords = coords - center
        theta = np.radians(angle)
        cos_theta, sin_theta = np.cos(theta), np.sin(theta)
        rotation_matrix = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
        rotated_coords = translated_coords @ rotation_matrix.T
        rotated_coords += center

        return rotated_coords
    
    def fit_ellipse(self, coords, outer=True):
        if len(coords) < 5:
            raise ValueError("Not enough points to fit an ellipse.")

        coords = np.array(coords, dtype=np.float32)
        ellipse = cv.fitEllipse(coords)
        if outer:
            self._outer_ellipse = ellipse
        else:
            self._inner_ellipse = ellipse

        return ellipse

class PerspectiveTransform(BoardTransform):
    def __init__(self, board):
        self._center = board._center
        self._img = board._img
        self._outer_ellipse = board._outer_ellipse
        self._inner_ellipse = board._inner_ellipse
        self._img_size = board._img_size

    def generate_ellipse_points(self, ellipse, num_points=120):
        center = np.array(ellipse[0], dtype=np.float32)
        a = ellipse[1][0] / 2
        b = ellipse[1][1] / 2
        angle = np.radians(ellipse[2])

        points = []
        for t in np.linspace(0, 2*np.pi, num_points, endpoint=False):
            x = a * np.cos(t)
            y = b * np.sin(t)
            
            x_rot = center[0] + x * np.cos(angle) - y * np.sin(angle)
            y_rot = center[1] + x * np.sin(angle) + y * np.cos(angle)
            
            points.append([x_rot, y_rot])

        return np.array(points, dtype=np.float32)

    def transform_ellipses_to_circle(self, outer_ellipse, inner_ellipse, center, predictions):

        destination_center = center
        points_src_outer = self.generate_ellipse_points(outer_ellipse)
        points_src_outer = np.vstack([points_src_outer, outer_ellipse[0]])
        points_dst_outer = []
        for point in points_src_outer[:-1]:
            angle = np.arctan2(point[1] - destination_center[1], point[0] - destination_center[0])
            radius = ((outer_ellipse[1][0] + outer_ellipse[1][1]) / 4) * 1.0
            dst_x = destination_center[0] + radius * np.cos(angle)
            dst_y = destination_center[1] + radius * np.sin(angle)
            points_dst_outer.append([dst_x, dst_y])
        points_dst_outer.append(destination_center)
        points_dst_outer = np.array(points_dst_outer, dtype=np.float32)


        points_src_inner = self.generate_ellipse_points(inner_ellipse)
        points_src_inner = np.vstack([points_src_inner, inner_ellipse[0]])
        points_dst_inner = []
        for point in points_src_inner[:-1]:
            angle = np.arctan2(point[1] - destination_center[1], point[0] - destination_center[0])
            radius = ((inner_ellipse[1][0] + inner_ellipse[1][1]) / 4) * 1.0
            dst_x = destination_center[0] + radius * np.cos(angle)
            dst_y = destination_center[1] + radius * np.sin(angle)
            points_dst_inner.append([dst_x, dst_y])
        points_dst_inner.append(destination_center)
        points_dst_inner = np.array(points_dst_inner, dtype=np.float32)

        points_src = np.vstack([points_src_outer, points_src_inner])
        points_dst = np.vstack([points_dst_outer, points_dst_inner])

        homography_matrix, _ = cv.findHomography(points_src, points_dst)
        transformed_image = cv.warpPerspective(self._img, homography_matrix, (self._img.shape[1], self._img.shape[0]))

        center_homogeneous = np.array([center[0], center[1], 1.0], dtype=np.float32).reshape(-1, 1)
        transformed_center_homogeneous = homography_matrix @ center_homogeneous
        transformed_center = transformed_center_homogeneous[:2] / transformed_center_homogeneous[2]

        transformed_predictions = []
        for pred in predictions:
            pred_homogeneous = np.array([pred[0], pred[1], 1.0], dtype=np.float32).reshape(-1, 1)
            transformed_pred_homogeneous = homography_matrix @ pred_homogeneous
            transformed_pred = transformed_pred_homogeneous[:2] / transformed_pred_homogeneous[2]
            transformed_predictions.append((int(transformed_pred[0][0]), int(transformed_pred[1][0])))

        self._img = transformed_image
        self._center = (int(transformed_center[0][0]), int(transformed_center[1][0]))
        # print(self._img.shape)
        return transformed_image, transformed_predictions

class Scores():
    def __init__(self, board):
        self.board = board
        self.final_score = 0
        self._scores = []
        self.deltas = []

    def define_zones(self, show=True):
        img = self.board._img
        outer_ellipse = self.board._outer_ellipse
        inner_ellipse = self.board._inner_ellipse

        self.triple_outer = inner_ellipse
        self.double_outer = outer_ellipse
        self.triple_inner = (inner_ellipse[0], 
                        (inner_ellipse[1][0] * (97/107), inner_ellipse[1][1] * (97/107)), 
                        inner_ellipse[2])
        self.double_inner = (outer_ellipse[0], 
                        (outer_ellipse[1][0] * (159.5/170), outer_ellipse[1][1] * (159.5/170)), 
                        outer_ellipse[2])
        self.bulls_eye_inner_radius = int((outer_ellipse[1][0]/170)*3.6)
        self.bulls_eye_outer_radius = int((outer_ellipse[1][0]/170)*8.2)

        self.triple_inner_radius = int((self.triple_inner[1][0] + self.triple_inner[1][1]) / 4)
        self.triple_outer_radius = int((self.triple_outer[1][0] + self.triple_outer[1][1]) / 4)
        self.double_inner_radius = int((self.double_inner[1][0] + self.double_inner[1][1]) / 4)
        self.double_outer_radius = int((self.double_outer[1][0] + self.double_outer[1][1]) / 4)

        self._center = self.board._center
 
    def calculate_distance_and_angle(self, prediction):
        distance = np.sqrt((prediction[0] - self._center[0]) ** 2 + (prediction[1] - self._center[1]) ** 2)
        angle = np.degrees(np.arctan2(prediction[1] - self._center[1], prediction[0] - self._center[0]))
        angle = angle if angle >= 0 else angle + 360
        return distance, angle

    def determine_sector(self, angle, shift_angle):
        angle_per_sector = 360 / 20
        adjusted_angle = (angle + 9 + shift_angle) % 360
        sector_index = int(adjusted_angle // angle_per_sector)
        sector_scores = [6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5, 20, 1, 18, 4, 13]
        sector = sector_scores[sector_index]
        return sector

    def check_bulls_eye(self, distance):
        if distance <= self.bulls_eye_inner_radius:
            return 50
        elif self.bulls_eye_inner_radius <= distance <= self.bulls_eye_outer_radius:
            return 25
        else:
            return None

    def check_triple(self, distance):
        return self.triple_inner_radius <= distance <= self.triple_outer_radius

    def check_double(self, distance):
        deltas = [abs(distance - self.double_inner_radius), abs(self.double_outer_radius - distance)]
        if min(deltas) < 1:
            return True 
        return self.double_inner_radius <= distance <= self.double_outer_radius

    def get_scores(self, predictions, shift_angle=0):
        scores = []
        for prediction in predictions:
            distance, angle = self.calculate_distance_and_angle(prediction)
            if distance > self.double_outer_radius + 1:
                score = 0
            else:
                score = self.check_bulls_eye(distance)
                if score is None:
                    sector = self.determine_sector(angle, shift_angle=shift_angle)
                    if self.check_triple(distance):
                        score = sector * 3
                    elif self.check_double(distance):
                        score = sector * 2
                    else:
                        score = sector
            scores.append(score)
        self._scores = scores
        self.final_score += sum(scores)
        return scores

    def draw_distances_and_angles(self, predictions, show=False):
        img = self.board._img.copy()
        mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        cv.circle(mask, (int(self._center[0]), int(self._center[1])), int(self.double_outer_radius), 255, thickness=-1)
        if img.shape[2] == 3:
            img = cv.cvtColor(img, cv.COLOR_BGR2BGRA)
        img[:, :, 3] = np.where(mask == 255, 255, 0)
        
        occupied_boxes = []

        for prediction, score in zip(predictions, self._scores):
            distance, angle = self.calculate_distance_and_angle(prediction)
            text = f'{score}'
            font_scale = 1.0
            thickness = 2
            font = cv.FONT_HERSHEY_TRIPLEX
            text_size = cv.getTextSize(text, font, font_scale, thickness)[0]
            
            x = int(prediction[0]) + 15
            y = int(prediction[1]) - 15
            
            text_box = (x, y - text_size[1], x + text_size[0], y)
            
            if img.shape[2] == 4:
                color = (69, 69, 255, 255)
            else:
                color = (56, 40, 215)
            
            cv.putText(img, text, (x, y), font, font_scale, color, thickness, cv.LINE_AA)

            occupied_boxes.append(text_box)

        return img

    def _check_intersection(self, box1, box2):
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2

        return not (x1_max < x2_min or x2_max < x1_min or y1_max < y2_min or y2_max < y1_min)

def yolo_predict(image, model):
    results = model(image, verbose=False)
    centers = []
    n_classes = 0
    for result in results:
        boxes = result.boxes.xyxy
        labels = result.boxes.cls
        n_classes = len(boxes)

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = [int(coord) for coord in box]
            label = int(labels[i])
            confidence = result.boxes.conf[i]

            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            centers.append((center_x, center_y))

    return centers, n_classes

def yolo_crop(image, model, padding=25, padding_box=0, blackout_boxes=[]):
    results = model(image, verbose=False)
    n_classes = []
    metal_boxes = []
    class_0_count = 0
    class_1_count = 0
    
    for result in results:
        boxes = result.boxes.xyxy
        labels = result.boxes.cls
        
        class_0_count += (labels == 0).sum().item()
        class_1_count += (labels == 1).sum().item()
        
        metal_boxes.extend(boxes[labels == 1])

    n_classes = [class_0_count, class_1_count]

    mask = np.zeros_like(image)

    for box in metal_boxes:
        x1, y1, x2, y2 = [int(coord) for coord in box]
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(image.shape[1], x2 + padding)
        y2 = min(image.shape[0], y2 + padding)

        mask[y1:y2, x1:x2] = image[y1:y2, x1:x2]

    return mask, n_classes

def yolo_init(image):
    model = YOLO('./yolos/yolo_crop.pt')
    results = model(image)
    boxes = results[0].boxes
    centers = []

    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2
        centers.append((x_center.item(), y_center.item()))

    return centers

def predict(board, part1, part2):
    padding = 20
    img = board._img
    cropped_img, n_classes_initial = yolo_crop(img, part1, padding=padding)
    predictions, n_classes_predictions = yolo_predict(cropped_img, part2)
    if n_classes_predictions>min(n_classes_initial) or n_classes_predictions>3:
        while n_classes_predictions>min(n_classes_initial) or n_classes_predictions>3:
            cropped_img, n_classes_initial = yolo_crop(img, part1, padding=padding)
            predictions, n_classes_predictions = yolo_predict(cropped_img, part2)
            padding -= 2
    elif n_classes_predictions<min(n_classes_initial):
        while n_classes_predictions<min(n_classes_initial):
            cropped_img, n_classes_initial = yolo_crop(img, part1, padding=padding)
            predictions, n_classes_predictions = yolo_predict(cropped_img, part2)
            padding += 2
    if padding > 50 or padding < 4:
        print("Take picture from another angle")
        raise ValueError("Bad prediction")

    return predictions


def find_bulls_eye(board, crop_eye=0.25, min_radius=8, max_radius=30, param1=70, param2=20, plots=False, centered=False):

    if centered:
        try:
            initial_center = (board._img.shape[1] // 2, board._img.shape[0] // 2)
            crop_mask = BoardTransform(board)
            crop_mask.crop_radial(center=initial_center, crop_factor=crop_eye)
            crop_mask.apply_masks(colors=['red', 'green'], lower_scale=1.0, upper_scale=1.0)
            precise_center, radius, circles = crop_mask.find_center(min_radius=min_radius, max_radius=max_radius, param1=param1, param2=param2)

            return precise_center
        except ValueError:
            initial_center = board._center
    else:
        initial_center = board._center
    angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    for step in np.linspace(0.0, 0.15, 6):
        iter = 0
        for angle in angles:
            radius_step = min(board._img.shape[:2]) * step
            iter += 1
            if iter > 8:
                print("I can't find the center. Please take another image.")
                raise ValueError("Center not found.")
            dx = radius_step * np.cos(angle)
            dy = radius_step * np.sin(angle)
            new_center = (int(initial_center[0] + dx), int(initial_center[1] + dy))

            crop_mask = BoardTransform(board)
            try:
                crop_mask.crop_radial(center=new_center, crop_factor=crop_eye)
                crop_mask.apply_masks(colors=['red', 'green'], lower_scale=1.0, upper_scale=1.0)
                bulls_eye, _ = crop_mask.dbscan_clustering(plot=False, eps=5, min_samples=10, threshold=10)
                center = crop_mask.calc_circle_center_least_squares(bulls_eye)

                try:
                    crop_mask_res = BoardTransform(board)
                    crop_mask_res.crop_radial(center=center, crop_factor=crop_eye)
                    crop_mask_res.apply_masks(colors=['red', 'green'], lower_scale=1.0, upper_scale=1.0)
                    precise_center, radius, circles = crop_mask_res.find_center(min_radius=min_radius, max_radius=max_radius, param1=param1, param2=param2)


                    return precise_center
                except ValueError:
                    return center
            except ValueError:
                raise ValueError("Center not found.")
    print("Tough to find a center... Let's give another try...")
    return initial_center


def ellipses(board, eps=10, min_samples=7, threshold=10):
    dbscan = BoardTransform(board)
    dbscan.apply_masks(colors=['red', 'green'], lower_scale=1.0, upper_scale=1.0)
    outer_ring, inner_ring = dbscan.dbscan_clustering(eps=10, min_samples=7, threshold=0)
 
    extreme_points = board.gather_extreme_points(coords=outer_ring)
    outer_ellipse = board.fit_ellipse(extreme_points, outer=True)
    extreme_points = board.gather_extreme_points(coords=inner_ring)
    inner_ellipse = board.fit_ellipse(extreme_points, outer=False) 

    return board, outer_ellipse, inner_ellipse

# Singleton pattern!

# That's an example why I love python.
# This pattern clearly shows how Python unites ideas of
# both Object-oriented and Functional programming 
# in itself!

class DbscanParams():
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(DbscanParams, cls).__new__(cls)   # ?
            # Can you make so in Java with its constructors? :)
            # How do you think, what is super(), if no parent class exists?
            # Answer: 
            # If you aren't familiar, at the first look it seems that when 
            # .__new__ is called inside .__new__,
            # we call function recursively. But that .__new__ doesnt belong to the the same class, 
            # super() returns PARENT class which even currently doesnt exist! 
            # Actually, it calls object.__new__  - which is literally default object ITSELF!
            # Next this object assinged to cls.__instance,
            # which is THIS class attribute, which is shared between all instances (objects) of this class.
            # Finally, _instance is not None 
            # What a power of Python!
        return  cls._instance
    
    def __init__(self, eps, min_samples, threshold):
        if not hasattr(self, 'exists'):
            self.eps = eps
            self.min_samples = min_samples
            self.threshold = threshold
            self.exists = True

# This pattern saved me 60% of execution time   
def find_ellipse_res(board, eps=10, min_samples=7, threshold=10, plot_ellipse=False, padding=0.03):
    def __find_ellipse1(board, eps=10, min_samples=7, threshold=10, plot_ellipse=False, padding=0.03):
        first_try = None
        second_try = None
        try:
            eps_ratio_semis = []
            ratio_outer_inner = None
            while ratio_outer_inner is None or ratio_outer_inner > 1.015 and eps > 4:
                board, outer_ellipse, inner_ellipse = ellipses(board, eps=eps, min_samples=min_samples, threshold=threshold)
                ratio_outer_inner = max((outer_ellipse[1][0]/outer_ellipse[1][1]),(inner_ellipse[1][0]/inner_ellipse[1][1]))\
                    /min((outer_ellipse[1][0]/outer_ellipse[1][1]),(inner_ellipse[1][0]/inner_ellipse[1][1]))
                ratio_outer = (max(outer_ellipse[1][0],outer_ellipse[1][1]))/(min(outer_ellipse[1][0],outer_ellipse[1][1]))
                eps_ratio_semis.append((eps, ratio_outer_inner, ratio_outer))
                eps -= 0.2                

        except (IndexError, ValueError):
            print(f"Unsure with results...Let's try to adjust parameters little bit...")
            filtered = [i for i in eps_ratio_semis if i[2]<1.5] # Angles limited to 1.5 semis ratio
            if filtered:
                eps = min(filtered, key=lambda x: x[1])[0] # Choosing best eps based on similarity of ellipses
                # print(f"Chosen eps: {eps}, semis ratio: {min(filtered, key=lambda x: x[1])[1]}, ratio_outer: {min(filtered, key=lambda x: x[1])[2]}")
                first_try = [eps, min_samples, min(filtered, key=lambda x: x[1])[1], min(filtered, key=lambda x: x[1])[2]]
            else:
                filtered = [i for i in eps_ratio_semis if i[2]<2] # Angles limited to 2 semis ratio
                if filtered:
                    eps = min(filtered, key=lambda x: x[1])[0] # Choosing best eps based on similarity of ellipses
                    # print(f"Chosen eps: {eps}, semis ratio: {min(filtered, key=lambda x: x[1])[1]}, ratio_outer: {min(filtered, key=lambda x: x[1])[2]}")
                    first_try = [eps, min_samples, min(filtered, key=lambda x: x[1])[1], min(filtered, key=lambda x: x[1])[2]]
                else:
                    print("One more try...")
        
            # Now iterate over min_samples
            try: 
                min_samples_ratio_semis = []
                eps = first_try[0]
                while ratio_outer_inner > 1.03 and min_samples < 100:
                    board, outer_ellipse, inner_ellipse = ellipses(board, eps=eps, min_samples=min_samples, threshold=threshold)
                    ratio_outer_inner = max((outer_ellipse[1][0]/outer_ellipse[1][1]),(inner_ellipse[1][0]/inner_ellipse[1][1]))\
                        /min((outer_ellipse[1][0]/outer_ellipse[1][1]),(inner_ellipse[1][0]/inner_ellipse[1][1]))
                    ratio_outer = (max(outer_ellipse[1][0],outer_ellipse[1][1]))/(min(outer_ellipse[1][0],outer_ellipse[1][1]))
                    min_samples_ratio_semis.append((min_samples, ratio_outer_inner, ratio_outer, threshold))
                    if threshold > 0:
                        threshold -= 1
                    min_samples += 0.5
                filtered = [i for i in min_samples_ratio_semis if i[2]<2]
                min_samples = min(filtered, key=lambda x: x[1])[0]
                second_try = [eps, min_samples, min(min_samples_ratio_semis, key=lambda x: x[1])[1], min(min_samples_ratio_semis, key=lambda x: x[1])[2]]
            
            except (IndexError, ValueError):
                filtered = [i for i in min_samples_ratio_semis if i[2]<2]
                print("Hmm...")
                if filtered:
                    min_samples = min(filtered, key=lambda x: x[1])[0]
                    second_try = [first_try[0], min_samples, min(filtered, key=lambda x: x[1])[1], min(filtered, key=lambda x: x[1])[2]]
                else:
                    print("Cant detect board, please take image from another angle")
                    raise ValueError("Board not detected")

        if (first_try is not None) and (second_try is not None):
            eps = second_try[0]
            min_samples = second_try[1]
            threshold = second_try[3]
        board, outer_ellipse, inner_ellipse = ellipses(board, eps=eps, min_samples=min_samples, threshold=threshold)
        board._outer_ellipse = outer_ellipse
        board._inner_ellipse = inner_ellipse
        board.crop_ellipse(outer_ellipse, outer_padding_factor=padding)

        # Here we go, sigleton pattern usage:
        _ = DbscanParams(eps=eps, min_samples=min_samples, threshold=threshold)
        # We will not use the name of the object
        return board
    
    def __find_ellipse2(board, eps=10, min_samples=7, threshold=10, plot_ellipse=False, padding=0.03):

        board, outer_ellipse, inner_ellipse = ellipses(board, eps=eps, min_samples=min_samples, threshold=threshold)
                
        board._outer_ellipse = outer_ellipse
        board._inner_ellipse = inner_ellipse
        board.crop_ellipse(outer_ellipse, outer_padding_factor=padding)

        return board
    
    if DbscanParams._instance is None:
        return __find_ellipse2(board, eps=eps, min_samples=min_samples, threshold=threshold, plot_ellipse=plot_ellipse, padding=padding)
    else:
        params = DbscanParams._instance
        eps = params.eps
        min_samples = params.min_samples
        threshold = params.threshold
        return __find_ellipse2(board, eps=eps, min_samples=min_samples, threshold=threshold, plot_ellipse=plot_ellipse, padding=padding)

def find_ellipse(board, eps=10, min_samples=7, threshold=10, plot_ellipse=False):
    board, outer_ellipse, inner_ellipse = ellipses(board, eps=eps, min_samples=min_samples, threshold=threshold)
    board._outer_ellipse = outer_ellipse
    board._inner_ellipse = inner_ellipse
    board.crop_ellipse(outer_ellipse, outer_padding_factor=0.035)
    return board


def initial_prepare(board, crop_eye=0.25, crop_scale=1.0, size=None):
    """Receives basic BoardTransform with ._img
    Resizes image if needed
    Finds center, crops around it with circle
    Should be passed to find_ellipse further
    Return cropped board back with defined center, if center found"""
    if size:
        board.resize_image(target_size=size)
    try:
        bulls_eye = find_bulls_eye(board, crop_eye=crop_eye)
    except Exception as e:
        print(f"Center was not found: {e}")
        raise ValueError("Center was not found")
    if bulls_eye:
        board._center = bulls_eye
        board.crop_radial(bulls_eye, crop_factor=1.0*crop_scale)
    else:
        board.crop_radial(crop_factor=1.0*crop_scale)
    
    return board

def transform_perspective(board, predictions, plots=False, crop_eye=0.25, shifts=None):
    """Receives prepared board with defined center, ellipses
    Returns transformed image, with refined center, if found"""
    transform = PerspectiveTransform(board)
    _,predictions = transform.transform_ellipses_to_circle(transform._outer_ellipse, transform._inner_ellipse, transform._center, predictions)
    try:
        new_center = find_bulls_eye(transform, crop_eye=crop_eye)
        old_center = transform._center
        transform._center = new_center

        shift_x = new_center[0] - old_center[0]
        shift_y = new_center[1] - old_center[1]
        if shifts is not None:
            shifts.append((shift_x, shift_y))
        compensated_predictions = []
        for pred in predictions:
            compensated_x = pred[0] + shift_x/2
            compensated_y = pred[1] + shift_y/2
            compensated_predictions.append((compensated_x, compensated_y))

        predictions = compensated_predictions

    except (ValueError, TypeError):
        print("Center was not refined")

    return transform, predictions, shifts

def iterative_transform(board,
                    predictions,
                    eps=10, min_samples=7, 
                    threshold=10, 
                    iterations=3,
                    crop_eye=0.25,
                    plot_steps=False):

    shifts = []
    for i in range(1,iterations+1):
        
        # 0.6 sec for 1 block of find_ellipse and transform_perspective
        board = find_ellipse(board, eps=eps, min_samples=min_samples, threshold=threshold, plot_ellipse=False)
        board, predictions, shifts = transform_perspective(board, predictions, plots=plot_steps, crop_eye=crop_eye, shifts=shifts)
        
        
        if i == 3 or i==4 or i == 5:
            model_dir = os.path.join(os.path.dirname(__file__), 'yolos')
            model2 = os.path.join(model_dir, 'part2.pt')
            part2 = YOLO(model2)
            cropped = board.crop_around(predictions, square_size=40)
            predictions_new, n_classes_predictions = yolo_predict(cropped, part2)
            if len(predictions_new) == len(predictions):
                predictions = predictions_new
                if i==3:
                    print('Hold on, we are getting closer...')
                if i==4:
                    print('Almost there...')
        if i == iterations-1:
            outer_ellipse = board._outer_ellipse
            inner_ellipse = board._inner_ellipse
            outer_ellipse = (
                outer_ellipse[0],
                (min(outer_ellipse[1][0], outer_ellipse[1][1]), min(outer_ellipse[1][0], outer_ellipse[1][1])),
                outer_ellipse[2]
            )
            inner_ellipse = (
                inner_ellipse[0],
                (min(inner_ellipse[1][0], inner_ellipse[1][1]), min(inner_ellipse[1][0], inner_ellipse[1][1])),
                inner_ellipse[2]
            )
            board._outer_ellipse = outer_ellipse
            board._inner_ellipse = inner_ellipse
            predictions = board.final_crop(board._outer_ellipse, predictions=predictions, padding=0.01)
            new_center = find_bulls_eye(board, crop_eye=0.25, plots=False, centered=True, max_radius=20, min_radius=13, param1=50, param2=15)
            old_center = board._center
            board._center = new_center
      
    total_shift_x = 0
    total_shift_y = 0
    for shift in shifts:
        total_shift_x += shift[0]
        total_shift_y += shift[1]
    avg_shift_x = total_shift_x / len(shifts)
    avg_shift_y = total_shift_y / len(shifts)

    compensated_predictions = []
    for pred in predictions:
        compensated_x = pred[0] + avg_shift_x
        compensated_y = pred[1] + avg_shift_y
        compensated_predictions.append((compensated_x, compensated_y))
    predictions = compensated_predictions

    board = find_ellipse(board, eps=eps, min_samples=min_samples, threshold=threshold, plot_ellipse=False)
    
    return board, predictions

def find20_center(board):
    dbscan = BoardTransform(board)
    angles_to_crop = [261-8, 279+8]
    radius_outer = (board._outer_ellipse[1][0] + board._outer_ellipse[1][1])/4
    radius_inner = (board._inner_ellipse[1][0] + board._inner_ellipse[1][1])/4
    radiuses_to_crop = [(radius_outer + radius_inner)/2, radius_outer]
    center = board._center
    
    angles_to_crop_rad = [np.deg2rad(angle) for angle in angles_to_crop]
    
    mask = np.zeros(board._img.shape[:2], dtype=np.uint8)

    cv.ellipse(mask, center, 
               (int(radiuses_to_crop[1]), int(radiuses_to_crop[1])), 
               0, 
               np.rad2deg(angles_to_crop_rad[0]), 
               np.rad2deg(angles_to_crop_rad[1]), 
               255, 
               thickness=-1)
    cv.ellipse(mask, center, 
               (int(radiuses_to_crop[0]), int(radiuses_to_crop[0])), 
               0, 
               np.rad2deg(angles_to_crop_rad[0]), 
               np.rad2deg(angles_to_crop_rad[1]), 
               0, 
               thickness=-1)

    cropped_img = cv.bitwise_and(dbscan._img, dbscan._img, mask=mask)
    dbscan._img = cropped_img

    dbscan.apply_masks(colors=['red'], lower_scale=1.0, upper_scale=1.0)
    ring20, _ = dbscan.dbscan_clustering(eps=10, min_samples=7, threshold=10, plot=False)

    def find_center_point(coords):
        x_coords = [coord[0] for coord in coords]
        y_coords = [coord[1] for coord in coords]

        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)

        center_x = (min_x + max_x) // 2
        center_y = (min_y + max_y) // 2

        center_point = (center_x, center_y)
        
        return center_point

    center20 = find_center_point(ring20)
    
    return center20

def transform(img_path, predictions, eps=10, min_samples=7, threshold=10, crop_scale=1.0, crop_eye=0.25, size_transform=(1100,1100), iterations=5):
    print(f"Processing image...")
    try:
        original = Board(img_path)
        board = BoardTransform(original)
        board = initial_prepare(board, crop_eye=crop_eye, crop_scale=crop_scale, size=size_transform)

        points = yolo_init(board._img)
        board.fit_ellipse(points, outer=True)
        board.crop_ellipse(board._outer_ellipse, outer_padding_factor=0.08)

        # img_rgb = cv.cvtColor(board._img, cv.COLOR_BGR2RGB)
        # plt.imshow(img_rgb)
        # plt.axis('off')
        # plt.show()    

# JUST OKAY HERE
        transformed, predictions = iterative_transform(
            board,
            predictions,
            iterations=iterations,
            eps=eps, min_samples=min_samples, threshold=threshold,
            crop_eye=crop_eye,
        )
        outer_ellipse = transformed._outer_ellipse
        inner_ellipse = transformed._inner_ellipse
        outer_ellipse = (
            outer_ellipse[0],
            (min(outer_ellipse[1][0], outer_ellipse[1][1]), min(outer_ellipse[1][0], outer_ellipse[1][1])),
            outer_ellipse[2]
        )
        inner_ellipse = (
            inner_ellipse[0],
            (min(inner_ellipse[1][0], inner_ellipse[1][1]), min(inner_ellipse[1][0], inner_ellipse[1][1])),
            inner_ellipse[2]
        )
        transformed._outer_ellipse = outer_ellipse
        transformed._inner_ellipse = inner_ellipse
        
        predictions = transformed.final_crop(transformed._outer_ellipse, predictions=predictions, padding=0)
        
        new_center = find_bulls_eye(transformed, crop_eye=0.25, plots=False, centered=True, max_radius=20, min_radius=13, param1=50, param2=15)
        transformed._center = new_center
        
        # THIS IS TO COMPENSATE ROTATION OF THE BOARD FURTHER. WORKS WITHIN 8 DEGREES.
        center20 = find20_center(transformed)

        return transformed, predictions, center20
    
    except Exception as e:
        print(f"Error processing {img_path}: {str(e)}")
        raise ValueError("Error processing image")

def process_image(img_path, size=(1000, 1000), accuracy=0.05, iterations=5, show=False, test=False, test_n=None):
    if test:
        img_path = f'./test/{test_n}.jpg'

    original = Board(img_path)
    board = BoardTransform(original)
    size = (1000, 1000)

    board = initial_prepare(board, size=size, crop_scale=1.3)

    # Models
    model_dir = os.path.join(os.path.dirname(__file__), 'yolos')
    model1 = os.path.join(model_dir, 'part1.pt')
    model2 = os.path.join(model_dir, 'part2.pt')
    part1 = YOLO(model1)
    part2 = YOLO(model2)
    predictions = predict(board, part1, part2)


    transformed, predictions, center20 = transform(img_path, predictions, size_transform=size, iterations=5)
    
    angle = np.degrees(np.arctan2(center20[1] - transformed._center[1], center20[0] - transformed._center[0]))
    angle = angle if angle >= 0 else angle + 360
    shift_angle = 270 - angle
    scores = Scores(transformed)
    scores.define_zones(show=False)
    scores.get_scores(predictions, shift_angle=shift_angle)
    score = scores._scores

    img = scores.draw_distances_and_angles(predictions)

    if img.shape[2] == 4:
        processed_image_rgba = cv.cvtColor(img, cv.COLOR_BGRA2RGBA)
    else:
        processed_image_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    processed_image = Image.fromarray(processed_image_rgba if img.shape[2] == 4 else processed_image_rgb)

    processed_image_path = img_path.replace('.jpg', '_processed.png')

    processed_image.save(processed_image_path, format='PNG')


    return processed_image_path, score


if __name__ == '__main__':
    pass
    # for i in range(1, 2):
    #     img_path = f'./test_images/{i}.jpg'
    #     processed_image_path, score = process_image(img_path, test=False)
    #     print(f"Image: {i}.jpg")
    #     print(f"Score: {score}")
    #     print(f"Processed image saved as {processed_image_path}\n")


    # img_path = './test_images/5.jpg'
    # processed_image_path, score = process_image(img_path, test=False)
    # print(f"Score: {score}")
    # print(f"Processed image saved as {processed_image_path}")