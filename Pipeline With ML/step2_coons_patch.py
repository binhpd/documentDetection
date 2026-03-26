import cv2
import numpy as np

class CoonsPatchDewarper:
    def __init__(self, output_width=None, output_height=None):
        self.output_width = output_width
        self.output_height = output_height

    def _get_contour_segment(self, contour, idxA, idxB, other1, other2):
        n = len(contour)
        if idxA <= idxB:
            path1 = list(range(idxA, idxB + 1))
        else:
            path1 = list(range(idxA, n)) + list(range(0, idxB + 1))
            
        path1_set = set(path1)
        if other1 in path1_set or other2 in path1_set:
            if idxA >= idxB:
                path2 = list(range(idxA, idxB - 1, -1))
            else:
                path2 = list(range(idxA, -1, -1)) + list(range(n - 1, idxB - 1, -1))
            return contour[path2]
        else:
            return contour[path1]

    def _resample_curve(self, curve, num_points):
        # curve shape: (N, 2)
        if len(curve) == 0:
            return np.zeros((num_points, 2), dtype=np.float32)
        if len(curve) == 1:
            return np.tile(curve[0], (num_points, 1)).astype(np.float32)
            
        diffs = np.diff(curve, axis=0)
        dists = np.linalg.norm(diffs, axis=1)
        cum_dists = np.zeros(len(curve))
        cum_dists[1:] = np.cumsum(dists)
        
        total_dist = cum_dists[-1]
        if total_dist == 0:
            return np.tile(curve[0], (num_points, 1)).astype(np.float32)
            
        t_target = np.linspace(0, total_dist, num_points)
        
        resampled_x = np.interp(t_target, cum_dists, curve[:, 0])
        resampled_y = np.interp(t_target, cum_dists, curve[:, 1])
        
        return np.column_stack((resampled_x, resampled_y)).astype(np.float32)

    def _order_points(self, pts):
        pts = np.array(pts, dtype=np.float32)
        rect = np.zeros((4, 2), dtype=np.float32)
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def dewarp_via_contour(self, img, mask, corners, save_prefix=None):
        """
        Warp an image based on the curved mask contours to a completely flat rectangle.
        corners: 4 document corners (any order).
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return img
        
        if save_prefix is not None:
            vis_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            cv2.imwrite(f"{save_prefix}_step2_coons_1a_mask.jpg", vis_mask)

        c = max(contours, key=cv2.contourArea).squeeze()
        if c.ndim == 1:
            c = c.reshape(-1, 2)
            
        TL, TR, BR, BL = self._order_points(corners)
        
        # Find closest contour points to the mathematical corners
        i_TL = int(np.argmin(np.linalg.norm(c - TL, axis=1)))
        i_TR = int(np.argmin(np.linalg.norm(c - TR, axis=1)))
        i_BR = int(np.argmin(np.linalg.norm(c - BR, axis=1)))
        i_BL = int(np.argmin(np.linalg.norm(c - BL, axis=1)))
        
        P_TL = c[i_TL]
        P_TR = c[i_TR]
        P_BR = c[i_BR]
        P_BL = c[i_BL]

        # Extract segments
        top_curve = self._get_contour_segment(c, i_TL, i_TR, i_BR, i_BL)
        right_curve = self._get_contour_segment(c, i_TR, i_BR, i_TL, i_BL)
        bottom_curve = self._get_contour_segment(c, i_BL, i_BR, i_TL, i_TR)
        left_curve = self._get_contour_segment(c, i_TL, i_BL, i_TR, i_BR)
        
        width_top = np.linalg.norm(P_TL - P_TR)
        width_bottom = np.linalg.norm(P_BL - P_BR)
        height_left = np.linalg.norm(P_TL - P_BL)
        height_right = np.linalg.norm(P_TR - P_BR)
        
        W = self.output_width if self.output_width else int(max(width_top, width_bottom))
        H = self.output_height if self.output_height else int(max(height_left, height_right))
        
        if W <= 0 or H <= 0:
            return img
        
        T = self._resample_curve(top_curve, W)
        B = self._resample_curve(bottom_curve, W)
        L = self._resample_curve(left_curve, H)
        R = self._resample_curve(right_curve, H)
        
        # Coons Patch mapping
        u = np.linspace(0, 1, W, dtype=np.float32)
        v = np.linspace(0, 1, H, dtype=np.float32)
        ug, vg = np.meshgrid(u, v)
        
        ug_3d = ug[..., np.newaxis]
        vg_3d = vg[..., np.newaxis]
        
        T_grid = np.tile(T, (H, 1, 1))
        B_grid = np.tile(B, (H, 1, 1))
        L_grid = np.tile(L[:, np.newaxis, :], (1, W, 1))
        R_grid = np.tile(R[:, np.newaxis, :], (1, W, 1))
        
        term1 = (1 - vg_3d) * T_grid
        term2 = vg_3d * B_grid
        term3 = (1 - ug_3d) * L_grid
        term4 = ug_3d * R_grid
        
        term5 = (1 - ug_3d) * (1 - vg_3d) * P_TL + \
                ug_3d * (1 - vg_3d) * P_TR + \
                (1 - ug_3d) * vg_3d * P_BL + \
                ug_3d * vg_3d * P_BR
                
        map_coords = term1 + term2 + term3 + term4 - term5
        
        map_x = map_coords[..., 0].astype(np.float32)
        map_y = map_coords[..., 1].astype(np.float32)
        
        dewarped = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
        return dewarped
