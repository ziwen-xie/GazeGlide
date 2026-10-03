"""Map webcam eye features to normalized coordinates on one calibrated display."""
import numpy as np

TARGETS = [(x, y) for y in (.15, .5, .85) for x in (.15, .5, .85)]

def head_features(matrix, center, face_width):
    """Camera-space yaw/pitch in radians and normalized face position/size."""
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all() or face_width <= 0:
        raise ValueError('Head pose is unavailable.')
    forward = matrix[:3, 2]
    yaw = np.arctan2(forward[0], forward[2])
    pitch = -np.arctan2(forward[1], np.hypot(forward[0], forward[2]))
    return np.array([yaw, pitch, *center, face_width])

def head_adjustment(current, reference, strength=1.):
    """Experimental eye-feature correction; gains are approximate, not camera calibration."""
    delta = np.asarray(current) - np.asarray(reference)
    # Large pose/distance changes are beyond this local approximation.
    ratio = current[4] / reference[4]
    if np.any(np.abs(delta[:2]) > .65) or not .65 < ratio < 1.5:
        return None
    rotation = delta[:2] * .20
    translation = delta[2:4] / reference[4] * .08
    return float(strength) * (rotation + translation)

class GazeMap:
    def __init__(self, features, targets=TARGETS):
        features = np.asarray(features, dtype=float)
        targets = np.asarray(targets, dtype=float)
        if features.shape != (9, 2) or not np.isfinite(features).all():
            raise ValueError('Nine clear eye measurements are required.')
        self.mean = features.mean(axis=0)
        self.scale = features.std(axis=0)
        if np.any(self.scale < .004):
            raise ValueError('Eye movement was too small. Improve lighting and try again.')
        design = np.column_stack(((features-self.mean)/self.scale, np.ones(9)))
        if np.linalg.cond(design) > 20:
            raise ValueError('Horizontal and vertical gaze were not distinct. Please retry.')
        # Held-out targets detect calibrations that merely fit noisy eye measurements.
        errors = []
        for i in range(9):
            mask = np.arange(9) != i
            weights = np.linalg.lstsq(design[mask], targets[mask], rcond=None)[0]
            errors.append(np.linalg.norm(design[i] @ weights - targets[i]))
        self.error = float(np.mean(errors))
        if self.error > .23:
            raise ValueError('Calibration was inconsistent. Keep your head still and follow each dot again.')
        self.weights = np.linalg.lstsq(design, targets, rcond=None)[0]

    def predict(self, features):
        row = np.r_[(np.asarray(features)-self.mean)/self.scale, 1.]
        return np.clip(row @ self.weights, 0, 1)
