"""Calibrated gaze classification and dwell gate, independent of the camera."""
class ScrollControl:
    def __init__(self, up, center, down, dwell=0.7, interval=0.14):
        if abs(up - center) < 0.004 or abs(down - center) < 0.004:
            raise ValueError('Eye movement was too small. Try calibration again.')
        if (up - center) * (down - center) >= 0:
            raise ValueError('Top and bottom were not distinct. Keep your head still and retry.')
        self.center = center
        self.up = up
        self.down = down
        self.dwell = dwell
        self.interval = interval
        self.reset()

    def reset(self):
        self.direction = 0
        self.since = 0
        self.last = 0

    def step(self, value, now, allowed=True):
        direction = 0
        if allowed and value is not None:
            delta = value - self.center
            if delta * (self.up - self.center) > 0 and abs(delta) > abs(self.up - self.center) * 0.65:
                direction = 1
            elif delta * (self.down - self.center) > 0 and abs(delta) > abs(self.down - self.center) * 0.65:
                direction = -1
        if direction != self.direction:
            self.direction, self.since = direction, now
        if direction and now - self.since >= self.dwell and now - self.last >= self.interval:
            self.last = now
            return direction
        return 0
