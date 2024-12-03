class climber:
    def __init__(self, height):     # Height in cm
        # User height
        self.height = height
        # User arm, leg length
        self.leg_length = height * 0.45
        self.arm_length = height * 0.35
        self.torso_length = height * 0.35
        # Center of body
        self.body_center = (0, height * 0.55)
        # User reach, vertical and horizontal
        self.horizontal_reach = height
        self.vertical_reach = height * 1.35

