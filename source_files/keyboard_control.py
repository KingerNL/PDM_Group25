import numpy as np
from pynput.keyboard import Key, Listener

class KeyboardController:
    def __init__(self):
        # keys that are currently pressed
        self.active_keys = set()

        # the "state" the simulator will receive every timestep
        self.state = np.zeros(2)   # [velocity, steering]

        # how strongly keys push the state
        self.key_force = {
            Key.left:   np.array([0.0,  2.0]),   # steering rate   (rad/sec)
            Key.right:  np.array([0.0, -2.0]),
            Key.up:     np.array([3.0,  0.0]),   # acceleration    (m/s^2)
            Key.down:   np.array([-3.0, 0.0]),
        }

        # drag applied every timestep
        self.drag = np.array([1.0, 1.5])   # velocity drag, steering drag

        # limits
        self.max_speed = 8.0
        self.max_steer = 0.6   # rad

    def on_press(self, key):
        self.active_keys.add(key)

    def on_release(self, key):
        if key in self.active_keys:
            self.active_keys.remove(key)

    def start(self):
        Listener(on_press=self.on_press, on_release=self.on_release).start()

    def step(self, dt):
        """
        Update internal state each timestep.
        Returns: [velocity, steering_angle]
        """

        # 1) compute desired input from active keys (acceleration + steering rate)
        input_acc = np.zeros(2)
        for key in self.active_keys:
            if key in self.key_force:
                input_acc += self.key_force[key]

        # 2) apply acceleration to state
        self.state += input_acc * dt

        # 3) apply drag (simple exponential-like decay)
        self.state -= self.drag * self.state * dt

        # 4) clip
        self.state[0] = np.clip(self.state[0], -self.max_speed, self.max_speed)
        self.state[1] = np.clip(self.state[1], -self.max_steer, self.max_steer)

        return self.state.copy()

