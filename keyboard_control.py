import numpy as np
from pynput.keyboard import Key, Listener

class KeyboardController:
    def __init__(self):
        self.active_keys = set()
        self.action = np.zeros(2)

        self.mapping = {
            Key.left: np.array([0.0, 2.0]),      # steering left
            Key.right: np.array([0.0, -2.0]),      # steering right
            Key.up: np.array([3.0, 0.0]),         # forward
            Key.down: np.array([-3.0, 0.0]),      # backward
        }

    def compute_action(self):
        total = np.zeros(2)
        for key in self.active_keys:
            if key in self.mapping:
                total += self.mapping[key]

        self.action = total

    def on_press(self, key):
        self.active_keys.add(key)
        self.compute_action()

    def on_release(self, key):
        if key in self.active_keys:
            self.active_keys.remove(key)
        self.compute_action()

    def start(self):
        Listener(on_press=self.on_press, on_release=self.on_release).start()

