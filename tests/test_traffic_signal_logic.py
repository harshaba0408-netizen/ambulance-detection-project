import pytest


class TrafficSignalCoordinator:
    """Core decision logic mirror from new.py for headless testing."""

    def __init__(self, num_roads=4):
        self.num_roads = num_roads
        self.audio_detected = {f"Road {i+1}": False for i in range(num_roads)}
        self.video_detected = {f"Road {i+1}": False for i in range(num_roads)}
        self.road_policy = {f"Road {i+1}": "AUDIO" for i in range(num_roads)}
        self.signal_state = {f"Road {i+1}": "RED" for i in range(num_roads)}

    def set_policy(self, road, policy):
        assert policy in ["AUDIO", "VIDEO", "BOTH"]
        self.road_policy[road] = policy

    def update_detection(self, road, is_audio=None, is_video=None):
        if is_audio is not None:
            self.audio_detected[road] = is_audio
        if is_video is not None:
            self.video_detected[road] = is_video
        self.evaluate_signal(road)

    def evaluate_signal(self, road):
        policy = self.road_policy[road]
        audio = self.audio_detected[road]
        video = self.video_detected[road]

        should_be_green = False
        if policy == "AUDIO" and audio:
            should_be_green = True
        elif policy == "VIDEO" and video:
            should_be_green = True
        elif policy == "BOTH" and audio and video:
            should_be_green = True

        self.signal_state[road] = "GREEN" if should_be_green else "RED"

    def clear_all(self):
        for road in self.signal_state:
            self.audio_detected[road] = False
            self.video_detected[road] = False
            self.signal_state[road] = "RED"


def test_default_all_red():
    coord = TrafficSignalCoordinator()
    for road in coord.signal_state:
        assert coord.signal_state[road] == "RED"


def test_audio_policy_triggers_green():
    coord = TrafficSignalCoordinator()
    coord.set_policy("Road 1", "AUDIO")
    coord.update_detection("Road 1", is_audio=True)
    assert coord.signal_state["Road 1"] == "GREEN"
    assert coord.signal_state["Road 2"] == "RED"


def test_video_policy_triggers_green():
    coord = TrafficSignalCoordinator()
    coord.set_policy("Road 2", "VIDEO")
    coord.update_detection("Road 2", is_video=True)
    assert coord.signal_state["Road 2"] == "GREEN"
    assert coord.signal_state["Road 1"] == "RED"


def test_both_policy_requires_audio_and_video():
    coord = TrafficSignalCoordinator()
    coord.set_policy("Road 3", "BOTH")

    # Only video -> stays RED
    coord.update_detection("Road 3", is_video=True, is_audio=False)
    assert coord.signal_state["Road 3"] == "RED"

    # Only audio -> stays RED
    coord.update_detection("Road 3", is_video=False, is_audio=True)
    assert coord.signal_state["Road 3"] == "RED"

    # Both -> switches to GREEN
    coord.update_detection("Road 3", is_video=True, is_audio=True)
    assert coord.signal_state["Road 3"] == "GREEN"


def test_revert_to_red_on_exit():
    coord = TrafficSignalCoordinator()
    coord.set_policy("Road 1", "AUDIO")
    coord.update_detection("Road 1", is_audio=True)
    assert coord.signal_state["Road 1"] == "GREEN"

    # Ambulance leaves / siren stops
    coord.update_detection("Road 1", is_audio=False)
    assert coord.signal_state["Road 1"] == "RED"


def test_clear_all_resets_all_signals():
    coord = TrafficSignalCoordinator()
    coord.set_policy("Road 1", "AUDIO")
    coord.update_detection("Road 1", is_audio=True)
    coord.set_policy("Road 2", "VIDEO")
    coord.update_detection("Road 2", is_video=True)

    assert coord.signal_state["Road 1"] == "GREEN"
    assert coord.signal_state["Road 2"] == "GREEN"

    coord.clear_all()
    for road in coord.signal_state:
        assert coord.signal_state[road] == "RED"

