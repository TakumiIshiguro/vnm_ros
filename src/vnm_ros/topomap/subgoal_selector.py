import numpy as np


class SubgoalSelector:
    def __init__(self, goal_node: int, search_radius: int, close_threshold: float):
        self.goal_node = goal_node
        self.search_radius = search_radius
        self.close_threshold = close_threshold
        self.closest_node = 0
        self.selected_node = 0
        self.localized_node = 0
        self.target_node = 0
        self.closest_distance = float("inf")
        self.state = "initializing"

    def select(self, distances, waypoints, waypoint_index: int):
        start = max(self.closest_node - self.search_radius, 0)
        distances = np.asarray(distances).reshape(-1)
        min_idx = int(np.argmin(distances))
        self.localized_node = min(start + min_idx, self.goal_node)
        self.closest_distance = float(distances[min_idx])

        if distances[min_idx] > self.close_threshold:
            selected_idx = min_idx
            self.closest_node = self.localized_node
            self.state = "approaching"
        else:
            selected_idx = min(min_idx + 1, len(waypoints) - 1)
            self.closest_node = min(start + min_idx + 1, self.goal_node)
            self.state = "advanced"

        self.selected_node = min(start + selected_idx, self.goal_node)
        self.target_node = self.selected_node
        return waypoints[selected_idx][waypoint_index]

    def window_bounds(self):
        start = max(self.closest_node - self.search_radius, 0)
        end = min(self.closest_node + self.search_radius + 1, self.goal_node)
        return start, end

    def reached_goal(self) -> bool:
        return self.closest_node >= self.goal_node
