class RunningGaitCycleCounter:
    def __init__(self):
        self.cycle_count = 0
        self.previous_right_phase = None
        self.previous_left_phase = None
        self.subphase_counts = {
            "Initial Contact": 0,
            "Midstance": 0,
            "Terminal Stance": 0,
            "Pre-Swing": 0,
            "Initial Swing": 0,
            "Mid Swing": 0,
            "Terminal Swing": 0
        }
        self.total_phases = 0

    def get_running_phase(self, right_ankle, right_knee, right_hip, left_ankle, left_knee, left_hip):
        flight_threshold = 5

        if right_ankle[1] > right_knee[1] and right_knee[1] > right_hip[1]:
            right_side_phase = "Stance"
        elif right_ankle[1] > flight_threshold and left_ankle[1] > flight_threshold:
            right_side_phase = "Flight"
        else:
            right_side_phase = "Swing"

        if left_ankle[1] > left_knee[1] and left_knee[1] > left_hip[1]:
            left_side_phase = "Stance"
        elif right_ankle[1] > flight_threshold and left_ankle[1] > flight_threshold:
            left_side_phase = "Flight"
        else:
            left_side_phase = "Swing"

        return {"right_phase": right_side_phase, "left_phase": left_side_phase}

    def get_subphase(self, ankle, knee, hip):
        if ankle[1] > knee[1] and knee[1] > hip[1]:
            # Determine subphase of Stance
            if ankle[1] - knee[1] > 0.1:  # Adjust the threshold as needed
                return "Initial Contact"
            elif knee[1] - hip[1] > 0.1:  # Adjust the threshold as needed
                return "Midstance"
            elif hip[1] - knee[1] > 0.05:  # Adjust the threshold for Pre-Swing
                return "Preswing"
            else:
                return "Terminal Stance"
        else:
            # Determine subphase of Swing
            if knee[1] - ankle[1] > 0.1:  # Adjust the threshold as needed
                return "Initial Swing"
            elif hip[1] - knee[1] > 0.1:  # Adjust the threshold as needed
                return "Mid Swing"
            else:
                return "Terminal Swing"

    def process_phases(self, right_ankle, right_knee, right_hip, left_ankle, left_knee, left_hip):
        phases = self.get_running_phase(right_ankle, right_knee, right_hip, left_ankle, left_knee, left_hip)
        right_subphase = self.get_subphase(right_ankle, right_knee, right_hip)
        left_subphase = self.get_subphase(left_ankle, left_knee, left_hip)

        self.subphase_counts[right_subphase] += 1
        self.subphase_counts[left_subphase] += 1
        self.total_phases += 2

        if self.previous_right_phase == "Swing" and phases["right_phase"] == "Stance":
            self.cycle_count += 1

        self.previous_right_phase = phases["right_phase"]
        self.previous_left_phase = phases["left_phase"]

    def get_cycle_count(self):
        return self.cycle_count

    def get_subphase_percentages(self):
        return {phase: (count / self.total_phases) * 100 for phase, count in self.subphase_counts.items()}

