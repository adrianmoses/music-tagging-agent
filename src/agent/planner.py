
class PlanningModule:
    def __init__(self, confidence_threshold=0.85):
        self.memory = MemoryModule()
        self.threshold = confidence_threshold

    def create_plan(self, content):
        content_type = self.classify_content(content)
        return {
            'steps': self.determine_steps(content_type),
            'models': self.select_models(content_type),
            'confidence_req': self.threshold
        }