class BaseScenario:
    def make_world(self):
        raise NotImplementedError()

    def reset_world(self, world, np_random):
        raise NotImplementedError()
