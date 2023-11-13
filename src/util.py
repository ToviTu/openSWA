import inspect


class HyperPrameters:
    def save_hyperparameters(self, ignore=[]):
        """
        This function saves the arguments of the last frame as the attributes of this instance
        """
        frame = inspect.currentframe().f_back  # access the frame of last function call
        _, _, _, local_vars = inspect.getargvalues(frame)
        for k, v in local_vars.items():
            if k not in ignore:
                setattr(self, k, v)
