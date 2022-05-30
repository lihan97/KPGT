class Result_Tracker():
    def __init__(self,metric_name):
        self.metric_name = metric_name
    def init(self):
        if self.metric_name in ['rmse', 'mae']:
            init_value = 999
        else:
            init_value = -999
        return init_value

    def update(self, old_result, new_result):
        if self.metric_name in ['rmse', 'mae']:
            if new_result < old_result:
                return True
            else:
                return False

        else:
            if new_result > old_result:
                return True
            else:
                return False