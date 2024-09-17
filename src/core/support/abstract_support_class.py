class AbstractSupportClass:
    """
    This class acts as an abstract base class for all supporting classes to the learner.
    In the case that the support classes are not instantiated by the calling class, they simply pass when called.
    """
    def before_fit(self, **kwargs): pass
    def after_fit(self, **kwargs): pass
    def before_epoch(self, **kwargs): pass
    def after_epoch(self, **kwargs): pass
    def before_epoch_train(self, **kwargs): pass
    def after_epoch_train(self, **kwargs): pass
    def before_epoch_valid(self, **kwargs): pass
    def after_epoch_valid(self, **kwargs): pass
    def before_batch_train(self, **kwargs): pass
    def after_batch_train(self, **kwargs): pass
    def before_batch_valid(self, **kwargs): pass
    def after_batch_valid(self, **kwargs): pass
    def before_batch_predict(self, **kwargs): pass
    def after_batch_predict(self, **kwargs): pass
    def before_batch_test(self, **kwargs): pass
    def after_batch_eval(self, **kwargs): pass
    def before_forward(self, **kwargs): pass
    def after_forward(self, **kwargs): pass
