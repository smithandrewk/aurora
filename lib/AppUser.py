from flask_login import UserMixin

class User(UserMixin):
    def __init__(self, id):
        self.id = id
    #     self._is_active = True
    #     self._is_authenticated = False
    #     self._is_anonymous = False
    #     self._id = None

    # @property
    # def is_active(self):
    #     return self._is_active

    # @property
    # def is_authenticated(self):
    #     return self._is_active

    # @property
    # def is_anonymous(self):
    #     return self._is_anonymous

    # def get_id(self):
    #     try:
    #         return str(self.id)
    #     except AttributeError:
    #         raise NotImplementedError("No `id` attribute - override `get_id`") from None