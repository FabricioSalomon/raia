class UserRepository:
    def __init__(self, db):
        self.db = db

    def get_by_id(self, user_id: int):
        user = self.db.query(UserModel).filter_by(id=user_id).first()
        return user
