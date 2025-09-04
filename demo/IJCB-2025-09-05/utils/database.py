
class CFGDatabase:
    def __init__(self, cfg):
        self.cfg = cfg

    def get_avatar(self, user_id):
        return self.cfg['users'].get(user_id, {}).get('avatar')
    
    def check_pass(self, user_id, password: str) -> bool:
        user = self.cfg['users'].get(user_id, None)
        if user is None:
            return False
        return user.get('pass') == password