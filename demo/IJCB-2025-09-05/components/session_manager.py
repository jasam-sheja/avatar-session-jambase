import random
from pathlib import Path
from typing import Any, Dict


class AvatarSessionManager:
    """
    A class to manage the avatar session, including saving the session ID and
    handling the configuration.
    """

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg

    def _get_avatar(self, side: str):
        seen = self.seen_avatars_left if side == "left" else self.seen_avatars_right
        if self.session_num > len(self.avatar_pools):
            raise ValueError(
                f"Session number {self.session_num} exceeds the number of available avatar pools."
            )
        pool = [
            ava for ava in self.avatar_pools[self.session_num - 1] if ava not in seen
        ]
        if not pool:
            raise ValueError("Avatar pool is empty. Please check the configuration.")
        avatar = random.choice(pool)
        seen.add(avatar)
        return avatar

    def get_avatar(self, user: str):
        return self.cfg[f"{user}_user"]["avatar"]

