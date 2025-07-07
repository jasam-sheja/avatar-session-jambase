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
        if "save_path" not in cfg:
            raise ValueError("Save path is not specified in the configuration.")
        self.save_path = Path(cfg["save_path"])
        self.save_path.mkdir(parents=True, exist_ok=True)

        avatar_pools = [
            v
            for k, v in sorted(cfg.get("avatar_pool", {}).items(), key=lambda kv: kv[0])
        ]
        if not avatar_pools:
            raise ValueError("Avatar pool is empty. Please check the configuration.")

        self.session_num = 1
        self.avatar_pools = avatar_pools
        self.seen_avatars_left = set()
        self.seen_avatars_right = set()

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

    def get_left_avatar(self):
        return self._get_avatar("left")

    def get_right_avatar(self):
        return self._get_avatar("right")

    def next_session(self):
        """
        Increment the session number.
        """
        self.session_num += 1

    def available_sessions(self):
        """
        Check if there are available sessions.
        Returns True if there are more sessions, False otherwise.
        """
        return self.session_num < len(self.avatar_pools)
