
from pathlib import Path
import datetime


def get_stream_path(exp_name: str, user_id: str, root: Path, date: str = None) -> Path:
    """
    Get the path for the stream video file based on experiment name and user ID.

    Args:
        exp_name (str): The name of the experiment.
        user_id (str): The ID of the user.

    Returns:
    """
    if date is None:
        date = datetime.datetime.now().strftime("%Y-%m-%d")
    date = datetime.datetime.now().strftime("%Y-%m-%d")
    return root.joinpath(f"{date}/{exp_name}-{user_id}/stream.mp4")

def link_other_subject(exp_name: str, user1: str, user2: str, root: Path, date: str = None):
    """
    Write a text file in each subject's folder to indicate that they are linked.

    Args:
        exp_name (str): The name of the experiment.
        user1 (str): The ID of the first user.
        user2 (str): The ID of the second user.

    """
    if date is None:
        date = datetime.datetime.now().strftime("%Y-%m-%d")
    stream_1 = get_stream_path(exp_name, user1, root, date).with_name("linked_with.txt")
    stream_2 = get_stream_path(exp_name, user2, root, date).with_name("linked_with.txt")
    stream_1.parent.mkdir(parents=True, exist_ok=True)
    stream_2.parent.mkdir(parents=True, exist_ok=True)
    stream_1.write_text(f"{stream_2.parent.relative_to(root).as_posix()}\n", encoding='utf-8')
    stream_2.write_text(f"{stream_1.parent.relative_to(root).as_posix()}\n", encoding='utf-8')


def get_reenactment_path(exp_name: str, user_id: str, session_num: int, root: Path, date: str = None) -> Path:
    """
    Get the path for the reenactment video file based on experiment name, user ID, and session number.

    Args:
        exp_name (str): The name of the experiment.
        user_id (str): The ID of the user.
        session_num (int): The session number.

    Returns:
    """
    if date is None:
        date = datetime.datetime.now().strftime("%Y-%m-%d")
    return root.joinpath(f"{date}/{exp_name}-{user_id}/reenactment-session{session_num}.mp4")

def get_questionnaire_path(exp_name: str, user_id: str, root: Path, date: str = None) -> Path:
    """
    Get the path for the questionnaire file based on experiment name and user ID.

    Args:
        exp_name (str): The name of the experiment.
        user_id (str): The ID of the user.

    Returns:
    """
    if date is None:
        date = datetime.datetime.now().strftime("%Y-%m-%d")
    return root.joinpath(f"{date}/{exp_name}-{user_id}/questionnaire.json")