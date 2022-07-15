import random
from typing import List

def fill_random(l: List, size: int) -> List:
    """Fills a list untile size is reached. It uses elements from same list.

    Args:
        l (List): list
        size (int): final size 

    Returns:
        List: list with len==size
    """
    while len(l) < size:
        el = random.choice(l)
        l.append(el)
    return l