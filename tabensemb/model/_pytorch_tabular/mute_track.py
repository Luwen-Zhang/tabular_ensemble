from tabensemb.utils import *


def _iterator(iterator, *args, **kwargs):
    for item in iterator:
        yield item


def mute_track():
    rich = reload_module("rich")
    rich.progress.track = _iterator
