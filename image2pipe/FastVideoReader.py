from typing import NamedTuple, Tuple, Optional
from fractions import Fraction
from multiprocessing import Queue

import numpy as np
import image2pipe
from image2pipe import ffprobe
from image2pipe.utils import yield_from_queue


class Frame(NamedTuple):
    index: int
    image: np.ndarray


class VideoInfo(NamedTuple):
    frame_count: int
    width: int
    height: int
    fps: float


class FastVideoReader:
    queue: Queue

    frame_count: int
    width: int
    height: int
    fps: float

    _frame_index_start: int
    _frame_index_end: int

    def __init__(self,
                 filename: str,
                 frame_range: Optional[Tuple[int, int]] = None,
                 process_nth_frames: Optional[int] = None,
                 min_size: Optional[int] = None,
                 video_info: Optional[VideoInfo] = None):
        if video_info is not None:
            frame_count, width, height, fps = video_info
            self.frame_count = frame_count
            self.width = width
            self.height = height
            self.fps = fps
        else:
            info = ffprobe(filename)['streams'][0]
            frame_count = int(info['nb_frames'])
            self.frame_count = frame_count
            width = int(info['width'])
            height = int(info['height'])
            self.width = width
            self.height = height
            fps = Fraction(info['avg_frame_rate'])
            self.fps = float(fps)

        self.queue = Queue()
        if frame_range is not None:
            frame_index_start, frame_index_end = frame_range

            if frame_index_end == -1:
                frame_index_end = frame_count

            if frame_index_start < 0:
                print(f"WARNING! Invalid frame_range {frame_range}. frame_index_start cant be less then 0. Using default value of 0 instead.")
                frame_index_start = 0
            if frame_index_end > frame_count:
                print(f"WARNING! Invalid frame_range {frame_range}. frame_index_end cant be greater then {frame_count }. Using default value of {frame_count - 1} instead.")
                frame_index_end = frame_count
        else:
            frame_index_start = 0
            frame_index_end = frame_count

        ss = str(float(frame_index_start / fps))
        to = str(float(frame_index_end / fps))

        self._frame_index_end = frame_index_end
        self._frame_index_start = frame_index_start

        if process_nth_frames:
            fps /= process_nth_frames

        if min_size is not None:
            if width < height:
                scale = (min_size, int(round(height / (width / min_size))))
            else:
                scale = (int(round(width / (height / min_size))), min_size)
        else:
            scale = (width, height)
        decoder = image2pipe.images_from_url(self.queue, filename, ss=ss, to=to, fps=fps, scale=scale, buffer_size=scale)
        decoder.start()

    def get_next_frame(self) -> Frame:
        for (i, frame) in yield_from_queue(self.queue):
            index = i + self._frame_index_start
            if index == self._frame_index_end:
                return
            ret = Frame(index, frame)
            yield ret

    def get_single_frame(self) -> Frame:
        for (i, frame) in yield_from_queue(self.queue):
            index = i + self._frame_index_start
            return Frame(index, frame)


def get_single_frame(filename: str, index: int, min_size: Optional[int] = None,
                     video_info: Optional[VideoInfo] = None) -> Frame:
    return FastVideoReader(filename=filename, frame_range=(index, index+1), min_size=min_size,
                           video_info=video_info).get_single_frame()

