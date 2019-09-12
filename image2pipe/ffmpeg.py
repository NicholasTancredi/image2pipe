#!/usr/bin/env python
import copy
import itertools
import logging
import multiprocessing
import subprocess
from typing import Optional

import numpy
from numpy.ma import frombuffer
from image2pipe.Timer import Timer

FFMPEG_BIN = "ffmpeg"
log = logging.getLogger(__name__)


def bgr24_from_stdin_subp(_fps, _scale):
    """
    Usage:
    sp = bgr24_from_stdin_subp(fps, scale)
    sp.stdin.write(rawvideo)

    :param _fps: 
    :param _scale: 
    :return: 
    """
    vf_scale = "fps=%s,scale=%sx%s" % (_fps, _scale[0], _scale[1])
    cmd = [FFMPEG_BIN, "-v", "error",
           "-i", "pipe:0",
           "-an", "-sn", "-f", "image2pipe",
           "-vcodec", "rawvideo", "-pix_fmt", "bgr24", "-vf", vf_scale,
           "-y", "pipe:1"]
    log.debug("popen %s" % " ".join(cmd))
    ffmpeg_p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    return ffmpeg_p


def images_from_url_subp(_fps,
                         _scale,
                         _url,
                         ss: Optional[str] = None,
                         to: Optional[str] = None,
                         image_format: str = 'bgr24',
                         vf: Optional[list] = None,
                         resize_mode: str = 'fast_bilinear'):
    """
    Usage:
    p = images_from_url_subp(fps, scale, url)
    :type ss: str

    :param vf: ffmpeg -vf filters
    :param image_format: str with ffmpeg's pix_fmt
    :param ss: "00:00:00"
    :param to: "00:00:00"
    :param _fps:
    :param _scale:
    :param _url:
    :param resize_mode:
    :return:
    """
    cmd = [FFMPEG_BIN, "-v", "error"]
    if ss:
        cmd.append("-ss")
        cmd.append(ss)
    if to:
        cmd.append("-to")
        cmd.append(to)
    cmd += ["-i", _url, '-an', '-sn', "-f", "image2pipe", "-vcodec", "rawvideo", "-pix_fmt", image_format]

    if vf:
        _vf = copy.copy(vf)
    else:
        _vf = []

    if _fps:
        _vf.append("fps=%s" % _fps)
    if _scale:
        _vf.append("scale=%sx%s" % (_scale[0], _scale[1]))
    if len(_vf) > 0:
        cmd.append("-vf")
        cmd.append(",".join(_vf))
    if resize_mode:
        cmd.append('-sws_flags')
        cmd.append(resize_mode)
    cmd.append("-")
    log.debug("popen %s" % " ".join(cmd))
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stdin=subprocess.DEVNULL, bufsize=10 ** 8)


def enqueue_frames_from_output(_proc, _qout, scale, use_timer=False):
    """

    :type scale: tuple
    :type _proc: subprocess.Popen
    :type _qout: queues.Queue
    """
    timer = None
    if use_timer:
        timer = Timer(title='enqueue_frames_from_output')
    e = None
    frame_counter = itertools.count()
    img_size = scale[0] * scale[1] * 3
    while multiprocessing.current_process().is_alive():
        if use_timer: timer.tic('_proc.stdout.read')
        bb = _proc.stdout.read(img_size)
        if use_timer: timer.toc('_proc.stdout.read')
        if len(bb) > 0:
            try:
                if use_timer: timer.tic('frombuffer')
                ndarr = frombuffer(bb, dtype=numpy.uint8)
                if use_timer: timer.toc('frombuffer')
                if use_timer: timer.tic('buffer reshape')
                ndarr = ndarr.reshape((scale[1], scale[0], 3))
                if use_timer: timer.toc('buffer reshape')
                fn = next(frame_counter)
                _qout.put((fn, ndarr))
            except Exception as err:
                log.error("%s" % err)

        e = _proc.poll()
        # log.debug("%s bb size %d" % (e, len(bb)))
        if e >= 0 and len(bb) == 0:
            break

    if use_timer:
        timer.print()

    log.debug("bye ffmpeg %d" % e)
    if e == 0:
        _qout.put(None)
        _qout.close()
    elif e > 0:
        _qout.put(None)
        _qout.close()
        raise RuntimeError("ffmpeg exits with code %d" % e)
