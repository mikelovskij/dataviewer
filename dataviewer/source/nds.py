# coding=utf-8
# Copyright (C) Duncan Macleod (2014)
#
# This file is part of GWDV
#
# GWDV is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# GWDV is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with GWDV.  If not, see <http://www.gnu.org/licenses/>

"""This module defines the `NDSDataBuffer`
"""


from numpy import ceil
from gwpy.io import nds as ndsio
from gwpy.timeseries import (TimeSeries, TimeSeriesDict, StateVector, StateVectorDict, StateTimeSeries)
from gwpy.segments import DataQualityDict, DataQualityFlag
from fractions import gcd
from time import sleep
from gwpy.time import tconvert
import operator as op
from numbers import Number

from .. import version
from ..log import Logger
from . import (register_data_source, register_data_iterator)

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'
__version__ = version.version


class NDSDataSource(object):
    """A data holder for fetching from NDS
    """
    source = 'nds2'

    def __init__(self, channels, host=None, port=None, connection=None,
                 logger=Logger('nds'), **kwargs):
        self.connection = None
        super(NDSDataSource, self).__init__(channels, logger=logger, **kwargs)
        # connect if the monitor hasn't done it already
        if not self.connection:
            self.connect(connection=connection, host=host, port=port)

    def connect(self, connection=None, host=None, port=None, force=False):
        """Connect to an NDS server

        If a connection is given, it is simply attached, otherwise a new
        connection will be opened

        Parameters
        ----------
        connection : `nds2.connection`, optional
            an existing open connection to an NDS server
        host : `str`, optional
            the name of the NDS server host to connect
        port : `int`, optional
            the port number for the NDS server connection
        force : `bool`, optional
            force a new connection even if one is already open

        Returns
        -------
        connection : `nds2.connection`
            the attached connection
        """
        if force or (self.connection and (connection or host)):
            del self.connection
            self.logger.debug('Closed existing connection')

        # set up connection
        if connection:
            self.connection = connection
            self.logger.debug('Attached open connection to %s:%d...'
                              % (self.connection.get_host(),
                                 self.connection.get_port()))
        else:
            if not host:
                ifos = list(set([c.ifo for c in self.channels if c.ifo]))
                if len(ifos) == 1:
                    hosts = ndsio.host_resolution_order(ifos[0])
                else:
                    hosts = ndsio.host_resolution_order(None)
                try:
                    host, port = hosts[0]
                except IndexError:
                    raise ValueError("Cannot auto-select NDS server for "
                                     "ifos: %s" % ifos)
            if port:
                self.logger.debug('Opening connection to %s:%d...'
                                  % (host, port))
            else:
                self.logger.debug('Opening connection to %s...' % host)
            self.connection = ndsio.auth_connect(host, port)
            self.logger.debug('Connection open')
        return self.connection

    def fetch(self, channels, start, end, **kwargs):
        """Fetch new data from the NDS server

        Parameters
        ----------
        channels : `list`, `~gwpy.detectorChannelList`
            list of channels whose data are required
        start : `int`
            GPS start time of request
        end : `int`
            GPS end time of request
        **kwargs
            any other keyword arguments required to retrieve the
            data from the remote source

        Returns
        -------
        data : `TimeSeriesDict`
            new data
        """
        self.logger.info('Fetching data for [%s, %s)' % (start, end))
        kwargs.setdefault('connection', self.connection)
        uchannels = self._unique_channel_names(channels)
        data = self.RawDictClass.fetch(uchannels, start, end, **kwargs)
        out = type(data)()
        for chan in channels:
            out[chan] = data[self._channel_basename(chan)].copy()
        return out


register_data_source(NDSDataSource)
register_data_source(NDSDataSource, 'nds')


class NDSDataIterator(NDSDataSource):
    """Custom iterator to handle NDS1 update stride

    The NDS1 protocol iterator returns 1-second buffers under all
    user inputs, so we need to work around this and manually buffer the
    data.

    For NDS2 protocol connections, this wrapper is trivial.
    """
    def __init__(self, channels, duration=0, interval=2, host=None,
                 port=None, connection=None, logger=Logger('nds'),
                 gap='pad', pad=0.0, attempts=50, **kwargs):
        """Construct a new iterator
        """
        super(NDSDataIterator, self).__init__(channels, host=host, port=port,
                                              connection=connection,
                                              logger=logger, **kwargs)


        if self.connection.get_protocol() == 1:
            ndsstride = 1
        else:
            ndsstride = int(ceil(min(interval, 10)))
        self.interval = interval
        self.ndsstride = ndsstride
        self._duration = None
        self.duration = duration
        self.gap = gap
        self.pad = pad
        self.attempts = attempts
        self.flag_iterator = None
        self.start()

    def __iter__(self):
        return self

    def start(self):
        try:
            self.iterator = self.connection.iterate(
                self.ndsstride, self._unique_channel_names(self.allchannels))
            # if hasattr(self, 'flags'):
            #     self.flag_iterator = self.connection.iterate(
            #         self.ndsstride,
            #         self._unique_channel_names(self.flags.keys()))
        except RuntimeError as e:
            if e.message == 'Invalid channel name':
                self.logger.error('Invalid channel name %s' % str(
                    self._unique_channel_names(self.allchannels)))
            raise
        self.logger.debug('NDSDataIterator ready')
        return self.iterator

    def restart(self):
        del self.iterator
        self.connect(force=True)
        return self.start()

    def _next(self):
        # why not do it in __init__?
        uchannels = self._unique_channel_names(self.channels)
        if hasattr(self, 'flags'):
            uflags = self._unique_channel_names(self.flags.keys())
        else:
            uflags = []
        new = TimeSeriesDict()
        svd = StateVectorDict()
        new_flags = DataQualityDict()
        dq_dic = DataQualityDict()
        f_buffers = []
        span = 0
        epoch = 0
        att = 0
        self.logger.debug('Waiting for next NDS2 packet...')
        while span < self.interval:
            try:
                buffers = next(self.iterator)
                ch_buffers = buffers[0:len(uchannels)]
                if hasattr(self, 'flags'):
                    f_buffers = buffers[-len(uflags):]
            except RuntimeError as e:
                self.logger.error('RuntimeError caught: %s' % str(e))
                if att < self.attempts:
                    att += 1
                    wait_time = att / 3 + 1
                    self.logger.warning(
                        'Attempting to reconnect to the nds server... %d/%d'
                        % (att, self.attempts))
                    self.logger.warning('Next attempt in minimum %d seconds' %
                                        wait_time)
                    self.restart()
                    sleep(wait_time - tconvert('now') % wait_time)
                    continue
                else:
                    self.logger.critical(
                        'Maximum number of attempts reached, exiting')
                    break
            att = 0
            for f_buffer, flg in zip(f_buffers, uflags):
                svd[flg] = StateVector.from_nds2_buffer(f_buffer)
                dq_dic = self.flag_converter(svd)
            for key, val in dq_dic.iteritems():
                if key not in new_flags:
                    new_flags[key] = val
                else:
                    for seg in val.active:
                        new_flags[key].active.append(seg)
                    for seg in val.known:
                        new_flags[key].known.append(seg)

            for buff, c in zip(ch_buffers, uchannels):
                ts = TimeSeries.from_nds2_buffer(buff)
                try:
                    new.append({c: ts}, gap=self.gap, pad=self.pad)
                except ValueError as e:
                    if 'discontiguous' in str(e):
                        e.message = (
                            'NDS connection dropped data between %d and '
                            '%d, restarting building the buffer from %d ') \
                            % (epoch, ts.span[0], ts.span[0])
                        self.logger.warning(str(e))
                        new = TimeSeriesDict()
                        new[c] = ts.copy()
                    elif ('starts before' in str(e)) or \
                            ('overlapping' in str(e)):
                        e.message = (
                            'Overlap between old data and new data in the '
                            'nds buffer, only the new data will be kept.')
                        self.logger.warning(str(e))
                        new = TimeSeriesDict()
                        new[c] = ts.copy()
                    else:
                        raise
                span = abs(new[c].span)
                epoch = new[c].span[-1]
                self.logger.debug('%ds data for %s received'
                                  % (abs(ts.span), str(c)))
        out = type(new)()
        for chan in self.channels:
            out[chan] = new[self._channel_basename(chan)].copy()
        dq_out = type(new_flags)()
        if hasattr(self, 'flags'):  # todo: sto if c'è troppe volte, come lo tolgo?
            for fname, flag in new_flags.iteritems():
                dq_out[fname] = flag.copy()
        return out, dq_out

    def next(self):
        """Get the next data iteration

        For NDS1 connections this method simply loops over the underlying
        iterator until we have collected enough 1-second chunks.

        Returns
        -------
        data : :class:`~gwpy.timeseries.TimeSeriesDict`
           a new `TimeSeriesDict` with the concatenated, buffered data
        """
        # get new data
        new, new_dq = self._next()
        if (not new) & (not new_dq):
            self.logger.warning('No data were received')
            return self.data, self. s_data
        epoch = None
        if new:
            epoch = new.values()[0].span[-1]  # todo: why [0] and not [-1]?
            self.logger.debug('%d seconds of data received up to epoch %s'
                          % (epoch - new.values()[0].span[0], epoch))
            # record in buffer
            self.append(new)
            if abs(self.segments) > self.duration:
                self.crop(start=epoch - self.duration)
        if new_dq:
            if not epoch:
                epoch = (reduce(op.or_,
                                (flag.active for flag in new_dq.values())))\
                    .extent()[-1]
            self.seg_append(new_dq)
            if abs(self.s_segments) > self.duration:
            # todo: understand why the int( is necessary, due to small errors in epoch?
                print '%.15f' % epoch
                self.seg_crop(start=int(epoch - self.duration))
        return self.data, self.s_data

    def fetch(self, *args, **kwargs):
        try:
            return super(NDSDataIterator, self).fetch(*args, **kwargs)
        except RuntimeError as e:
            if 'Another transfer' in str(e):
                self.connect(force=True)
                return super(NDSDataIterator, self).fetch(*args, **kwargs)
            else:
                raise

    @property
    def duration(self):
        return float(self._duration)

    @duration.setter
    def duration(self, d):
        rinterval = ceil(
            self.interval / float(self.ndsstride)) * self.ndsstride
        self._duration = rinterval + d - gcd(rinterval, d)
        self.logger.debug(
            'The buffer has been set to store %d s of data' % self.duration)

    def flag_converter(self, svd):
        dqdict = DataQualityDict()
        # todo: check the unique things if requires some adjustment
        for (f, sv), kwargs in zip(svd.iteritems(),
                                   self.flags.itervalues()):
            cond = kwargs.get('condition', None)
            name = kwargs.get('name')
            prec = kwargs.get('precision', None)
            dq_kwargs = {'round': True}  # todo: add to config?
            if sv.unit:
                sv.override_unit(None)
            if prec:
                    dq_kwargs['minlen'] = int(prec*sv.samplerate)
            if isinstance(name, (list, tuple)):
                dq_kwargs['bits'] = name
                dqdict.update(sv.to_dqflags(**dq_kwargs))
            elif isinstance(name, basestring):
                dq_kwargs['name'] = name
                if cond:
                    operator, value = self.eval_condition(cond)
                else:
                    operator = op.eq
                    value = True
                print sv
                sts = operator(sv, value)
                dqdict[name] = sts.to_dqflag(**dq_kwargs)
            else:
                raise ValueError('Name parameter for flag {0} not valid'
                                 .format(f))
        return dqdict

    @staticmethod
    def eval_condition(condition):
        if isinstance(condition, basestring):
                opdict = {'>': op.gt,
                          '>=': op.ge,
                          '<': op.lt,
                          '<=': op.le,
                          '==': op.eq,
                          '!=': op.ne}
                for s, oper in opdict.iteritems():
                    if s in condition:
                        return oper, eval(condition.split(s)[-1])
                raise ValueError('Condition "{0}" not valid'.format(condition))
        elif isinstance(condition, Number):
            return op.eq, condition
        else:
            raise ValueError('Condition not valid')

register_data_iterator(NDSDataIterator)
register_data_iterator(NDSDataIterator, 'nds')
