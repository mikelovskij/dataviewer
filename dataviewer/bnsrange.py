# coding=utf-8
# Copyright (C) Duncan Macleod and Shivaraj kandhasamy (2014)
#
# This file is part of GWDV.
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
# along with GWDV.  If not, see <http://www.gnu.org/licenses/>.

"""DataMonitor for for BNS range.
"""

from epics import PV

import re
from itertools import (cycle, izip_longest)

from astropy.time import Time
from gwpy.timeseries import (TimeSeries, TimeSeriesDict)
from gwpy.spectrogram import (Spectrogram, SpectrogramList)
from gwpy.plotter import (SpectrogramPlot, TimeSeriesAxes)
from gwpy.astro.range import inspiral_range_psd

import cPickle

from . import version
from .buffer import (OrderedDict, DataBuffer)

from .registry import register_monitor
from .timeseries import TimeSeriesMonitor

__author__ = 'Shivaraj Kandhasamy <shivaraj.kandhasamy@ligo.org>'
__version__ = version.version

__all__ = ['BNSRangeSpectrogramMonitor']

# -----------------------------------------------------------------------------
#
# Data mixin
#
# -----------------------------------------------------------------------------

stateDQ = 1  # defines whether the data is meaningful or not (redefined below)


class SpectrogramBuffer(DataBuffer):
    DictClass = OrderedDict
    SeriesClass = Spectrogram
    ListClass = SpectrogramList

    def __init__(self, channels, stride=1, fftlength=1, overlap=0,
                 method='welch', filter=None, cflags=None,
                 fhigh=8000, flow=0, statechannel=list(), **kwargs):
        super(SpectrogramBuffer, self).__init__(channels, **kwargs)
        self.method = method
        if 'window' in kwargs:
            self.window = {'window': kwargs.pop('window')}
        else:
            self.window = {}
        # todo: maybe it is better to pass some of these kwargs as kwargs to ts.spectrogram
        self.stride = self._param_dict(stride)
        self.fftlength = self._param_dict(fftlength)
        self.overlap = self._param_dict(overlap)
        self.filter = self._param_dict(filter)
        self.fhigh = self._param_dict(fhigh)
        self.cflags = self._param_dict(cflags)
        self.flow = self._param_dict(flow)
        # define state vector
        global stateDQ
        self.pv = None
        if isinstance(statechannel, str):
            statechannel = (statechannel,)
        if len(statechannel) == 1:
            self.pv = PV(statechannel[0])
            stateDQ = self.pv.get()
        elif len(statechannel) == 2:
            statechannels = statechannel[0].split(",")
            statecondition = statechannel[1].split(",")
            self.pv = OrderedDict()
            for cond, lockchanls in zip(statecondition, statechannels):
                pvt = PV(lockchanls)
                self.pv[pvt] = cond
                stateDQ = stateDQ and eval(str(pvt.get()) + cond)
        elif len(statechannel) > 2:
            raise UserException("Unknown state channels/ conditions")

    def _param_dict(self, param):
        # format parameters
        if not isinstance(param, dict):
            return dict((c, param) for c in self.channels)
        return param

    def fetch(self, channels, start, end, **kwargs):
        # set params
        fftparams = dict()
        for param in ['stride', 'fftlength', 'overlap', 'method', 'filter']:
            fftparams[param] = kwargs.pop(param, getattr(self, param))
        # get data
        tsd = super(SpectrogramBuffer, self).fetch(
            channels, start, end, **kwargs)
        return self.from_timeseriesdict(tsd, **fftparams)

    @staticmethod
    def flag_check(ts, flagnames, flagbuffer):
        if flagnames:
            for name in flagnames:
                try:
                    if ts.span not in flagbuffer[name].active:
                        return False
                except KeyError as e:
                    e.message = 'cflag {0} does not exist!'.format(name)
                    raise
        return True

    def from_timeseriesdict(self, tsd, fld, **kwargs):
        # format parameters
        stride = self._param_dict(kwargs.pop('stride', self.stride))
        fftlength = self._param_dict(kwargs.pop('fftlength', self.fftlength))
        overlap = self._param_dict(kwargs.pop('overlap', self.overlap))
        filter = self._param_dict(kwargs.pop('filter', self.filter))
        fhigh = self._param_dict(kwargs.pop('fhigh', self.fhigh))
        flow = self._param_dict(kwargs.pop('flow', self.flow))

        # calculate spectrograms only if state conditon(s) is satisfied
        data = self.DictClass()
        global stateDQ
        stateDQ = 1
        if isinstance(self.pv, PV):
            stateDQ = self.pv.get()
        elif isinstance(self.pv, OrderedDict):
            for pvt, cond in self.pv.iteritems():
                stateDQ = stateDQ and eval(str(pvt.get()) + cond)
        if stateDQ:
            for channel, ts in zip(self.channels, tsd.values()):
                if self.flag_check(ts, self.cflags.get(channel, None), fld):
                    spec = ts.asd(fftlength=fftlength[channel],
                                  overlap=overlap[channel],
                                  method=self.method,
                                  **self.window) \
                        .crop(flow[channel], fhigh[channel])
                    spec.epoch = ts.epoch
                    self.logger.debug('TimeSeries span: {0},'
                                      ' TimeSeries length: {1}, Stride: {2}'
                                      .format(ts.span,
                                              ts.span[-1] - ts.span[0],
                                              stride[channel]))
                    if hasattr(channel, 'resample') \
                            and channel.resample is not None:
                        nyq = float(channel.resample) / 2.
                        nyqidx = int(nyq / spec.df.value)
                        spec = spec[:nyqidx]
                    if channel in filter and filter[channel]:
                        self.logger.debug('Filtering ASD')
                        spec = spec.filter(*filter[channel]).copy()
                    self.logger.debug('Calculating insp. range psd')
                    range_spec = inspiral_range_psd(spec ** 2)
                    self.logger.debug('Insp. range psd calculated')
                    ranges = (range_spec * range_spec.df) ** 0.5
                    data[channel] = Spectrogram.from_spectra(
                        ranges, epoch=spec.epoch, dt=self.stride[channel],
                        frequencies=spec.frequencies)
                    self.logger.debug('From_timeseries completed...')
        return data


class SpectrogramIterator(SpectrogramBuffer):
    def _next(self):
        new = super(SpectrogramIterator,
                    self)._next()  # todo:  is this iterator even used?
        return self.from_timeseriesdict(
            new, method=self.method, stride=self.stride,
            fftlength=self.fftlength, overlap=self.overlap, filter=self.filter)


# -----------------------------------------------------------------------------
#
# Monitor
#
# -----------------------------------------------------------------------------

class UserException(Exception):
    pass


class BNSRangeSpectrogramMonitor(TimeSeriesMonitor):
    """Monitor some spectra
    """
    type = 'bnsrangespectrogram'
    FIGURE_CLASS = SpectrogramPlot
    AXES_CLASS = TimeSeriesAxes

    def __init__(self, *channels, **kwargs):
        # get FFT parameters
        stride = kwargs.pop('stride', 20)
        fftlength = kwargs.pop('fftlength', 1)
        overlap = kwargs.pop('overlap', 0)
        method = kwargs.pop('method', 'welch')
        filter = kwargs.pop('filter', None)
        ratio = kwargs.pop('ratio', None)
        resample = kwargs.pop('resample', None)
        cflags = kwargs.pop('cflags', [])
        kwargs.setdefault('interval', stride)
        flow = kwargs.pop('flow')
        fhigh = kwargs.pop('fhigh')
        picklefile = kwargs.pop('picklefile', None)
        statechannel = kwargs.pop('statechannel', [])
        if kwargs['interval'] % stride:
            raise ValueError("%s interval must be exact multiple of the stride"
                             % type(self).__name__)

        # build 'data' as SpectrogramBuffer
        self.spectrograms = SpectrogramIterator(
            channels, stride=stride, method=method, overlap=overlap,
            fftlength=fftlength, flow=flow, fhigh=fhigh,
            statechannel=statechannel)
        if isinstance(filter, list):
            self.spectrograms.filter = dict(zip(self.spectrograms.channels,
                                                filter))
        else:
            self.spectrograms.filter = filter
        self.spectrograms.cflags = dict(zip(self.spectrograms.channels,
                                            cflags))
        self.picklefile = picklefile
        self.fftlength = fftlength
        self.stride = stride
        self.overlap = overlap
        self.olepoch = None
        self.duration = kwargs['duration']
        self.coloraxes = OrderedDict()
        # build monitor
        kwargs.setdefault('yscale', 'log')
        kwargs.setdefault('gap', 'raise')
        self.plots = kwargs.pop('plots')
        if isinstance(self.plots, str):
            self.plots = (self.plots,)

        super(BNSRangeSpectrogramMonitor, self).__init__(*channels,
                                                         **kwargs)
        self.buffer.channels = self.spectrograms.channels

        # reset buffer duration to store a single stride
        self.buffer.duration = kwargs['interval']

        if ratio is not None:
            if not isinstance(ratio, (list, tuple)):
                ratio = [ratio] * len(self.channels)
            for c, r in izip_longest(self.channels, ratio):
                c.ratio = r
        if resample is not None:
            if not isinstance(resample, (list, tuple)):
                resample = [resample] * len(self.channels)
            for c, r in izip_longest(self.channels, resample):
                c.resample = r

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, d):
        self._data = d

    @property
    def duration(self):
        return self.spectrograms.duration

    @duration.setter
    def duration(self, t):
        self.spectrograms.duration = t

    def init_figure(self):
        self._fig = self.FIGURE_CLASS(**self.params['figure'])
        if self.params['draw']['norm'][0] == "linear":
            self.params['draw']['norm'][0] = ''

        def _new_axes():
            ax = self._fig._add_new_axes(self._fig._DefaultAxesClass.name)
            ax.set_epoch(float(self.epoch))
            ax.set_xlim(float(self.epoch - self.duration), float(self.epoch))
            return ax

        for n in range(len(self.plots)):
            _new_axes()
        self.set_params('init')
        for ax in self._fig.get_axes(self.AXES_CLASS.name)[:-1]:
            ax.set_xlabel('')
        self.set_params('refresh')
        return self._fig

    def update_data(self, new):
        """Update the `SpectrogramMonitor` data
        This method only applies a ratio, if configured
        """
        new, new_f = new
        # check that the stored epoch is bigger then the first buffered data
        if new[self.channels[0]][0].span[0] > self.epoch:
            s = ('The available data starts at gps %d '
                 'which is after the end of the last spectrogram(gps %d)'
                 ': a segment is missing and will be skipped!')
            self.logger.warning(s, new[self.channels[0]][0].span[0],
                                self.epoch)
            self.epoch = new[self.channels[0]][0].span[0]

        if not self.spectrograms.data:
            # be sure that the first cycle is syncronized with the buffer
            self.epoch = new[self.channels[0]][0].span[0]
            # TODO : use a database instead to not resave the whole specgram?
            if self.picklefile:
                try:
                    #    load the saved spectrograms if there are any
                    picklehandle = open(self.picklefile, 'r')
                    tempspect = cPickle.load(picklehandle)
                    picklehandle.close()
                    self.spectrograms.data[self.channels[0]] = tempspect
                except:
                    self.logger.warning('Error in loading the picklefile, '
                                        'old spectrum will not be loaded')
        self.olepoch = self.epoch
        while int(new[self.channels[0]][0].span[-1]) >= int(
                        self.epoch + self.stride):
            # data buffer will return dict of 1-item lists, so reform to tsd
            _new = TimeSeriesDict((key, val[0].crop(self.epoch, self.epoch +
                                                    self.stride))
                                  for key, val in new.iteritems())
            self.logger.debug('Calculating spectrogram from epoch {0}'
                              .format(self.epoch))
            self.spectrograms.append(
                self.spectrograms.from_timeseriesdict(_new, new_f))
            self.logger.debug('New stride appended')
            self.epoch += self.stride
            self.spectrograms.crop(self.epoch - self.duration)
            self.logger.debug('Spectrogram cropped')
        self.data = type(self.spectrograms.data)()
        if self.spectrograms.data:  # is this if necessary?
            # TODO: any way to avoid looping since there is only one channel?
            for channel in self.channels:
                try:
                    self.data[channel] = type(self.spectrograms.data[channel])(
                        *self.spectrograms.data[channel])
                    nepoch = self.data[channel][-1].span[-1]
                    self.epoch = max(nepoch, self.epoch)
                except KeyError:
                    self.data[channel] = SpectrogramList()
                self.logger.debug('Data copied in buffer')
                if self.picklefile:
                    picklehandle = open(self.picklefile, 'w')
                    cPickle.dump(self.spectrograms.data[channel], picklehandle,
                                 cPickle.HIGHEST_PROTOCOL)
                    picklehandle.close()
                    self.logger.debug('Pickle saved')
        return self.data

    def refresh(self):
        # extract data
        # replot all spectrogram
        if self.data:
            # if self.data[self.channels[0]]:
            axes = cycle(self._fig.get_axes(self.AXES_CLASS.name))
            coloraxes = self._fig.colorbars
            params = self.params['draw']
            # if only one channel given then proceed
            if len(self.data.keys()) == 1:
                channel = self.data.keys()[0]
            else:
                raise UserException(
                    "Only one channel is accepted for BNSrange Monitor")

            for i, plottype in enumerate(self.plots):
                ax = next(axes)

                # plot new data
                pparams = {}
                for key in params:
                    try:
                        if params[key][i]:
                            pparams[key] = params[key][i]
                    except IndexError:
                        pass
                if plottype == "timeseries":
                    newspectrogram = self.data[channel]
                    # todo: do these calculations only for the new data?
                    for spec in newspectrogram:
                        rangintegrand = spec ** 2
                        rangesimeseriessquare = rangintegrand.sum(axis=1)
                        rangetimeseries = TimeSeries(
                            rangesimeseriessquare.value ** 0.5,
                            epoch=rangesimeseriessquare.epoch,
                            name=rangesimeseriessquare.name,
                            sample_rate=1.0 / rangesimeseriessquare.dt.value,
                            unit='Mpc', channel=rangesimeseriessquare.name)
                        coll = ax.plot(rangetimeseries, color='b',
                                       linewidth=3.0, marker='o')
                elif plottype == "spectrogram":
                    # this part allows to replot only the last spectrogram
                    if len(ax.collections):
                        newspectrogram = self.data[channel][-1:]
                        # remove old spectrogram if the new one contains
                        # the same data
                        if float(newspectrogram[-1].span[0]) < self.olepoch:
                            ax.collections.remove(ax.collections[-1])
                    else:
                        newspectrogram = self.data[channel]
                    # coll = ax.plot(newspectrogram, label=label, **pparams)
                    for spec in newspectrogram:
                        # the .copy() is necessary for some reason to avoid a
                        #  weird error in the .sum at line 343 that happens if
                        # the spectrogram is plotted before the timeseries
                        coll = ax.plot(spec.copy(), **pparams)

                    # rescale all the colormaps to the last one plotted
                    new_clim = None
                    for co in ax.collections:
                        new_clim = coll.get_clim()
                        co.set_clim(new_clim)
                    if coll:
                        if i not in self.coloraxes:
                            cbparams = {}
                            for key, val in self.params[
                                'colorbar'].iteritems():
                                if not (isinstance(val, (list, tuple)) and
                                        isinstance(val[0], (
                                            list, tuple, basestring))):
                                    cbparams[key] = self.params['colorbar'][
                                        key]
                                else:
                                    cbparams[key] = \
                                    self.params['colorbar'][key][i]
                            try:
                                self._fig.add_colorbar(mappable=coll,
                                                       ax=ax, **cbparams)
                                self.coloraxes[i] = self._fig.colorbars[-1]
                            except Exception as e:
                                self.logger.error(str(e))
                        elif (new_clim is not None) and \
                                (self.coloraxes[i].get_clim() != new_clim):
                            for cax in self._fig.colorbars:
                                if cax == self.coloraxes[i]:
                                    cax.set_clim(new_clim)
            for ax in self._fig.get_axes(self.AXES_CLASS.name):
                if not k:
                    l, b, w, h = ax.get_position().bounds
                    k = 1
                else:
                    ll, bb, ww, hh = ax.get_position().bounds
                    if w != ww:
                        ax.set_position([ll, bb, w, h])
                ax.relim()
                # ax.autoscale_view(scalex=False)

            self.logger.debug('Figure data updated')
            # add suptitle
            if 'suptitle' not in self.params['init']:
                prefix = ('FFT length: %ss, Overlap: %ss, Stride: %ss -- '
                          % (self.fftlength, self.overlap, self.stride))
                utc = re.sub('\.0+', '',
                             Time(self.epoch, format='gps', scale='utc').iso)
                suffix = 'Last updated: %s UTC (%s)' % (utc, self.epoch)
                self.suptitle = self._fig.suptitle(prefix + suffix)
        for ax in self._fig.get_axes(self.AXES_CLASS.name):
            ax.set_xlim(float(self.epoch - self.duration),
                        float(self.epoch))
            ax.set_epoch(self.epoch)
        self.set_params('refresh')
        self._fig.refresh()
        self.logger.debug('Figure refreshed')


register_monitor(BNSRangeSpectrogramMonitor)
