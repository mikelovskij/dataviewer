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

from epics import caget

import re
from itertools import (cycle, izip_longest)

from astropy.time import Time
from gwpy.timeseries import (TimeSeries, TimeSeriesDict)
from gwpy.spectrogram import (Spectrogram, SpectrogramList)
from gwpy.plotter import (SpectrogramPlot, TimeSeriesAxes)
from gwpy.spectrum import Spectrum
from gwpy.astro.range import inspiral_range_psd

import pickle

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
                 method='welch', filter=None, fhigh=8000, flow=0,
                 window=None, **kwargs):
        super(SpectrogramBuffer, self).__init__(channels, **kwargs)
        self.method = method
        self.window = window
        # todo: maybe it is better to pass some of these kwargs as kwargs to ts.spectrogram
        self.stride = self._param_dict(stride)
        self.fftlength = self._param_dict(fftlength)
        self.overlap = self._param_dict(overlap)
        self.filter = self._param_dict(filter)
        self.fhigh = self._param_dict(fhigh)
        self.flow = self._param_dict(flow)

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

    def from_timeseriesdict(self, tsd, **kwargs):
        # format parameters
        stride = self._param_dict(kwargs.pop('stride', self.stride))
        fftlength = self._param_dict(kwargs.pop('fftlength', self.fftlength))
        overlap = self._param_dict(kwargs.pop('overlap', self.overlap))
        filter = self._param_dict(kwargs.pop('filter', self.filter))
        fhigh = self._param_dict(kwargs.pop('fhigh', self.fhigh))
        flow = self._param_dict(kwargs.pop('flow', self.flow))

        # calculate spectrograms only if state conditon(s) is satisfied
        data = self.DictClass()
        if stateDQ:
            for channel, ts in zip(self.channels, tsd.values()):
                try:
                    specgram = ts.spectrogram(stride[channel],
                                              fftlength=fftlength[channel],
                                              overlap=overlap[channel],
                                              method=self.method,
                                              window=self.window) ** (1 / 2.)
                except ZeroDivisionError:
                    if stride[channel] == 0:
                        raise ZeroDivisionError("Spectrogram stride is 0")
                    elif fftlength[channel] == 0:
                        raise ZeroDivisionError("FFT length is 0")
                    else:
                        raise
                except ValueError:
                    self.logger.error('TimeSeries span: {0},'
                                      ' TimeSeries length: {1}, Stride: {2}'
                                      .format(ts.span,
                                              ts.span[-1] - ts.span[0],
                                              stride[channel]))
                    raise
                if hasattr(channel,
                           'resample') and channel.resample is not None:
                    nyq = float(channel.resample) / 2.
                    nyqidx = int(nyq / specgram.df.value)
                    specgram = specgram[:, :nyqidx]
                if channel in filter and filter[channel]:
                    specgram = specgram.filter(*filter[channel]).copy()
                asd = (Spectrum(specgram.value[-1, :],
                                frequencies=specgram.frequencies,
                                channel=specgram.channel,
                                unit=specgram.unit)) \
                    .crop(flow[channel], fhigh[channel])
                range_spec = inspiral_range_psd(asd ** 2)
                ranges = (range_spec * range_spec.df) ** 0.5
                data[channel] = type(specgram).from_spectra(
                    ranges, epoch=specgram.epoch, dt=specgram.dt,
                    frequencies=asd.frequencies)
        return data


class SpectrogramIterator(SpectrogramBuffer):
    def _next(self):
        new = super(SpectrogramIterator, self)._next()  #todo:  is this iterator even used?
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
        global stateDQ
        # get FFT parameters
        stride = kwargs.pop('stride', 20)
        fftlength = kwargs.pop('fftlength', 1)
        overlap = kwargs.pop('overlap', 0)
        method = kwargs.pop('method', 'welch')
        window = kwargs.pop('window', None)
        filter = kwargs.pop('filter', None)
        ratio = kwargs.pop('ratio', None)
        resample = kwargs.pop('resample', None)
        kwargs.setdefault('interval', stride)
        flow = kwargs.pop('flow')
        fhigh = kwargs.pop('fhigh')
        if kwargs['interval'] % stride:
            raise ValueError("%s interval must be exact multiple of the stride"
                             % type(self).__name__)

        # build 'data' as SpectrogramBuffer
        self.spectrograms = SpectrogramIterator(
            channels, stride=stride, method=method, overlap=overlap,
            fftlength=fftlength, flow=flow, fhigh=fhigh, window=window)
        if isinstance(filter, list):
            self.spectrograms.filter = dict(zip(self.spectrograms.channels,
                                                filter))
        else:
            self.spectrograms.filter = filter
        self.fftlength = fftlength
        self.stride = stride
        self.overlap = overlap
        self.duration = kwargs['duration']

        # build monitor
        kwargs.setdefault('yscale', 'log')
        kwargs.setdefault('gap', 'raise')
        self.plots = kwargs.pop('plots')
        if isinstance(self.plots, str):
            self.plots = (self.plots,)
        # define state vector
        self.stateChannel = kwargs.pop('statechannel', [])
        if isinstance(self.stateChannel, str):
            self.stateChannel = (self.stateChannel,)
        if len(self.stateChannel) == 1:
            stateDQ = caget(self.stateChannel[0])  # is this performed only one time?
        elif len(self.stateChannel) == 2:
            stateChannels = self.stateChannel[0].split(",")
            stateCondition = self.stateChannel[1].split(",")
            for i, lockChanls in enumerate(stateChannels):
                stateDQ = stateDQ and eval(
                    str(caget(lockChanls)) + stateCondition[i])
        elif len(self.stateChannel) > 2:
            raise UserException("Unknown state channels/ conditions")

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
        # sharex = None
        if self.params['draw']['norm'][0] == "linear":
            self.params['draw']['norm'][0] = ''

        def _new_axes():
            ax = self._fig._add_new_axes(self._fig._DefaultAxesClass.name)
            #                                       sharex=sharex)
            ax.set_epoch(float(self.epoch))
            ax.set_xlim(float(self.epoch - self.duration), float(self.epoch))
            return ax

        for n in range(len(self.plots)):
            _new_axes()
            # sharex = _new_axes()
        yscale = self.params['init'].pop('yscale')
        self.set_params('init')
        for ax in self._fig.get_axes(self.AXES_CLASS.name)[:-1]:
            ax.set_xlabel('')
        for i, ax in enumerate(self._fig.get_axes(self.AXES_CLASS.name)):
            if isinstance(yscale, str):
                yscalePlot = yscale
            else:
                yscalePlot = yscale[i]
            ax.set_yscale(yscalePlot)
        self.set_params('refresh')
        # ax.set_xlim(float(self.epoch - self.duration), float(self.epoch))
        return self._fig

    def update_data(self, new, gap='pad', pad=0):
        """Update the `SpectrogramMonitor` data
        This method only applies a ratio, if configured
        """
        pickleFile = 'rangefile'  # to store data at each step

        # check that the stored epoch is bigger then the first buffered data
        if new[self.channels[0]][0].span[0] > self.epoch:
            s = ('The available data starts at gps {0} '
                 'which. is after the end of the last spectrogram(gps {1})'
                 ': a segment is missing and will be skipped!')
            self.logger.warning(s.format(new[self.channels[0]][0].span[0],
                                         self.epoch))
            self.epoch = new[self.channels[0]][0].span[0]

        if not self.spectrograms.data:
            # be sure that the first cycle is syncronized with the buffer
            self.epoch = new[self.channels[0]][0].span[0]
            try:
                #    load the saved spectrograms if there are any # TODO: rework this
                pickleHandle = open(pickleFile, 'r')
                tempSpect = pickle.load(pickleHandle)
                pickleHandle.close()
                self.spectrograms.data[self.channels[0]] = tempSpect
            except:
                pass

        while int(new[self.channels[0]][0].span[-1]) >= int(
                        self.epoch + self.stride):
            # data buffer will return dict of 1-item lists, so reform to tsd
            _new = TimeSeriesDict((key, val[0].crop(self.epoch, self.epoch +
                                                    self.stride))
                                  for key, val in new.iteritems())
            self.logger.debug('Calculating spectrogram from epoch {0}'
                              .format(self.epoch))
            self.spectrograms.append(
                self.spectrograms.from_timeseriesdict(_new))
            self.epoch += self.stride
            self.spectrograms.crop(self.epoch - self.duration)
        self.data = type(self.spectrograms.data)()
        if self.spectrograms.data:  # is this if necessary?
            for channel in self.channels:  # TODO: any way to avoid looping since there is only one channel?
                self.data[channel] = type(self.spectrograms.data[channel])(
                    *self.spectrograms.data[channel])
                pickleHandle = open(pickleFile, 'w')
                pickle.dump(self.spectrograms.data[channel], pickleHandle)
                pickleHandle.close()
        self.epoch = self.data[self.channels[0]][-1].span[-1]
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

            # plot spectrograms
            # newSpectrogram = self.data[channel]
            for i, plotType in enumerate(self.plots):
                ax = next(axes)
                if len(ax.collections):
                    newSpectrogram = self.data[channel][-1:]
                    # remove old spectrogram
                    if float(abs(
                            newSpectrogram[-1].span)) > self.buffer.interval:
                        ax.collections.remove(ax.collections[-1])
                else:
                    newSpectrogram = self.data[channel]
                # plot new data
                pparams = {}
                for key in params:
                    try:
                        if params[key][i]:
                            pparams[key] = params[key][i]
                    except IndexError:
                        pass
                if plotType == "timeseries":
                    for spec in newSpectrogram:
                        rangIntegrand = spec ** 2
                        rangeTimeseriesSquare = rangIntegrand.sum(axis=1)
                        rangeTimeseries = TimeSeries(
                            rangeTimeseriesSquare.value ** 0.5,
                            epoch=rangeTimeseriesSquare.epoch,
                            name=rangeTimeseriesSquare.name,
                            sample_rate=1.0 / rangeTimeseriesSquare.dt.value,
                            unit='Mpc', channel=rangeTimeseriesSquare.name)
                        coll = ax.plot(rangeTimeseries, color='b',
                                       linewidth=3.0, marker='o')
                elif plotType == "spectrogram":
                    # coll = ax.plot(newSpectrogram, label=label, **pparams)
                    for spec in newSpectrogram:
                        coll = ax.plot(spec.copy(), **pparams)
                        # the .copy() is necessary for some reason to avoid a
                        #  weird error in the .sum at line 343 that happens if
                        # the spectrogram is plotted before the timeseries
                    try:
                        coloraxes[i]
                    except IndexError:
                        cbparams = {}
                        for key in self.params['colorbar']:
                            try:
                                if self.params['colorbar'][key][i]:
                                    cbparams[key] = self.params[
                                        'colorbar'][key][i]
                            except IndexError:
                                pass
                        try:
                            self._fig.add_colorbar(mappable=coll, ax=ax,
                                                   **cbparams)
                        except Exception as e:
                            self.logger.error(str(e))
                else:
                    raise UserException("Unknown plot option")
                    # label = None
            k = 0  # for resizing plots to look better
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
