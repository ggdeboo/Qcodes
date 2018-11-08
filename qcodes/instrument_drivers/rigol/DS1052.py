import numpy as np
import re, logging, warnings

from qcodes import VisaInstrument, validators as vals
from qcodes.utils.validators import Ints
from qcodes import ArrayParameter
from qcodes.instrument.channel import InstrumentChannel, ChannelList
from distutils.version import LooseVersion

log = logging.getLogger(__name__)

class ScopeArray(ArrayParameter):
    """Parameter to read out a waveform from the oscilloscope."""
    def __init__(self, name, instrument, channel, raw=False):
        super().__init__(name=name,
                         instrument=instrument,
                         shape=(1400,),
                         label='Voltage',
                         unit='V',
                         setpoint_names=('Time', ),
                         setpoint_labels=('Time', ),
                         setpoint_units=('s',),
                         docstring='holds an array from scope')
        self.channel = channel
        if channel == 1:
            self.chan_str = 'CHAN1'
        elif channel == 2:
            self.chan_str = 'CHAN2'
        else:
            self.chan_str = channel # MATH or FFT

        self.raw = raw
        self.max_read_step = 50
        self.trace_ready = False

    def get_raw(self, return_int=False):
        """Get the waveform data from the oscilloscope.
        
        Returns:
            data (numpy.array):

        Notes:
            The number of points depends on the run mode.
        """
        self.instrument.write(':WAV:DATA? {0}'.format(self.chan_str))
        wvf_data = self.instrument._parent.visa_handle.read_raw()
        data_raw = np.frombuffer(wvf_data[10:], dtype=np.uint8)

        if return_int:
            return data_raw

        if self.chan_str in ['CHAN1', 'CHAN2']:
            midpoint = 125
            points_per_division = 256/10
            vscale = self.instrument.vertical_scale()
            data = -1.0*(data_raw.astype(float) - midpoint) / points_per_division * vscale
            return data
        else:
            return data_raw

    def trace_length(self) -> int:
        """Determine the trace length based on instrument settings."""
        if self.instrument._parent.waveform_points_mode() == 'MAXIMUM':
            if self.instrument._parent.trigger_status() == 'STOP':
                points_mode='RAW'
            else:
                points_mode='NORMAL'

        if points_mode == 'RAW':
            if self.instrument._parent.acquire_mem_depth() == 'LONG':
                if self.instrument._parent.half_channel():
                    return 1048576
                else:
                    return 524288
            else:
                if self.instrument._parent.half_channel():
                    return 8192
                else:
                    return 16384
        else:
            return 600

class RigolDS1000Channel(InstrumentChannel):

    def __init__(self, parent, name, channel):
        super().__init__(parent, name)
        if channel == 1:
            self.chan_str = 'CHAN1'
        elif channel == 2:
            self.chan_str = 'CHAN2'
        else:
            self.chan_str = channel # MATH or FFT

        self.add_parameter("coupling",
                            label="Input Coupling",
                            get_cmd=":CHAN{}:COUP?".format(channel),
                            set_cmd=":CHAN{}:COUP {}".format(channel, "{}"),
                            post_delay=0.2,
                            )
        self.add_parameter("amplitude",
                           get_cmd=":MEASure:VAMP? chan{}".format(channel),
                           get_parser = float,
                           unit='V',
                          )
        self.add_parameter("vertical_scale",
                           get_cmd=":CHANnel{}:SCALe?".format(channel),
                           set_cmd=":CHANnel{}:SCALe {}".format(channel, "{}"),
                           get_parser=float,
                           unit='V',
                          )
        if self.chan_str == 'FFT':
            self.add_parameter("display",
                            get_cmd=":{}:DISP?".format(self.chan_str),
                            set_cmd=":{}:DISP {}".format(self.chan_str, "{}"),
                            val_mapping={False : 'OFF',
                                         True  : 'ON',
                                         })
        else:
            self.add_parameter("display",
                            get_cmd=":{}:DISP?".format(self.chan_str),
                            set_cmd=":{}:DISP {}".format(self.chan_str, "{}"),
                            val_mapping={False : '0',
                                       True : '1',
                                       })
        self.add_parameter("vertical_offset",
                            get_cmd = ":CHAN{}:OFFS?".format(channel),
                            set_cmd = ":CHAN{}:OFFS {}".format(channel, "{}"),
                            get_parser = float
                            )
        #measure commands
        self.add_parameter("VPP",
                    get_cmd=":MEAS:VPP? chan{}".format(channel),
                    get_parser = float
                    )
        self.add_parameter("maximum",
                    get_cmd=":MEAS:VMAX? chan{}".format(channel),
                    get_parser = float
                    )
        self.add_parameter("minimum",
                    get_cmd=":MEAS:VMIN? chan{}".format(channel),
                    get_parser = float
                    )

        # Return the waveform displayed on the screen
        self.add_parameter('memory_depth',
                    get_cmd=':CHAN{}:MEMD?'.format(channel),
                    get_parser=int)
        self.add_parameter('waveform_data',
                           channel=channel,
                           parameter_class=ScopeArray,
                           raw=False
                           )

class RigolDS1000(VisaInstrument):
    """This is an instrument driver for the Rigol series 1000 Oscilloscope."""
    def __init__(self, name: str, address, timeout=2, **kwargs):
        """
        Initialises the Rigol DS1000
        
        Args:
            name (str) of the instrument
            address (string) Visa address for the instrument
            timeout (float) visa timeout
        """
        super().__init__(name, address, device_clear=False, timeout=timeout,
            **kwargs)
        self.connect_message()


        #acquire mode parameters
        self.add_parameter("acquire_type",
                label='type of aquire being used by oscilloscope',
                get_cmd=':ACQ:TYPE?',
                set_cmd='ACQ:TYPE {}',
                vals=vals.Enum('NORMAL', 'AVERAGE', 'PEAKDETECT'))
        self.add_parameter("acquire_mode",
                label="mode of aquisition being used by oscillioscope",
                get_cmd=':ACQ:MODE?',
                set_cmd=':ACQ:MODE {}',
                vals = vals.Enum('RTIM', 'ETIM')
                )
        self.add_parameter("acquire_averages",
                label="the average number in averages mode",
                get_cmd=':ACQ:AVER?',
                set_cmd=':ACQ:AVER {}',
                vals=vals.Enum(2, 4, 8, 16, 32, 64, 128, 256))
        self.add_parameter("acquire_mem_depth",
                label='depth of memory being used by oscilloscope',
                get_cmd=':ACQ:MEMDepth?',
                set_cmd=':ACQ:MEMDepth {}',
                vals = vals.Enum('LONG', 'NORM'))

        #display parameters
        self.add_parameter("display_type",
                label='display type between samples, either vector or dot',
                get_cmd=':DISP:TYPE?',
                set_cmd=':DISP:TYPE {}',
                vals = vals.Enum('VECT', 'DOTS'))
        self.add_parameter("display_grid",
                label='controls/queries if the oscilloscope display has a grid',
                get_cmd=':DISP:GRID?',
                set_cmd=':DISP:GRID {}',
                vals = vals.Enum('FULL', 'HALF', 'NONE'))
        self.add_parameter("display_persist",
                label="controls/queries if the waveform record point persists or refreshes",
                get_cmd=':DISP:PERS?',
                set_cmd=':DiSP:PERS {}',
                vals = vals.Enum('ON', 'OFF'))
        self.add_parameter("display_mnud",
                label="timing for menus hiding automatically",
                get_cmd=':DISP:MNUD?',
                set_cmd=':DISP:MNUD {}',
                vals = vals.Enum('1', '2', '5', '10', '20', 'Infinite')
                )
        self.add_parameter("display_mnus",
                label="status of the menus",
                get_cmd=':DISP:MNUS?',
                set_cmd=':DISP:MNUS {}',
                vals = vals.Enum('ON', 'OFF'))
        #note to self - add display clear parameter
        self.add_parameter("display_brightness",
                label="set and get the brightness of the grid for the oscilloscope",
                get_cmd=':DISP:BRIG?',
                set_cmd=':DISP:BRIG {}',
                get_parser=int,
                vals=vals.Ints(0,32))
        self.add_parameter("display_intensity",
                label="set the brightness of the waveform",
                get_cmd=':DISP:INT?',
                set_cmd=':DISP:INT {}',
                get_parser=int,
                vals=vals.Ints(0,32))

        # Channel parameters
        channels = ChannelList(self, "Channels", RigolDS1000Channel, snapshotable=False)

        for channel_number in [1, 2]:
            channel = RigolDS1000Channel(self, "ch{}".format(channel_number),
                                            channel_number)
            channels.append(channel)

        for channel_number in ['MATH', 'FFT']:
            channel = RigolDS1000Channel(self, "ch_{}".format(channel_number),
                                            channel_number)
            channels.append(channel)

        channels.lock()
        self.add_submodule('channels', channels)

        #timebase parameters
        self.add_parameter("time_base_mode",
                label="the scan mode of the horizontal time base",
                get_cmd=":TIM:MODE?",
                set_cmd=":TIM:MODE {}",
                vals = vals.Enum('MAIN', 'DEL'))
        #note to self, decide how to deal with timebase offset parameter.
        self.add_parameter("time_base_offset",
                label="controls the main and delayed mode offset",
                get_cmd=":TIM:OFFS?",
                set_cmd=":TIM:OFFS {}",
                get_parser=float,
                unit="s")
        
        self.add_parameter("time_base_scale",
                label="the scale of the horizontal time base",
                get_cmd=":TIM:SCAL?",
                set_cmd=":TIM:SCAL {}",
                get_parser=float,
                unit="s/div")
        self.add_parameter("time_base_format",
                label="the format of the time base",
                get_cmd=":TIM:FORM?",
                set_cmd=":TIM:FORM {}",
                vals=vals.Enum('X-Y', 'Y-T', 'SCANNING')
                )
        #trigger commands
        self.add_parameter("trigger_mode",
                label="sets and queries the trigger mode",
                get_cmd=":TRIG:MODE?",
                set_cmd=":TRIG:MODE {}",
                vals = vals.Enum('EDGE', 'PULSE', 'VIDEO', 'SLOPE',
                    'PATTERN','DURATION', 'ALTERNATION')
                )
        self.add_parameter("trigger_source",
                label="sets and queries the trigger source",
                get_cmd = ":TRIG:{}:SOUR?".format(self.trigger_mode()),
                set_cmd = ":TRIG:{}:SOUR {}".format(self.trigger_mode(), "{}")
                )
        self.add_parameter("trigger_level",
                label="sets and queries the trigger level",
                get_cmd=":TRIG:{}:LEV?".format(self.trigger_mode()),
                set_cmd=":TRIG:{}:LEV {}".format(self.trigger_mode(), "{}"),
                get_parser = float,
                unit = "V",
                #vals = vals.permissiveInts(-6*#vertical scale, 6*#vertical scale())
                )
        self.add_parameter("trigger_sweep",
                label="sets and queries the trigger sweep",
                get_cmd =":TRIG:{}:SWE?".format(self.trigger_mode()),
                set_cmd =":TRIG:{}:SWE {}".format(self.trigger_mode(), "{}"),
                #get_cmd = self.sweep_get_cmd,
                #set_cmd = self.sweep_set_cmd,
                vals = vals.Enum("AUTO","NORM","SING")
                )
        self.add_parameter("trigger_coupling",
                label="sets and queries the coupling",
                get_cmd = ":TRIG:{}:COUP?".format(self.trigger_mode()),
                set_cmd = ":TRIG:{}:COUP {}".format(self.trigger_mode(), "{}"),
                vals = vals.Enum("DC","AC","HF","LF")
                )
        self.add_parameter("trigger_holdoff",
                label="sets and queries the trigger holdoff",
                get_cmd = ":TRIG:HOLD?",
                set_cmd = ":TRIG:HOLD {}",
                get_parser = float,
                unit = 's'
               )
        self.add_parameter("trigger_status",
                label="Trigger Status",
                get_cmd=":TRIG:STAT?")

        # Waveform
        self.add_parameter("waveform_points_mode",
                   label="Number of the waveform points",
                   get_cmd=":WAVEFORM:POINTS:MODE?",
                   set_cmd=":WAVEFORM:POINTS:MODE {}",
                   vals = vals.Enum("NORM","MAX","RAW"),
                   get_parser=str,
                  )

    def run(self):
        """Start the acquisition."""
        self.write(':RUN')

    def stop(self):
        """Stop the acquisition."""
        self.write(':STOP')

    def single(self):
        """Single trace acquisition."""
        self.write(':SINGle')

    def force_trigger(self):
        """Force trigger event."""
        self.write('TFORce')

    def beep(self):
        self.write(':BEEP:ACT')

    def half_channel(self):
        """Determine whether the instrument is operating in half channel mode."""
        return (self.channels.ch1.display() != self.channels.ch2.display())
