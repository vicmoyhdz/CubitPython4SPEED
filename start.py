#############################################################################
# start.py
# Modified by Victor Hernandez from GEOCUBIT (Emanuele Casarotti)           #                                             #
# Copyright (c) 2008 Istituto Nazionale di Geofisica e Vulcanologia         #
#                                                                           #
#############################################################################
#                                                                           #
# This program is free software; you can redistribute it and/or modify      #
# it under the terms of the GNU General Public License as published by      #
# the Free Software Foundation; either version 3 of the License, or         #
# (at your option) any later version.                                       #
#                                                                           #
# This program is distributed in the hope that it will be useful,           #
# but WITHOUT ANY WARRANTY; without even the implied warranty of            #
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the             #
# GNU General Public License for more details.                              #
#                                                                           #
# You should have received a copy of the GNU General Public License along   #
# with this program; if not, write to the Free Software Foundation, Inc.,   #
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.               #
#                                                                           #
#############################################################################
from __future__ import print_function

# method to call the library

def start_mpi():
    """
    start mpi, fakempi mimick the mpi function when the run is serial.
    The object mpi is based upon pyMPI.
    it returns: mpiflag,iproc,numproc,mpi
    where
        mpiflag is True if parallel mesh is on
        iproc is the id of the processor (0 if the run is serial)
        numproc is the number of the processor (1 if the run is serial)
        mpi is the mpi object
    """
    try:
        if __name__ != '__console__':
            iproc = 0
    except:
        iproc = 0

    try:
        import mpi
        numproc = mpi.size
        mpiflag = True
        if numproc == 1:
            mpiflag = False
        else:
            iproc = mpi.rank
    except:
        class fakempi(object):

            def __init__(self):
                self.size = 1
                self.rank = 0

            def barrier(self):
                pass

            def allgather(self, value):
                return value

            def gather(self, value):
                return value

            def bcast(self, value):
                return value

            def scatter(self, value):
                return value

            def send(self, value, values):
                return

            def recv(self, value):
                return value, value
        mpi = fakempi()
        numproc = 1
        mpiflag = False
    return mpiflag, iproc, numproc, mpi


def start_cubit(init=False):
    """
    start cubit, it return the cubit object
    init argument set the monitotr files
    """
    import sys
    try:
        cubit.silent_cmd('comment')
    except:
        try:
            import cubit
            import utilities
            cubit.init([""])
        except:
            print('error importing cubit')
            sys.exit()
        try:
            if init:
                from start import start_cfg, start_mpi
                cfg = start_cfg()
                mpiflag, iproc, numproc, mpi = start_mpi()
                version_cubit = utilities.get_cubit_version()
                cubit.cmd('set logging on file "' + cfg.working_dir +
                          '/cubit_proc_' + str(iproc) + '.log"')
                cubit.cmd("set echo on")
                cubit.cmd("set info on")
                if iproc == cfg.monitored_cpu:
                    #print('here3')
                    cubit.cmd("record '" + cfg.working_dir +
                              "/monitor_" + str(cfg.monitored_cpu) + ".jou'")
                    cubit.cmd("set journal on")
                    if version_cubit < 16:
                        cubit.cmd("journal error on")
                    d = cfg.__dict__
                    ks = d.keys()
                    ks.sort()
                    for k in ks:
                        if '__' not in k and '<' not in str(d[k]):
                            if d[k] is not None:
                                txt = str(k) + ' -----> ' + str(d[k])
                                txt = txt.replace("'", "").replace('"', '')
                                cubit.cmd('comment "' + txt + '"')
                else:
                    cubit.cmd("set journal " + cfg.jou_info)
                    if version_cubit < 16:
                        cubit.cmd("journal error " + cfg.jer_info)
                    d = cfg.__dict__
                    ks = d.keys()
                    ks.sort()
                    for k in ks:
                        if '__' not in k and '<' not in str(d[k]):
                            if d[k] is not None:
                                txt = str(k) + ' -----> ' + str(d[k])
                                txt = txt.replace("'", "").replace('"', '')
                                cubit.cmd('comment "' + txt + '"')
                cubit.cmd("set echo " + cfg.echo_info)
                cubit.cmd("set info " + cfg.cubit_info)
                if version_cubit > 13 and version_cubit < 15:
                    print('VERSION CUBIT ', version_cubit)
                    print('VERSIONs of CUBIT > 13 have bugs with merge node')
                    print('the merge option is not operative with this version')
                    print('please download CUBIT 15+')
                else:
                    print('VERSION CUBIT ', version_cubit)
        except:
            print('error start cubit')
            sys.exit()
    return cubit


def start_cfg(filename=None):
    """
    return the object cfg with the parameters of the mesh
    """
    import read_parameter_cfg
    #mpiflag, iproc, numproc, mpi = start_mpi()
    mpiflag=False

    cfg = read_parameter_cfg.readcfg(
            filename=filename, mpiflag=mpiflag)

    import os
    try:
        os.makedirs(cfg.working_dir)
    except OSError:
        pass
    try:
        os.makedirs(cfg.output_dir)
    except OSError:
        pass

    return cfg


def start_numpy():
    """
    import numpy and check if it is installed
    """
    import sys
    try:
        import numpy
    except:
        print('error importing numpy')
        print('please check if numpy is correctly installed')
        sys.exit(1)
    return numpy
