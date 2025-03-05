#############################################################################
# main.py                                                                   #
# Modified by Victor Hernandez from GEOCUBIT (Emanuele Casarotti)           #
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

try:
    import start as start
    cubit = start.start_cubit()
except:
    try:
        import cubit
    except:
        print('error importing cubit, check if cubit is installed')
        pass


def volumes(filename=None):
    """create the volumes"""
    import time
    import datetime
    starttime=time.time()

    import start as start
    print('Starting volume creation')
    cfg = start.start_cfg(filename=filename)
    #
    sandwich = 'verticalsandwich_from_ascii_regulargrid'
    if cfg.volume_type == 'layercake_from_ascii_regulargrid':
        if cfg.onlysurface:
            layercake_from_ascii_regulargrid(
            filename=filename, onlysurface=True)
        else:
            layercake_from_ascii_regulargrid(filename=filename)
    elif cfg.volume_type == sandwich:
        layercake_from_ascii_regulargrid(
            filename=filename, verticalsandwich=True)
    else:
        print('volumen_type not defined/wrong')
        
    if cfg.block_firstlayer:
        blockTop()

    if cfg.onlysurface:
        if cfg.enlarge_boundary:
            cubit.cmd('set node constraint off')
            
            cubit.cmd('group \'left_boundary\' add \
            Node in edge in group edge_left_boundary')
            cubit.cmd('node in left_boundary move x -5')

            cubit.cmd('group \'right_boundary\' add \
            Node in edge in group edge_right_boundary')
            cubit.cmd('node in right_boundary move x 5')

            cubit.cmd('group \'lower_boundary\' add \
            Node in edge in group edge_lower_boundary')
            cubit.cmd('node in lower_boundary move y -5')

            cubit.cmd('group \'upper_boundary\' add \
            Node in edge in group edge_upper_boundary')
            cubit.cmd('node in upper_boundary move y 5')

    #Save volumes meshed unmerged
    if cfg.onlysurface:
        cubitcommand = 'save as "' + cfg.output_dir + '/' + \
                'surface.cub5' + '"  overwrite'
        cubit.cmd(cubitcommand) 
    else:
        cubitcommand = 'save as "' + cfg.output_dir + '/' + \
                'vol.cub5' + '"  overwrite'
        cubit.cmd(cubitcommand) 

    if cfg.merging:
        numproces = (cfg.end_chunk_xi - cfg.start_chunk_xi) * (cfg.end_chunk_eta - cfg.start_chunk_eta)
        if numproces > 1:
            if cfg.onlysurface:
                merging_surface()
                cubit.cmd('compress all')
                cubitcommand = 'save as "' + cfg.output_dir + '/' + \
                    'surface_merged.cub5' + '"  overwrite'
                cubit.cmd(cubitcommand) 
            else:
                # merging()
                cubit.cmd('compress all')
                # cubitcommand = 'save as "' + cfg.output_dir + '/' + \
                #      'vol_merged.cub5' + '"  overwrite'
                # cubit.cmd(cubitcommand)    

    #Curve refinement
    if cfg.curve_refinement:
        if not cfg.merging:
            cubit.cmd('disassociate mesh from volume all')
            cubit.cmd('del vol all')
        try:
            if cfg.onlysurface:
                if isinstance(cfg.curvename, str):
                    surface_refinement(filename)  
            else:
                if isinstance(cfg.curvename, str):
                    basin_refinement(filename)     
        except:
            pass
     
    #Export mesh
    if cfg.export_mesh:
        if cfg.onlysurface:
            from utilities import export_surfacemesh
            export_surfacemesh(1,filename)
        else:
            block_listt=cubit.parse_cubit_list('block', 'all')
            block_list_unsorted=block_listt[:-2]
            block_list = tuple(sorted(block_list_unsorted))
            from utilities import export_mesh
            export_mesh(block_list,[100],filename)
    #Export LS
    if cfg.export_ls:
        if cfg.onlysurface:
            from utilities import export_LS
            export_LS(1,1,filename)
        else:
            print('LS exporting available just when using onlysurface = True')

    endtime=time.time()
    convert = str(datetime.timedelta(seconds = endtime-starttime))
    print('Total elapsed time: ',convert)


def ordering_surfaces(list_surfaces):
    list_z = []
    for s in list_surfaces:
        _, _, z = cubit.get_center_point("surface", s)
        list_z.append(z)
    ord_list_surfaces = [s for s, _z in sorted(
        zip(list_surfaces, list_z), key=lambda x: (x[1]))]
    return ord_list_surfaces


def onlyvolumes():
    list_surfaces = cubit.parse_cubit_list("surface", "all")
    ord_list_surfaces = ordering_surfaces(list_surfaces)
    for s1, s2 in zip(ord_list_surfaces[:-1], ord_list_surfaces[1:]):
        create_volume(s1, s2, method=None)
    cubitcommand = 'del surface all'
    cubit.cmd(cubitcommand)
    list_vol = cubit.parse_cubit_list("volume", "all")
    if len(list_vol) > 1:
        cubitcommand = 'imprint volume all'
        cubit.cmd(cubitcommand)
        cubitcommand = 'merge all'
        cubit.cmd(cubitcommand)


def surfaces(filename=None,):
    """create the volumes"""
    import start as start
    sandwich = 'verticalsandwich_from_ascii_regulargrid'
    print('volume')
    cfg = start.start_cfg(filename=filename)
    if cfg.volume_type == 'layercake_from_ascii_regulargrid':
        layercake_from_ascii_regulargrid(
            filename=filename, onlysurface=True)
    elif cfg.volume_type == sandwich:
        layercake_from_ascii_regulargrid(
            filename=filename, verticalsandwich=True, onlysurface=True)
    else:
        print('volumen_type not defined/wrong')


def layercake_from_ascii_regulargrid(filename=None,
                                                     verticalsandwich=False,
                                                     onlysurface=False):
    import time
    import datetime
    starttimem=time.time()
    
    import start as start
    #
    mpiflag, ipro, numproc, mpi = start.start_mpi()
    mpiflag = False
    ipro = 0
    #
    numpy = start.start_numpy()
    cfg = start.start_cfg(filename=filename)

    from utilities import geo2utm, cubit_command_check
    #
    print('Start x:',cfg.start_chunk_xi)
    print('Start y:',cfg.start_chunk_eta)
    print('end x:',cfg.end_chunk_xi)
    print('end y:',cfg.end_chunk_eta)

    numprocesx=(cfg.end_chunk_xi - cfg.start_chunk_xi)
    numprocesy=(cfg.end_chunk_eta - cfg.start_chunk_eta)
    numproces =  numprocesx*numprocesy
    print('Number of Chunks:', numproces)

    command = "comment '" + "PROC: " + str(ipro) + "/" + str(numproces) + " '"
    cubit_command_check(ipro, command, stop=True)
    if verticalsandwich:
        cubit.cmd("comment 'Starting Vertical Sandwich'")
    #
    # number of points in the files that describe the topography

    if cfg.geometry_format == 'ascii':
        import local_volume as lvolume
        coordx, coordy, elev, nx, ny = lvolume.read_grid(filename, ipro)

        print(' end of receving grd files ')
        if cfg.nproc_xi==1:
            nx_segment=nx
        else:
            nx_segment = int(nx / cfg.nproc_xi) + 1
        if cfg.nproc_eta==1:
            ny_segment=ny
        else:
            ny_segment = int(ny / cfg.nproc_eta) + 1

    elif cfg.geometry_format == 'regmesh':  # !!!!!!!

        if cfg.depth_bottom != cfg.zlayer[0]:
            print ('the bottom of the block is at different \
                depth than zlayer[0] in the configuration file')
        nx = cfg.nproc_xi + 1
        ny = cfg.nproc_eta + 1
        nx_segment = 2
        ny_segment = 2

        elev = numpy.zeros([nx, ny, cfg.nz], float)
        coordx = numpy.zeros([nx, ny], float)
        coordy = numpy.zeros([nx, ny], float)

        # length of x slide for chunk
        xlength = (cfg.xmax - cfg.xmin) / float(cfg.nproc_xi)
        # length of y slide for chunk
        ylength = (cfg.ymax - cfg.ymin) / float(cfg.nproc_eta)

        for i in range(0, cfg.nz):
            elev[:, :, i] = cfg.zlayer[i]

        icoord = 0
        for iy in range(0, ny):
            for ix in range(0, nx):
                icoord = icoord + 1
                coordx[ix, iy] = cfg.xmin + xlength * (ix)
                coordy[ix, iy] = cfg.ymin + ylength * (iy)
        
        from utilities import DoRotation
        coordx,coordy = DoRotation(cfg.xmin,cfg.ymin,coordx, coordy, -1*cfg.rot_deg)

    #
    print('end of building grid ' + str(ipro))
    print('number of point: ', numpy.size(coordx,0) * numpy.size(coordx,1))

    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # for each CHUNK
    indexy=0
    for iypro in range(cfg.start_chunk_eta,cfg.end_chunk_eta):
        indexx=0
        for ixpro in range(cfg.start_chunk_xi,cfg.end_chunk_xi):
            ipro = (cfg.nproc_xi*iypro) + ixpro
            print('C H U N K:',ipro)

            # get icpuy,icpux values
            icpux = ipro % cfg.nproc_xi
            icpuy = int(ipro / cfg.nproc_xi)
            #
            nxmin_cpu = (nx_segment - 1) * (icpux)
            nymin_cpu = (ny_segment - 1) * (icpuy)
            if ixpro == cfg.end_chunk_xi -1:
                if cfg.end_chunk_xi == cfg.nproc_xi:
                    nxmax_cpu = nx - 1
                else:
                    nxmax_cpu = min(nx - 1, (nx_segment - 1) * (icpux + 1))
            else:
                nxmax_cpu = (nx_segment - 1) * (icpux + 1)
            if iypro == cfg.end_chunk_eta -1:
                if cfg.end_chunk_eta == cfg.nproc_eta:
                    nymax_cpu = ny - 1
                else:
                    nymax_cpu = min(ny - 1, (ny_segment - 1) * (icpuy + 1))
            else:
                nymax_cpu = (ny_segment - 1) * (icpuy + 1)

            isurf = 0
            isurfaces = []

            # create vertex
            if verticalsandwich:
                nlayer = cfg.nxvol
            else:
                nlayer = cfg.nz

            for inz in range(0, nlayer):
                if cfg.sea and inz == cfg.nz - 1:  # sea layer
                    sealevel = True
                    bathymetry = False
                elif cfg.sea and inz == cfg.nz - 2:  # bathymetry layer
                    sealevel = False
                    bathymetry = True
                else:
                    sealevel = False
                    bathymetry = False
                #print(sealevel, bathymetry)

                if cfg.bottomflat and inz == 0:  # bottom layer
                    #
                    if cfg.geometry_format == 'ascii' and not verticalsandwich:
                        lv = cubit.get_last_id("vertex")

                        x_current, y_current = (coordx[nxmin_cpu, nymin_cpu], coordy[
                                                nxmin_cpu, nymin_cpu])
                        cubitcommand = 'create vertex ' + \
                            str(x_current) + ' ' + str(y_current) + \
                            ' ' + str(cfg.depth_bottom)
                        cubit.cmd(cubitcommand)
                        #
                        x_current, y_current = (coordx[nxmin_cpu, nymax_cpu], coordy[
                                                nxmin_cpu, nymax_cpu])
                        cubitcommand = 'create vertex ' + \
                            str(x_current) + ' ' + str(y_current) + \
                            ' ' + str(cfg.depth_bottom)
                        cubit.cmd(cubitcommand)
                        #
                        x_current, y_current = (coordx[nxmax_cpu, nymax_cpu], coordy[
                                                nxmax_cpu, nymax_cpu])
                        cubitcommand = 'create vertex ' + \
                            str(x_current) + ' ' + str(y_current) + \
                            ' ' + str(cfg.depth_bottom)
                        cubit.cmd(cubitcommand)
                        #
                        x_current, y_current = (coordx[nxmax_cpu, nymin_cpu], coordy[
                                                nxmax_cpu, nymin_cpu])
                        cubitcommand = 'create vertex ' + \
                            str(x_current) + ' ' + str(y_current) + \
                            ' ' + str(cfg.depth_bottom)
                        cubit.cmd(cubitcommand)
                        #
                        lv2 = cubit.get_last_id("vertex")

                        cubitcommand = 'create surface vertex ' + \
                            str(lv + 1) + ' to ' + str(lv2)
                        cubit.cmd(cubitcommand)
                        #
                        lastsurface = cubit.get_last_id("surface")
                        isurfaces.append(lastsurface)
                        isurf = isurf + 1
                    else:
                        lv = cubit.get_last_id("vertex")
                        x_current, y_current = geo2utm(coordx[nxmin_cpu, nymin_cpu],
                                                    coordy[nxmin_cpu, nymin_cpu],
                                                    'utm')
                        cubitcommand = 'create vertex ' + \
                            str(x_current) + ' ' + str(y_current) + \
                            ' ' + str(cfg.depth_bottom)
                        cubit.cmd(cubitcommand)
                        #
                        x_current, y_current = geo2utm(coordx[nxmin_cpu, nymax_cpu],
                                                    coordy[nxmin_cpu, nymax_cpu],
                                                    'utm')
                        cubitcommand = 'create vertex ' + \
                            str(x_current) + ' ' + str(y_current) + \
                            ' ' + str(cfg.depth_bottom)
                        cubit.cmd(cubitcommand)
                        #
                        if verticalsandwich:
                            x_current, y_current = geo2utm(
                                coordx[nxmin_cpu, nymax_cpu],
                                coordy[nxmin_cpu, nymax_cpu],
                                'utm')
                            cubitcommand = 'create vertex ' + \
                                str(x_current) + ' ' + str(y_current) + ' ' + str(0)
                            cubit.cmd(cubitcommand)
                            #
                            x_current, y_current = geo2utm(
                                coordx[nxmin_cpu, nymin_cpu],
                                coordy[nxmin_cpu, nymin_cpu],
                                'utm')
                            cubitcommand = 'create vertex ' + \
                                str(x_current) + ' ' + str(y_current) + ' ' + str(0)
                            cubit.cmd(cubitcommand)
                        else:
                            x_current, y_current = geo2utm(
                                coordx[nxmax_cpu, nymax_cpu],
                                coordy[nxmax_cpu, nymax_cpu],
                                'utm')
                            cubitcommand = 'create vertex ' + \
                                str(x_current) + ' ' + str(y_current) + \
                                ' ' + str(cfg.depth_bottom)
                            cubit.cmd(cubitcommand)
                            #
                            x_current, y_current = geo2utm(
                                coordx[nxmax_cpu, nymin_cpu],
                                coordy[nxmax_cpu, nymin_cpu],
                                'utm')
                            cubitcommand = 'create vertex ' + \
                                str(x_current) + ' ' + str(y_current) + \
                                ' ' + str(cfg.depth_bottom)
                            cubit.cmd(cubitcommand)
                        #
                        lv2 = cubit.get_last_id("vertex")
                        cubitcommand = 'create surface vertex ' + \
                            str(lv + 1) + ' to ' + str(lv2)
                        cubit.cmd(cubitcommand)
                        #
                        lastsurface = cubit.get_last_id("surface")
                        isurfaces.append(lastsurface)
                        isurf = isurf + 1

                elif cfg.topflat and inz == nlayer - 1:
                    if cfg.geometry_format == 'ascii' and not verticalsandwich:
                        lv = cubit.get_last_id("vertex")

                        x_current, y_current = (coordx[nxmin_cpu, nymin_cpu], coordy[
                                                nxmin_cpu, nymin_cpu])
                        cubitcommand = 'create vertex ' + \
                            str(x_current) + ' ' + str(y_current) + \
                            ' ' + str(cfg.depth_top)
                        cubit.cmd(cubitcommand)
                        #
                        x_current, y_current = (coordx[nxmin_cpu, nymax_cpu], coordy[
                                                nxmin_cpu, nymax_cpu])
                        cubitcommand = 'create vertex ' + \
                            str(x_current) + ' ' + str(y_current) + \
                            ' ' + str(cfg.depth_top)
                        cubit.cmd(cubitcommand)
                        #
                        x_current, y_current = (coordx[nxmax_cpu, nymax_cpu], coordy[
                                                nxmax_cpu, nymax_cpu])
                        cubitcommand = 'create vertex ' + \
                            str(x_current) + ' ' + str(y_current) + \
                            ' ' + str(cfg.depth_top)
                        cubit.cmd(cubitcommand)
                        #
                        x_current, y_current = (coordx[nxmax_cpu, nymin_cpu], coordy[
                                                nxmax_cpu, nymin_cpu])
                        cubitcommand = 'create vertex ' + \
                            str(x_current) + ' ' + str(y_current) + \
                            ' ' + str(cfg.depth_top)
                        cubit.cmd(cubitcommand)
                        #
                        lv2 = cubit.get_last_id("vertex")

                        cubitcommand = 'create surface vertex ' + \
                            str(lv + 1) + ' to ' + str(lv2)
                        cubit.cmd(cubitcommand)
                        #
                        lastsurface = cubit.get_last_id("surface")
                        isurfaces.append(lastsurface)
                        isurf = isurf + 1
                    else:
                        print("top_flat is not a valid option for sandwich")

                else:
                    print(inz, 'layer')
                    if cfg.geometry_format == 'regmesh':
                        if verticalsandwich:
                            zvertex = cfg.xwidth[inz]
                            lv = cubit.get_last_id("vertex")
                            x_current, y_current = geo2utm(
                                coordx[nxmax_cpu, nymin_cpu],
                                coordy[nxmax_cpu, nymin_cpu],
                                'utm')
                            x_current = x_current + zvertex
                            cubitcommand = 'create vertex ' + \
                                str(x_current) + ' ' + str(y_current) + \
                                ' ' + str(cfg.depth_bottom)
                            cubit.cmd(cubitcommand)
                            #
                            x_current, y_current = geo2utm(
                                coordx[nxmax_cpu, nymax_cpu],
                                coordy[nxmax_cpu, nymax_cpu],
                                'utm')
                            x_current = x_current + zvertex
                            cubitcommand = 'create vertex ' + \
                                str(x_current) + ' ' + str(y_current) + \
                                ' ' + str(cfg.depth_bottom)
                            cubit.cmd(cubitcommand)
                            #
                            x_current, y_current = geo2utm(
                                coordx[nxmax_cpu, nymax_cpu],
                                coordy[nxmax_cpu, nymax_cpu],
                                'utm')
                            x_current = x_current + zvertex
                            cubitcommand = 'create vertex ' + \
                                str(x_current) + ' ' + str(y_current) + ' ' + str(0)
                            cubit.cmd(cubitcommand)
                            #
                            x_current, y_current = geo2utm(
                                coordx[nxmax_cpu, nymin_cpu],
                                coordy[nxmax_cpu, nymin_cpu],
                                'utm')
                            x_current = x_current + zvertex
                            cubitcommand = 'create vertex ' + \
                                str(x_current) + ' ' + str(y_current) + ' ' + str(0)
                            cubit.cmd(cubitcommand)
                        else:
                            zvertex = cfg.zlayer[inz]
                            lv = cubit.get_last_id("vertex")
                            x_current, y_current = geo2utm(
                                coordx[nxmin_cpu, nymin_cpu],
                                coordy[nxmin_cpu, nymin_cpu],
                                'utm')
                            cubitcommand = 'create vertex ' + \
                                str(x_current) + ' ' + \
                                str(y_current) + ' ' + str(zvertex)
                            cubit.cmd(cubitcommand)
                            #
                            x_current, y_current = geo2utm(
                                coordx[nxmin_cpu, nymax_cpu],
                                coordy[nxmin_cpu, nymax_cpu],
                                'utm')
                            cubitcommand = 'create vertex ' + \
                                str(x_current) + ' ' + \
                                str(y_current) + ' ' + str(zvertex)
                            cubit.cmd(cubitcommand)
                            #
                            x_current, y_current = geo2utm(
                                coordx[nxmax_cpu, nymax_cpu],
                                coordy[nxmax_cpu, nymax_cpu],
                                'utm')
                            cubitcommand = 'create vertex ' + \
                                str(x_current) + ' ' + \
                                str(y_current) + ' ' + str(zvertex)
                            cubit.cmd(cubitcommand)
                            #
                            x_current, y_current = geo2utm(
                                coordx[nxmax_cpu, nymin_cpu],
                                coordy[nxmax_cpu, nymin_cpu],
                                'utm')
                            cubitcommand = 'create vertex ' + \
                                str(x_current) + ' ' + \
                                str(y_current) + ' ' + str(zvertex)
                            cubit.cmd(cubitcommand)
                        #
                        cubitcommand = 'create surface vertex ' + \
                            str(lv + 1) + ' ' + str(lv + 2) + ' ' + \
                            str(lv + 3) + ' ' + str(lv + 4)
                        cubit.cmd(cubitcommand)
                        #
                        lastsurface = cubit.get_last_id("surface")
                        isurfaces.append(lastsurface)
                        isurf = isurf + 1
                        
                    elif cfg.geometry_format == 'ascii':
                        
                        if len(cfg.filename) == 1 and inz < nlayer - 1:
                            zvertex = cfg.zlayer[inz]
                            lv = cubit.get_last_id("vertex")

                            x_current, y_current = (coordx[nxmin_cpu, nymin_cpu], coordy[
                                                    nxmin_cpu, nymin_cpu])
                            cubitcommand = 'create vertex ' + \
                                str(x_current) + ' ' + str(y_current) + \
                                ' ' + str(zvertex)
                            cubit.cmd(cubitcommand)
                            #
                            x_current, y_current = (coordx[nxmin_cpu, nymax_cpu], coordy[
                                                    nxmin_cpu, nymax_cpu])
                            cubitcommand = 'create vertex ' + \
                                str(x_current) + ' ' + str(y_current) + \
                                ' ' + str(zvertex)
                            cubit.cmd(cubitcommand)
                            #
                            x_current, y_current = (coordx[nxmax_cpu, nymax_cpu], coordy[
                                                    nxmax_cpu, nymax_cpu])
                            cubitcommand = 'create vertex ' + \
                                str(x_current) + ' ' + str(y_current) + \
                                ' ' + str(zvertex)
                            cubit.cmd(cubitcommand)
                            #
                            x_current, y_current = (coordx[nxmax_cpu, nymin_cpu], coordy[
                                                    nxmax_cpu, nymin_cpu])
                            cubitcommand = 'create vertex ' + \
                                str(x_current) + ' ' + str(y_current) + \
                                ' ' + str(zvertex)
                            cubit.cmd(cubitcommand)
                            #
                            lv2 = cubit.get_last_id("vertex")

                            cubitcommand = 'create surface vertex ' + \
                                str(lv + 1) + ' to ' + str(lv2)
                            cubit.cmd(cubitcommand)
                            #
                            lastsurface = cubit.get_last_id("surface")
                            isurfaces.append(lastsurface)
                            isurf = isurf + 1   
                        
                        else: #topo surface
                            vertex = []

                            for iy in range(nymin_cpu, nymax_cpu + 1):
                                for ix in range(nxmin_cpu, nxmax_cpu + 1):
                                    zvertex = elev[ix, iy, inz]
                                    x_current, y_current = (coordx[ix, iy], coordy[ix, iy])
                                    #
                                    vertex.append(' location ' + str(x_current) +
                                                ' ' + str(y_current) + ' ' +
                                                str(zvertex))
                            #
                            print('proc', ipro, 'vertex list created....', len(vertex))

                            uline = []
                            vline = []
                            iv = 0

                            cubit.cmd('set info off')
                            cubit.cmd('set echo off')
                            cubit.cmd('set journal off')

                            for iy in range(0, nymax_cpu - nymin_cpu + 1):
                                positionx = ''
                                for ix in range(0, nxmax_cpu - nxmin_cpu + 1):
                                    positionx = positionx + vertex[iv]
                                    iv = iv + 1
                                command = 'create curve spline' + positionx
                                cubit.cmd(command)
                                uline.append(cubit.get_last_id("curve"))
                            for ix in range(0, nxmax_cpu - nxmin_cpu + 1):
                                positiony = ''
                                for iy in range(0, nymax_cpu - nymin_cpu + 1):
                                    positiony = positiony + \
                                        vertex[ix + iy * (nxmax_cpu - nxmin_cpu + 1)]
                                command = 'create curve spline ' + positiony
                                cubit.cmd(command)
                                vline.append(cubit.get_last_id("curve"))
                            #
                            cubit.cmd("set info  " + cfg.cubit_info)
                            cubit.cmd("set echo " + cfg.echo_info)
                            cubit.cmd("set journal " + cfg.jou_info)
                            #
                            print('proc', ipro, 'lines created....',
                                len(uline), '*', len(vline))
                            umax = max(uline)
                            umin = min(uline)
                            vmax = max(vline)
                            vmin = min(vline)
                            ner = cubit.get_error_count()
                            cubitcommand = 'create surface net u curve ' + \
                                str(umin) + ' to ' + str(umax) + ' v curve ' + \
                                str(vmin) + ' to ' + str(vmax) + ' heal'
                            cubit.cmd(cubitcommand)
                            ner2 = cubit.get_error_count()
                            if ner == ner2:
                                command = "del curve all"
                                cubit.cmd(command)
                                lastsurface = cubit.get_last_id("surface")
                                isurfaces.append(lastsurface)
                                isurf = isurf + 1
                            else:
                                raise ValueError('error creating the surface')
                        #
                    else:
                        raise ValueError(
                            'check geometry_format, it should be ascii or regmesh')
                        #
                cubitcommand = 'del vertex all'
                cubit.cmd(cubitcommand)
            #!create volume
            ivol = []
            if not onlysurface:
                if nlayer == 1:
                    nsurface = 2
                else:
                    nsurface = nlayer
                for inz in range(0, nsurface-1):
                    ner = cubit.get_error_count()
                    basesurf = isurfaces[inz]
                    topsurf = isurfaces[inz+1]
                    create_volume(basesurf, topsurf, method=cfg.volumecreation_method)
                    ner2 = cubit.get_error_count()
                    if ner != ner2:
                        if cfg.volumecreation_method != 'loft':
                            ner = cubit.get_error_count()
                            create_volume(inz, inz + 1, method='loft')
                            ner2 = cubit.get_error_count()
                            if ner != ner2:
                                print('ERROR creating volume')
                                break
                        else:
                            print('ERROR creating volume')
                            break
                    lastvol = cubit.get_last_id("volume")
                    ivol.append(lastvol)
                
                cubitcommand = 'del surface all'
                cubit.cmd(cubitcommand)
                
                # Merge vertical chunks
                for inz in range(0, nsurface-2):
                    cmd = 'merge volume ' + str(ivol[inz]) + ' ' + str(ivol[inz+1])
                    cubit.cmd(cmd)
                    
            if cfg.meshing:
                if cfg.onlysurface:
                    # Mesh each surface chunk                 
                    ll = cubit.parse_cubit_list("surface", "all")
                    lprevious=len(ll)-nlayer
                    import mesh_volume
                    mesh_volume.meshsurface(filename,ipro,lprevious)

                    definesurface_blocks(nx_segment,ny_segment,lprevious,filename)

                else:       
                    # Mesh each chunk                   
                    ll = cubit.parse_cubit_list("volume", "all")
                    lprevious=len(ll)-nsurface+1
                    import mesh_volume
                    mesh_volume.mesh(filename,ipro,lprevious)

                    if indexx > 0:
                        cubit.cmd("group 'coincident_lateral_nodes' add node in group next_lateral_nodes") 
                        cubit.cmd("delete group next_lateral_nodes") 

                    # Assign blocks
                    define_blocks(nx_segment,ny_segment,lprevious,filename)  

                if cfg.merging and not cfg.onlysurface:
                    cubit.cmd("disassociate mesh from volume all")
                    cubit.cmd('del vol all')
                    cubit.cmd('delete group disassociated_elements')
                    if indexx > 0:
                        factor, minvalue, maxvalue = prepare_equivalence_new(name_group='intermediatelateral')
                        tol = minvalue / 20.
                        step_tol = minvalue / 20.

                        cubit.cmd("delete group intermediatelateral") 

                        isempty = False
                        while not isempty:
                            isempty = merging_node_new(tol, clean=True, graphic_debug=False)
                            tol = tol + step_tol
                            if tol > maxvalue * 1.5:
                                raise MergingError(
                                'tolerance greater than the max length of the edges, \
                                please check the mesh')
                            
                    if indexy>0 and indexx == (numprocesx-1):
                        factor, minvalue, maxvalue = prepare_equivalence_new(name_group='lateral')
                        tol = minvalue / 20.
                        step_tol = minvalue / 20.
                        cubit.cmd("group 'coincident_lateral_nodes' add node in face in group lateral")
                        cubit.cmd("delete group lateral") 

                        isempty = False
                        while not isempty:
                            isempty = merging_node_new(tol, clean=True, graphic_debug=False)
                            tol = tol + step_tol
                            if tol > maxvalue * 1.5:
                                raise MergingError(
                                'tolerance greater than the max length of the edges, \
                                please check the mesh')

                    if indexx == (numprocesx-1) and indexy < (numprocesy-1):
                        cubit.cmd("group 'lateral' add face in group next_lateral")
                        cubit.cmd("delete group next_lateral")
                                
                if cfg.disassociate:
                    cubit.cmd("disassociate mesh from volume all")
                    cubit.cmd('del vol all')
                    cubit.cmd("disassociate mesh from surface all")
                    cubit.cmd('del surface all')
                    cubit.cmd('delete group disassociated_elements')

            indexx=indexx+1
        indexy=indexy+1


    endtimem=time.time()
    convertm = str(datetime.timedelta(seconds = endtimem-starttimem))
    print('Elapsed time geometry and meshing:',convertm)

###################################################################################################

def basin_refinement(filename):
    print('Basin refinement')
    import time
    import datetime
    import start as start
    starttimer=time.time()
    cfg = start.start_cfg(filename)
    curvename = cfg.curvename
    if cfg.num_curve_refinement>1:
        curvename2 = cfg.curvename2

    from utilities import load_curves
    import mesh_volume

    block_listt=cubit.parse_cubit_list('block', 'all')
    block_list_unsorted=block_listt[:-2]
    block_list = tuple(sorted(block_list_unsorted))

    for i in range(0,cfg.num_curve_refinement):
        
        if i==0:
            curves=load_curves(curvename)
            mesh_volume.refine_inside_curve(curves, ntimes=1, depth=1, block=1001)
            cmd = 'delete curve all'
            cubit.cmd(cmd)
            lv=cubit.parse_cubit_list("hex","with not block_assigned")
            command = "block " + str(block_list[-1]+1) + " add hex " + ' '.join(str(x) for x in lv)
            cubit.cmd(command)
            command ="group \'ntop\' add node in hex in block " + str(block_list[-1]+1)
            cubit.cmd(command)
            group1 = cubit.get_id_from_name("ntop")
            nodes50 = list(cubit.get_group_nodes(group1))
            nodes=[]
            for inodes in nodes50:
                nh = cubit.parse_cubit_list('hex', 'in node ' + str(inodes))
                if len(nh)==4:
                    nodes.append(inodes)
            command="group \'topRefined\' add node " + ' '.join(str(x) for x in nodes)
            cubit.cmd(command)
            cubit.cmd('delete group ntop')
        else:
            cubit.cmd('delete group topRefined')
            curves=load_curves(curvename2)
            mesh_volume.refine_inside_curve(curves, ntimes=1, depth=1, block=1001, nodes=nodes)
            cmd = 'delete curve all'
            cubit.cmd(cmd)
            lv=cubit.parse_cubit_list("hex","with not block_assigned")
            command = "block " + str(block_list[-1]+1) + " add hex " + ' '.join(str(x) for x in lv)
            cubit.cmd(command)
            command ="group \'ntop\' add node in hex in block " + str(block_list[-1]+1)
            cubit.cmd(command)
            group1 = cubit.get_id_from_name("ntop")
            nodes50 = list(cubit.get_group_nodes(group1))
            nodes=[]
            for inodes in nodes50:
                nh = cubit.parse_cubit_list('hex', 'in node ' + str(inodes))
                if len(nh)==4:
                    nodes.append(inodes)
            command="group \'topRefined\' add node " + ' '.join(str(x) for x in nodes)
            cubit.cmd(command)
            cubit.cmd('delete group ntop')

    endtimer=time.time()
    convertr = str(datetime.timedelta(seconds = endtimer-starttimer))
    print('Elapsed time refinement:',convertr)

    #Save merged REFINED
    cubitcommand = 'save as "' + cfg.output_dir + '/' + \
                'vol_merged_refined.cub5' + '"  overwrite'
    cubit.cmd(cubitcommand) 

def surface_refinement(filename):
    print('Basin refinement')
    import time
    import datetime
    import start as start
    starttimer=time.time()
    cfg = start.start_cfg(filename)
    curvename = cfg.curvename
    if cfg.num_curve_refinement>1:
        curvename2 = cfg.curvename2

    from utilities import load_curves
    import mesh_volume

    for i in range(0,cfg.num_curve_refinement):
        
        if i==0:
            curves=load_curves(curvename)
            mesh_volume.refine_surface_curve(curves, ntimes=1, depth=1, block=1)
            cmd = 'delete curve all'
            cubit.cmd(cmd)
            lv=cubit.parse_cubit_list("tri","with not block_assigned")
            command = "block 1 add tri " + ' '.join(str(x) for x in lv)
            cubit.cmd(command)
        else:
            curves=load_curves(curvename2)
            mesh_volume.refine_surface_curve(curves, ntimes=2, depth=1, block=1)
            cmd = 'delete curve all'
            cubit.cmd(cmd)
            lv=cubit.parse_cubit_list("tri","with not block_assigned")
            command = "block 1 add tri " + ' '.join(str(x) for x in lv)
            cubit.cmd(command)

    endtimer=time.time()
    convertr = str(datetime.timedelta(seconds = endtimer-starttimer))
    print('Elapsed time refinement:',convertr)

    #Save merged REFINED
    cubitcommand = 'save as "' + cfg.output_dir + '/' + \
                'surface_merged_refined.cub5' + '"  overwrite'
    cubit.cmd(cubitcommand)                               
    
class MergingError(Exception):
    pass

def merging_surface():
    import time
    import datetime
    starttime_mer=time.time()

    cubit.cmd('disassociate mesh from volume all')
    cubit.cmd('del vol all') 

    length = {}
    cmd = "group 'tmpn' add edge in face in group lateral"
    cubit.cmd(cmd)
    ge = cubit.get_id_from_name("tmpn")
    e1 = cubit.get_group_edges(ge)
    lengthmin = 1e9
    for e in e1:
        lengthmin = min(lengthmin, cubit.get_mesh_edge_length(e))
        length[e] = lengthmin * 0.5
    cubit.cmd('delete group ' + str(ge))

    #print('  equivalence edge lengths ',length)
    if len(length) > 0:
        minvalue = min(length.values())
        maxvalue = max(length.values())
    else:
        minvalue = 100.
        maxvalue = 100.

    tol = minvalue / 20.
    step_tol = minvalue / 20.

    print('  min length: ', minvalue, 'max length: ', maxvalue)

    cubit.cmd('group \'coincident_lateral_nodes\' add \
        Node in edge in group lateral')
    
    isempty = False
    while not isempty:
        isempty = merging_node_new(tol, clean=True, graphic_debug=False)
        tol = tol + step_tol
        if tol > maxvalue * 1.5:
            raise MergingError('tolerance greater than the max length of the edges, \
                    please check the mesh')
    # cubit.cmd('del group all')        
    # cubit.cmd('set info on')
    # cubit.cmd('set echo on')

    endtime_mer=time.time()
    convert_mer = str(datetime.timedelta(seconds = endtime_mer-starttime_mer))
    print('Elapsed time merging:',convert_mer)

def merging():
    import time
    import datetime
    starttime_mer=time.time()

    cubit.cmd('disassociate mesh from volume all')
    cubit.cmd('del vol all')
    
    factor, minvalue, maxvalue = prepare_equivalence_new()
    # cubit.cmd('set info off')
    # cubit.cmd('set echo off')
    # cubit.cmd('set journal off')

    tol = minvalue / 20.
    step_tol = minvalue / 20.
    #print('tolerance ', tol)

    cubit.cmd('group \'coincident_lateral_nodes\' add \
                   Node in face in group lateral')
    
    #cubit.cmd('set info on')
    isempty = False
    while not isempty:
        isempty = merging_node_new(tol, clean=True, graphic_debug=False)
        tol = tol + step_tol
        if tol > maxvalue * 1.5:
            raise MergingError(
                'tolerance greater than the max length of the edges, \
                    please check the mesh')

    endtime_mer=time.time()
    convert_mer = str(datetime.timedelta(seconds = endtime_mer-starttime_mer))
    
    print('Elapsed time merging:',convert_mer)

    
            
def prepare_equivalence_new(name_group='lateral'):
    print('equivalence group ',name_group)
    length = {}
    cmd = "group 'tmpn' add edge in face in group " + name_group
    cubit.cmd(cmd)
    ge = cubit.get_id_from_name("tmpn")
    e1 = cubit.get_group_edges(ge)
    lengthmin = 1e9
    for e in e1:
        lengthmin = min(lengthmin, cubit.get_mesh_edge_length(e))
        length[e] = lengthmin * 0.5
    cubit.cmd('delete group ' + str(ge))

    #print('  equivalence edge lengths ',length)
    if len(length) > 0:
        minvalue = min(length.values())
        maxvalue = max(length.values())
    else:
        minvalue = 100.
        maxvalue = 100.
    #
    print('  min length: ', minvalue, 'max length: ', maxvalue)
    if minvalue != 0:
        nbin = int((maxvalue / minvalue)) + 1
        factor = (maxvalue - minvalue) / nbin
    else:
        nbin = 0
        factor = 0.0

    return factor, minvalue, maxvalue


def merging_node_new(tol, clean=True, graphic_debug=False):
    empty = False
    print('tolerance ', tol)
    command = "topology check coincident node node in \
              group coincident_lateral_nodes tolerance " + str(tol) + " highlight brief result \
              group 'merging_lateral_nodes'"
    cubit.cmd(command)
    group_exist = cubit.get_id_from_name("merging_lateral_nodes")
    if not group_exist:
        print('no nodes in this tolerance range')
    else:
        merging_nodes = cubit.get_group_nodes(group_exist)
        if graphic_debug:
            cubit.cmd('draw group lateral')
            cubit.cmd('high group merging_lateral_nodes')
        print('merging ', len(merging_nodes), ' nodes.....')
        cmd = "equivalence node in merging_lateral_nodes \
                  tolerance " + str(tol * 2)
        cubit.cmd(cmd)
        if clean:
            cubit.cmd("group coincident_lateral_nodes \
                      remove node in group merging_lateral_nodes")
            cubit.cmd("delete Group merging_lateral_nodes")
        ic_nodes = cubit.get_id_from_name('coincident_lateral_nodes')
        c_nodes = cubit.get_group_nodes(ic_nodes)
        print('Coincident nodes after merging attempt: ',len(c_nodes))
        if len(c_nodes) == 0:
            empty = True
        if graphic_debug:
            cubit.cmd('draw group lateral')
            cubit.cmd('high group coincident_lateral_nodes')
            cubit.cmd('quality hex all jacobian \
                      global high 0 draw mesh draw add')
        return empty
    

def definesurface_blocks(nx_segment,ny_segment,lprevious,filename):
    import start as start
    cfg = start.start_cfg(filename)

    list_comp_sur = cubit.parse_cubit_list("surface", "all")
    list_sur=list_comp_sur[lprevious:]

    if cfg.geometry_format == 'regmesh':
        nx = cfg.nproc_xi + 1
        ny = cfg.nproc_eta + 1
    else:
        ny=cfg.ny
        nx=cfg.nx

    xmin_box = cfg.xmin+(nx_segment-1)*(cfg.start_chunk_xi)*(cfg.xmax-cfg.xmin)/(nx-1)
    ymin_box = cfg.ymin+(ny_segment-1)*(cfg.start_chunk_eta)*(cfg.ymax-cfg.ymin)/(ny-1)

    if cfg.end_chunk_xi == cfg.nproc_xi:
        xmax_box=cfg.xmax
    else:
        xmax_box = cfg.xmin+(nx_segment-1)*(cfg.end_chunk_xi)*(cfg.xmax-cfg.xmin)/(nx-1)
    if cfg.end_chunk_eta == cfg.nproc_eta:
        ymax_box=cfg.ymax
    else:
        ymax_box = cfg.ymin+(ny_segment-1)*(cfg.end_chunk_eta)*(cfg.ymax-cfg.ymin)/(ny-1)
    
    tol=2

    command = "block 1 add tri in surface " + str(list_sur[0])
    cubit.cmd(command)
    
    lcurve = cubit.get_relatives("surface", list_sur[0], "curve")
    list_curve_or=list(lcurve)

    from utilities import DoRotation
    import numpy
        
    for iv in list_curve_or:
        pv = cubit.get_center_point("curve", iv)
        x_rotated,y_rotated = DoRotation(cfg.xmin,cfg.ymin,numpy.array([pv[0]]), numpy.array([pv[1]]), -1*cfg.rot_deg)
        
        if xmin_box-tol < x_rotated < xmin_box+tol:
            cubit.cmd("group 'edge_left_boundary' add edge in curve " + str(iv))
        elif xmax_box-tol < x_rotated < xmax_box+tol:
            cubit.cmd("group 'edge_right_boundary' add edge in curve " + str(iv))
        elif ymin_box-tol < y_rotated < ymin_box+tol:
            cubit.cmd("group 'edge_lower_boundary' add edge in curve " + str(iv))
        elif ymax_box-tol < y_rotated < ymax_box+tol:
            cubit.cmd("group 'edge_upper_boundary' add edge in curve " + str(iv))
        else:
                # command = "block 200 add edge in curve " + str(iv)
                # cubit.cmd(command)
                cubit.cmd("group 'lateral' add edge in curve " + str(iv))


def define_blocks(nx_segment,ny_segment,lprevious,filename):
    import start as start
    cfg = start.start_cfg(filename)

    list_comp_vol = cubit.parse_cubit_list("volume", "all")
    list_vol=list_comp_vol[lprevious:]

    if cfg.geometry_format == 'regmesh':
        nx = cfg.nproc_xi + 1
        ny = cfg.nproc_eta + 1
    else:
        ny=cfg.ny
        nx=cfg.nx

    zmin_box = cfg.depth_bottom
    xmin_box = cfg.xmin+(nx_segment-1)*(cfg.start_chunk_xi)*(cfg.xmax-cfg.xmin)/(nx-1)
    ymin_box = cfg.ymin+(ny_segment-1)*(cfg.start_chunk_eta)*(cfg.ymax-cfg.ymin)/(ny-1)     

    if cfg.end_chunk_xi == cfg.nproc_xi:
        xmax_box=cfg.xmax
    else:
        xmax_box = cfg.xmin+(nx_segment-1)*(cfg.end_chunk_xi)*(cfg.xmax-cfg.xmin)/(nx-1)
    if cfg.end_chunk_eta == cfg.nproc_eta:
        ymax_box=cfg.ymax
    else:
        ymax_box = cfg.ymin+(ny_segment-1)*(cfg.end_chunk_eta)*(cfg.ymax-cfg.ymin)/(ny-1)

    tol=2
    
    from utilities import get_v_h_list, DoRotation
    surf_or, surf_vertical, list_curve_or, list_curve_vertical, \
        bottom, top = get_v_h_list(list_vol)
    import numpy
        
    for iv in surf_vertical:
        pv = cubit.get_center_point("surface", iv)
        x_rotated,y_rotated = DoRotation(cfg.xmin,cfg.ymin,numpy.array([pv[0]]), numpy.array([pv[1]]), -1*cfg.rot_deg)
        normal = cubit.get_surface_normal(iv)
        normx_rotated,normy_rotated = DoRotation(0,0,numpy.array([normal[0]]), numpy.array([normal[1]]), -1*cfg.rot_deg)
        if xmin_box-tol < x_rotated < xmin_box+tol or xmax_box-tol < x_rotated < xmax_box+tol :
                command = "block 100 add face in surface " + str(iv)
                cubit.cmd(command)
        elif ymin_box-tol < y_rotated < ymin_box+tol or ymax_box-tol < y_rotated < ymax_box+tol :
                command = "block 100 add face in surface " + str(iv)
                cubit.cmd(command)
        elif 0.9 <= normx_rotated <= 1.1:
                cubit.cmd("group 'intermediatelateral' add face in surface " + str(iv))      
                cubit.cmd("group 'next_lateral_nodes' add node in surface " + str(iv))
        elif -1.1 <= normx_rotated <= -0.9:
                cubit.cmd("group 'intermediatelateral' add face in surface " + str(iv))
                cubit.cmd("group 'coincident_lateral_nodes' add node in surface " + str(iv))
        elif 0.9 <= normy_rotated <= 1.1:
                cubit.cmd("group 'next_lateral' add face in surface " + str(iv))
        else:
                cubit.cmd("group 'lateral' add face in surface " + str(iv))
    

    if cfg.geometry_format == 'regmesh':
        zmax=cfg.zlayer[-1]
        for ih in surf_or:
            ph = cubit.get_center_point("surface", ih)
            if zmin_box-tol < ph[2] < zmin_box+tol :
                command = "block 100 add face in surface " + str(ih)
                cubit.cmd(command)
            elif zmax-tol < ph[2] < zmax+tol :
                command = "block 1001 add face in surface " + str(ih)
                cubit.cmd(command)            
            else:
                pass

    else:
        zmax=cfg.zlayer[-1]
        for ih in surf_or:
            ph = cubit.get_center_point("surface", ih)
            if zmin_box-tol < ph[2] < zmin_box+tol :
                    command = "block 100 add face in surface " + str(ih)
                    cubit.cmd(command)
            elif ph[2] > zmax + tol :
                    zmax=ph[2]
                    surTopo=ih             
            else:
                pass
        command = "block 1001 add face in surface " + str(surTopo)
        cubit.cmd(command)

    volNOSORTED = []
    vol = []
    for id_vol in list_vol:
        p = cubit.get_center_point("volume", id_vol)
        volNOSORTED.append(cubitvolume(id_vol, p[2]))
    vol=sorted(volNOSORTED, key=keyfunction)
    nvol = len(list_vol)

    cubit.cmd("set warning off")        
       
    for iii in range(nvol, 0, -1):
        command = "block " + str(iii) + " add hex in volume " + str(vol[-1*iii].ID)
        cubit.cmd(command)  
    cubit.cmd("set warning on")  


def blockTop():

    block_listt=cubit.parse_cubit_list('block', 'all')
    block_list_unsorted=block_listt[:-2]
    block_list = tuple(sorted(block_list_unsorted)) 

    quads_in_surf = cubit.get_block_faces(1001)
    list_all_hex = cubit.parse_cubit_list('hex', 'in block 1')
    Nhex = len(list_all_hex)
    s = set(quads_in_surf)
    listTop=[]
    for h in list_all_hex:
        faces = cubit.get_sub_elements('hex', h, 2)
        for f in faces:
            if f in s:
                    listTop.append(h)
    command = "block 1 remove hex " + ' '.join(str(x) for x in listTop)
    cubit.cmd(command)
    command = "block " + str(block_list[-1]+1) + " add hex " + ' '.join(str(x) for x in listTop)
    cubit.cmd(command)

def define_top(filename,ltprevious,block):
    import start as start
    cfg = start.start_cfg(filename=filename)
    
    list_comp_vol = cubit.parse_cubit_list("volume", "all")
    list_vol=list_comp_vol[ltprevious:]

    tol=1
    
    from utilities import get_v_h_list
    surf_or, surf_vertical, list_curve_or, list_curve_vertical, \
        bottom, top = get_v_h_list(list_vol)
        
    for iht in surf_or:
        ph = cubit.get_center_point("surface", iht)
        if  ph[2] > cfg.zlayer[-1] + tol :
                command = "block " + str(block) + " add face in surface " + str(iht)
                cubit.cmd(command)
        else:
            pass     


def hor_distance(c1, c2):
    p1 = cubit.get_center_point("curve", c1)
    p2 = cubit.get_center_point("curve", c2)
    d = (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2
    return d


def coupling_curve(lcurve1, lcurve2):
    """get the couple of curve that we  use to get the skin surface"""
    import operator
    couples = []
    for c1 in lcurve1:
        distance = []
        for c2 in lcurve2:
            d = hor_distance(c1, c2)
            distance.append(d)
        min_index, min_value = min(
            enumerate(distance), key=operator.itemgetter(1))
        couples.append((c1, lcurve2[min_index]))
    return couples


def create_volume(surf1, surf2, method='loft'):
    if method == 'loft':
        cmd = 'create volume loft surface ' + str(surf1) + ' ' + str(surf2)
        cubit.cmd(cmd)
    else:
        lcurve1 = cubit.get_relatives("surface", surf1, "curve")
        lcurve2 = cubit.get_relatives("surface", surf2, "curve")
        couples = coupling_curve(lcurve1, lcurve2)
        is_start = cubit.get_last_id('surface') + 1
        for cs in couples:
            cmd = 'create surface skin curve ' + str(cs[1]) + ' ' + str(cs[0])
            cubit.cmd(cmd)
        is_stop = cubit.get_last_id('surface')
        cmd = "create volume surface " + \
            str(surf1) + ' ' + str(surf2) + ' ' + str(is_start) + \
            ' to ' + str(is_stop) + "  heal keep"
        cubit.cmd(cmd)

class cubitvolume:

    def __init__(self, ID, centerpoint):
        self.ID = ID
        self.centerpoint = centerpoint

def keyfunction(item):
    return item.centerpoint