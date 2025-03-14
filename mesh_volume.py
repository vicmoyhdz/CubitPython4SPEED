#############################################################################
# mesh_volume.py
# Modified by Victor Hernandez from GEOCUBIT (Emanuele Casarotti)           #                                            #
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
# from __future__ import print_function

try:
    import start as start
    cubit = start.start_cubit()
except:
    try:
         import cubit
    except:
         print("error importing cubit, check if cubit is installed")
         pass

# try:
#     set
# except NameError:
#     from sets import Set as set

def mesh(filename,iproc,lprevious):
    """create the mesh"""
    import start as start
    cfg = start.start_cfg(filename)
    #mpiflag, iproc, numproc, mpi = start.start_mpi()
    #
    if cfg.map_meshing_type == 'regularmap':
        mesh_layercake_regularmap(filename,iproc,lprevious)
    else:
        print('error: map_meshing_type ', cfg.map_meshing_type, \
              ' not implemented')
        
def meshsurface(filename,iproc,lprevious):
    """create the mesh"""
    import start as start
    cfg = start.start_cfg(filename)
    #mpiflag, iproc, numproc, mpi = start.start_mpi()
    #
    if cfg.map_meshing_type == 'regularmap':
        meshsurface_layercake_regularmap(filename,iproc,lprevious)
    else:
        print('error: map_meshing_type ', cfg.map_meshing_type, \
              ' not implemented')


class cubitvolume:

    def __init__(self, ID, intervalv, centerpoint, dimension):
        self.ID = ID
        self.intervalv = intervalv
        self.centerpoint = centerpoint
        self.dim = dimension

    def __repr__(self):
        msg = "(vol:%3i, vertical interval: %4i, centerpoint: %8.2f)" % (
            self.ID, self.intervalv, self.centerpoint)
        return msg

# def cmp(a, b):
#     return (a > b) - (a < b) 

# def by_z(x, y):
#     return cmp(x.centerpoint, y.centerpoint)

def keyfunction(item):
    return item.centerpoint

def meshsurface_layercake_regularmap(filename,iproc,lprevious):
    import math
    import start as start
    # mpiflag, iproc, numproc, mpi = start.start_mpi()
    from utilities import cubit_command_check, get_cubit_version
    #
    # numpy = start.start_numpy()
    cfg = start.start_cfg(filename)
    # from math import sqrt

    version_cubit = get_cubit_version()

    list_comp_sur = cubit.parse_cubit_list("surface", "all")
    if len(list_comp_sur) != 0:
        pass
    else:
        geometryfile = 'geometry_vol_' + str(iproc) + '.cub'
        importgeometry(geometryfile, iproc=iproc)
    
    command = 'compress all'
    cubit.cmd(command)
    list_comp_sur = cubit.parse_cubit_list("surface", "all")
    list_sur=list_comp_sur[lprevious:]
    nvol = len(list_sur)
    
    command = "surface " + str(list_sur[0]) + " scheme " + cfg.or_mesh_scheme
    cubit.cmd(command)
    #
    lcurve = cubit.get_relatives("surface", list_sur[0], "curve")
    list_curve_or=list(lcurve)

    ucurve, vcurve = get_uv_curve(list_curve_or,filename)
    schemepave = False

    #interval = int(round(length / cfg.size))*3**(cfg.ntripl)

    ucurve_interval = {}
    for k in ucurve:
        vertexends=cubit.get_relatives("curve", k, "vertex")
        pos1=cubit.get_center_point("vertex",vertexends[0])
        pos2=cubit.get_center_point("vertex",vertexends[1])
        length = ((pos1[0]-pos2[0])**2+(pos1[1]-pos2[1])**2)**0.5
        interval = int(round(length / cfg.size))*3**(cfg.ntripl)
        ucurve_interval[k] = interval
    for k in ucurve:
        intervalu=min(ucurve_interval.values())
        command = "curve " + str(k) + " interval " + str(intervalu)
        cubit.cmd(command)
        # cubit_error_stop(iproc,command,ner)
        command = "curve " + str(k) + " scheme equal"
        cubit.cmd(command)
        # cubit_error_stop(iproc,command,ner)
    #
    vcurve_interval = {}
    for k in vcurve:
        vertexends=cubit.get_relatives("curve", k, "vertex")
        pos1=cubit.get_center_point("vertex",vertexends[0])
        pos2=cubit.get_center_point("vertex",vertexends[1])
        length = ((pos1[0]-pos2[0])**2+(pos1[1]-pos2[1])**2)**0.5
        interval = int(round(length / cfg.size))*3**(cfg.ntripl)
        vcurve_interval[k] = interval
    for k in vcurve:
        intervalv=min(vcurve_interval.values())
        command = "curve " + str(k) + " interval " + str(intervalv)
        cubit.cmd(command)
        # cubit_error_stop(iproc,command,ner)
        command = "curve " + str(k) + " scheme equal"
        cubit.cmd(command)
        # cubit_error_stop(iproc,command,ner)
    

    print('meshing...')

    command = "mesh surface " + str(list_sur[0])
    status = cubit_command_check(iproc, command, stop=False)

    # if cfg.ntripl != 0:
    #     cubitcommand = 'refine node in surf  ' + \
    #                 str(list_sur[0]) + ' numsplit '+ str(cfg.ntripl) +' bias 1.0'
    #     cubit.cmd(cubitcommand)
    
    command='QTri surface ' + str(list_sur[0])
    cubit.cmd(command)


def mesh_layercake_regularmap(filename,iproc,lprevious):
    import math
    import start as start
    # mpiflag, iproc, numproc, mpi = start.start_mpi()
    from utilities import get_v_h_list, cubit_command_check, get_cubit_version
    #
    # numpy = start.start_numpy()
    cfg = start.start_cfg(filename)
    # from math import sqrt

    version_cubit = get_cubit_version()

    list_comp_vol = cubit.parse_cubit_list("volume", "all")
    if len(list_comp_vol) != 0:
        pass
    else:
        print('error: no volumes to mesh')

    #  command = 'composite create curve all'
    # cubit.cmd(command)
    # print('###"No valid composites can be created from the specified curves." \
    #      is NOT a critical ERROR.')
    
    command = 'compress all'
    cubit.cmd(command)
    list_comp_vol = cubit.parse_cubit_list("volume", "all")
    list_vol=list_comp_vol[lprevious:]
    nvol = len(list_vol)
    volNOSORTED = []
    vol = []
    for id_vol in list_vol:
        p = cubit.get_center_point("volume", id_vol)
        volNOSORTED.append(cubitvolume(id_vol, 1, p[2], 0))
    #vol.sort(by_z)
    vol=sorted(volNOSORTED, key=keyfunction)
    #print(vol)
    #
    for id_vol in range(0, nvol):
        vol[id_vol].intervalv = cfg.iv_interval[id_vol]
    #
    surf_vertical = []
    surf_or = []

    surf_or, surf_vertical, list_curve_or, list_curve_vertical, \
        bottom, top = get_v_h_list(list_vol, chktop=cfg.chktop)


    for k in surf_vertical:
        command = "surface " + str(k) + " scheme submap"
        cubit.cmd(command)
    for k in surf_or:
        command = "surface " + str(k) + " scheme " + cfg.or_mesh_scheme
        cubit.cmd(command)
    #
    ucurve, vcurve = get_uv_curve(list_curve_or,filename)
    schemepave = False
    #
    # print('ucurve',ucurve)
    # print('vcurve',vcurve)
    ucurve_interval = {}
    for k in ucurve:
        length_r = cubit.get_curve_length(k)
        length=round(length_r)
        interval = int(round(length / cfg.size))
        ucurve_interval[k] = interval
    #if lprevious==1:
    intervalu=min(ucurve_interval.values())   
       # print('First surface')
    for k in ucurve:
        command = "curve " + str(k) + " interval " + str(intervalu)
        cubit.cmd(command)
        command = "curve " + str(k) + " scheme equal"
        cubit.cmd(command)
        # cubit_error_stop(iproc,command,ner)
    #
    vcurve_interval = {}
    for k in vcurve:
        length_r = cubit.get_curve_length(k)
        length=round(length_r)
        interval = int(round(length / cfg.size))
        vcurve_interval[k] = interval
    intervalv=min(vcurve_interval.values())   
    for k in vcurve:
        command = "curve " + str(k) + " interval " + str(intervalv)
        cubit.cmd(command)
        # cubit_error_stop(iproc,command,ner)
        command = "curve " + str(k) + " scheme equal"
        cubit.cmd(command)
        # cubit_error_stop(iproc,command,ner)
    #
    for s in surf_vertical:
        lcurve = cubit.get_relatives("surface", s, "curve")
        interval_store = []
        for k in lcurve:
            interval_curve = cubit.get_mesh_intervals('curve', k)
            if k in list_curve_vertical:
                volume_id = cubit.get_owning_volume("curve", k)
                for idv in range(0, nvol):
                    if vol[idv].ID == volume_id:
                        int_v = vol[idv].intervalv
                command = "curve " + str(k) + " interval " + str(int_v)
                cubit.cmd(command)
                # cubit_error_stop(iproc,command,ner)
                command = "curve " + str(k) + " scheme equal"
                cubit.cmd(command)
                # cubit_error_stop(iproc,command,ner)
            else:
                interval_store.append((k, interval_curve))
            if len(interval_store) != 0:
                interval_min = min([iv[1] for iv in interval_store])
                str_interval = [str(iv[0]) for iv in interval_store]
                cmd = "curve " + ' '.join(str_interval) + \
                      " interval " + str(interval_min)
                cubit.cmd(cmd)
                command = "curve " + \
                    ' '.join(str(iv[0])
                             for iv in interval_store) + " scheme equal"
                cubit.cmd(command)
        command = "surface " + str(s) + " scheme submap"
        cubit.cmd(command)

    print('meshing...')
    if cfg.or_mesh_scheme == 'pave' or schemepave:
        command = 'mesh surf ' + ' '.join(str(t) for t in top)
        status = cubit_command_check(iproc, command, stop=True)
        # cubit.cmd(command)
    elif cfg.or_mesh_scheme == 'map':
        command = 'mesh surf ' + ' '.join(str(t) for t in bottom)
        status = cubit_command_check(iproc, command, stop=True)
        # cubit.cmd(command)
    for id_volume in range(nvol - 1, -1, -1):
        command = "mesh vol " + str(vol[id_volume].ID)
        status = cubit_command_check(iproc, command, stop=False)
        if not status:
            for s in surf_vertical:
                command_surf = "mesh surf " + str(s)
                cubit.cmd(command_surf)
            if version_cubit < 16:
                command_set_meshvol = '''volume all redistribute nodes on
                volume all autosmooth target off
                volume all scheme Sweep Vector 0 0 -1
                volume all sweep smooth Auto
                '''
            else:
                command_set_meshvol = '''volume all redistribute nodes on
                volume all autosmooth target off
                volume all scheme Sweep Vector 0 0 -1
                '''
            status2 = cubit_command_check(
                iproc, command_set_meshvol, stop=False)
            status2 = cubit_command_check(iproc, command, stop=False)
            if not status2:
                _surf = cubit.get_relatives(
                    'volume', vol[id_volume].ID, 'surface')
                local_o_surf = [x for x in _surf if x in surf_or]
                cubit.cmd('volume ' + str(vol[id_volume].ID) +
                          ' redistribute nodes off')
                cubit.cmd('volume ' + str(vol[id_volume].ID) +
                          ' scheme Sweep  source surface ' +
                          ' '.join(str(x) for x in local_o_surf[0:-1]) +
                          ' target surface ' + str(local_o_surf[-1]))
                cubit.cmd('volume ' + str(vol[id_volume].ID) +
                          ' autosmooth_target off')
                status = cubit_command_check(iproc, command, stop=True)

    #
    # smoothing
    print(iproc, 'untangling...')
    cmd = "volume " + ' '.join(str(x) for x in list_vol) + " smooth scheme untangle beta 0.02 cpu 10"
    cubit.cmd(cmd)
    cmd = "smooth volume " + ' '.join(str(x) for x in list_vol)
    cubit.cmd(cmd)

    if cfg.smoothing:
        print('smoothing .... ' + str(cfg.smoothing))
        cubitcommand = 'surf all smooth scheme laplacian '
        cubit.cmd(cubitcommand)
        cubitcommand = 'smooth surf all'
        cubit.cmd(cubitcommand)
        #
        cubitcommand = 'vol all smooth scheme laplacian '
        cubit.cmd(cubitcommand)
        cubitcommand = 'smooth vol all'
        cubit.cmd(cubitcommand)
    #
    #
    # vertical refinement
    # for nvol = 3
    ##
    # ___________________________ interface 4
    ##
    # vol 2
    # ___________________________ interface 3
    ##
    # vol 1
    # ___________________________ interface 2
    ##
    # vol 0
    # ___________________________ interface 1
    ##
    refinement(nvol, vol, filename=filename)
    #
    # top layer vertical coarsening
    #print('coarsening top layer... ', cfg.coarsening_top_layer)
    if cfg.coarsening_top_layer:

        cubit.cmd("set warning off") 
        topvol=[vol[-1].ID]
        surftop_or, surftop_vertical, curvetop_or, curvetop_vertical, \
        bottomtop, toptop = get_v_h_list(topvol, chktop=cfg.chktop)

        cubitcommand = 'del mesh vol ' + str(vol[-1].ID) + ' propagate'
        cubit.cmd(cubitcommand)
        cubitcommand = 'reset vol ' + str(vol[-1].ID)
        cubit.cmd(cubitcommand)

        cubit.cmd("set warning on") 
        
        # s1 = set(list_curve_vertical)
        # command = "group 'list_curve_tmp' add curve " + \
        #     "in vol " + str(vol[-1].ID)
        # cubit.cmd(command)
        # group = cubit.get_id_from_name("list_curve_tmp")
        # list_curve_tmp = cubit.get_group_curves(group)
        # command = "delete group " + str(group)
        # cubit.cmd(command)
        # s2 = set(list_curve_tmp)
        # lc = list(s1 & s2)
        #
        cubitcommand = 'curve ' + \
            ' '.join(str(x) for x in curvetop_vertical) + ' interval ' + \
            str(cfg.actual_vertical_interval_top_layer)
        cubit.cmd(cubitcommand)
        cubitcommand = 'mesh vol ' + str(vol[-1].ID)
        cubit.cmd(cubitcommand)
    #
    # n = cubit.get_sideset_id_list()
    # if len(n) != 0:
    #     command = "del sideset all"
    #     cubit.cmd(command)
    # n = cubit.get_block_id_list()
    # if len(n) != 0:
    #     command = "del block all"
    #     cubit.cmd(command)
    #
    # import boundary_definition
    # entities = ['face']
    # #print(iproc, 'hex block definition...')
    # boundary_definition.define_bc(entities, parallel=True,
    #                               cpux=cfg.cpux, cpuy=cfg.cpuy,
    #                               cpuxmin=0, cpuymin=0, optionsea=False)
    # # save mesh

    # print(iproc, 'untangling...')
    # cmd = "volume all smooth scheme untangle beta 0.02 cpu 10"
    # cubit.cmd(cmd)
    # cmd = "smooth volume all"
    # cubit.cmd(cmd)

    # print(iproc, 'saving...')
    # savemesh(mpiflag, iproc=iproc, filename=filename)
    #


def refinement(nvol, vol, filename=None):
    import start as start
    cfg = start.start_cfg(filename=filename)
    from utilities import get_v_h_list
    #
    # vertical refinement
    # for nvol = 3
    #
    # ___________________________ interface 4
    #
    # vol 2
    # ___________________________ interface 3
    #
    # vol 1
    # ___________________________ interface 2
    #
    # vol 0
    # ___________________________ interface 1
    #
    #
    if cfg.ntripl != 0:
        if len(cfg.refinement_depth) != 0:
            _, _, _, _, _, tsurf = get_v_h_list([vol[nvol - 1].ID])
            tsurf = ' '.join(str(x) for x in tsurf)
            for idepth in cfg.refinement_depth:
                cubitcommand = 'refine node in surf  ' + \
                    str(tsurf) + ' numsplit 1 bias 1.0 depth ' + str(idepth)
                cubit.cmd(cubitcommand)
        else:
            for ir in cfg.tripl:
                if ir == 1:
                    print('interface = 1 means that the refinement'
                          'interface is at the bottom of the volume')
                    txt = ' all '
                    idepth = 1
                    cubitcommand = 'refine hex in vol  ' + txt
                elif ir != nvol + 1:
                    txt = ''
                    for id_vol_ref in range(ir - 1, nvol):
                        txt = txt + str(vol[id_vol_ref].ID) + ' '
                    cubitcommand = 'refine hex in vol  ' + txt
                else:
                    # refinement on the top surface
                    _, _, _, _, _, tsurf = get_v_h_list([vol[ir - 2].ID])
                    tsurf = ' '.join(str(x) for x in tsurf)
                    idepth = cfg.refinement_depth_top
                    cubitcommand = 'refine node in surf ' + str(tsurf) + \
                    ' numsplit '+ str(cfg.nsplit) +' bias 1.0 depth ' + str(idepth)
                cubit.cmd(cubitcommand)

    sandwich = 'verticalsandwich_volume_ascii_regulargrid_mpiregularmap'
    if not nvol and cfg.volume_type == sandwich:
            # AAA
            # Volume 2 is always in between the 2nd and 3rd vertical
            # surfaces from the left
        cubitcommand = "refine node in volume 2 numsplit 1 depth 0"
        cubit.cmd(cubitcommand)


def pinpoly(x, y, polyx, polyy):
    """point in polygon using ray tracing"""
    n = len(polyx)
    inside = False
    poly = list(zip(polyx, polyy))
    px_0, py_0 = poly[0]
    for i in range(n + 1):
        px_1, py_1 = poly[i % n]
        if y > min(py_0, py_1):
            if y <= max(py_1, py_0):
                if x <= max(px_1, px_0):
                    if py_0 != py_1:
                        intersect = (y - py_0) * (px_1 - px_0) / \
                            (py_1 - py_0) + px_0
                    if px_1 == px_0 or x <= intersect:
                        inside = not inside
        px_0, py_0 = px_1, py_1
    return inside


def curve2poly(line):
    curve = int(line)
    cubit.cmd('curve ' + str(curve) + ' size 100')
    cubit.cmd('curve ' + str(curve) + ' scheme equal')
    cubit.cmd('mesh curve ' + str(curve))
    n = cubit.get_curve_nodes(curve)
    orientnode = []
    vertex_list = cubit.get_relatives("curve", curve, "vertex")
    if len(vertex_list) != 0:
        startnode = cubit.get_vertex_node(vertex_list[0])
        cubit.cmd("group 'pgon' add edge in node " + str(startnode))
        group1 = cubit.get_id_from_name("pgon")
        edges = list(cubit.get_group_edges(group1))
        edgestart = edges[0]
    else:
        startnode = n[0]
        cubit.cmd("group 'pgon' add edge in node " + str(startnode))
        group1 = cubit.get_id_from_name("pgon")
        edges = list(cubit.get_group_edges(group1))
        edgestart = edges[0]
    begin = startnode
    orientnode.append(begin)
    node_id_list = list(cubit.get_connectivity("edge", edgestart))
    node_id_list.remove(startnode)
    startnode = node_id_list[0]
    orientnode.append(startnode)
    stopflag = False
    while startnode != begin and not stopflag:
        cubit.cmd('del group pgon')
        cubit.cmd("group 'pgon' add edge in node " + str(startnode))
        group1 = cubit.get_id_from_name("pgon")
        edges = list(cubit.get_group_edges(group1))
        if len(edges) != 1:
            edges.remove(edgestart)
            edgestart = edges[0]
            node_id_list = list(cubit.get_connectivity("edge", edgestart))
            node_id_list.remove(startnode)
            orientnode.append(node_id_list[0])
            startnode = node_id_list[0]
        else:
            stopflag = True
    vx = []
    vy = []
    for n in orientnode:
        v = cubit.get_nodal_coordinates(n)
        vx.append(v[0])
        vy.append(v[1])

    cubit.cmd('del group pgon')
    return vx, vy, orientnode


def get_nodes_inside_curve(nodes, curve):
    vx, vy, n = curve2poly(curve)
    nodes_inside = []
    for n in nodes:
        vp = cubit.get_nodal_coordinates(n)
        if pinpoly(vp[0], vp[1], vx, vy):
            nodes_inside.append(n)
    return nodes_inside


def refine_inside_curve(curves, ntimes=1, depth=1, block=1, nodes=[], surface=False):
    sflag = surface
    if not isinstance(curves, list):
        if isinstance(curves, str):
            curves = list(map(int, curves.split()))
        else:
            curves = [curves]
    for curve in curves:

        if len(nodes) > 0:
            nodes=nodes
        else:
            if not surface:
                cubit.cmd("group 'ntop' add node in face in block " + str(block))
            else:
                cubit.cmd("group 'ntop' add node in face in surface " + str(sflag))
            group1 = cubit.get_id_from_name("ntop")
            nodes = list(cubit.get_group_nodes(group1))
        ni = get_nodes_inside_curve(nodes, curve)
        # if ntimes > 1:
        #     cmd = 'del group hex_ref'
        #     cubit.cmd(cmd)
        #     command = "group 'hex_ref' add hex propagate face in node " +\
        #               ' '.join(str(x) for x in ni) + "times " + str(ntimes)
        #     cubit.cmd(command)
        #     # id_group = cubit.get_id_from_name('hex_ref')
        #     cmd = 'refine hex in group "hex_ref" numsplit 1 bias 1 depth ' + \
        #         str(depth) + ' smooth'
        #     cubit.cmd(cmd)
        # else:
        print('Refining nodes')
        command = 'refine node ' + \
            " ".join(str(x) for x in ni) + \
            ' numsplit ' + str(ntimes) + ' bias 1.0 depth ' + str(depth) + ' smooth'
        cubit.cmd(command)
        cubit.cmd('del group ntop')
    #
    #
    cmd = 'group "negativejac" add quality hex all Jacobian high'
    cubit.cmd(cmd)
    group_id_1 = cubit.get_id_from_name("negativejac")
    n1 = cubit.get_group_nodes(group_id_1)
    if len(n1) != 0:
        print('error, negative jacobian after the refining')

def refine_surface_curve(curves, ntimes=1, depth=1, block=1, nodes=[], surface=False):
    sflag = surface
    if not isinstance(curves, list):
        if isinstance(curves, str):
            curves = list(map(int, curves.split()))
        else:
            curves = [curves]
    for curve in curves:

        if len(nodes) > 0:
            nodes=nodes
        else:
            if not surface:
                cubit.cmd("group 'ntop' add node in tri in block " + str(block))
            else:
                cubit.cmd("group 'ntop' add node in tri in surface " + str(sflag))
            group1 = cubit.get_id_from_name("ntop")
            nodes = list(cubit.get_group_nodes(group1))
        ni = get_nodes_inside_curve(nodes, curve)
        # if ntimes > 1:
        #     cmd = 'del group hex_ref'
        #     cubit.cmd(cmd)
        #     command = "group 'hex_ref' add hex propagate face in node " +\
        #               ' '.join(str(x) for x in ni) + "times " + str(ntimes)
        #     cubit.cmd(command)
        #     # id_group = cubit.get_id_from_name('hex_ref')
        #     cmd = 'refine hex in group "hex_ref" numsplit 1 bias 1 depth ' + \
        #         str(depth) + ' smooth'
        #     cubit.cmd(cmd)
        # else:
        print('Refining nodes')
        command = 'refine node ' + \
            " ".join(str(x) for x in ni) + \
            ' numsplit ' + str(ntimes) + ' bias 1.0 depth ' + str(depth) + ' smooth'
        cubit.cmd(command)
        cubit.cmd('del group ntop')
    #
    #
    cmd = 'group "negativejac" add quality hex all Jacobian high'
    cubit.cmd(cmd)
    group_id_1 = cubit.get_id_from_name("negativejac")
    n1 = cubit.get_group_nodes(group_id_1)
    if len(n1) != 0:
        print('error, negative jacobian after the refining')


def get_uv_curve(list_curve_or,filename):
    import math
    import numpy as np
    from utilities import DoRotation
    import start as start
    cfg = start.start_cfg(filename)
    klen = {}
    for curve in list_curve_or:
        vertex_list = cubit.get_relatives("curve", curve, "vertex")
        coord0_rot = cubit.get_center_point('vertex', vertex_list[0])
        coord1_rot = cubit.get_center_point('vertex', vertex_list[1])
        coord0x,coord0y = DoRotation(cfg.xmin,cfg.ymin,np.array([coord0_rot[0]]), np.array([coord0_rot[1]]), -1*cfg.rot_deg)
        coord1x,coord1y = DoRotation(cfg.xmin,cfg.ymin,np.array([coord1_rot[0]]), np.array([coord1_rot[1]]), -1*cfg.rot_deg)
        klen[curve] = np.array([coord1x[0,0],coord1y[0,0],coord1_rot[2]]) - np.array([coord0x[0,0],coord0y[0,0],coord0_rot[2]])
    #
    l0 = list_curve_or[0]
    c0 = klen[l0]
    angles = {}
    angles[l0] = 0
    for curve in list_curve_or[1:]:
        c1 = klen[curve]
        angletmp = np.dot(c0, c1) / (np.dot(c0, c0)**.5 * np.dot(c1, c1)**.5)
        if -1 < angletmp < 1:
            angle = math.sin(np.arccos(angletmp))
        else:
            angle = 0.
        angles[curve] = angle
    #a = angles.values()
    #diff = max(a) - min(a)
    ucurve = []
    vcurve = []
    for curve in list_curve_or:
        if  abs(angles[curve]) < 0.707:
            vcurve.append(curve)
        else:
            ucurve.append(curve)
    #
    return ucurve, vcurve
