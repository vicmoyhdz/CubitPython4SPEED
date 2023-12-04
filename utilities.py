#############################################################################
# utilities.py                                                              #
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

try:
    import start as start
    cubit = start.start_cubit()
except:
    try:
        import cubit
    except:
        print('error importing cubit, check if cubit is installed')
        pass


def get_cubit_version():
    v = cubit.get_version()
    try:
        v = float(v[0:4])
    except:
        v = float(v[0:2])
    return v


def cubit_command_check(iproc, command, stop=True):
    """
    Run a cubit command, checking if it performs correctly.
    If the command fails, it writes the result on a file
    "error_[processor number]" and stop the meshing process

    iproc = process number
    command = cubit command
    stop = if command fails, stop the meshing process (Default: True)

    return status variable (0 ok, -1 fail)

    """
    flag = True
    er = cubit.get_error_count()
    cubit.cmd(command)
    print(command)
    ner = cubit.get_error_count()
    if ner > er:
        text = '"Proc: ' + str(iproc) + ' ERROR ' + str(command) + \
            ' number of error ' + str(er) + '/' + str(ner) + '"'
        cubitcommand = 'comment ' + text
        cubit.cmd(cubitcommand)
        f = open('error_' + str(iproc) + '.log', 'a')
        f.write("CUBIT ERROR: \n" + text)
        f.close()
        if stop:
            raise Exception("CUBIT ERROR: " + text)
        flag = False
    return flag

def export_mesh(block_list,filename=None):

    if filename:
        import start
        cfg = start.start_cfg(filename=filename)

    face_list_unsorted = cubit.get_block_faces(100)

    if len(face_list_unsorted)>0:

        face_list = tuple(sorted(face_list_unsorted))
        num_faces=len(face_list_unsorted)

        print('exporting SPEED mesh file...')
        cubit.cmd('compress node hex face')
        node_list_unsorted = cubit.parse_cubit_list('node', 'in hex in block '+ '  '.join(str(x) for x in block_list))
        node_list = tuple(sorted(node_list_unsorted))
        num_nodes = len(node_list)
        print(' total number of nodes:', str(num_nodes))
        num_elems = 0
        for iblock in block_list:
            num_elems=num_elems + len(cubit.get_block_hexes(iblock))

        print(' total number of elements:', str(num_elems+num_faces))

        if filename:
            meshfile = open(cfg.output_dir + '\Meshfile.mesh', 'w')
        else:
            meshfile = open('Meshfile.mesh', 'w')
        
        txt = ('   %i  %i  %i   %i   %i\n') % (num_nodes, num_elems+num_faces, 0, 0, 0)
        meshfile.write(txt)

        for node in node_list:
            x, y, z = cubit.get_nodal_coordinates(node)
            txt = ('%i  %+0.7e  %+0.7e  %+0.7e\n') % (node, x, y, z)
            meshfile.write(txt)

        idf = 1
            
        for face in face_list:
            nodesf= cubit.get_connectivity('face', face)
            txt = str(idf) + '  100  quad  ' + '  '.join(str(x) for x in nodesf)
            txt = txt + '\n'
            meshfile.write(txt)
            idf=idf+1
        
        idh=1

        hex_quality = []
        for block in block_list:
            hex_list_unsorted = cubit.get_block_hexes(block)
            hex_list = tuple(sorted(hex_list_unsorted))
  
            for hex in hex_list:
                hex_quality.append(cubit.get_quality_value("hex", hex, "scaled jacobian"))
                nodes= cubit.get_connectivity('hex', hex)
                txt = str(idh) + '  ' + str(block) + '   hex  ' + '  '.join(str(x) for x in nodes)
                txt = txt + '\n'
                meshfile.write(txt)
                idh=idh+1
        sj = [x for x in hex_quality if x <= 0.2]
        sj1 = [x for x in hex_quality if x <= 0.1]
        print('Minimum scaled jacobian is ', str(min(hex_quality)))
        if len(sj)>0:  
            print('Warning: ',str(round(100*len(sj)/len(hex_quality),1)),'% of elements (',str(len(sj)),') have scaled jacobian (sj) <0.2, from which ',str(len(sj1)),' elements have sj<0.1')       

        meshfile.close()
    else:
        print('Block 100 (ABS) is empty or missing. This block should include the faces of the ABS')
    

def export_surfacemesh(bl=1,filename=None):

    if filename:
        import start
        cfg = start.start_cfg(filename=filename)

    print('exporting SPEED surface file...')
    cubit.cmd('compress all')
    node_list_unsorted = cubit.parse_cubit_list('node', 'in tri in block ' +str(bl))
    node_list = tuple(sorted(node_list_unsorted))
    num_nodes = len(node_list)
    print(' total number of nodes:', str(num_nodes))

    face_list_unsorted = cubit.get_block_tris(bl)
    face_list = tuple(sorted(face_list_unsorted))
    num_faces=len(face_list_unsorted)

    print(' total number of faces:', str(num_faces))
    if filename:
        meshfile = open(cfg.output_dir + '\XYZ.out', 'w')
    else:
        meshfile = open('XYZ.out', 'w')
    txt = ('    %i     %i\n') % (num_nodes, num_faces)
    meshfile.write(txt)

    for node in node_list:
        x, y, z = cubit.get_nodal_coordinates(node)
        txt = ('%i  %+0.7e  %+0.7e  %+0.7e\n') % (node, x, y, z)
        meshfile.write(txt)

    idf = 1
        
    for trin in face_list:
        nodesf= cubit.get_connectivity('tri', trin)
        txt = str(idf) + '   ' + '  '.join(str(x) for x in nodesf) + '    1' 
        txt = txt + '\n'
        meshfile.write(txt)
        idf=idf+1

    meshfile.close()

def export_LS(block=1,istart=1,filename=None):

    print('exporting LS surface file...')

    cubit.cmd('compress all')
    node_list_unsorted = cubit.parse_cubit_list('node', 'in tri in block ' +str(block))
    node_list = tuple(sorted(node_list_unsorted))
    num_nodes = len(node_list)
    print(' total number of nodes:', str(num_nodes))

    if filename:
        import start
        cfg = start.start_cfg(filename=filename)
        LSfile = open(cfg.output_dir + '\LS.input', 'w')
    else:
        LSfile = open('LS.input', 'w')

    txt = ('%i \n') % (num_nodes)
    LSfile.write(txt)

    for node in node_list:
        x, y, z = cubit.get_nodal_coordinates(node)
        txt = ('%i  %+0.7e  %+0.7e  %+0.7e\n') % (istart, x, y, z)
        LSfile.write(txt)
        istart = istart + 1

    LSfile.close()

def DoRotation(xmin,ymin,x1, y1, RotDeg=0):
    """Generate a meshgrid and rotate it by RotRad radians."""
    import math
    import numpy
    RotRad=math.radians(RotDeg)
    # Clockwise, 2D rotation matrix
    RotMatrix = numpy.array([[numpy.cos(RotRad),  -numpy.sin(RotRad), xmin*(1-numpy.cos(RotRad))+ymin*numpy.sin(RotRad)],
                          [numpy.sin(RotRad), numpy.cos(RotRad), ymin*(1-numpy.cos(RotRad))-xmin*numpy.sin(RotRad)],
                          [0,0,1]])

    #x, y = numpy.meshgrid(xspan, yspan)
    z1 = numpy.ones(x1.shape)
    b=numpy.dstack([x1, y1, z1])
    a=numpy.einsum('ji, mni -> jmn', RotMatrix, b)
    #a=numpy.einsum('ijk, mnjk -> imn', RotMatrix, b)
    
    x=a[0]
    y=a[1]
    return x,y

def load_curves(acis_filename):
    """
    load the curves from acis files
    """
    import os
    #
    #print(acis_filename)
    tmp_curve = cubit.get_last_id("curve")
    command = "import acis '" + acis_filename + "'"
    cubit.cmd(command)
    tmp_curve_after = cubit.get_last_id("curve")
    curves = ' '.join(str(x)
        for x in range(tmp_curve + 1, tmp_curve_after + 1))

    return [curves]


def project_curves(curves, top_surface):
    """
    project curves on surface
    """
    if not isinstance(curves, list):
        curves = curves.split()
    tmpc = []
    for curve in curves:
        command = "project curve " + \
            str(curve) + " onto surface " + str(top_surface)
        cubit.cmd(command)
        tmp_curve_after = cubit.get_last_id("curve")
        tmpc.append(tmp_curve_after)
        command = "del curve " + str(curve)
        cubit.cmd(command)
    return tmpc


def geo2utm(lon, lat, unit, ellipsoid=23):
    """conversion geocoodinates from geographical to utm

    usage: x,y=geo2utm(lon,lat,unit,ellipsoid=23)

    dafault ellipsoid is 23 = WGS-84,
        ellipsoid:
        1, "Airy"
        2, "Australian National"
        3, "Bessel 1841"
        4, "Bessel 1841 (Nambia] "
        5, "Clarke 1866"
        6, "Clarke 1880"
        7, "Everest"
        8, "Fischer 1960 (Mercury] "
        9, "Fischer 1968"
        10, "GRS 1967"
        11, "GRS 1980"
        12, "Helmert 1906"
        13, "Hough"
        14, "International"
        15, "Krassovsky"
        16, "Modified Airy"
        17, "Modified Everest"
        18, "Modified Fischer 1960"
        19, "South American 1969"
        20, "WGS 60"
        21, "WGS 66"
        22, "WGS-72"
        23, "WGS-84"

    unit:  'geo' if the coordinates of the model (lon,lat) are geographical
           'utm' if the coordinates of the model (lon,lat) are utm

    x,y: the function return the easting, northing utm coordinates
    """
    import LatLongUTMconversion
    if unit == 'geo':
        (zone, x, y) = LatLongUTMconversion.LLtoUTM(ellipsoid, lat, lon)
    elif unit == 'utm':
        x = lon
        y = lat
    return x, y

def get_v_h_list(vol_id_list, chktop=False):
    """
    return the lists of the cubit ID of vertical/horizontal
    surface and vertical/horizontal curves
    where v/h is defined by the distance of the z normal component from
    the axis direction the parameter cfg.tres is the threshold as
    for example if
    -tres <= normal[2] <= tres
    then the surface is vertical
    #
    usage: surf_or,surf_vertical,list_curve_or,
        list_curve_vertical,bottom,top = get_v_h_list(list_vol,chktop=False)
    """
    #
    tres = 0.1

    try:
        _ = len(vol_id_list)
    except:
        vol_id_list = [vol_id_list]
    surf_vertical = []
    surf_or = []
    list_curve_vertical = []
    list_curve_or = []
    ltotsurf = []
    
    for id_vol in vol_id_list:
        lsurf = cubit.get_relatives("volume", id_vol, "surface")
        for k in lsurf:
            ltotsurf.append(k)
            normal = cubit.get_surface_normal(k)
            center_point = cubit.get_center_point("surface", k)
            if -1 * tres <= normal[2] <= tres:
                surf_vertical.append(k)
                lcurve = cubit.get_relatives("surface", k, "curve")
                list_curve_vertical = list_curve_vertical + list(lcurve)
            else:
                surf_or.append(k)
                lcurve = cubit.get_relatives("surface", k, "curve")
                list_curve_or = list_curve_or + list(lcurve)
    for x in list_curve_or:
        try:
            list_curve_vertical.remove(x)
        except:
            pass

    # find the top and the bottom surfaces
    k = surf_or[0]
    center_point = cubit.get_center_point("surface", k)[2]
    center_point_top = center_point
    center_point_bottom = center_point
    top = k
    bottom = k
    for k in surf_or[1:]:
        center_point = cubit.get_center_point("surface", k)[2]
        if center_point > center_point_top:
            center_point_top = center_point
            top = k
        elif center_point < center_point_bottom:
            center_point_bottom = center_point
            bottom = k
    # check that a top surface exists
    # it assume that the z coord of the center point
    if chktop:
        k = lsurf[0]
        vertical_centerpoint_top = cubit.get_center_point("surface", k)[2]
        vertical_zmax_box_top = cubit.get_bounding_box('surface', k)[7]
        normal_top = cubit.get_surface_normal(k)
        top = k
        for k in lsurf:
            vertical_centerpoint = cubit.get_center_point("surface", k)[2]
            vertical_zmax_box = cubit.get_bounding_box('surface', k)[7]
            normal = cubit.get_surface_normal(k)
            check = (vertical_centerpoint >= vertical_centerpoint_top) and (
                     vertical_zmax_box >= vertical_zmax_box_top) and (
                     normal >= normal_top)
            if check:
                top = k
        if top in surf_vertical:
            surf_vertical.remove(top)
        if top not in surf_or:
            surf_or.append(top)
    # if more than one surf is on the top, I get all the surfaces that are in
    # touch with top surface but not the vertical surfaces
    surftop = list(cubit.get_adjacent_surfaces(
        "surface", top))  # top is included in the list
    for s in surf_vertical:
        try:
            surftop.remove(s)
        except:
            pass
    top = surftop
    # check that all the surf are Horizontal or vertical
    surf_all = surf_vertical + surf_or
    if len(surf_all) != len(ltotsurf):
        print('not all the surf are horizontal or vertical, check the normals')
        print('list of surfaces: ', surf_all)
        print('list of vertical surface', surf_vertical)
        print('list of horizontal surface', surf_or)

    bottom = [bottom]
    return surf_or, surf_vertical, list_curve_or, \
        list_curve_vertical, bottom, top


def list2str(l):
    if not isinstance(l, list):
        l = list(l)
    return ' '.join(str(x) for x in l)


def highlight(ent, l):
    txt = list2str(l)
    txt = 'highlight ' + ent + ' ' + txt
    cubit.cmd(txt)
