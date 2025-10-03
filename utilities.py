#############################################################################
# utilities.py                                                              #
# By Victor Hernandez (victorh@hi.is)                                       #
# University of Iceland and Politecnico di Milano                           #
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

def export_mesh(block_list,quad_list=[100],filename=None):

    if filename:
        import start
        cfg = start.start_cfg(filename=filename)

    cubit.cmd('compress all')
    
    print('exporting SPEED mesh file...')  
    
    node_list_unsorted = cubit.parse_cubit_list('node', 'all')
    node_list = tuple(sorted(node_list_unsorted))
    num_nodes = len(node_list)
    print(' total number of nodes:', str(num_nodes))
    num_faces = 0
    for iquad in quad_list:
            num_faces=num_faces + len(cubit.get_block_faces(iquad))
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
    
    for iquad in quad_list:
        face_list_unsorted=cubit.get_block_faces(iquad)
        face_list = tuple(sorted(face_list_unsorted))
                  
        for face in face_list:
            nodesf= cubit.get_connectivity('face', face)
            txt = str(idf) + '  '  + str(iquad) + '  quad  ' + '  '.join(str(x) for x in nodesf)
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

def separateABC(z_bottom,bl=100):

    cubit.cmd('set duplicate block elements on')
    face_list_unsorted = cubit.get_block_faces(bl)
    face_list = tuple(sorted(face_list_unsorted))
        
    lateral_faces=[]
    for face in face_list:
        loc_center= cubit.get_center_point('face', face)
        if loc_center[2]>z_bottom:
            lateral_faces.append(face)
             
    command = "block 101 add face " + ' '.join(str(x) for x in lateral_faces)
    cubit.cmd(command)
    command = "block 100 remove face " + ' '.join(str(x) for x in lateral_faces)
    cubit.cmd(command)


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
        txt = ('%i  %+0.7e  %+0.7e  %+0.7e\n') % (istart, x, y, z+20)
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

def BlockBoundary(blockabsorbing=100,blockapply=[1],thick_cushion=1):
    quads_in_surf = cubit.get_block_faces(blockabsorbing)
    list_all_hex = cubit.parse_cubit_list('hex', 'in block ' + ' '.join(str(x) for x in blockapply))
    Nhex = len(list_all_hex)
    s = set(quads_in_surf)
    listTop=[]
    for h in list_all_hex:
        faces = cubit.get_sub_elements('hex', h, 2)
        for f in faces:
            if f in s:
                    listTop.append(h)
    for ih in blockapply:
        command = "block " + str(ih) + " remove hex " + ' '.join(str(x) for x in listTop)
        cubit.cmd(command)
    command = "block 8 add hex " + ' '.join(str(x) for x in listTop)
    cubit.cmd(command)

    for i in range(thick_cushion-1):
        cubit.cmd('group \'boundaryhex\' add node in hex in block 8')
        groupb = cubit.get_id_from_name("boundaryhex")
        nodesb = list(cubit.get_group_nodes(groupb))
        hexesl=[]
        for inodes in nodesb:
                nh = cubit.parse_cubit_list('hex', 'in node ' + str(inodes))
                hexesadd = list(nh)
                hexesl.append(hexesadd)
        for ih in blockapply:
            command = "block " + str(ih) + " remove hex " + ' '.join(str(x) for x in hexesl)
            cubit.cmd(command)
        command = "block 8 add hex " + ' '.join(str(x) for x in hexesl)
        cubit.cmd(command)


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
    
###################################

def DRM_ALL(block_inp,box_1,z_1,box_2,z_2):
    
    DRM(block_inp,box_1,z_1)  
    #removing block 90
    cubit.cmd("delete block 90")
    lv=cubit.parse_cubit_list("hex","with not block_assigned")
    command = "block 1 add hex " + ' '.join(str(x) for x in lv)
    cubit.cmd(command)
    
    interface_block(91,block_inp,92)
    DRM(91,box_2,z_2,93,94)
    interface_block(94,93,95)       

def DRM(block_inp,box_xy,z_base,block_id_boundary=90,block_id_inside=91,
        alpha=0.95, beta=0.95,corner_gamma=0.95, tol_base=0.95, per_z_layer=True):
    
        # ===================== inputs =====================
    # block_inp = block where the box is embedded (it could be extended to take more than 1 block)
    #box_xy = [(-10,-10),(10,-10),(10,10),(-10,10)]   # CLOSED polygon vertices (no repeat at end)
    #z_base = coordinate z of the base of the box
    #the strip of elements (edge) with width equal to 1 element,
    # closest to the box defined by box_xy and z_base are selected
    #block_id_boundary = number of output block with the "edge"
    #block_id_inside = number of output block with hexes inside the box
    
    #beta   = 1.0   # along-edge stride ~ beta * local h_xy
    #per_z_layer = True -> keep one element per edge and per depth layer
    # =================================================

    from collections import namedtuple, Counter
    Candidate = namedtuple("Candidate", "eid s d hxy z seg_i")
    
    Lcum = precompute_edges(box_xy) #creates polygon from box_xy
    nvert = len(box_xy)

    #extracts all elements from input block
    all_elems_unsorted = cubit.get_block_hexes(block_inp) 
    all_elems = tuple(sorted(all_elems_unsorted))
        
    elems=[] #takes only hexes above the coordinate z_base (to reduce calculations)
    for hex in all_elems:
        #loc_center has coordinates x,y,z of the center of each hex
        loc_center= cubit.get_center_point('hex', hex)
        if loc_center[2]>=z_base-1e-9:
            elems.append(hex)
    
    sample = elems[:min(200, len(elems))]
    hz = estimate_hz_from_nodes(sample)   # estimates median height of elements from 200 hexes
    
    # polygon bbox for quick reject
    minx = min(p[0] for p in box_xy); maxx = max(p[0] for p in box_xy)
    miny = min(p[1] for p in box_xy); maxy = max(p[1] for p in box_xy)

    seg_buckets = {}      # (seg_i, zlayer) -> list[Candidate]
    corner_buckets = {}   # (vtx_i, zlayer) -> list[Candidate]
    inside_cache = []     # (eid, x, y, z, hxy)

    # Gather edge candidates + cache all inside elements
    for eid in elems: #loop over the hexes
        x,y,z = cubit.get_center_point("hex",eid)
        if x < minx-1e-9 or x > maxx+1e-9 or y < miny-1e-9 or y > maxy+1e-9:
            continue # skip if outside the limits of the box + tolerance
        if not point_in_poly(x, y, box_xy, include_boundary=True):
            continue #verify that the center of the hex is inside the polygon

        hxy = hxy_bbox(eid) #gets minimum edge size of the hex
        inside_cache.append((eid, x, y, z, hxy)) #gathers elements inside the polygon

        #nearest edge of the box to the centroid of the hex
        d, s, seg_i, t = nearest_edge_raw(x, y, box_xy, Lcum)
        if d > (alpha * hxy): #defines if the hex (centroid) is located at less that 1 element distance to the edge
            continue  # skip: not in the first ring next to the edge
        
        zk = 0 if not per_z_layer else int(round(z / max(1e-9, hz)))
        
        # absolute corner band ~ corner_gamma * hxy. Here we define if the hex is at a corner
        edge_s0 = Lcum[seg_i]; edge_s1 = Lcum[seg_i+1]
        dist_to_end = min(s - edge_s0, edge_s1 - s)
        is_corner = dist_to_end <= (corner_gamma * hxy)
        vtx_i = seg_i if (s - edge_s0) <= (edge_s1 - s) else (seg_i + 1) % nvert

        cand = Candidate(eid=eid, s=s, d=d, hxy=hxy, z=z, seg_i=seg_i)
        #separates hexes in corners or along edges
        if is_corner:
            corner_buckets.setdefault((vtx_i, zk), []).append(cand)
        else:
            seg_buckets.setdefault((seg_i, zk), []).append(cand)

    # Edge keep (greedy per segment + one per corner)
    kept_edge = []
    for key, lst in seg_buckets.items():
        lst.sort(key=lambda c: (c.s, c.d))
        i = 0
        while i < len(lst):
            best = lst[i]
            j = i + 1
            s_next = best.s + beta * best.hxy
            while j < len(lst) and lst[j].s < s_next:
                if lst[j].d < best.d:
                    best = lst[j]
                    s_next = best.s + beta * best.hxy
                j += 1
            kept_edge.append(best)
            i = j
    for key, lst in corner_buckets.items():
        kept_edge.append(min(lst, key=lambda c: c.d))

    side_ids = {c.eid for c in kept_edge}

    # Selection of hexes at the base:
    base_ids = set()

    z_base_eff = float(z_base)
    z_tol_eff = tol_base * hz
    for (eid, _x, _y, z, _h) in inside_cache:
        if abs(z - z_base_eff) <= z_tol_eff:
             base_ids.add(eid)
    
    # Inside = all inside elements minus the EDGE set and BASE set
    inside_ids = {eid for (eid, _x, _y, _z, _h) in inside_cache}
    inside_ids -= side_ids
    inside_ids -= base_ids
    
    # Make blocks
    make_block(block_id_boundary, block_inp, side_ids, "TMP_EDGE")
    make_block(block_id_boundary, block_inp, base_ids, "TMP_EDGE")
    make_block(block_id_inside, block_inp, inside_ids, "TMP_INSIDE")
    cubit.cmd("delete group TMP_EDGE")
    cubit.cmd("delete group TMP_INSIDE")
   
   ######################################################### 
    #Create block with faces between EDGE and INSIDE
def interface_block(block_id_inside, block_id_boundary,block_face=92):
        from collections import Counter
        cubit.cmd('set info off')
        cubit.cmd('set echo off')

        inside_hex = get_hexes_in_block(block_id_inside)
        edge_hex   = get_hexes_in_block(block_id_boundary)

        # 1) EDGE face keys (set of 4-corner frozensets)
        edge_face_keys = set()
        for eid in edge_hex:
            for fn in hex_local_faces_hex8(hex_nodes(eid)):
                edge_face_keys.add(frozenset(fn))

        # 2) INSIDE boundary face keys (count==1 within INSIDE)
        cnt = Counter()
        for eid in inside_hex:
            for fn in hex_local_faces_hex8(hex_nodes(eid)):
                cnt[frozenset(fn)] += 1
        inside_boundary = {k for k, c in cnt.items() if c == 1}

        # 3) Interface keys = intersection
        iface_keys = inside_boundary & edge_face_keys
        if not iface_keys:
            raise RuntimeError("No interface faces found (INSIDE and EDGE may not touch).")

        # Consistent ordered 4-tuples for each key (use INSIDE orientation)
        ordered = {}
        for eid in inside_hex:
            n = hex_nodes(eid)
            for fn in hex_local_faces_hex8(n):
                k = frozenset(fn)
                if k in iface_keys and k not in ordered:
                    ordered[k] = fn

        # 4) Snapshot existing faces ONCE
        before_faces = set(cubit.parse_cubit_list("face", "all"))

        # 5) Create missing faces (skip ones that already exist)
        to_create = list(ordered.values())
        for (n1, n2, n3, n4) in to_create:
            cubit.cmd(f"create face node {n1} {n2} {n3} {n4}")

        # 6) Map interface keys -> face IDs using only the *new* faces
        after_faces = set(cubit.parse_cubit_list("face", "all"))
        new_face_ids = list(after_faces - before_faces)
        # Build node-sets for new faces
        new_faces_nodes = [(fid, set(cubit.get_connectivity("face", fid))) for fid in new_face_ids]

        key_to_fid = {}
        exact_map = {frozenset(nodes): fid for fid, nodes in new_faces_nodes}
        for k in iface_keys:
            if k in exact_map:
                key_to_fid[k] = exact_map[k]

        # If some interface faces already existed before, map them too (superset over *all* faces)
        '''
        unresolved = [k for k in iface_keys if k not in key_to_fid]
        if unresolved:
            all_faces = list(after_faces)
            all_faces_nodes = [(fid, set(cubit.get_connectivity("face", fid))) for fid in all_faces]
            for k in list(unresolved):
                for fid, nodes in all_faces_nodes:
                    if k.issubset(nodes):
                        key_to_fid[k] = fid
                        break
        '''

        iface_fids = sorted(key_to_fid.values())
        if not iface_fids:
            raise RuntimeError("Could not resolve any interface face IDs.")

        # 7) Build a FACE BLOCK
        part = " ".join(map(str, iface_fids))
        cubit.cmd(f"block {block_face} add face {part}")
        cubit.cmd(f"block {block_face} element type QUAD4")

#------- Helper functions for DRM---------#
# create blocks for DRM
def make_block(block_id,block_inp, ids, tmpname):
        #cubit.cmd(f"group '{tmpname}' delete")
        step = 5000
        ids = sorted(ids)
        for k in range(0, len(ids), step):
            part = " ".join(map(str, ids[k:k+step]))
            cubit.cmd(f"group '{tmpname}' add hex {part}")
        #cubit.cmd(f"block {block_id} delete")
        ge = cubit.get_id_from_name(tmpname)
        e1 = cubit.get_group_hexes(ge)
        command = "block " + str(block_inp) + " remove hex " + ' '.join(str(x) for x in e1)
        cubit.cmd(command)
        command = "block " + str(block_id) + " add hex " + ' '.join(str(x) for x in e1)
        cubit.cmd(command)
        
def hex_nodes(eid):
    return cubit.get_connectivity("hex", eid)

def hex_local_faces_hex8(n):
    # six faces as 4-corner tuples (HEX8 order)
    return [
        (n[0], n[1], n[2], n[3]),
        (n[4], n[5], n[6], n[7]),
        (n[0], n[1], n[5], n[4]),
        (n[1], n[2], n[6], n[5]),
        (n[2], n[3], n[7], n[6]),
        (n[3], n[0], n[4], n[7]),
    ]

def get_hexes_in_block(bid):
    return list(cubit.parse_cubit_list("hex", f"in block {bid}"))

def build_node_to_faces_index():
    from collections import defaultdict
    """node_id -> set(face_ids) for all existing faces"""
    node2faces = defaultdict(set)
    all_faces = cubit.parse_cubit_list("face", "all")
    for fid in all_faces:
        for nid in cubit.get_connectivity("face", fid):
            node2faces[nid].add(fid)
    return node2faces

def find_existing_face_by_four_nodes(node2faces, n1, n2, n3, n4):
    """Return an existing face id that contains all 4 nodes, or None."""
    s1 = node2faces.get(n1); s2 = node2faces.get(n2)
    if not s1 or not s2: return None
    cand = s1 & s2
    if not cand: return None
    s3 = node2faces.get(n3); s4 = node2faces.get(n4)
    if not s3 or not s4: return None
    cand &= s3; 
    if not cand: return None
    cand &= s4
    # pick any (should be unique)
    return next(iter(cand)) if cand else None

# --- geometry helpers ---
def seg_point_dist2(px, py, x1, y1, x2, y2):
    vx, vy = x2-x1, y2-y1
    wx, wy = px-x1, py-y1
    vv = vx*vx + vy*vy
    if vv <= 0.0:
        return (wx*wx + wy*wy), 0.0
    t = (wx*vx + wy*vy) / vv
    t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
    dx, dy = (x1 + t*vx) - px, (y1 + t*vy) - py
    return dx*dx + dy*dy, t

def precompute_edges(poly):
    import math
    L = [0.0]
    for i in range(len(poly)):
        x1,y1 = poly[i]
        x2,y2 = poly[(i+1) % len(poly)]
        L.append(L[-1] + math.hypot(x2-x1, y2-y1))
    return L

def nearest_edge_raw(px, py, poly, Lcum):
    import math
    best = (float("inf"), 0.0, -1, 0.0)  # (d2, s, seg_i, t)
    for i in range(len(poly)):
        x1,y1 = poly[i]; x2,y2 = poly[(i+1)%len(poly)]
        d2, t = seg_point_dist2(px, py, x1,y1, x2,y2)
        if d2 < best[0]:
            s = Lcum[i] + t*(Lcum[i+1]-Lcum[i])
            best = (d2, s, i, t)
    d2, s, seg_i, t = best
    return math.sqrt(d2), s, seg_i, t

def hxy_bbox(eid):
    nids = cubit.get_connectivity("hex", eid)
    xs, ys = [], []
    for nid in nids:
        x,y,_ = cubit.get_nodal_coordinates(nid)
        xs.append(x); ys.append(y)
    dx = max(xs)-min(xs); dy = max(ys)-min(ys)
    return max(1e-9, min(dx, dy))

# defines if the centroid of the hex is inside the poly
def point_in_poly(px, py, poly, include_boundary=True, eps=1e-12):
    inside = False
    n = len(poly)
    for i in range(n):
        x1,y1 = poly[i]; x2,y2 = poly[(i+1)%n]

        if include_boundary:
            cross = (x2 - x1)*(py - y1) - (y2 - y1)*(px - x1)
            if abs(cross) <= eps * max(1.0, abs(x2 - x1), abs(y2 - y1)):
                if (min(x1, x2) - eps <= px <= max(x1, x2) + eps and
                    min(y1, y2) - eps <= py <= max(y1, y2) + eps):
                    return True

        if ((y1 > py) != (y2 > py)):
            xints = x1 + (py - y1) * (x2 - x1) / (y2 - y1)
            if px < xints:
                inside = not inside
    return inside

def estimate_hz_from_nodes(eids, min_valid=1e-9):
    import statistics
    hz_vals = []
    for eid in eids:
        nids = cubit.get_connectivity("hex", eid)
        zs = [cubit.get_nodal_coordinates(n)[2] for n in nids]
        hz_e = max(zs) - min(zs)  # element's vertical thickness
        if hz_e > min_valid:
            hz_vals.append(hz_e)
    if not hz_vals:
        return 1.0
    return statistics.median(hz_vals)


