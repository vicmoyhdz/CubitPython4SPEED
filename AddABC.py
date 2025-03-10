def add_faces(xmin,ymin,rot_deg,coord):
    from utilities import DoRotation
    import numpy
    import cubit
    cubit.cmd('set info on')
    cubit.cmd('set echo off')
    cubit.cmd('set journal off')
    tol=2
    all_faces=cubit.parse_cubit_list('face','all')
    face_list = tuple(sorted(all_faces))
    listFace=[]
    pvx=[]
    pvy=[]
    for face in face_list:
        pv = cubit.get_center_point("face", face)
        pvx.append(pv[0])
        pvy.append(pv[1])
    print('finished face list')
        
    x_rotated,y_rotated = DoRotation(xmin,ymin,numpy.array(pvx), numpy.array(pvy), -1*rot_deg)
    print('finished rotations')
    y_list=list(y_rotated[0])

    index=0
    for y in y_list:      
        if coord-tol < y < coord+tol :
            listFace.append(face_list[index])
        index=index+1

    
    command = "block 100 add face " + ' '.join(str(x) for x in listFace)
    cubit.cmd(command)
    print('finished')
    

