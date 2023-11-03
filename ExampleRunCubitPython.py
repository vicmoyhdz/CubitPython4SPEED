import cubit
cubit.init(['cubit','-nojournal'])
cubit.reset()

cubit.cmd('cd \'C:\\Users\\vmh5\\Documents\\CubitPython\'')
f="Files\TopoExample.cfg"
#f="Files\RegmeshExample.cfg"
#f="Files\TopoSurfaceExample.cfg"
#f="Files\TopoSurfaceLSExample.cfg"
#f="Files\TopoGeoExample.cfg"

import main
main.volumes(f)
